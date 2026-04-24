"""
Real-time switch head inference with DINOv2 feature caching.

Core idea: instead of encoding T=20 frames every step (~30ms), cache one
DINOv2 feature per frame in a FIFO queue and mean-pool on demand (~2ms).

Usage patterns
--------------

1. Drop-in integration (replace Gemini value scoring in existing eval):

    from switch_head_inference import load_switch_head

    switch = load_switch_head("checkpoints/switch_head/best_model.pt")

    for t in range(max_steps):
        img = extract_image(obs)
        state = extract_state(obs)
        result = switch.step([img], state)          # ~2ms

        if not action_plan:
            if result.rescue:                       # replaces _check_rescue_needed()
                ...  # SDE rescue
            else:
                ...  # normal ODE

2. Standalone eval (replaces Gemini value scoring with trained switch head):

    python switch_head_inference.py eval \
        --model_path nvidia/GR00T-N1.6-3B \
        --switch_head_ckpt checkpoints/switch_head/best_model.pt \
        --robot_type google \
        --dd_base_port 8020

3. Latency benchmark:

    python switch_head_inference.py benchmark \
        --switch_head_ckpt checkpoints/switch_head/best_model.pt
"""

import argparse
import collections
import logging
import os
import pathlib
import time
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from train_switch_head_gemini import (
    DINOv2SwitchHead,
    GOOGLE_FRACTAL_TASKS,
    WIDOWX_BRIDGE_TASKS,
    ACTION_KEYS,
    ROLLOUT_FPS,
    _build_groot_obs_google,
    _build_groot_obs_widowx,
    _extract_image_from_obs,
    _extract_state_from_obs,
    _convert_to_simpler_actions,
)


# ============================================================================
#  Inference class
# ============================================================================

class StepResult(NamedTuple):
    """Returned by SwitchHeadInference.step() every frame."""
    rescue: bool          # whether rescue should be triggered right now
    score: float | None   # rescue probability (None if not queried this frame)
    frame_count: int      # total frames seen since last reset


class SwitchHeadInference:
    """
    Feature-cached switch head for real-time rescue detection.

    Latency breakdown (FP16, V100):
      - DINOv2 encode 1 frame:   ~2-4ms   (every frame)
      - Temporal mean pool:       ~0.01ms  (every query_interval frames)
      - MLP classifier:           ~0.1ms   (every query_interval frames)
      - Total per VLA step:       ~2-4ms   (< 2% of GR00T's ~150ms)

    The rescue decision uses two conditions (matching Gemini-based logic):
      1. Absolute threshold: rescue_prob > rescue_threshold
      2. Score spike:        rescue_prob increased by > score_spike_threshold
         compared to a prior reading (sudden deterioration)
    """

    def __init__(
        self,
        model: DINOv2SwitchHead,
        *,
        clip_len: int = 20,
        query_interval: int = 5,
        rescue_threshold: float = 0.5,
        score_spike_threshold: float = 0.15,
        device: torch.device | str = "cuda",
    ):
        self.model = model
        self.model.eval()
        self.clip_len = clip_len
        self.query_interval = query_interval
        self.rescue_threshold = rescue_threshold
        self.score_spike_threshold = score_spike_threshold
        self.device = torch.device(device)

        self.num_cameras = model.num_cameras
        self.feature_dim = model.feature_dim

        # Per-camera feature queues
        self._feat_queues: list[collections.deque] = []
        # Score history: list of (frame_idx, rescue_prob)
        self._score_history: list[tuple[int, float]] = []
        self._frame_count = 0
        self._latest_rescue = False

        self.reset()

    def reset(self):
        """Call at the start of each episode to clear cached state."""
        self._feat_queues = [
            collections.deque(maxlen=self.clip_len)
            for _ in range(self.num_cameras)
        ]
        self._score_history = []
        self._frame_count = 0
        self._latest_rescue = False

    @torch.no_grad()
    def _encode_single_image(self, img_np: np.ndarray) -> torch.Tensor:
        """
        Encode one (H, W, 3) uint8 image → (1, feature_dim) feature vector.
        Uses the model's frozen DINOv2 backbone.
        """
        # (H, W, 3) uint8 → (1, 3, H, W) float [0, 1]
        img = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img = img.to(self.device)

        img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
        img = (img - self.model.img_mean) / self.model.img_std
        feat = self.model.backbone(img)  # (1, feature_dim)
        return feat

    @torch.no_grad()
    def _compute_score(self, state_np: np.ndarray) -> float:
        """
        Mean-pool cached features across time for each camera, concat with
        state, and run the MLP classifier.  Returns rescue probability [0, 1].
        """
        pooled_feats = []
        for queue in self._feat_queues:
            if len(queue) == 0:
                pooled_feats.append(torch.zeros(1, self.feature_dim, device=self.device))
            else:
                stacked = torch.cat(list(queue), dim=0)  # (T', feature_dim)
                pooled_feats.append(stacked.mean(dim=0, keepdim=True))  # (1, feature_dim)

        state = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0).to(self.device)
        combined = torch.cat(pooled_feats + [state], dim=-1)  # (1, num_cameras*D + state_dim)
        logit = self.model.classifier(combined).squeeze(-1)   # (1,)
        return torch.sigmoid(logit).item()

    def _check_rescue(self) -> bool:
        """Check rescue conditions against score history."""
        if not self._score_history:
            return False

        _, latest_prob = self._score_history[-1]

        # Condition 1: absolute threshold
        if latest_prob >= self.rescue_threshold:
            return True

        # Condition 2: sudden spike in rescue probability
        if len(self._score_history) >= 2:
            _, prev_prob = self._score_history[-2]
            if (latest_prob - prev_prob) >= self.score_spike_threshold:
                return True

        return False

    def step(
        self,
        images: list[np.ndarray],
        state: np.ndarray,
    ) -> StepResult:
        """
        Process one frame.  Call this every environment step.

        Args:
            images: list of (H, W, 3) uint8 arrays, one per camera.
            state:  (state_dim,) float32 robot state vector.

        Returns:
            StepResult with rescue decision, score, and frame count.
        """
        assert len(images) == self.num_cameras, (
            f"Expected {self.num_cameras} images, got {len(images)}"
        )

        # 1. Encode each camera's frame and push to feature queue
        for cam_idx, img_np in enumerate(images):
            feat = self._encode_single_image(img_np)
            self._feat_queues[cam_idx].append(feat)

        self._frame_count += 1
        score = None

        # 2. Run classifier at query_interval
        if self._frame_count % self.query_interval == 0:
            score = self._compute_score(state)
            self._score_history.append((self._frame_count, score))
            self._latest_rescue = self._check_rescue()

            if self._latest_rescue:
                logging.info(
                    f"[SwitchHead] Rescue triggered at frame {self._frame_count}: "
                    f"score={score:.3f} (threshold={self.rescue_threshold})"
                )

        return StepResult(
            rescue=self._latest_rescue,
            score=score,
            frame_count=self._frame_count,
        )

    @property
    def latest_score(self) -> float | None:
        """Most recent rescue probability, or None if never queried."""
        if self._score_history:
            return self._score_history[-1][1]
        return None

    def should_rescue(self) -> bool:
        """Current rescue decision (updated every query_interval frames)."""
        return self._latest_rescue


# ============================================================================
#  Loading utility
# ============================================================================

def load_switch_head(
    checkpoint_path: str,
    *,
    device: str = "cuda",
    dinov2_model: str = "dinov2_vitb14",
    hidden_dim: int = 256,
    state_dim: int = 8,
    num_cameras: int = 1,
    freeze_backbone: bool = True,
    clip_len: int = 20,
    query_interval: int = 5,
    rescue_threshold: float = 0.5,
    score_spike_threshold: float = 0.15,
) -> SwitchHeadInference:
    """
    Load a trained switch head checkpoint and wrap it for inference.

    Returns a ready-to-use SwitchHeadInference instance.
    """
    model = DINOv2SwitchHead(
        dinov2_model=dinov2_model,
        hidden_dim=hidden_dim,
        state_dim=state_dim,
        num_cameras=num_cameras,
        freeze_backbone=freeze_backbone,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    logging.info(
        f"Loaded switch head from {checkpoint_path} "
        f"(dinov2={dinov2_model}, cameras={num_cameras}, state_dim={state_dim})"
    )

    return SwitchHeadInference(
        model,
        clip_len=clip_len,
        query_interval=query_interval,
        rescue_threshold=rescue_threshold,
        score_spike_threshold=score_spike_threshold,
        device=device,
    )


# ============================================================================
#  Eval: SimplerEnv rollout with switch head rescue
# ============================================================================

def eval_with_switch_head(args):
    """
    Run SimplerEnv evaluation using the trained switch head for rescue detection.

    When rescue is triggered, switches GR00T from ODE (deterministic) to SDE
    (stochastic) sampling.  If --dd_base_port is set, also uses DreamDojo +
    Gemini for candidate selection (full pipeline).  Otherwise, uses a single
    SDE resample (lightweight mode).
    """
    os.environ.setdefault("DISPLAY", "")

    import gymnasium as gym
    import imageio
    import tqdm
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model.gr00t_n1d7.gr00t_n1d6_sde import Gr00tN1d6SDEActionHead
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    np.random.seed(args.seed)

    # -- Register envs --
    from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs
    register_simpler_envs()

    if args.robot_type == "google":
        task_names = GOOGLE_FRACTAL_TASKS
        env_prefix = "simpler_env_google"
        build_groot_obs_fn = _build_groot_obs_google
        embodiment_tag = EmbodimentTag.OXE_GOOGLE
    elif args.robot_type == "widowx":
        task_names = WIDOWX_BRIDGE_TASKS
        env_prefix = "simpler_env_widowx"
        build_groot_obs_fn = _build_groot_obs_widowx
        embodiment_tag = EmbodimentTag.OXE_WIDOWX
    else:
        raise ValueError(f"Unknown robot_type: {args.robot_type}")

    if args.skip_tasks > 0:
        task_names = task_names[args.skip_tasks:]

    output_dir = pathlib.Path(args.video_out_path or f"data/switch_head_eval_{args.robot_type}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load GR00T policy (ODE normal + SDE rescue) --
    logging.info(f"Loading GR00T policy from {args.model_path} ...")
    base_policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device=args.device,
        strict=True,
    )
    groot_policy = Gr00tSimPolicyWrapper(base_policy)
    ode_action_head = base_policy.model.action_head

    sde_config = base_policy.model.config
    sde_config.noise_level = args.noise_level
    sde_config.noise_method = "flow_sde"
    sde_action_head = Gr00tN1d6SDEActionHead(sde_config)
    sde_action_head.load_state_dict(ode_action_head.state_dict())
    sde_action_head.to(device=args.device, dtype=torch.bfloat16)
    sde_action_head.eval()
    logging.info("GR00T policy loaded (ODE normal + SDE rescue).")

    # -- Load switch head --
    switch = load_switch_head(
        args.switch_head_ckpt,
        device=f"cuda:{args.device}" if isinstance(args.device, int) else args.device,
        dinov2_model=args.dinov2_model,
        state_dim=8,
        num_cameras=1,
        clip_len=args.clip_len,
        query_interval=args.query_interval,
        rescue_threshold=args.rescue_threshold,
    )
    logging.info(
        f"Switch head loaded (query every {args.query_interval} frames, "
        f"threshold={args.rescue_threshold})"
    )

    # -- Optional: DreamDojo rescue (import from existing eval script) --
    use_dreamdojo = args.dd_base_port > 0
    if use_dreamdojo:
        from eval_with_gemini_rescue_dreamdojo_groot import _rescue_select_action
        logging.info(f"DreamDojo enabled (base port {args.dd_base_port})")
    else:
        logging.info("DreamDojo disabled — using SDE resampling only for rescue")

    # -- Rollout loop --
    total_episodes, total_successes = 0, 0

    for task_name in tqdm.tqdm(task_names, desc="tasks"):
        env_id = f"{env_prefix}/{task_name}"
        logging.info(f"\n{'='*60}\nTask: {env_id}\n{'='*60}")

        task_episodes, task_successes = 0, 0

        for trial_idx in tqdm.tqdm(range(args.num_trials_per_task), desc="episodes", leave=False):
            task_segment = task_name.replace(" ", "_")

            # Skip if already completed
            existing = [
                d for d in output_dir.glob(f"rollout_{task_segment}_ep{trial_idx}_*")
                if d.name.endswith("_success") or d.name.endswith("_failure")
            ]
            if existing:
                logging.info(f"  Skip: {task_segment} ep {trial_idx}")
                if "success" in existing[0].name:
                    task_successes += 1
                    total_successes += 1
                task_episodes += 1
                total_episodes += 1
                continue

            rollout_dir = output_dir / f"rollout_{task_segment}_ep{trial_idx}_running"
            rollout_dir.mkdir(parents=True, exist_ok=True)

            env = gym.make(env_id)
            obs, info = env.reset(seed=args.seed + trial_idx)

            task_description = str(obs.get(
                "annotation.human.action.task_description", task_name,
            ))

            action_plan = collections.deque()
            clean_images = []
            rescue_log = []

            switch.reset()
            done, truncated = False, False

            logging.info(f"  Episode {trial_idx+1}: {task_description}")

            for t in range(args.max_episode_steps):
                if done or truncated:
                    break

                try:
                    img = _extract_image_from_obs(obs, args.robot_type)
                    state_vec = _extract_state_from_obs(obs, args.robot_type)
                    clean_images.append(img.copy())

                    # -- Switch head: encode every frame, query every N frames --
                    result = switch.step([img], state_vec)

                    # -- Replan when action queue is empty --
                    if not action_plan:
                        if result.rescue:
                            rescue_log.append(len(clean_images))
                            logging.info(
                                f"    [Rescue] frame={len(clean_images)} "
                                f"score={switch.latest_score:.3f}"
                            )

                            if use_dreamdojo:
                                # Full pipeline: SDE + DreamDojo + Gemini selection
                                step_save_dir = (
                                    rollout_dir / "rescue_steps"
                                    / f"frame{len(clean_images)}"
                                )
                                best_actions, _ = _rescue_select_action(
                                    obs=obs,
                                    replay_images_for_history=clean_images,
                                    task_description=task_description,
                                    groot_policy=groot_policy,
                                    base_policy=base_policy,
                                    sde_action_head=sde_action_head,
                                    ode_action_head=ode_action_head,
                                    build_groot_obs_fn=build_groot_obs_fn,
                                    replan_steps=args.rescue_action_horizon,
                                    step_save_dir=step_save_dir,
                                    dd_base_port=args.dd_base_port,
                                    robot_type=args.robot_type,
                                    num_samples=args.num_rescue_samples,
                                )
                                action_plan.extend(best_actions)
                            else:
                                # Lightweight: single SDE resample
                                base_policy.model.action_head = sde_action_head
                                seed_i = int(time.time() * 1e6) % (2**31)
                                torch.manual_seed(seed_i)
                                torch.cuda.manual_seed(seed_i)
                                groot_obs = build_groot_obs_fn(obs)
                                action_dict, _ = groot_policy.get_action(groot_obs)
                                action_list = _convert_to_simpler_actions(
                                    action_dict, args.rescue_action_horizon,
                                )
                                action_plan.extend(action_list)
                                base_policy.model.action_head = ode_action_head
                        else:
                            # Normal ODE inference
                            groot_obs = build_groot_obs_fn(obs)
                            action_dict, _ = groot_policy.get_action(groot_obs)
                            action_list = _convert_to_simpler_actions(
                                action_dict, args.action_horizon,
                            )
                            action_plan.extend(action_list)

                    action = action_plan.popleft()
                    obs, reward, done, truncated, info = env.step(action)

                except Exception as e:
                    logging.error(f"  Step exception: {e}")
                    break

            success = done
            if success:
                task_successes += 1
                total_successes += 1
            task_episodes += 1
            total_episodes += 1

            suffix = "success" if success else "failure"
            final_dir = output_dir / f"rollout_{task_segment}_ep{trial_idx}_{suffix}"
            rollout_dir.rename(final_dir)

            if clean_images:
                imageio.mimwrite(
                    str(final_dir / "complete_video.mp4"),
                    [np.asarray(x) for x in clean_images],
                    fps=ROLLOUT_FPS,
                )

            # Save rescue metadata
            import json
            with open(final_dir / "switch_head_log.json", "w") as f:
                json.dump({
                    "task": task_description,
                    "episode_idx": trial_idx,
                    "success": success,
                    "num_steps": len(clean_images),
                    "rescue_frames": rescue_log,
                    "num_rescues": len(rescue_log),
                    "score_history": [
                        {"frame": fr, "score": round(sc, 4)}
                        for fr, sc in switch._score_history
                    ],
                }, f, indent=2)

            env.close()

            logging.info(
                f"  -> {suffix.upper()} | steps={len(clean_images)} "
                f"rescues={len(rescue_log)}"
            )

        rate = task_successes / max(task_episodes, 1)
        logging.info(f"Task success rate: {rate:.1%} ({task_successes}/{task_episodes})")

    rate = total_successes / max(total_episodes, 1)
    logging.info(f"\nTotal success rate: {rate:.1%} ({total_successes}/{total_episodes})")


# ============================================================================
#  Benchmark
# ============================================================================

def benchmark(args):
    """Measure per-step latency of the switch head with feature caching."""
    device = f"cuda:{args.device}" if isinstance(args.device, int) else args.device
    switch = load_switch_head(
        args.switch_head_ckpt,
        device=device,
        dinov2_model=args.dinov2_model,
        clip_len=args.clip_len,
        query_interval=args.query_interval,
    )

    # Warmup
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    dummy_state = np.zeros(8, dtype=np.float32)
    for _ in range(10):
        switch.step([dummy_img], dummy_state)
    switch.reset()

    # Synchronize before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    num_frames = args.benchmark_frames
    timings_encode = []
    timings_query = []

    for i in range(num_frames):
        dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        result = switch.step([dummy_img], dummy_state)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000
        if result.score is not None:
            timings_query.append(elapsed_ms)
        else:
            timings_encode.append(elapsed_ms)

    def _stats(vals, label):
        if not vals:
            return
        arr = np.array(vals)
        logging.info(
            f"  {label}: mean={arr.mean():.2f}ms  "
            f"median={np.median(arr):.2f}ms  "
            f"p95={np.percentile(arr, 95):.2f}ms  "
            f"p99={np.percentile(arr, 99):.2f}ms  "
            f"(n={len(arr)})"
        )

    logging.info(f"\n{'='*60}")
    logging.info(f"Switch Head Latency Benchmark ({num_frames} frames)")
    logging.info(f"  Device: {device}")
    logging.info(f"  Backbone: {args.dinov2_model}")
    logging.info(f"  clip_len={args.clip_len}, query_interval={args.query_interval}")
    logging.info(f"{'='*60}")
    _stats(timings_encode, "Encode-only frames (DINOv2 only)     ")
    _stats(timings_query,  "Query frames (DINOv2 + pool + MLP)   ")
    _stats(timings_encode + timings_query, "All frames (overall)                 ")
    logging.info(f"{'='*60}")

    # Compare to naive approach (encode T frames every step)
    logging.info(f"\nFor comparison — naive approach (encode {args.clip_len} frames per step):")
    switch.reset()
    # Fill queue first
    for _ in range(args.clip_len):
        switch.step([dummy_img], dummy_state)

    naive_times = []
    for _ in range(50):
        imgs = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                for _ in range(args.clip_len)]
        stacked = np.stack(imgs)  # (T, H, W, 3)
        img_t = torch.from_numpy(stacked).permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0
        img_t = img_t.to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            switch.model._encode_image(img_t)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        naive_times.append((t1 - t0) * 1000)

    _stats(naive_times, f"Naive (encode {args.clip_len} frames every step)")
    if timings_encode:
        speedup = np.mean(naive_times) / np.mean(timings_encode)
        logging.info(f"\n  Feature caching speedup: {speedup:.1f}x")


# ============================================================================
#  CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Switch head inference & eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- Shared switch head args --
    def _add_switch_args(p):
        p.add_argument("--switch_head_ckpt", type=str, required=True,
                        help="Path to trained switch head checkpoint (.pt)")
        p.add_argument("--dinov2_model", type=str, default="dinov2_vitb14",
                        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"])
        p.add_argument("--clip_len", type=int, default=20,
                        help="Feature queue length (match training clip_len)")
        p.add_argument("--query_interval", type=int, default=5,
                        help="Run classifier every N frames (default 5 = 0.5s at 10Hz)")
        p.add_argument("--rescue_threshold", type=float, default=0.5,
                        help="Rescue probability threshold (0.5 for hard labels, "
                        "0.6 for soft labels trained with 1-gemini_score)")
        p.add_argument("--device", type=int, default=0)

    # -- eval --
    p_eval = subparsers.add_parser("eval", help="Eval SimplerEnv with switch head rescue")
    _add_switch_args(p_eval)
    p_eval.add_argument("--model_path", type=str, default="nvidia/GR00T-N1.6-3B")
    p_eval.add_argument("--robot_type", type=str, default="google",
                        choices=["google", "widowx"])
    p_eval.add_argument("--num_trials_per_task", type=int, default=20)
    p_eval.add_argument("--max_episode_steps", type=int, default=120)
    p_eval.add_argument("--action_horizon", type=int, default=1)
    p_eval.add_argument("--rescue_action_horizon", type=int, default=8)
    p_eval.add_argument("--noise_level", type=float, default=0.3)
    p_eval.add_argument("--num_rescue_samples", type=int, default=4)
    p_eval.add_argument("--dd_base_port", type=int, default=0,
                        help="DreamDojo base port (0 = disabled, use SDE-only rescue)")
    p_eval.add_argument("--video_out_path", type=str, default="")
    p_eval.add_argument("--seed", type=int, default=7)
    p_eval.add_argument("--skip_tasks", type=int, default=0)

    # -- benchmark --
    p_bench = subparsers.add_parser("benchmark", help="Measure switch head latency")
    _add_switch_args(p_bench)
    p_bench.add_argument("--benchmark_frames", type=int, default=200,
                         help="Number of frames to benchmark")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.command == "eval":
        eval_with_switch_head(args)
    elif args.command == "benchmark":
        benchmark(args)


if __name__ == "__main__":
    main()
