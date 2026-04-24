"""
SimplerEnv evaluation with trained DINOv2 switch head for rescue detection.

Drop-in replacement for eval_with_gemini_rescue_dreamdojo_groot.py.
The ONLY change: Gemini value scoring (~2-5s latency per query, requires API)
is replaced by the trained switch head (~2ms per frame, runs locally on GPU).

DreamDojo + Gemini is still used for rescue action *selection* (choosing the
best among SDE candidates).  To run without DreamDojo entirely, set
--dd_base_port 0 and rescue falls back to a single SDE resample.

Latency comparison (per frame):
  Gemini value scoring : ~2000-5000ms async  (network round-trip + VLM inference)
  Switch head (cached) : ~2-4ms sync         (1x DINOv2 encode + feature queue)

Usage:
    # Start DreamDojo servers first (same as before):
    CUDA_VISIBLE_DEVICES=0 python dreamdojo_server.py --port 8020 ...
    CUDA_VISIBLE_DEVICES=1 python dreamdojo_server.py --port 8021 ...

    # With DreamDojo candidate selection (recommended)
    python eval_with_switch_head_rescue_dreamdojo_groot.py \
        --model_path nvidia/GR00T-N1.6-3B \
        --switch_head_ckpt checkpoints/switch_head/best_model.pt \
        --robot_type google \
        --dd_base_port 8020

    # Without DreamDojo (SDE resampling only — no GOOGLE_API_KEY needed)
    python eval_with_switch_head_rescue_dreamdojo_groot.py \
        --model_path nvidia/GR00T-N1.6-3B \
        --switch_head_ckpt checkpoints/switch_head/best_model.pt \
        --robot_type google \
        --dd_base_port 0
"""

import collections
import dataclasses
import json
import logging
import os
import pathlib
import time

import imageio
import numpy as np
import torch
import tqdm
import tyro

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.gr00t_n1d6.gr00t_n1d6_sde import Gr00tN1d6SDEActionHead
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

# Reuse helpers from existing scripts (avoid duplication)
from eval_with_gemini_rescue_dreamdojo_groot import (
    _build_groot_obs_google,
    _build_groot_obs_widowx,
    _extract_image_from_obs,
    _convert_to_simpler_actions,
    _rescue_select_action,
    GOOGLE_FRACTAL_TASKS,
    WIDOWX_BRIDGE_TASKS,
    ACTION_KEYS,
    ROLLOUT_FPS,
)
from train_switch_head_gemini import _extract_state_from_obs
from switch_head_inference import load_switch_head


# ============================================================================
#  Args
# ============================================================================

@dataclasses.dataclass
class Args:
    # -- GR00T policy --
    model_path: str = "nvidia/GR00T-N1.6-3B"
    device: int = 0
    noise_level: float = 0.3
    """SDE diffusion noise strength (only used during rescue sampling)."""
    num_inference_timesteps: int | None = None
    """Number of denoising steps for SDE (None uses model default, only during rescue)."""
    action_horizon: int = 1
    """Number of actions to execute per normal policy query."""
    rescue_action_horizon: int = 8
    """Number of actions in each rescue candidate chunk for DreamDojo generation."""

    # -- Switch head (replaces Gemini value scoring) --
    switch_head_ckpt: str = ""
    """Path to trained switch head checkpoint (.pt)."""
    dinov2_model: str = "dinov2_vitb14"
    """DINOv2 backbone variant (must match training)."""
    clip_len: int = 20
    """Feature queue length (must match training clip_len)."""
    query_interval: int = 5
    """Run classifier every N frames (5 = 0.5s at 10Hz)."""
    rescue_threshold: float = 0.5
    """Rescue probability threshold. Use 0.5 for hard-label models,
    ~0.6 for soft-label models (since target was 1 - gemini_score,
    and RESCUE_SCORE_ABSOLUTE was 0.40 → 1-0.40 = 0.60)."""
    score_spike_threshold: float = 0.15
    """Trigger rescue if score spikes by this much (mirrors RESCUE_SCORE_DROP)."""

    # -- Environment --
    robot_type: str = "google"
    num_trials_per_task: int = 20
    max_episode_steps: int = 120

    # -- DreamDojo rescue selection --
    dd_base_port: int = 8020
    """DreamDojo base port. Set to 0 to disable DreamDojo (SDE-only rescue)."""
    num_rescue_samples: int = 4

    # -- Output --
    video_out_path: str = ""
    seed: int = 7
    skip_tasks: int = 0


# ============================================================================
#  Eval loop
# ============================================================================

def eval_simpler_env(args: Args) -> None:
    os.environ.setdefault("DISPLAY", "")

    import gymnasium as gym
    from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs

    np.random.seed(args.seed)
    register_simpler_envs()

    # ---- Environment setup ----
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
        logging.info(f"Skipping first {args.skip_tasks} tasks")
        task_names = task_names[args.skip_tasks:]

    if not args.video_out_path:
        args.video_out_path = f"data/switch_head_rescue_{args.robot_type}"
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # ---- Load GR00T policy (ODE normal + SDE rescue) ----
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
    if args.num_inference_timesteps is not None:
        sde_config.num_inference_timesteps = args.num_inference_timesteps
    sde_action_head = Gr00tN1d6SDEActionHead(sde_config)
    sde_action_head.load_state_dict(ode_action_head.state_dict())
    sde_action_head.to(device=args.device, dtype=torch.bfloat16)
    sde_action_head.eval()
    logging.info("GR00T policy loaded (ODE normal + SDE rescue).")

    # ---- Load switch head (replaces Gemini value scoring) ----
    assert args.switch_head_ckpt, "--switch_head_ckpt is required"
    device_str = f"cuda:{args.device}" if isinstance(args.device, int) else str(args.device)
    switch = load_switch_head(
        args.switch_head_ckpt,
        device=device_str,
        dinov2_model=args.dinov2_model,
        state_dim=8,
        num_cameras=1,
        clip_len=args.clip_len,
        query_interval=args.query_interval,
        rescue_threshold=args.rescue_threshold,
        score_spike_threshold=args.score_spike_threshold,
    )
    logging.info(
        f"Switch head loaded: query every {args.query_interval} frames, "
        f"threshold={args.rescue_threshold:.2f}"
    )

    use_dreamdojo = args.dd_base_port > 0
    if use_dreamdojo:
        logging.info(f"DreamDojo enabled (base port {args.dd_base_port})")
    else:
        logging.info("DreamDojo disabled — using single SDE resample for rescue")

    # ---- Rollout ----
    total_episodes, total_successes = 0, 0

    for task_name in tqdm.tqdm(task_names, desc="tasks"):
        env_id = f"{env_prefix}/{task_name}"
        logging.info(f"\n=== Task: {env_id} ===")

        task_episodes, task_successes = 0, 0

        for trial_idx in tqdm.tqdm(range(args.num_trials_per_task), desc="episodes", leave=False):
            task_segment = task_name.replace(" ", "_")

            # Skip if already completed
            existing_dirs = [
                d for d in pathlib.Path(args.video_out_path).glob(
                    f"rollout_{task_segment}_ep{trial_idx}_*"
                )
                if d.name.endswith("_success") or d.name.endswith("_failure")
            ]
            if existing_dirs:
                logging.info(f"Skip: {task_segment} (Episode {trial_idx})")
                if "success" in existing_dirs[0].name:
                    task_successes += 1
                    total_successes += 1
                task_episodes += 1
                total_episodes += 1
                continue

            rollout_dir = (
                pathlib.Path(args.video_out_path)
                / f"rollout_{task_segment}_ep{trial_idx}_running"
            )
            rollout_dir.mkdir(parents=True, exist_ok=True)

            env = gym.make(env_id)
            obs, info = env.reset(seed=args.seed + trial_idx)

            task_description = str(obs.get(
                "annotation.human.action.task_description",
                task_name,
            ))
            logging.info(f"\nTask: {task_description}")

            action_plan = collections.deque()
            clean_images = []
            rescue_log = []
            rescue_selections = []

            # Reset switch head state for new episode
            switch.reset()

            done, truncated = False, False

            logging.info(f"Starting episode {task_episodes + 1}...")
            for t in range(args.max_episode_steps):
                if done or truncated:
                    break

                try:
                    img = _extract_image_from_obs(obs, args.robot_type)
                    state_vec = _extract_state_from_obs(obs, args.robot_type)
                    clean_images.append(img.copy())
                    num_frames = len(clean_images)

                    # ---- Switch head: encode every frame (~2ms) ----
                    # Classifier runs every query_interval frames automatically
                    result = switch.step([img], state_vec)

                    if result.score is not None:
                        logging.info(
                            f"[SwitchHead] frame={num_frames} "
                            f"rescue_prob={result.score:.3f} "
                            f"rescue={result.rescue}"
                        )

                    # ---- Replan when action queue is empty ----
                    if not action_plan:
                        if result.rescue:
                            logging.info(f"[Rescue] Activating at frame {num_frames}...")
                            rescue_log.append(num_frames)

                            if use_dreamdojo:
                                # Full pipeline: SDE + DreamDojo + Gemini selection
                                from datetime import datetime as _dt
                                _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
                                step_save_dir = (
                                    rollout_dir / "rescue_steps"
                                    / f"{task_segment}_ep{trial_idx}_frame{num_frames}_{_ts}"
                                )
                                best_actions, sel_record = _rescue_select_action(
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
                                sel_record["frame"] = num_frames
                                rescue_selections.append(sel_record)
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
                            # Normal GR00T ODE inference (deterministic)
                            groot_obs = build_groot_obs_fn(obs)
                            action_dict, _ = groot_policy.get_action(groot_obs)
                            action_list = _convert_to_simpler_actions(
                                action_dict, args.action_horizon,
                            )
                            action_plan.extend(action_list)

                    action = action_plan.popleft()
                    obs, reward, done, truncated, info = env.step(action)

                except Exception as e:
                    logging.error(f"Step exception: {e}")
                    break

            # ---- Episode summary ----
            success = done
            if success:
                task_successes += 1
                total_successes += 1
            task_episodes += 1
            total_episodes += 1

            suffix = "success" if success else "failure"
            final_rollout_dir = (
                pathlib.Path(args.video_out_path)
                / f"rollout_{task_segment}_ep{trial_idx}_{suffix}"
            )
            rollout_dir.rename(final_rollout_dir)
            rollout_dir = final_rollout_dir

            # Save switch head log (replaces gemini_values.txt / gemini_results.json)
            with open(rollout_dir / "switch_head_log.json", "w") as f:
                json.dump({
                    "task": task_description,
                    "episode_index": trial_idx,
                    "outcome": suffix,
                    "num_steps": len(clean_images),
                    "rescue_activations": rescue_log,
                    "num_rescues": len(rescue_log),
                    "rescue_selections": rescue_selections,
                    "score_history": [
                        {"frame": fr, "rescue_prob": round(sc, 4)}
                        for fr, sc in switch._score_history
                    ],
                    "config": {
                        "switch_head_ckpt": args.switch_head_ckpt,
                        "rescue_threshold": args.rescue_threshold,
                        "query_interval": args.query_interval,
                        "clip_len": args.clip_len,
                    },
                }, f, indent=2)

            if clean_images:
                imageio.mimwrite(
                    rollout_dir / "complete_video.mp4",
                    [np.asarray(x) for x in clean_images],
                    fps=ROLLOUT_FPS,
                )

            env.close()

            logging.info(f"Success: {success}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(
                f"# successes: {total_successes} "
                f"({total_successes / total_episodes * 100:.1f}%)"
            )

        logging.info(
            f"Current task success rate: "
            f"{float(task_successes) / float(max(task_episodes, 1))}"
        )
        logging.info(
            f"Current total success rate: "
            f"{float(total_successes) / float(max(total_episodes, 1))}"
        )

    logging.info(
        f"Total success rate: "
        f"{float(total_successes) / float(max(total_episodes, 1))}"
    )
    logging.info(f"Total episodes: {total_episodes}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    eval_simpler_env(tyro.cli(Args))
