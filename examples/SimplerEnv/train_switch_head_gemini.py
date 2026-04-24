"""
Two-phase pipeline for training a switch head on SimplerEnv data with Gemini labels.

Phase 1 – collect:
  Run SimplerEnv rollouts (Google Fractal / WidowX Bridge) with the GR00T policy,
  query Gemini for dense value scores every 2 seconds, and apply the rescue trigger
  logic to produce binary switch labels.  Each replanning step is saved as an .npz.

Phase 2 – train:
  Train a standalone DINOv2-based binary classifier on the collected data.
  Alternatively, the saved switch_label can be injected into a dataset for
  integrated training.

Usage
-----
  # 1) Collect labeled data (needs GOOGLE_API_KEY set)
  python train_switch_head_gemini.py collect \
      --model_path /path/to/groot/checkpoint \
      --robot_type google \
      --output_dir data/simpler_env_switch_labels \
      --num_trials_per_task 20

  # 2) Train standalone switch head (single GPU)
  python train_switch_head_gemini.py train \
      --data_dir data/simpler_env_switch_labels \
      --output_dir checkpoints/switch_head \
      --epochs 30

  # 2b) Multi-GPU training via torchrun (e.g. 8 GPUs)
  torchrun --nproc_per_node=8 train_switch_head_gemini.py train \
      --data_dir data/simpler_env_switch_labels \
      --output_dir checkpoints/switch_head \
      --batch_size 8 \
      --epochs 30

  # 3) (Optional) Export labels into LeRobot v2 dataset for integrated training
  python train_switch_head_gemini.py export \
      --data_dir data/simpler_env_switch_labels \
      --lerobot_dataset fractal20220817_data_lerobot \
      --output_dir fractal20220817_data_lerobot_with_switch
"""

import argparse
import collections
import concurrent.futures
import glob
import json
import logging
import os
import pathlib
import sys
import tempfile
import threading
import time

import imageio
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb
except ImportError:
    wandb = None

# ---------------------------------------------------------------------------
# Constants (match eval_with_gemini_rescue_dreamdojo_groot.py)
# ---------------------------------------------------------------------------
ROLLOUT_FPS = 10

GEMINI_QUERY_INTERVAL_FRAMES = 20
GEMINI_HISTORY_FRAMES = 200
GEMINI_VALUE_MODEL = "gemini-3.1-flash-lite-preview"

RESCUE_SCORE_ABSOLUTE = 0.40
RESCUE_SCORE_DROP = 0.15

GOOGLE_FRACTAL_TASKS = [
    "google_robot_pick_coke_can",
    "google_robot_pick_object",
    "google_robot_move_near",
    "google_robot_open_drawer",
    "google_robot_close_drawer",
]

WIDOWX_BRIDGE_TASKS = [
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
    "widowx_put_eggplant_in_sink",
    "widowx_open_drawer",
    "widowx_close_drawer",
]

ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

GOOGLE_STATE_KEYS = ["x", "y", "z", "rx", "ry", "rz", "rw", "gripper"]
WIDOWX_STATE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]


# ============================================================================
#  Phase 1: Data collection with Gemini labeling
# ============================================================================

_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(http_options={"api_version": "v1alpha"})
    return _gemini_client


def _query_gemini_value(frames: list, task_description: str, step_idx: int,
                        score_history: list, lock: threading.Lock) -> dict:
    """Query Gemini for a value score on a video clip. Thread-safe."""
    from google import genai
    from google.genai import types
    from pydantic import BaseModel

    class ValueEvaluation(BaseModel):
        reasoning: str
        score: float
        status: str

    client = _get_gemini_client()
    tmp_path = None
    video_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name
        imageio.mimwrite(tmp_path, [np.asarray(x) for x in frames], fps=ROLLOUT_FPS)

        video_file = client.files.upload(file=tmp_path)
        file_info = client.files.get(name=video_file.name)
        while file_info.state.name == "PROCESSING":
            time.sleep(2)
            file_info = client.files.get(name=video_file.name)
        if file_info.state.name == "FAILED":
            return {"step": step_idx, "error": "Video processing failed"}

        prompt = (
            f'You are a top-tier robot action evaluation expert responsible for constructing a '
            f'Dense Value Function for an RL model. '
            f'The robot is performing the task: "{task_description}". '
            f'Based on the provided video sequence (including the past history), please '
            f'evaluate the robot\'s state **over the most recent 2s** and provide a **Value Score** '
            f'between **0.00** and **1.00**.\n'
            f'IMPORTANT: Focus on the **final frames** of the video to judge the current state. '
            f'Do NOT give a high score just because the robot appeared to be on the right track earlier.\n'
            f'Rigorous Scoring Scale:\n'
            f'- 0.00 - 0.20 (Disengaged/Failure State): The robot is not in contact with the target '
            f'object, is moving in the wrong direction, has knocked the object away, or the object '
            f'has slipped out of the gripper.\n'
            f'- 0.20 - 0.40 (Approach State): The robot\'s end-effector is moving correctly toward '
            f'the target object, but has not yet made contact.\n'
            f'- 0.40 - 0.60 (Initial Interaction State): The gripper is touching or closing on the '
            f'object, but the object is NOT yet securely grasped or lifted.\n'
            f'- 0.60 - 0.80 (Critical Execution State): The object is securely grasped and being '
            f'lifted, but has not yet reached the goal height or position.\n'
            f'- 0.80 - 1.00 (Completion State): The task goal is fully achieved — for pick tasks, '
            f'the object is clearly lifted off the surface and stably held in the gripper.\n'
            f'Common failure patterns to watch for:\n'
            f'- Gripper closes but misses the object → score 0.10-0.20\n'
            f'- Object touched but not grasped (slides away) → score 0.20-0.30\n'
            f'- Object grasped but slips during lift → score 0.30-0.40\n'
            f'- Robot arm moving aimlessly or oscillating → score 0.05-0.15\n'
            f'Output strictly in **JSON array format**. Include reasoning, score (two decimal places) '
            f'and status. Example: [{{"reasoning": "...", "score": 0.35, "status": "Approach State"}}]'
        )

        response = client.models.generate_content(
            model=GEMINI_VALUE_MODEL,
            contents=[prompt, "\n[Current Video]:", video_file],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[ValueEvaluation],
                temperature=0.0,
            ),
        )
        result = json.loads(response.text)

        if result:
            score = result[0].get("score")
            if score is not None:
                with lock:
                    score_history.append((step_idx, float(score)))
                logging.info(
                    f"[Gemini Value] frame={step_idx} score={score:.2f} "
                    f"status={result[0].get('status')}"
                )

        return {"step": step_idx, "result": result}

    except Exception as e:
        logging.error(f"[Gemini Value] frame={step_idx} error: {e}")
        return {"step": step_idx, "error": str(e)}
    finally:
        if video_file is not None:
            try:
                client.files.delete(name=video_file.name)
            except Exception:
                pass
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _check_rescue_needed(score_history: list, lock: threading.Lock) -> bool:
    """Return True if rescue should be triggered based on the Gemini score history."""
    with lock:
        if not score_history:
            return False
        sorted_scores = sorted(score_history, key=lambda x: x[0])

    latest_frame, latest_score = sorted_scores[-1]

    # Condition 1: absolute low score
    if latest_score <= RESCUE_SCORE_ABSOLUTE:
        logging.info(f"[Rescue] Triggered: score {latest_score:.2f} <= {RESCUE_SCORE_ABSOLUTE}")
        return True

    # Condition 2: score drop >= RESCUE_SCORE_DROP compared to prior
    prev_score = None
    for frame_idx, score in reversed(sorted_scores[:-1]):
        if latest_frame - frame_idx >= GEMINI_QUERY_INTERVAL_FRAMES:
            prev_score = score
            break
    if prev_score is not None and (latest_score - prev_score) <= -RESCUE_SCORE_DROP:
        logging.info(
            f"[Rescue] Triggered: score dropped {prev_score:.2f} -> {latest_score:.2f} "
            f"(drop={prev_score - latest_score:.2f} >= {RESCUE_SCORE_DROP})"
        )
        return True

    return False


# ---------------------------------------------------------------------------
# SimplerEnv helpers
# ---------------------------------------------------------------------------

def _build_groot_obs_google(obs: dict):
    img = np.asarray(obs["video.image"])
    return {
        "video.image": img[np.newaxis, np.newaxis].astype(np.uint8),
        "state.x":  np.array([[[float(obs["state.x"][0])]]], dtype=np.float32),
        "state.y":  np.array([[[float(obs["state.y"][0])]]], dtype=np.float32),
        "state.z":  np.array([[[float(obs["state.z"][0])]]], dtype=np.float32),
        "state.rx": np.array([[[float(obs["state.rx"][0])]]], dtype=np.float32),
        "state.ry": np.array([[[float(obs["state.ry"][0])]]], dtype=np.float32),
        "state.rz": np.array([[[float(obs["state.rz"][0])]]], dtype=np.float32),
        "state.rw": np.array([[[float(obs["state.rw"][0])]]], dtype=np.float32),
        "state.gripper": np.array([[[float(obs["state.gripper"][0])]]], dtype=np.float32),
        "annotation.human.action.task_description": (
            str(obs["annotation.human.action.task_description"]),
        ),
    }


def _build_groot_obs_widowx(obs: dict):
    img = np.asarray(obs["video.image_0"])
    return {
        "video.image_0": img[np.newaxis, np.newaxis].astype(np.uint8),
        "state.x":     np.array([[[float(obs["state.x"][0])]]], dtype=np.float32),
        "state.y":     np.array([[[float(obs["state.y"][0])]]], dtype=np.float32),
        "state.z":     np.array([[[float(obs["state.z"][0])]]], dtype=np.float32),
        "state.roll":  np.array([[[float(obs["state.roll"][0])]]], dtype=np.float32),
        "state.pitch": np.array([[[float(obs["state.pitch"][0])]]], dtype=np.float32),
        "state.yaw":   np.array([[[float(obs["state.yaw"][0])]]], dtype=np.float32),
        "state.pad":   np.array([[[float(obs["state.pad"][0])]]], dtype=np.float32),
        "state.gripper": np.array([[[float(obs["state.gripper"][0])]]], dtype=np.float32),
        "annotation.human.action.task_description": (
            str(obs["annotation.human.action.task_description"]),
        ),
    }


def _extract_image_from_obs(obs: dict, robot_type: str) -> np.ndarray:
    if robot_type == "google":
        return np.asarray(obs["video.image"])
    else:
        return np.asarray(obs["video.image_0"])


def _extract_state_from_obs(obs: dict, robot_type: str) -> np.ndarray:
    if robot_type == "google":
        keys = GOOGLE_STATE_KEYS
    else:
        keys = WIDOWX_STATE_KEYS
    return np.array([float(obs[f"state.{k}"][0]) for k in keys], dtype=np.float32)


def _convert_to_simpler_action(action_chunk: dict, idx: int = 0) -> dict:
    return {
        f"action.{key}": np.atleast_1d(action_chunk[f"action.{key}"][0, idx]).flatten()[:1]
        for key in ACTION_KEYS
    }


def _convert_to_simpler_actions(action_chunk: dict, action_horizon: int) -> list[dict]:
    first_key = f"action.{ACTION_KEYS[0]}"
    avail = action_chunk[first_key].shape[1]
    num_steps = min(action_horizon, avail)
    return [_convert_to_simpler_action(action_chunk, i) for i in range(num_steps)]


def _actions_list_to_array(action_list: list[dict]) -> np.ndarray:
    rows = []
    for a in action_list:
        rows.append(np.concatenate([a[f"action.{k}"] for k in ACTION_KEYS]))
    return np.array(rows, dtype=np.float32)


def collect(args):
    """Phase 1: Run SimplerEnv rollouts with Gemini scoring and save labeled data."""
    os.environ.setdefault("DISPLAY", "")

    import gymnasium as gym
    import tqdm
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    np.random.seed(args.seed)
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
        raise ValueError(f"Unknown robot_type: {args.robot_type}. Use 'google' or 'widowx'.")

    if args.skip_tasks > 0:
        task_names = task_names[args.skip_tasks:]

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading GR00T policy from {args.model_path} ...")
    base_policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device=args.device,
        strict=True,
    )
    groot_policy = Gr00tSimPolicyWrapper(base_policy)
    logging.info("GR00T policy loaded.")

    # ---- Multi-GPU sharding ----
    # Build flat work list: (task_name, trial_idx) for all combinations
    all_work = [
        (task_name, trial_idx)
        for task_name in task_names
        for trial_idx in range(args.num_trials_per_task)
    ]
    if args.num_shards > 1:
        all_work = all_work[args.shard_id::args.num_shards]
        logging.info(
            f"Shard {args.shard_id}/{args.num_shards}: "
            f"{len(all_work)} work items on device {args.device}"
        )

    stats = {"total_episodes": 0, "total_successes": 0,
             "total_rescue_steps": 0, "total_normal_steps": 0}
    all_episode_meta = []

    for task_name, trial_idx in tqdm.tqdm(all_work, desc="rollouts"):
        env_id = f"{env_prefix}/{task_name}"
        task_segment = task_name.replace(" ", "_")

        # Skip if already completed
        existing_dirs = [
            d for d in output_dir.glob(f"rollout_{task_segment}_ep{trial_idx}_*")
            if d.name.endswith("_success") or d.name.endswith("_failure")
        ]
        if existing_dirs:
            logging.info(f"  Skip: {task_segment} ep {trial_idx} (already collected)")
            continue

        rollout_dir = output_dir / f"rollout_{task_segment}_ep{trial_idx}_running"
        rollout_dir.mkdir(parents=True, exist_ok=True)

        env = gym.make(env_id)
        obs, info = env.reset(seed=args.seed + trial_idx)

        task_description = str(obs.get(
            "annotation.human.action.task_description",
            task_name,
        ))

        action_plan = collections.deque()
        clean_images = []
        replan_records = []

        # Gemini scoring state
        score_history = []
        score_lock = threading.Lock()
        gemini_futures = []
        gemini_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        done, truncated = False, False

        logging.info(f"  Episode {trial_idx+1}/{args.num_trials_per_task}: {task_description}")

        for t in range(args.max_episode_steps):
            if done or truncated:
                break

            try:
                img = _extract_image_from_obs(obs, args.robot_type)
                clean_images.append(img.copy())
                num_frames = len(clean_images)

                # ---- Async Gemini value query ----
                if num_frames % GEMINI_QUERY_INTERVAL_FRAMES == 0:
                    clip = list(clean_images[-GEMINI_HISTORY_FRAMES:])
                    future = gemini_executor.submit(
                        _query_gemini_value,
                        clip, task_description, num_frames,
                        score_history, score_lock,
                    )
                    gemini_futures.append(future)

                # ---- Replan when action queue is empty ----
                if not action_plan:
                    rescue = _check_rescue_needed(score_history, score_lock)
                    state_vec = _extract_state_from_obs(obs, args.robot_type)

                    groot_obs = build_groot_obs_fn(obs)
                    action_dict, _ = groot_policy.get_action(groot_obs)
                    action_list = _convert_to_simpler_actions(
                        action_dict, args.action_horizon
                    )
                    action_arr = _actions_list_to_array(action_list)

                    # Extract video clip of recent clip_len frames
                    clip_len = args.clip_len
                    img_clip = list(clean_images[-clip_len:])
                    if len(img_clip) < clip_len:
                        pad_n = clip_len - len(img_clip)
                        img_clip = [img_clip[0]] * pad_n + img_clip

                    replan_records.append({
                        "frame_idx": num_frames,
                        "image": img.copy(),
                        "image_clip": np.stack(img_clip),
                        "state": state_vec.copy(),
                        "actions": action_arr,
                        "rescue": rescue,
                    })

                    action_plan.extend(action_list)

                action = action_plan.popleft()
                obs, reward, done, truncated, info = env.step(action)

            except Exception as e:
                logging.error(f"  Step exception: {e}")
                break

        # Wait for all Gemini queries to finish
        gemini_executor.shutdown(wait=True)

        # ---- Post-process: re-label with complete score history ----
        sorted_scores = sorted(score_history, key=lambda x: x[0])

        for rec in replan_records:
            frame = rec["frame_idx"]
            scores_up_to = [(f, s) for f, s in sorted_scores if f <= frame]
            if not scores_up_to:
                rec["switch_label"] = 0.0
                rec["gemini_score"] = 0.5  # default when no Gemini score available yet
                continue

            latest_f, latest_s = scores_up_to[-1]
            rec["gemini_score"] = latest_s  # preserve raw score for soft-label training
            should_rescue = False

            # Condition 1: absolute low
            if latest_s <= RESCUE_SCORE_ABSOLUTE:
                should_rescue = True
            # Condition 2: score drop
            if not should_rescue:
                prev_s = None
                for f, s in reversed(scores_up_to[:-1]):
                    if latest_f - f >= GEMINI_QUERY_INTERVAL_FRAMES:
                        prev_s = s
                        break
                if prev_s is not None and (latest_s - prev_s) <= -RESCUE_SCORE_DROP:
                    should_rescue = True

            rec["switch_label"] = 1.0 if should_rescue else 0.0

        # ---- Label shifting: propagate rescue labels backward for anticipation ----
        if args.label_shift_steps > 0:
            # Walk forward; when a rescue label is found, mark the preceding N steps too
            for i in range(len(replan_records)):
                if replan_records[i]["switch_label"] > 0.5:
                    for j in range(max(0, i - args.label_shift_steps), i):
                        replan_records[j]["switch_label"] = 1.0

        # ---- Finalize rollout directory ----
        n_rescue = sum(1 for r in replan_records if r.get("switch_label", 0) > 0.5)
        n_normal = len(replan_records) - n_rescue

        stats["total_episodes"] += 1
        if done:
            stats["total_successes"] += 1
        stats["total_rescue_steps"] += n_rescue
        stats["total_normal_steps"] += n_normal

        success = done
        suffix = "success" if success else "failure"
        final_rollout_dir = output_dir / f"rollout_{task_segment}_ep{trial_idx}_{suffix}"
        rollout_dir.rename(final_rollout_dir)

        if clean_images:
            imageio.mimwrite(
                str(final_rollout_dir / "complete_video.mp4"),
                [np.asarray(x) for x in clean_images],
                fps=ROLLOUT_FPS,
            )

        # ---- Save per-step .npz files inside the rollout directory ----
        for step_i, rec in enumerate(replan_records):
            save_path = final_rollout_dir / f"step_{step_i:04d}.npz"
            np.savez_compressed(
                save_path,
                image=rec["image"],                              # (H, W, 3) uint8
                image_clip=rec["image_clip"],                    # (T, H, W, 3) uint8
                state=rec["state"],                              # (8,) float32
                actions=rec["actions"],                          # (action_horizon, 7) float32
                switch_label=np.float32(rec["switch_label"]),    # 0.0 or 1.0
                gemini_score=np.float32(rec["gemini_score"]),    # raw Gemini value score
                clip_len=np.int32(args.clip_len),
                prompt=np.array(task_description),
                task=np.array(task_name),
                episode_idx=np.int32(trial_idx),
                frame_idx=np.int32(rec["frame_idx"]),
            )

        episode_meta = {
            "task": task_name,
            "task_description": task_description,
            "episode_idx": trial_idx,
            "success": success,
            "num_steps": len(replan_records),
            "num_rescue": n_rescue,
            "gemini_scores": [(int(f), float(s)) for f, s in sorted_scores],
        }
        all_episode_meta.append(episode_meta)

        env.close()

        logging.info(
            f"  -> {suffix.upper()} | "
            f"steps={len(replan_records)} rescue={n_rescue} normal={n_normal}"
        )

    # Save collection metadata
    meta_name = f"collection_meta_s{args.shard_id}.json" if args.num_shards > 1 else "collection_meta.json"
    meta_path = output_dir / meta_name
    with open(meta_path, "w") as f:
        json.dump({"stats": stats, "episodes": all_episode_meta}, f, indent=2)

    logging.info(f"\nCollection complete.")
    total_samples = stats['total_rescue_steps'] + stats['total_normal_steps']
    logging.info(f"  Total samples: {total_samples}")
    logging.info(f"  Rescue steps : {stats['total_rescue_steps']}")
    logging.info(f"  Normal steps : {stats['total_normal_steps']}")
    logging.info(f"  Saved to     : {output_dir}")


# ============================================================================
#  Phase 2: Training
# ============================================================================

class DINOv2SwitchHead(nn.Module):
    """
    Standalone DINOv2-based binary classifier for switch/intervention prediction.

    Supports multiple data formats:
      - SimplerEnv (1 camera): single image
      - LIBERO (2 cameras): base_image + wrist_image
      - Agilex (3 cameras): top + right + left

    Encodes each image with a frozen DINOv2 backbone, concatenates features
    with the robot state, and outputs a switch probability.
    """

    def __init__(
        self,
        dinov2_model: str = "dinov2_vitb14",
        hidden_dim: int = 256,
        state_dim: int = 8,
        num_cameras: int = 1,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.num_cameras = num_cameras

        self.backbone = torch.hub.load("facebookresearch/dinov2", dinov2_model)
        self.feature_dim = self.backbone.embed_dim  # 768 for ViT-B/14
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        input_dim = num_cameras * self.feature_dim + state_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def _encode_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        Encode image(s) to feature vector with optional temporal mean pooling.

        Args:
            img: (B, 3, H, W) single frame or (B, T, 3, H, W) temporal clip, float [0,1]

        Returns:
            (B, feature_dim)
        """
        if img.dim() == 5:
            # Temporal clip: fold B and T, encode all frames, then mean-pool
            B, T, C, H, W = img.shape
            img_flat = img.reshape(B * T, C, H, W)
        else:
            img_flat = img
            B = img.shape[0]
            T = 1

        img_flat = F.interpolate(img_flat, size=(224, 224), mode="bilinear", align_corners=False)
        img_flat = (img_flat - self.img_mean) / self.img_std

        if self.freeze_backbone:
            with torch.no_grad():
                feat_flat = self.backbone(img_flat)
        else:
            feat_flat = self.backbone(img_flat)

        if T > 1:
            feat = feat_flat.view(B, T, -1)
            return feat.mean(dim=1)  # Temporal Mean Pool → (B, feature_dim)
        return feat_flat

    def forward(self, images: list[torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: list of (B, 3, H, W) or (B, T, 3, H, W) float [0,1] tensors,
                    length = num_cameras. Temporal clips are mean-pooled after encoding.
            state:  (B, state_dim)

        Returns:
            logit: (B,) — raw logit (apply sigmoid for probability)
        """
        feats = [self._encode_image(img) for img in images]  # list of (B, D)
        combined = torch.cat(feats + [state], dim=-1)
        return self.classifier(combined).squeeze(-1)

    def predict_switch_prob(
        self, images: list[torch.Tensor], state: torch.Tensor
    ) -> torch.Tensor:
        """Returns switch probability in [0, 1]."""
        return torch.sigmoid(self.forward(images, state))


class SwitchLabelDataset(Dataset):
    """
    Loads .npz files produced by the 'collect' phase.

    Auto-detects format:
      - SimplerEnv: key "image" only            → 1 camera
      - LIBERO: keys "image", "wrist_image"     → 2 cameras
      - Agilex: keys "top", "right", "left"     → 3 cameras
    """

    def __init__(self, data_dir: str, use_clip: bool = True, use_soft_label: bool = False):
        # Support both new layout (rollout_*/step_*.npz) and legacy flat layout (sample_*.npz)
        self.files = sorted(glob.glob(os.path.join(data_dir, "rollout_*", "step_*.npz")))
        if not self.files:
            self.files = sorted(glob.glob(os.path.join(data_dir, "sample_*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No step_*.npz or sample_*.npz files found in {data_dir}")
        logging.info(f"Found {len(self.files)} samples in {data_dir}")

        # Detect format from first file
        first = np.load(self.files[0], allow_pickle=True)
        if "top" in first:
            self.format = "agilex"
            self.num_cameras = 3
            self.image_keys = ["top", "right", "left"]
        elif "wrist_image" in first:
            self.format = "libero"
            self.num_cameras = 2
            self.image_keys = ["image", "wrist_image"]
        else:
            self.format = "simpler_env"
            self.num_cameras = 1
            self.image_keys = ["image"]
        self.state_dim = first["state"].shape[0]

        # Temporal clip support: use image_clip (T, H, W, 3) instead of single image
        self.use_clip = use_clip and "image_clip" in first
        if self.use_clip:
            self.clip_len = int(first.get("clip_len", 1))
            logging.info(f"  Using temporal clips (clip_len={self.clip_len})")
        else:
            self.clip_len = 1

        # Soft label support: regress Gemini score instead of binary 0/1
        self.use_soft_label = use_soft_label and "gemini_score" in first
        if use_soft_label and not self.use_soft_label:
            logging.warning("  --use_soft_label requested but 'gemini_score' not found in data, falling back to hard labels")
        if self.use_soft_label:
            logging.info("  Using soft labels (1 - gemini_score)")

        logging.info(
            f"  Detected format: {self.format} ({self.num_cameras} cameras, "
            f"state_dim={self.state_dim})"
        )

        # Count class balance (skip corrupt files)
        labels = []
        valid_files = []
        for f in self.files:
            try:
                d = np.load(f)
                labels.append(float(d["switch_label"]))
                valid_files.append(f)
            except Exception:
                logging.warning(f"  Skipping corrupt file: {f}")
        if len(valid_files) < len(self.files):
            logging.info(f"  Skipped {len(self.files) - len(valid_files)} corrupt files")
            self.files = valid_files
        n_pos = sum(1 for l in labels if l > 0.5)
        n_neg = len(labels) - n_pos
        logging.info(f"  Class balance: {n_pos} rescue (pos) / {n_neg} normal (neg)")
        self._labels = labels

    def __len__(self):
        return len(self.files)

    @property
    def pos_weight(self) -> float:
        """Compute pos_weight for BCE loss to handle class imbalance."""
        n_pos = sum(1 for l in self._labels if l > 0.5)
        n_neg = len(self._labels) - n_pos
        if n_pos == 0:
            return 1.0
        return n_neg / n_pos

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)

        images = []
        for key in self.image_keys:
            if self.use_clip and key == "image" and "image_clip" in data:
                # (T, H, W, 3) uint8 → (T, 3, H, W) float [0,1]
                clip = torch.from_numpy(
                    data["image_clip"].copy()
                ).permute(0, 3, 1, 2).float() / 255.0
                images.append(clip)
            else:
                # (H, W, 3) uint8 → (3, H, W) float [0,1]
                img = torch.from_numpy(
                    data[key].copy()
                ).permute(2, 0, 1).float() / 255.0
                images.append(img)

        state = torch.from_numpy(data["state"].astype(np.float32))

        if self.use_soft_label and "gemini_score" in data:
            # Convert value score → rescue probability (low score = high rescue need)
            label = torch.tensor(1.0 - float(data["gemini_score"]), dtype=torch.float32)
        else:
            label = torch.tensor(float(data["switch_label"]), dtype=torch.float32)

        return images, state, label


def _collate_switch(batch):
    """Custom collate for variable-length image lists."""
    images_list, states, labels = zip(*batch)
    num_cameras = len(images_list[0])
    batched_images = [torch.stack([img[c] for img in images_list]) for c in range(num_cameras)]
    return batched_images, torch.stack(states), torch.stack(labels)


def _setup_distributed():
    """Initialize distributed training if launched via torchrun / torch.distributed.launch."""
    if "RANK" not in os.environ:
        return 0, 0, 1  # rank, local_rank, world_size  (single-GPU fallback)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def train(args):
    """Phase 2: Train the standalone switch head (supports multi-GPU via torchrun)."""
    rank, local_rank, world_size = _setup_distributed()
    distributed = world_size > 1
    is_main = rank == 0

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main:
        logging.info(f"Using device: {device}  (world_size={world_size})")

    train_dataset = SwitchLabelDataset(
        args.data_dir, use_clip=args.use_clip, use_soft_label=args.use_soft_label,
    )
    val_dataset = (
        SwitchLabelDataset(
            args.val_dir, use_clip=args.use_clip, use_soft_label=args.use_soft_label,
        )
        if args.val_dir
        else None
    )

    num_cameras = train_dataset.num_cameras
    state_dim = train_dataset.state_dim
    if is_main:
        logging.info(
            f"Training with {num_cameras}-camera data ({train_dataset.format} format), "
            f"state_dim={state_dim}"
        )

    model = DINOv2SwitchHead(
        dinov2_model=args.dinov2_model,
        hidden_dim=args.hidden_dim,
        state_dim=state_dim,
        num_cameras=num_cameras,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                     find_unused_parameters=False)
    # Access the underlying module for saving state_dict without "module." prefix
    raw_model = model.module if distributed else model

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate_switch,
    )

    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed and val_dataset else None
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=_collate_switch,
        )
        if val_dataset
        else None
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if is_main:
        logging.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Loss function: MSE for soft-label regression, BCE for hard binary labels
    use_soft_label = args.use_soft_label and train_dataset.use_soft_label
    if use_soft_label:
        if is_main:
            logging.info("Using MSE loss for soft-label regression (target = 1 - gemini_score)")
        criterion = nn.MSELoss()
    else:
        pw = torch.tensor([train_dataset.pos_weight], device=device)
        if is_main:
            logging.info(f"BCE pos_weight: {pw.item():.2f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    output_dir = pathlib.Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    best_val_metric = -float("inf")

    # ---- wandb ----
    use_wandb = is_main and wandb is not None and args.wandb_project
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "dinov2_model": args.dinov2_model,
                "hidden_dim": args.hidden_dim,
                "freeze_backbone": args.freeze_backbone,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "use_clip": args.use_clip,
                "use_soft_label": args.use_soft_label,
                "world_size": world_size,
                "data_dir": args.data_dir,
                "num_samples": len(train_dataset),
                "num_cameras": num_cameras,
                "state_dim": state_dim,
            },
        )
        logging.info(f"wandb run: {wandb.run.url}")

    for epoch in range(args.epochs):
        # ---- Train ----
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        if args.freeze_backbone:
            backbone = raw_model.backbone
            backbone.eval()

        total_loss, num_batches = 0.0, 0
        train_correct, train_total = 0, 0

        for images, state, label in train_loader:
            images = [img.to(device) for img in images]
            state = state.to(device)
            label = label.to(device)

            logit = model(images, state)
            if use_soft_label:
                loss = criterion(torch.sigmoid(logit), label)
            else:
                loss = criterion(logit, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pred = (torch.sigmoid(logit) > 0.5).float()
            label_hard = (label > 0.5).float()
            train_correct += (pred == label_hard).sum().item()
            train_total += label.numel()

        avg_train_loss = total_loss / max(num_batches, 1)
        train_acc = train_correct / max(train_total, 1)
        scheduler.step()

        # ---- Val ----
        val_str = "N/A"
        if val_loader is not None:
            model.eval()
            vl, vn = 0.0, 0
            val_correct, val_total = 0, 0
            val_tp, val_fp, val_fn = 0, 0, 0

            with torch.no_grad():
                for images, state, label in val_loader:
                    images = [img.to(device) for img in images]
                    state = state.to(device)
                    label = label.to(device)

                    logit = model(images, state)
                    if use_soft_label:
                        vl += criterion(torch.sigmoid(logit), label).item()
                    else:
                        vl += criterion(logit, label).item()
                    vn += 1

                    pred = (torch.sigmoid(logit) > 0.5).float()
                    label_hard = (label > 0.5).float()
                    val_correct += (pred == label_hard).sum().item()
                    val_total += label.numel()
                    val_tp += ((pred == 1) & (label_hard == 1)).sum().item()
                    val_fp += ((pred == 1) & (label_hard == 0)).sum().item()
                    val_fn += ((pred == 0) & (label_hard == 1)).sum().item()

            avg_val_loss = vl / max(vn, 1)
            val_acc = val_correct / max(val_total, 1)
            val_precision = val_tp / max(val_tp + val_fp, 1)
            val_recall = val_tp / max(val_tp + val_fn, 1)
            val_f1 = (
                2 * val_precision * val_recall / max(val_precision + val_recall, 1e-8)
            )

            val_str = (
                f"loss={avg_val_loss:.4f} acc={val_acc:.3f} "
                f"P={val_precision:.3f} R={val_recall:.3f} F1={val_f1:.3f}"
            )

            # Save best model by F1 (more meaningful than accuracy for imbalanced data)
            if is_main and val_f1 > best_val_metric:
                best_val_metric = val_f1
                torch.save(raw_model.state_dict(), output_dir / "best_model.pt")
                logging.info(f"  -> Saved best model (F1={val_f1:.4f})")

        if is_main:
            logging.info(
                f"Epoch {epoch+1}/{args.epochs}  "
                f"train_loss={avg_train_loss:.4f} train_acc={train_acc:.3f}  "
                f"val=[{val_str}]  lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if use_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "train/acc": train_acc,
                "lr": scheduler.get_last_lr()[0],
            }
            if val_loader is not None:
                log_dict.update({
                    "val/loss": avg_val_loss,
                    "val/acc": val_acc,
                    "val/precision": val_precision,
                    "val/recall": val_recall,
                    "val/f1": val_f1,
                })
            wandb.log(log_dict, step=epoch + 1)

        if is_main and (epoch + 1) % args.save_every == 0:
            torch.save(raw_model.state_dict(), output_dir / f"model_epoch{epoch+1}.pt")

    if is_main:
        torch.save(raw_model.state_dict(), output_dir / "model_final.pt")
        logging.info(f"Training complete. Models saved to {output_dir}")

    if use_wandb:
        wandb.finish()

    _cleanup_distributed()


# ============================================================================
#  Phase 3 (optional): Export – inject switch_label into LeRobot v2 dataset
# ============================================================================

def export_for_training(args):
    """
    Inject switch_label into a local LeRobot v2 dataset.

    The LeRobot v2 layout:
        <dataset_dir>/
            meta/info.json          – feature schema + totals
            meta/tasks.jsonl        – task_index → task string
            meta/episodes.jsonl     – episode_index → tasks, length
            data/chunk-NNN/episode_NNNNNN.parquet  – per-episode frames

    This function:
      1. Loads collected .npz labels into a lookup keyed by (episode_index, frame_index).
      2. Copies the source dataset to output_dir.
      3. Adds a ``switch_label`` float32 column to every parquet file
         (0.0 for frames without a matching label).
      4. Updates meta/info.json to declare the new feature.
    """
    import shutil
    import pandas as pd

    data_dir = pathlib.Path(args.data_dir)
    src_dataset = pathlib.Path(args.lerobot_dataset)
    output_dir = pathlib.Path(args.output_dir)

    # ------------------------------------------------------------------
    # 1. Load collected switch labels
    # ------------------------------------------------------------------
    label_files = sorted(glob.glob(str(data_dir / "rollout_*" / "step_*.npz")))
    if not label_files:
        label_files = sorted(glob.glob(str(data_dir / "sample_*.npz")))
    if not label_files:
        logging.error(f"No step or sample files found in {data_dir}")
        return

    logging.info(f"Loading {len(label_files)} label files...")
    # key = (episode_index, frame_index) → switch_label
    # If collect produced multiple labels for the same frame, keep the last.
    label_lookup: dict[tuple[int, int], float] = {}
    for f in label_files:
        d = np.load(f, allow_pickle=True)
        ep = int(d["episode_idx"])
        fr = int(d["frame_idx"])
        label_lookup[(ep, fr)] = float(d["switch_label"])

    n_pos = sum(1 for v in label_lookup.values() if v > 0.5)
    n_neg = len(label_lookup) - n_pos
    logging.info(
        f"Loaded {len(label_lookup)} labels "
        f"({n_pos} rescue / {n_neg} normal)"
    )

    # ------------------------------------------------------------------
    # 2. Copy source dataset to output_dir
    # ------------------------------------------------------------------
    if output_dir.exists():
        logging.warning(f"Output dir {output_dir} already exists, will overwrite parquet files in-place")
    else:
        logging.info(f"Copying {src_dataset} -> {output_dir} ...")
        shutil.copytree(src_dataset, output_dir)
        logging.info("Copy done.")

    # ------------------------------------------------------------------
    # 3. Add switch_label column to every parquet file
    # ------------------------------------------------------------------
    parquet_files = sorted(glob.glob(str(output_dir / "data" / "chunk-*" / "*.parquet")))
    if not parquet_files:
        logging.error(f"No parquet files found in {output_dir / 'data'}")
        return

    logging.info(f"Processing {len(parquet_files)} parquet files...")
    total_frames = 0
    matched_frames = 0

    for pq_path in parquet_files:
        df = pd.read_parquet(pq_path)
        total_frames += len(df)

        labels = []
        for _, row in df.iterrows():
            ep = int(row["episode_index"])
            fr = int(row["frame_index"])
            key = (ep, fr)
            if key in label_lookup:
                labels.append(label_lookup[key])
                matched_frames += 1
            else:
                # No label for this frame: find closest frame in same episode
                ep_labels = [
                    (f, l) for (e, f), l in label_lookup.items() if e == ep
                ]
                if ep_labels:
                    closest_fr, closest_label = min(ep_labels, key=lambda x: abs(x[0] - fr))
                    labels.append(closest_label)
                else:
                    labels.append(0.0)

        df["switch_label"] = np.array(labels, dtype=np.float32)
        df.to_parquet(pq_path)

    logging.info(
        f"Processed {total_frames} frames, {matched_frames} exact matches "
        f"from collected labels"
    )

    # ------------------------------------------------------------------
    # 4. Update meta/info.json to include switch_label feature
    # ------------------------------------------------------------------
    info_path = output_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    if "switch_label" not in info.get("features", {}):
        info["features"]["switch_label"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None,
        }
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        logging.info("Updated meta/info.json with switch_label feature")

    logging.info(f"Done. Augmented dataset saved to {output_dir}")


# ============================================================================
#  CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train switch head with Gemini labels for SimplerEnv (collect → train)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- collect ----
    p_collect = subparsers.add_parser(
        "collect", help="Run SimplerEnv rollouts with Gemini scoring and save labeled data"
    )
    p_collect.add_argument("--model_path", type=str, default="nvidia/GR00T-N1.6-fractal",
                           help="Path to post-trained GR00T checkpoint "
                           "(e.g. nvidia/GR00T-N1.6-fractal for Google, "
                           "nvidia/GR00T-N1.6-bridge for WidowX)")
    p_collect.add_argument("--device", type=int, default=0)
    p_collect.add_argument("--action_horizon", type=int, default=1,
                           help="Number of actions to execute per policy query")
    p_collect.add_argument("--robot_type", type=str, default="google",
                           choices=["google", "widowx"])
    p_collect.add_argument("--num_trials_per_task", type=int, default=20)
    p_collect.add_argument("--max_episode_steps", type=int, default=120)
    p_collect.add_argument("--output_dir", type=str, default="data/simpler_env_switch_labels",
                           help="Directory to save .npz files with switch labels")
    p_collect.add_argument("--seed", type=int, default=7)
    p_collect.add_argument("--clip_len", type=int, default=20,
                           help="Number of recent frames for video clips (default 20 = 2s)")
    p_collect.add_argument("--skip_tasks", type=int, default=0)
    p_collect.add_argument("--label_shift_steps", type=int, default=0,
                           help="Shift rescue labels backward by N replan steps for anticipation "
                           "(e.g., 5 = teach model to predict rescue ~5 steps before failure)")
    p_collect.add_argument("--shard_id", type=int, default=0,
                           help="Shard index for multi-GPU parallel collection (0-indexed)")
    p_collect.add_argument("--num_shards", type=int, default=1,
                           help="Total number of shards (1 = single GPU, no sharding)")

    # ---- train ----
    p_train = subparsers.add_parser(
        "train", help="Train standalone DINOv2-based switch head"
    )
    p_train.add_argument("--data_dir", type=str, required=True)
    p_train.add_argument("--val_dir", type=str, default=None)
    p_train.add_argument("--output_dir", type=str, default="checkpoints/switch_head")
    p_train.add_argument(
        "--dinov2_model", type=str, default="dinov2_vitb14",
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
    )
    p_train.add_argument("--hidden_dim", type=int, default=256)
    p_train.add_argument("--freeze_backbone", action="store_true", default=True)
    p_train.add_argument(
        "--no_freeze_backbone", dest="freeze_backbone", action="store_false"
    )
    p_train.add_argument("--batch_size", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--weight_decay", type=float, default=1e-4)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--num_workers", type=int, default=4)
    p_train.add_argument("--save_every", type=int, default=5)
    p_train.add_argument("--use_clip", action="store_true", default=True,
                         help="Use temporal image clips (image_clip) instead of single frames")
    p_train.add_argument("--no_clip", dest="use_clip", action="store_false",
                         help="Fall back to single-frame images")
    p_train.add_argument("--use_soft_label", action="store_true", default=False,
                         help="Regress Gemini scores (MSE) instead of binary labels (BCE)")
    p_train.add_argument("--wandb_project", type=str, default=None,
                         help="W&B project name (omit to disable wandb logging)")
    p_train.add_argument("--wandb_run_name", type=str, default=None,
                         help="W&B run name (auto-generated if omitted)")

    # ---- export ----
    p_export = subparsers.add_parser(
        "export",
        help="Inject switch_label into local LeRobot v2 dataset",
    )
    p_export.add_argument("--data_dir", type=str, required=True,
                          help="Directory of collected .npz files")
    p_export.add_argument("--lerobot_dataset", type=str, required=True,
                          help="Path to source LeRobot v2 dataset "
                          "(e.g. fractal20220817_data_lerobot or bridge_orig_lerobot)")
    p_export.add_argument("--output_dir", type=str, required=True,
                          help="Path to save the augmented dataset copy")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.command == "collect":
        collect(args)
    elif args.command == "train":
        train(args)
    elif args.command == "export":
        export_for_training(args)


if __name__ == "__main__":
    main()
