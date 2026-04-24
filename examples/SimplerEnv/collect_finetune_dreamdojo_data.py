"""
Rollout GR00T on SimplerEnv and collect data in LeRobot format for DreamDojo fine-tuning.

Supports both Google Robot (Fractal) and WidowX (Bridge) environments.
Iterates over all registered tasks for the chosen robot type, runs multiple
trials per task, and saves both successful and failed trajectories.

Data layout produced:
  <output_dir>/
    meta/
      modality.json
      info.json
      episodes.jsonl
      tasks.jsonl
      stats.json
    data/
      chunk-000/
        episode_000000.parquet
        ...
    videos/
      chunk-000/
        observation.images.image/          (Google Fractal)
          episode_000000.mp4
        observation.images.image_0/        (WidowX Bridge)
          episode_000000.mp4

Action / state convention (matches DreamDojo's relative-action rebaselining):
  Google Fractal:
    observation.state  [8D]: eef_pos(3) + quat_xyzw(4) + gripper_closedness(1)  -- absolute
    action             [7D]: eef_pos(3) + quat_xyzw[:3](3) + gripper_closedness(1) -- absolute EE
  WidowX Bridge:
    observation.state  [8D]: eef_pos(3) + rpy(3) + pad(1) + gripper(1)  -- absolute
    action             [7D]: eef_pos(3) + rpy(3) + gripper(1)            -- absolute EE

Prerequisites:
    Start the GR00T server first (Terminal 1):
        # For Google Fractal:
        uv run --extra=gpu python gr00t/eval/run_gr00t_server.py \
            --model-path /path/to/checkpoint \
            --embodiment-tag OXE_GOOGLE \
            --use-sim-policy-wrapper

        # For WidowX Bridge:
        uv run --extra=gpu python gr00t/eval/run_gr00t_server.py \
            --model-path /path/to/checkpoint \
            --embodiment-tag OXE_WIDOWX \
            --use-sim-policy-wrapper

Usage (Terminal 2):
    python examples/SimplerEnv/collect_finetune_dreamdojo_data.py \
        --policy_client_host 127.0.0.1 \
        --policy_client_port 5555 \
        --robot_type google \
        --num_trials_per_task 50 \
        --action_horizon 1 \
        --output_dir data/simpler_env_dreamdojo
"""

import dataclasses
import json
import logging
import pathlib
from typing import Optional

import cv2
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ENV_FPS = 10
CHUNK_SIZE = 1000

# Task lists (matching simpler_env.py registration)
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

# ── Modality definitions (DreamDojo-compatible) ─────────────────────────────

GOOGLE_FRACTAL_MODALITY = {
    "state": {
        "eef_pos": {
            "original_key": "observation.state",
            "start": 0, "end": 3,
            "rotation_type": None, "absolute": True,
            "dtype": "float64", "range": None,
        },
        "eef_quat": {
            "original_key": "observation.state",
            "start": 3, "end": 7,
            "rotation_type": "quaternion_xyzw", "absolute": True,
            "dtype": "float64", "range": None,
        },
        "gripper": {
            "original_key": "observation.state",
            "start": 7, "end": 8,
            "rotation_type": None, "absolute": True,
            "dtype": "float64", "range": None,
        },
    },
    "action": {
        "arm": {
            "original_key": "action",
            "start": 0, "end": 7,
            "rotation_type": None, "absolute": True,
            "dtype": "float64", "range": None,
        },
    },
    "video": {
        "image": {
            "original_key": "observation.images.image",
        },
    },
    "annotation": {
        "language.task": {
            "original_key": "task_index",
        },
    },
}

WIDOWX_BRIDGE_MODALITY = {
    "state": {
        "eef_pos": {
            "original_key": "observation.state",
            "start": 0, "end": 3,
            "rotation_type": None, "absolute": True,
            "dtype": "float64", "range": None,
        },
        "eef_rot": {
            "original_key": "observation.state",
            "start": 3, "end": 6,
            "rotation_type": None, "absolute": True,
            "dtype": "float64", "range": None,
        },
        "pad": {
            "original_key": "observation.state",
            "start": 6, "end": 7,
            "rotation_type": None, "absolute": True,
            "dtype": "float64", "range": None,
        },
        "gripper": {
            "original_key": "observation.state",
            "start": 7, "end": 8,
            "rotation_type": None, "absolute": True,
            "dtype": "float64", "range": None,
        },
    },
    "action": {
        "arm": {
            "original_key": "action",
            "start": 0, "end": 7,
            "rotation_type": None, "absolute": True,
            "dtype": "float64", "range": None,
        },
    },
    "video": {
        "image_0": {
            "original_key": "observation.images.image_0",
        },
    },
    "annotation": {
        "language.task": {
            "original_key": "task_index",
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI args
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Args:
    # GR00T policy server
    policy_client_host: str = "127.0.0.1"
    policy_client_port: int = 5555
    action_horizon: int = 1
    """Number of actions to execute per policy query (official eval default: 1)."""

    # SimplerEnv
    robot_type: str = "google"          # "google" or "widowx"
    num_trials_per_task: int = 50
    max_episode_steps: int = 120        # Match official eval default (--episode_length)

    # Data output
    output_dir: str = "data/simpler_env_dreamdojo"

    skip_tasks: int = 0
    """Number of tasks to skip from the beginning (for resuming interrupted runs)"""

    resume: bool = False
    """Resume from existing data in output_dir. Scans existing parquets and
    skips already-completed task/episode pairs."""

    seed: int = 7


# ─────────────────────────────────────────────────────────────────────────────
# DreamDojo data writer
# ─────────────────────────────────────────────────────────────────────────────

class DreamDojoWriter:
    """Incrementally writes a LeRobot-format dataset compatible with DreamDojo."""

    def __init__(self, output_dir: pathlib.Path, video_key: str,
                 env_fps: int = ENV_FPS):
        self.root = pathlib.Path(output_dir)
        self.env_fps = env_fps
        self.video_key = video_key  # e.g. "observation.images.image"

        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        (self.root / "videos").mkdir(parents=True, exist_ok=True)

        # Running counters
        self.episode_index: int = 0
        self.global_frame_index: int = 0

        # Registered tasks: description -> task_index
        self.task_registry: dict[str, int] = {}

        # Per-episode buffers
        self._ep_states: list[np.ndarray] = []
        self._ep_actions: list[np.ndarray] = []
        self._ep_images: list[np.ndarray] = []
        self._ep_task_index: Optional[int] = None

        # Accumulated stats
        self._stat_accum: dict[str, list[np.ndarray]] = {
            "observation.state": [],
            "action": [],
        }

        # Episode metadata
        self._episodes_meta: list[dict] = []

    # ── episode lifecycle ────────────────────────────────────────────────────

    def begin_episode(self, task_description: str) -> None:
        if task_description not in self.task_registry:
            self.task_registry[task_description] = len(self.task_registry)
        self._ep_task_index = self.task_registry[task_description]
        self._ep_states = []
        self._ep_actions = []
        self._ep_images = []

    def record_step(
        self,
        image: np.ndarray,           # (H, W, 3) uint8
        state: np.ndarray,           # (8,) float64 -- absolute EE state
        action: np.ndarray,          # (7,) float64 -- absolute EE state for action
    ) -> None:
        self._ep_states.append(state.astype(np.float64))
        self._ep_actions.append(action.astype(np.float64))
        self._ep_images.append(image)

    def end_episode(self, success: bool) -> bool:
        """Flush episode to disk. Returns True if written."""
        n = len(self._ep_states)
        if n == 0:
            return False

        ep_idx = self.episode_index
        chunk_idx = ep_idx // CHUNK_SIZE

        # ── write video ──────────────────────────────────────────────────────
        vid_dir = self.root / "videos" / f"chunk-{chunk_idx:03d}" / self.video_key
        vid_dir.mkdir(parents=True, exist_ok=True)
        vid_path = vid_dir / f"episode_{ep_idx:06d}.mp4"
        imageio.mimwrite(
            vid_path,
            [np.asarray(f) for f in self._ep_images],
            fps=self.env_fps,
            codec="libx264",
            quality=8,
        )

        # ── write parquet ────────────────────────────────────────────────────
        data_dir = self.root / "data" / f"chunk-{chunk_idx:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for frame_idx, (state, action) in enumerate(
            zip(self._ep_states, self._ep_actions)
        ):
            rows.append({
                "observation.state": state,
                "action":            action,
                "timestamp":         frame_idx / self.env_fps,
                "frame_index":       frame_idx,
                "episode_index":     ep_idx,
                "index":             self.global_frame_index + frame_idx,
                "next.done":         (frame_idx == n - 1),
                "task_index":        self._ep_task_index,
                "success":           success,
            })

        df = pd.DataFrame(rows)
        df.to_parquet(data_dir / f"episode_{ep_idx:06d}.parquet", index=False)

        # ── accumulate stats ─────────────────────────────────────────────────
        self._stat_accum["observation.state"].extend(self._ep_states)
        self._stat_accum["action"].extend(self._ep_actions)

        # ── episode metadata ─────────────────────────────────────────────────
        self._episodes_meta.append({
            "episode_index": ep_idx,
            "tasks":         [list(self.task_registry.keys())[self._ep_task_index]],
            "length":        n,
            "success":       bool(success),
        })

        self.global_frame_index += n
        self.episode_index += 1
        return True

    def discard_episode(self) -> None:
        self._ep_states = []
        self._ep_actions = []
        self._ep_images = []
        self._ep_task_index = None

    def resume_from_existing(self, task_names: list[str]) -> set[tuple[str, int]]:
        """Scan existing parquets to restore writer state for resuming.

        Rebuilds episode_index, global_frame_index, task_registry, stats
        accumulators, and episode metadata from on-disk data.

        Args:
            task_names: Ordered list of task names used in collection.

        Returns:
            Set of (task_name, trial_idx) pairs already completed, so the
            caller can skip them.
        """
        existing_parquets = sorted(self.root.glob("data/chunk-*/episode_*.parquet"))
        if not existing_parquets:
            logging.info("[Resume] No existing episodes found, starting fresh.")
            return set()

        completed: set[tuple[str, int]] = set()
        # Build task_name -> task_index mapping (same order as fresh run)
        for name in task_names:
            if name not in self.task_registry:
                self.task_registry[name] = len(self.task_registry)

        # Track per-task episode counts to reconstruct trial_idx
        task_episode_counts: dict[int, int] = {}

        for pq_path in existing_parquets:
            df = pd.read_parquet(pq_path)
            n = len(df)
            ep_idx = int(df["episode_index"].iloc[0])
            task_idx = int(df["task_index"].iloc[0])

            # Restore stats
            states = np.stack(df["observation.state"].values)
            actions = np.stack(df["action"].values)
            self._stat_accum["observation.state"].extend(
                [s.astype(np.float64) for s in states]
            )
            self._stat_accum["action"].extend(
                [a.astype(np.float64) for a in actions]
            )

            # Restore episode metadata
            task_name = task_names[task_idx] if task_idx < len(task_names) else f"task_{task_idx}"
            success = bool(df["success"].iloc[-1]) if "success" in df.columns else False
            self._episodes_meta.append({
                "episode_index": ep_idx,
                "tasks": [task_name],
                "length": n,
                "success": success,
            })

            # Track which (task_name, trial_idx) are done
            trial_idx = task_episode_counts.get(task_idx, 0)
            task_episode_counts[task_idx] = trial_idx + 1
            completed.add((task_name, trial_idx))

            self.global_frame_index += n

        self.episode_index = len(existing_parquets)
        logging.info(
            f"[Resume] Restored {self.episode_index} episodes, "
            f"{self.global_frame_index} frames from existing data."
        )
        return completed

    # ── finalization ─────────────────────────────────────────────────────────

    def finalize(self, modality: dict, state_dim: int, action_dim: int,
                 state_names: list[str], action_names: list[str],
                 image_size: tuple[int, int]) -> None:
        """Write all meta files."""
        if self.episode_index == 0:
            logging.warning("No episodes were written; skipping finalize.")
            return

        total_ep = self.episode_index
        total_frames = self.global_frame_index

        # ── info.json ────────────────────────────────────────────────────────
        info = {
            "codebase_version": "v2.0",
            "robot_type":       "single_arm",
            "fps":              self.env_fps,
            "data_path":        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path":       "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "chunks_size":      CHUNK_SIZE,
            "total_episodes":   total_ep,
            "total_frames":     total_frames,
            "total_tasks":      len(self.task_registry),
            "features": {
                "observation.state": {
                    "shape":  [state_dim],
                    "names":  state_names,
                    "dtype":  "float64",
                },
                "action": {
                    "shape": [action_dim],
                    "names": action_names,
                    "dtype": "float64",
                },
                self.video_key: {
                    "shape":  [image_size[0], image_size[1], 3],
                    "names":  ["height", "width", "channel"],
                    "dtype":  "uint8",
                    "video_info": {"video.fps": float(self.env_fps)},
                },
            },
        }
        with open(self.root / "meta" / "info.json", "w") as f:
            json.dump(info, f, indent=2)

        # ── modality.json ────────────────────────────────────────────────────
        with open(self.root / "meta" / "modality.json", "w") as f:
            json.dump(modality, f, indent=2)

        # ── tasks.jsonl ──────────────────────────────────────────────────────
        with open(self.root / "meta" / "tasks.jsonl", "w") as f:
            for desc, idx in self.task_registry.items():
                f.write(json.dumps({"task_index": idx, "task": desc}) + "\n")

        # ── episodes.jsonl ───────────────────────────────────────────────────
        with open(self.root / "meta" / "episodes.jsonl", "w") as f:
            for meta in self._episodes_meta:
                f.write(json.dumps(meta) + "\n")

        # ── stats.json ───────────────────────────────────────────────────────
        stats = {}
        for col, arrays in self._stat_accum.items():
            if not arrays:
                continue
            data = np.stack(arrays, axis=0)
            stats[col] = {
                "mean": data.mean(0).tolist(),
                "std":  data.std(0).tolist(),
                "min":  data.min(0).tolist(),
                "max":  data.max(0).tolist(),
                "q01":  np.quantile(data, 0.01, axis=0).tolist(),
                "q99":  np.quantile(data, 0.99, axis=0).tolist(),
            }
        with open(self.root / "meta" / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        logging.info(
            f"Dataset written to {self.root}: "
            f"{total_ep} episodes, {total_frames} frames, "
            f"{len(self.task_registry)} tasks."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Observation extractors (per robot type)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_google_fractal(obs: dict):
    """Extract state / action / image from GoogleFractalEnv wrapped obs.

    Returns:
        image: (256, 320, 3) uint8
        state: (8,) float64 -- [x, y, z, qx, qy, qz, qw, gripper_closedness]
        action_state: (7,) float64 -- [x, y, z, qx, qy, qz, gripper_closedness]
    """
    image = np.asarray(obs["video.image"])

    x  = float(obs["state.x"][0])
    y  = float(obs["state.y"][0])
    z  = float(obs["state.z"][0])
    rx = float(obs["state.rx"][0])
    ry = float(obs["state.ry"][0])
    rz = float(obs["state.rz"][0])
    rw = float(obs["state.rw"][0])
    gripper = float(obs["state.gripper"][0])

    state = np.array([x, y, z, rx, ry, rz, rw, gripper], dtype=np.float64)
    # Action stored as absolute EE: pos(3) + quat_xyz(3) + gripper(1)
    # (qw dropped from action — same convention as 7D action space)
    action_state = np.array([x, y, z, rx, ry, rz, gripper], dtype=np.float64)

    return image, state, action_state


def _extract_widowx_bridge(obs: dict):
    """Extract state / action / image from WidowXBridgeEnv wrapped obs.

    Returns:
        image: (256, 256, 3) uint8
        state: (8,) float64 -- [x, y, z, roll, pitch, yaw, pad, gripper]
        action_state: (7,) float64 -- [x, y, z, roll, pitch, yaw, gripper]
    """
    image = np.asarray(obs["video.image_0"])

    x     = float(obs["state.x"][0])
    y     = float(obs["state.y"][0])
    z     = float(obs["state.z"][0])
    roll  = float(obs["state.roll"][0])
    pitch = float(obs["state.pitch"][0])
    yaw   = float(obs["state.yaw"][0])
    pad   = float(obs["state.pad"][0])
    gripper = float(obs["state.gripper"][0])

    state = np.array([x, y, z, roll, pitch, yaw, pad, gripper], dtype=np.float64)
    action_state = np.array([x, y, z, roll, pitch, yaw, gripper], dtype=np.float64)

    return image, state, action_state


# ─────────────────────────────────────────────────────────────────────────────
# GR00T observation builders (per robot type)
# ─────────────────────────────────────────────────────────────────────────────

def _build_groot_obs_google(obs: dict):
    """Build GR00T batched observation from GoogleFractalEnv obs."""
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
    """Build GR00T batched observation from WidowXBridgeEnv obs."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Action conversion from GR00T policy output -> env step
# ─────────────────────────────────────────────────────────────────────────────

ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def _convert_to_simpler_action(action_chunk: dict, idx: int = 0) -> dict:
    """Convert GR00T action chunk to a single env-step action dict.

    Matches the official eval's GR00TPolicy._convert_to_simpler_action:
    extracts action at index ``idx`` from the predicted chunk.
    Each value is a 1-element numpy array (required by GoogleFractalEnv.step
    which uses np.concatenate).
    """
    return {
        f"action.{key}": np.atleast_1d(action_chunk[f"action.{key}"][idx]).flatten()[:1]
        for key in ACTION_KEYS
    }


def _convert_to_simpler_actions(action_chunk: dict, action_horizon: int) -> list[dict]:
    """Convert GR00T action chunk to a list of env-step action dicts.

    When action_horizon == 1, returns a single-element list (official default).
    When action_horizon > 1, returns ``action_horizon`` actions from the chunk.
    """
    return [_convert_to_simpler_action(action_chunk, i) for i in range(action_horizon)]


# ─────────────────────────────────────────────────────────────────────────────
# Main collection loop
# ─────────────────────────────────────────────────────────────────────────────

def collect_simpler_env(args: Args) -> None:
    import os
    os.environ.setdefault("DISPLAY", "")

    import gymnasium as gym
    from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs
    from gr00t.policy.server_client import PolicyClient

    np.random.seed(args.seed)
    register_simpler_envs()

    # ── Select robot configuration ───────────────────────────────────────────
    if args.robot_type == "google":
        task_names = GOOGLE_FRACTAL_TASKS
        env_prefix = "simpler_env_google"
        extract_fn = _extract_google_fractal
        build_groot_obs_fn = _build_groot_obs_google
        modality = GOOGLE_FRACTAL_MODALITY
        video_key = "observation.images.image"
        image_size = (256, 320)
        state_dim = 8
        action_dim = 7
        state_names = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]
        action_names = ["x", "y", "z", "qx", "qy", "qz", "gripper"]
    elif args.robot_type == "widowx":
        task_names = WIDOWX_BRIDGE_TASKS
        env_prefix = "simpler_env_widowx"
        extract_fn = _extract_widowx_bridge
        build_groot_obs_fn = _build_groot_obs_widowx
        modality = WIDOWX_BRIDGE_MODALITY
        video_key = "observation.images.image_0"
        image_size = (256, 256)
        state_dim = 8
        action_dim = 7
        state_names = ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]
        action_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    else:
        raise ValueError(f"Unknown robot_type: {args.robot_type}. Use 'google' or 'widowx'.")

    logging.info(f"Robot type: {args.robot_type}, tasks: {len(task_names)}")

    policy = PolicyClient(host=args.policy_client_host, port=args.policy_client_port)
    writer = DreamDojoWriter(
        pathlib.Path(args.output_dir), video_key=video_key, env_fps=ENV_FPS,
    )

    total_episodes = 0
    total_successes = 0

    # ── Resume from existing data if requested ───────────────────────────────
    completed_episodes: set[tuple[str, int]] = set()
    if args.resume:
        all_task_names = task_names  # full list before skip_tasks
        completed_episodes = writer.resume_from_existing(all_task_names)
        total_episodes = writer.episode_index
        # Count successes from restored metadata
        total_successes = sum(1 for m in writer._episodes_meta if m["success"])
        logging.info(
            f"[Resume] Continuing from {total_episodes} episodes, "
            f"{total_successes} successes"
        )

    if args.skip_tasks > 0:
        logging.info(f"Skipping first {args.skip_tasks} tasks")
        task_names = task_names[args.skip_tasks:]

    for task_name in tqdm(task_names, desc="tasks"):
        env_id = f"{env_prefix}/{task_name}"
        logging.info(f"\n=== Task: {env_id} ===")

        # Reuse a single env per task (matching official eval behavior)
        env = gym.make(env_id)

        for trial_idx in tqdm(range(args.num_trials_per_task), desc="episodes", leave=False):
            # Skip already-completed episodes when resuming
            if (task_name, trial_idx) in completed_episodes:
                logging.info(f"[Resume] Skip: {task_name} trial {trial_idx} (already done)")
                continue

            # Pass seed through to the underlying ManiSkill2 env for
            # reproducible randomisation (simpler_env.py now forwards it).
            obs, info = env.reset(seed=args.seed + trial_idx)

            writer.begin_episode(task_name)

            done, truncated = False, False

            # ── Main episode loop (matches official eval_simpler.py) ─────
            for t in range(args.max_episode_steps):
                if done or truncated:
                    break

                try:
                    # ── record observation ────────────────────────────────
                    image, state_vec, action_state_vec = extract_fn(obs)
                    writer.record_step(
                        image=image,
                        state=state_vec,
                        action=action_state_vec,
                    )

                    # ── query GR00T policy (once per outer step) ──────────
                    groot_obs = build_groot_obs_fn(obs)
                    action_dict, _ = policy.get_action(groot_obs)
                    actions = _convert_to_simpler_actions(
                        action_dict, args.action_horizon
                    )

                    # ── execute action_horizon steps ──────────────────────
                    # With default action_horizon=1, this queries every step
                    # (matching official eval behavior).
                    for j in range(args.action_horizon):
                        action = actions[j]
                        obs, reward, done, truncated, info = env.step(action)
                        if done or truncated:
                            break

                except Exception as e:
                    logging.error(f"Step exception: {e}")
                    break

            # ── save episode (both success and failure) ──────────────────────
            success = done  # done=True means task succeeded
            if success:
                total_successes += 1
            writer.end_episode(success=success)
            total_episodes += 1

            logging.info(
                f"Episode {trial_idx+1}: {'SUCCESS' if success else 'FAIL'} | "
                f"total {total_successes}/{total_episodes} "
                f"({100*total_successes/max(total_episodes,1):.1f}%)"
            )

        env.close()

    # ── write all metadata files ─────────────────────────────────────────────
    writer.finalize(
        modality=modality,
        state_dim=state_dim,
        action_dim=action_dim,
        state_names=state_names,
        action_names=action_names,
        image_size=image_size,
    )

    logging.info(
        f"Final: {total_successes}/{total_episodes} "
        f"({100*total_successes/max(total_episodes,1):.1f}%)"
    )


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tyro
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    collect_simpler_env(args)
