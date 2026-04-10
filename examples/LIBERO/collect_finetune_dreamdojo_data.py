"""
Rollout GR00T on LIBERO and collect data in LeRobot format for DreamDojo fine-tuning.

Iterates over all tasks in a given LIBERO task suite (e.g. libero_spatial,
libero_10, …), runs multiple trials per task using a remote GR00T policy
server, and saves the collected trajectories.

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
        observation.images.agentview/
          episode_000000.mp4
          ...
        observation.images.wrist/
          episode_000000.mp4
          ...

Action / state convention (matches DreamDojo's relative-action rebaselining):
  observation.state  [8D]: eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)  -- absolute
  action             [7D]: eef_pos(3) + eef_axisangle(3) + gripper_qpos[0]  -- absolute EE state
    -> stored as absolute so WrappedLeRobotSingleDataset can compute chunk-relative deltas

In DreamDojo's 384-dim action vector the 7D arm action is placed in the reserved slot [169:176].
See DreamDojo/groot_dreams/data/dataset.py WrappedLeRobotSingleDataset.__getitem__.

Prerequisites:
    Start the GR00T server first (Terminal 1):
        uv run --extra=gpu python gr00t/eval/run_gr00t_server.py \
            --model-path /tmp/libero_spatial/checkpoint-20000/ \
            --embodiment-tag LIBERO_PANDA \
            --use-sim-policy-wrapper

Usage (Terminal 2):
    gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python \
        examples/LIBERO/collect_finetune_dreamdojo_data.py \
        --policy_client_host 127.0.0.1 \
        --policy_client_port 5555 \
        --task_suite_name libero_spatial \
        --num_trials_per_task 50 \
        --output_dir data/libero_dreamdojo
"""

import collections
import dataclasses
import json
import logging
import math
import pathlib
from typing import Optional

import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
ENV_FPS = 10
CHUNK_SIZE = 1000
STATE_DIM = 8   # eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)
ACTION_DIM = 7  # eef_pos(3) + eef_axisangle(3) + gripper_qpos[0](1)

# Modality JSON written alongside the dataset so DreamDojo can parse it.
LIBERO_MODALITY = {
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
        "gripper": {
            "original_key": "observation.state",
            "start": 6, "end": 8,
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
        "agentview": {
            "original_key": "observation.images.agentview",
        },
        "wrist": {
            "original_key": "observation.images.wrist",
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
    n_action_steps: int = 8

    # LIBERO
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    # Data output
    output_dir: str = "data/libero_dreamdojo"
    save_failed: bool = False   # also save failed episodes

    seed: int = 7


# ─────────────────────────────────────────────────────────────────────────────
# DreamDojo data writer
# ─────────────────────────────────────────────────────────────────────────────

class DreamDojoWriter:
    """Incrementally writes a LeRobot-format dataset compatible with DreamDojo."""

    def __init__(self, output_dir: pathlib.Path, env_fps: int = ENV_FPS):
        self.root = pathlib.Path(output_dir)
        self.env_fps = env_fps

        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        (self.root / "videos").mkdir(parents=True, exist_ok=True)

        # Running counters
        self.episode_index: int = 0
        self.global_frame_index: int = 0

        # Registered tasks: description -> task_index
        self.task_registry: dict[str, int] = {}

        # Per-episode buffers (populated by record_step)
        self._ep_states: list[np.ndarray] = []
        self._ep_actions: list[np.ndarray] = []
        self._ep_agentview: list[np.ndarray] = []
        self._ep_wrist: list[np.ndarray] = []
        self._ep_task_index: Optional[int] = None

        # Accumulated stats for normalization (per column, list of arrays)
        self._stat_accum: dict[str, list[np.ndarray]] = {
            "observation.state": [],
            "action": [],
        }

        # Written episode metadata (for episodes.jsonl)
        self._episodes_meta: list[dict] = []

    # ── episode lifecycle ────────────────────────────────────────────────────

    def begin_episode(self, task_description: str) -> None:
        if task_description not in self.task_registry:
            self.task_registry[task_description] = len(self.task_registry)
        self._ep_task_index = self.task_registry[task_description]
        self._ep_states = []
        self._ep_actions = []
        self._ep_agentview = []
        self._ep_wrist = []

    def record_step(
        self,
        agentview_img: np.ndarray,   # (H, W, 3) uint8
        wrist_img: np.ndarray,       # (H, W, 3) uint8
        eef_pos: np.ndarray,         # (3,)
        eef_quat: np.ndarray,        # (4,)  xyzw
        gripper_qpos: np.ndarray,    # (2,)
    ) -> None:
        axisangle = _quat2axisangle(eef_quat)                        # (3,)
        state = np.concatenate([eef_pos, axisangle, gripper_qpos])   # (8,)
        action = np.concatenate([eef_pos, axisangle, gripper_qpos[:1]])  # (7,)

        self._ep_states.append(state.astype(np.float64))
        self._ep_actions.append(action.astype(np.float64))
        self._ep_agentview.append(agentview_img)
        self._ep_wrist.append(wrist_img)

    def end_episode(self, success: bool) -> bool:
        """Flush episode to disk. Returns True if episode was written."""
        n = len(self._ep_states)
        if n == 0:
            return False

        ep_idx = self.episode_index
        chunk_idx = ep_idx // CHUNK_SIZE

        # ── write videos ────────────────────────────────────────────────────
        for cam_key, frames in [
            ("observation.images.agentview", self._ep_agentview),
            ("observation.images.wrist",     self._ep_wrist),
        ]:
            vid_dir = self.root / "videos" / f"chunk-{chunk_idx:03d}" / cam_key
            vid_dir.mkdir(parents=True, exist_ok=True)
            vid_path = vid_dir / f"episode_{ep_idx:06d}.mp4"
            imageio.mimwrite(
                vid_path,
                [np.asarray(f) for f in frames],
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
        """Discard the current in-progress episode without saving."""
        self._ep_states = []
        self._ep_actions = []
        self._ep_agentview = []
        self._ep_wrist = []
        self._ep_task_index = None

    # ── finalization ─────────────────────────────────────────────────────────

    def finalize(self) -> None:
        """Write all meta files after all episodes are collected."""
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
                    "shape":  [STATE_DIM],
                    "names":  ["eef_pos_x", "eef_pos_y", "eef_pos_z",
                               "eef_rot_x", "eef_rot_y", "eef_rot_z",
                               "gripper_0", "gripper_1"],
                    "dtype":  "float64",
                },
                "action": {
                    "shape": [ACTION_DIM],
                    "names": ["eef_pos_x", "eef_pos_y", "eef_pos_z",
                              "eef_rot_x", "eef_rot_y", "eef_rot_z",
                              "gripper"],
                    "dtype": "float64",
                },
                "observation.images.agentview": {
                    "shape":  [LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3],
                    "names":  ["height", "width", "channel"],
                    "dtype":  "uint8",
                    "video_info": {"video.fps": float(self.env_fps)},
                },
                "observation.images.wrist": {
                    "shape":  [LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3],
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
            json.dump(LIBERO_MODALITY, f, indent=2)

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
            data = np.stack(arrays, axis=0)  # (N, D)
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
# Main rollout loop
# ─────────────────────────────────────────────────────────────────────────────

def collect_libero(args: Args) -> None:
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from gr00t.policy.server_client import PolicyClient

    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}  ({num_tasks} tasks)")

    # max_steps per task suite (same as original eval script)
    max_steps_map = {
        "libero_spatial": 220,
        "libero_object":  280,
        "libero_goal":    300,
        "libero_10":      520,
        "libero_90":      400,
    }
    max_steps = max_steps_map.get(args.task_suite_name, 300)

    policy = PolicyClient(host=args.policy_client_host, port=args.policy_client_port)
    writer = DreamDojoWriter(pathlib.Path(args.output_dir), env_fps=ENV_FPS)

    total_episodes = 0
    total_successes = 0

    for task_id in tqdm(range(num_tasks), desc="tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        logging.info(f"\n=== Task {task_id}: {task_description} ===")

        for episode_idx in tqdm(range(args.num_trials_per_task), desc="episodes", leave=False):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            action_plan: collections.deque = collections.deque()

            writer.begin_episode(task_description)

            t = 0
            done = False

            while t < max_steps + args.num_steps_wait:
                try:
                    # Warm-up: let objects settle
                    if t < args.num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # ── build observation ────────────────────────────────────
                    agentview = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist     = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    # ── record step (absolute EE state for DreamDojo) ────────
                    writer.record_step(
                        agentview_img=agentview,
                        wrist_img=wrist,
                        eef_pos=obs["robot0_eef_pos"].copy(),
                        eef_quat=obs["robot0_eef_quat"].copy(),
                        gripper_qpos=obs["robot0_gripper_qpos"].copy(),
                    )

                    # ── query GR00T policy ───────────────────────────────────
                    if not action_plan:
                        groot_obs = _build_groot_observation(
                            agentview, wrist, obs, task_description
                        )
                        action_dict, _ = policy.get_action(groot_obs)
                        action_chunk = _groot_actions_to_env_list(
                            action_dict, args.n_action_steps
                        )
                        assert len(action_chunk) >= args.n_action_steps
                        action_plan.extend(action_chunk[: args.n_action_steps])

                    action = action_plan.popleft()
                    obs, _, done, _ = env.step(action.tolist())

                    if done:
                        total_successes += 1
                        break

                    t += 1

                except Exception as e:
                    logging.error(f"Step exception: {e}")
                    break

            # ── end of episode ───────────────────────────────────────────────
            if done or args.save_failed:
                writer.end_episode(success=done)
                total_episodes += 1
            else:
                writer.discard_episode()

            logging.info(
                f"Episode {episode_idx+1}: {'SUCCESS' if done else 'FAIL'} | "
                f"total {total_successes}/{total_episodes} "
                f"({100*total_successes/max(total_episodes,1):.1f}%)"
            )

        env.close()

    # ── write all metadata files ─────────────────────────────────────────────
    writer.finalize()

    logging.info(f"Final success rate: {total_successes}/{total_episodes} "
                 f"({100*total_successes/max(total_episodes,1):.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# GR00T policy helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_groot_observation(img, wrist_img, obs, task_description):
    """Build a batched observation dict for GR00T policy from raw LIBERO obs.

    GR00T expects the Gr00tSimPolicyWrapper format:
        video.image:   (B, T, H, W, C) uint8
        video.wrist_image: (B, T, H, W, C) uint8
        state.x/y/z/roll/pitch/yaw: (B, T, 1) float32
        state.gripper: (B, T, 2) float32
        annotation.human.action.task_description: tuple[str] of length B
    """
    xyz = obs["robot0_eef_pos"]
    rpy = _quat2axisangle(obs["robot0_eef_quat"])
    gripper = obs["robot0_gripper_qpos"]

    return {
        "video.image": img[np.newaxis, np.newaxis].astype(np.uint8),
        "video.wrist_image": wrist_img[np.newaxis, np.newaxis].astype(np.uint8),
        "state.x": np.array([[[xyz[0]]]], dtype=np.float32),
        "state.y": np.array([[[xyz[1]]]], dtype=np.float32),
        "state.z": np.array([[[xyz[2]]]], dtype=np.float32),
        "state.roll": np.array([[[rpy[0]]]], dtype=np.float32),
        "state.pitch": np.array([[[rpy[1]]]], dtype=np.float32),
        "state.yaw": np.array([[[rpy[2]]]], dtype=np.float32),
        "state.gripper": gripper[np.newaxis, np.newaxis].astype(np.float32),
        "annotation.human.action.task_description": (str(task_description),),
    }


def _groot_actions_to_env_list(action_dict: dict, num_steps: int) -> list[np.ndarray]:
    """Convert GR00T policy output to a list of 7-dim env action vectors.

    GR00T returns: dict[str, np.ndarray(B, T, D)]
    We need a list of T action vectors of shape (7,), with gripper
    normalized and inverted to match LIBERO env convention.
    """
    x = action_dict["action.x"][0]           # (T, 1)
    y = action_dict["action.y"][0]
    z = action_dict["action.z"][0]
    roll = action_dict["action.roll"][0]
    pitch = action_dict["action.pitch"][0]
    yaw = action_dict["action.yaw"][0]
    gripper = action_dict["action.gripper"][0]

    actions = np.concatenate([x, y, z, roll, pitch, yaw, gripper], axis=-1)  # (T, 7)

    # Normalize gripper from [0,1] -> [-1,+1] and invert sign for LIBERO
    actions = _normalize_gripper_action(actions)
    actions = _invert_gripper_action(actions)

    return [actions[t] for t in range(min(num_steps, len(actions)))]


# ─────────────────────────────────────────────────────────────────────────────
# LIBERO / math helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_libero_env(task, resolution: int, seed: int):
    import os
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) -> axis-angle (3,).
    Copied from robosuite transform_utils."""
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _normalize_gripper_action(action, binarize=True):
    """Changes gripper action (last dim) from [0,1] to [-1,+1]."""
    action = action.copy()
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def _invert_gripper_action(action):
    """Flips the sign of the gripper action (last dim)."""
    action = action.copy()
    action[..., -1] *= -1.0
    return action


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tyro
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    collect_libero(args)
