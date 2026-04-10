"""
Convert a HuggingFace LeRobot dataset to DreamDojo format.

Supports both Google Fractal (google_robot) and WidowX Bridge datasets.

The HF dataset has:
  - data/chunk-*/episode_*.parquet  (columns: observation.state, action, timestamp,
    frame_index, episode_index, index, task_index)
  - videos/chunk-*/observation.images.image/episode_*.mp4  (may be absent if not downloaded)

DreamDojo additionally requires:
  - meta/info.json, modality.json, tasks.jsonl, episodes.jsonl, stats.json
  - Parquet columns: next.done, success
  - float64 for observation.state / action

This script:
  1. Scans all existing parquet files
  2. Adds missing columns (next.done, success) and casts to float64
  3. Generates all meta/ files (adapted to the dataset's robot type)
  4. Symlinks (or copies) video files if they exist

Usage:
    # Google Fractal (auto-detected from dir name):
    python examples/SimplerEnv/convert_fractal_lerobot_to_dreamdojo.py \
        --input_dir examples/SimplerEnv/fractal20220817_data_lerobot \
        --output_dir examples/SimplerEnv/fractal20220817_data_dreamdojo

    # WidowX Bridge:
    python examples/SimplerEnv/convert_fractal_lerobot_to_dreamdojo.py \
        --input_dir examples/SimplerEnv/bridge_orig_lerobot \
        --output_dir examples/SimplerEnv/bridge_orig_dreamdojo

    # Explicit robot type:
    python examples/SimplerEnv/convert_fractal_lerobot_to_dreamdojo.py \
        --input_dir /path/to/data --output_dir /path/to/out --robot_type widowx

    # In-place (modifies input_dir directly):
    python examples/SimplerEnv/convert_fractal_lerobot_to_dreamdojo.py \
        --input_dir examples/SimplerEnv/fractal20220817_data_lerobot \
        --inplace
"""

import dataclasses
import json
import logging
import pathlib
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm
import tyro


CHUNK_SIZE = 1000

# ── Per-robot-type configurations ─────────────────────────────────────────────

ROBOT_CONFIGS = {
    "google": {
        "fps": 3,
        "state_dim": 8,
        "action_dim": 7,
        "state_names": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        "action_names": ["x", "y", "z", "qx", "qy", "qz", "gripper"],
        "image_size": (256, 320),
        "video_key": "observation.images.image",
        "robot_type": "google_robot",
        "modality": {
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
        },
    },
    "widowx": {
        "fps": 5,
        "state_dim": 8,
        "action_dim": 7,
        "state_names": ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"],
        "action_names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        "image_size": (256, 256),
        "video_key": "observation.images.image_0",
        "robot_type": "widowx",
        "modality": {
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
        },
    },
}


def _detect_robot_type(input_dir: str) -> str:
    """Auto-detect robot type from the input directory name."""
    name = pathlib.Path(input_dir).name.lower()
    if "bridge" in name or "widowx" in name:
        return "widowx"
    if "fractal" in name or "google" in name:
        return "google"
    raise ValueError(
        f"Cannot auto-detect robot type from directory name '{name}'. "
        f"Please specify --robot_type explicitly (one of: {list(ROBOT_CONFIGS.keys())})"
    )


@dataclasses.dataclass
class Args:
    input_dir: str = "examples/SimplerEnv/fractal20220817_data_lerobot"
    """Path to the downloaded HF LeRobot dataset."""

    output_dir: str = ""
    """Output directory for DreamDojo format. If empty, requires --inplace."""

    inplace: bool = False
    """Modify input_dir in-place (add meta/ and patch parquets)."""

    robot_type: str = "auto"
    """Robot type: 'google', 'widowx', or 'auto' (detect from directory name)."""

    fps: int = 0
    """Override FPS. If 0, uses the default for the robot type."""

    max_episodes: int = 0
    """If > 0, only process the first N episodes (for testing)."""


def convert(args: Args) -> None:
    input_dir = pathlib.Path(args.input_dir)
    if args.inplace:
        output_dir = input_dir
    elif args.output_dir:
        output_dir = pathlib.Path(args.output_dir)
    else:
        raise ValueError("Specify --output_dir or --inplace")

    if not (input_dir / "data").exists():
        raise FileNotFoundError(f"No data/ directory found in {input_dir}")

    # ── Resolve robot type and config ────────────────────────────────────────
    rtype = args.robot_type if args.robot_type != "auto" else _detect_robot_type(args.input_dir)
    if rtype not in ROBOT_CONFIGS:
        raise ValueError(f"Unknown robot_type '{rtype}'. Choose from: {list(ROBOT_CONFIGS.keys())}")
    cfg = ROBOT_CONFIGS[rtype]
    env_fps = args.fps if args.fps > 0 else cfg["fps"]
    logging.info(f"Robot type: {rtype}, fps: {env_fps}")

    # ── Discover all parquet files ───────────────────────────────────────────
    parquet_files = sorted(input_dir.glob("data/chunk-*/episode_*.parquet"))
    logging.info(f"Found {len(parquet_files)} parquet files in {input_dir}")

    if args.max_episodes > 0:
        parquet_files = parquet_files[:args.max_episodes]
        logging.info(f"Processing only first {args.max_episodes} episodes")

    if not parquet_files:
        raise FileNotFoundError("No parquet files found")

    # ── Prepare output directory ─────────────────────────────────────────────
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)

    if not args.inplace and output_dir != input_dir:
        # Copy data/ tree structure
        (output_dir / "data").mkdir(parents=True, exist_ok=True)
        (output_dir / "videos").mkdir(parents=True, exist_ok=True)

    # ── Process parquets, collect metadata ───────────────────────────────────
    task_registry: dict[int, str] = {}  # task_index -> task_description (placeholder)
    episodes_meta: list[dict] = []
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    global_frame_index = 0

    skipped = 0
    for pq_path in tqdm(parquet_files, desc="Processing episodes"):
        # ── Peek at episode_index to check if output already exists ──────────
        df_peek = pd.read_parquet(pq_path, columns=["episode_index"])
        ep_idx = int(df_peek["episode_index"].iloc[0])
        chunk_idx = ep_idx // CHUNK_SIZE
        out_data_dir = output_dir / "data" / f"chunk-{chunk_idx:03d}"
        out_path = out_data_dir / f"episode_{ep_idx:06d}.parquet"

        if out_path.exists() and not args.inplace:
            # ── Resume: read already-converted parquet for stats/metadata ─────
            df = pd.read_parquet(out_path)
            n = len(df)
            states = np.stack(df["observation.state"].values).astype(np.float64)
            actions = np.stack(df["action"].values).astype(np.float64)
            all_states.append(states)
            all_actions.append(actions)

            task_idx = int(df["task_index"].iloc[0])
            if task_idx not in task_registry:
                task_registry[task_idx] = f"{rtype}_task_{task_idx}"

            episodes_meta.append({
                "episode_index": ep_idx,
                "tasks": [task_registry[task_idx]],
                "length": n,
                "success": False,
            })
            global_frame_index += n
            skipped += 1
            continue

        # ── Full read & process ──────────────────────────────────────────────
        df = pd.read_parquet(pq_path)
        n = len(df)

        # ── Add missing columns ──────────────────────────────────────────────
        if "next.done" not in df.columns:
            done_flags = [False] * (n - 1) + [True]
            df["next.done"] = done_flags

        if "success" not in df.columns:
            # No ground-truth success label available; default to False
            df["success"] = False

        # ── Cast to float64 ──────────────────────────────────────────────────
        states = np.stack(df["observation.state"].values).astype(np.float64)
        actions = np.stack(df["action"].values).astype(np.float64)
        df["observation.state"] = list(states)
        df["action"] = list(actions)

        # ── Fix global frame index if processing a subset ────────────────────
        df["index"] = range(global_frame_index, global_frame_index + n)
        df["frame_index"] = range(n)
        df["episode_index"] = ep_idx

        # ── Collect stats ────────────────────────────────────────────────────
        all_states.append(states)
        all_actions.append(actions)

        # ── Task registry ────────────────────────────────────────────────────
        task_idx = int(df["task_index"].iloc[0])
        if task_idx not in task_registry:
            # We don't have the original task description from parquets alone;
            # use a placeholder that can be updated later.
            task_registry[task_idx] = f"{rtype}_task_{task_idx}"

        # ── Episode metadata ─────────────────────────────────────────────────
        episodes_meta.append({
            "episode_index": ep_idx,
            "tasks": [task_registry[task_idx]],
            "length": n,
            "success": False,
        })

        global_frame_index += n

        # ── Write patched parquet ────────────────────────────────────────────
        out_data_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)

    if skipped > 0:
        logging.info(f"Resumed: skipped {skipped} already-converted episodes, processed {len(parquet_files) - skipped} new episodes")

    # ── Symlink/copy videos if they exist ────────────────────────────────────
    src_videos = input_dir / "videos"
    dst_videos = output_dir / "videos"
    if src_videos.exists() and any(src_videos.rglob("*.mp4")):
        if not args.inplace and output_dir != input_dir:
            logging.info("Copying video files...")
            if dst_videos.exists():
                shutil.rmtree(dst_videos)
            shutil.copytree(src_videos, dst_videos)
        logging.info(f"Videos present at {dst_videos}")
    else:
        dst_videos.mkdir(parents=True, exist_ok=True)
        logging.warning(
            "No video files found in source dataset. "
            "DreamDojo training requires videos — download them separately:\n"
            "  huggingface-cli download <dataset_name> --include 'videos/**' "
            f"--local-dir {output_dir}"
        )

    # ── Try to load tasks.jsonl from HF metadata if available ────────────────
    hf_tasks_path = input_dir / "meta" / "tasks.jsonl"
    if hf_tasks_path.exists():
        logging.info("Found existing meta/tasks.jsonl, loading task descriptions...")
        with open(hf_tasks_path) as f:
            for line in f:
                entry = json.loads(line)
                task_registry[entry["task_index"]] = entry["task"]

    # ── Write meta/info.json ─────────────────────────────────────────────────
    total_ep = len(episodes_meta)
    total_frames = global_frame_index

    # Auto-detect actual dimensions from collected data
    state_dim = cfg["state_dim"]
    action_dim = cfg["action_dim"]
    if all_states:
        state_dim = all_states[0].shape[1]
    if all_actions:
        action_dim = all_actions[0].shape[1]

    info = {
        "codebase_version": "v2.0",
        "robot_type": cfg["robot_type"],
        "fps": env_fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "chunks_size": CHUNK_SIZE,
        "total_episodes": total_ep,
        "total_frames": total_frames,
        "total_tasks": len(task_registry),
        "features": {
            "observation.state": {
                "shape": [state_dim],
                "names": cfg["state_names"],
                "dtype": "float64",
            },
            "action": {
                "shape": [action_dim],
                "names": cfg["action_names"],
                "dtype": "float64",
            },
            cfg["video_key"]: {
                "shape": [cfg["image_size"][0], cfg["image_size"][1], 3],
                "names": ["height", "width", "channel"],
                "dtype": "uint8",
                "video_info": {"video.fps": float(env_fps)},
            },
        },
    }
    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    logging.info(f"Wrote meta/info.json (robot_type={cfg['robot_type']})")

    # ── Write meta/modality.json ─────────────────────────────────────────────
    with open(output_dir / "meta" / "modality.json", "w") as f:
        json.dump(cfg["modality"], f, indent=2)
    logging.info("Wrote meta/modality.json")

    # ── Write meta/tasks.jsonl ───────────────────────────────────────────────
    with open(output_dir / "meta" / "tasks.jsonl", "w") as f:
        for idx in sorted(task_registry.keys()):
            f.write(json.dumps({"task_index": idx, "task": task_registry[idx]}) + "\n")
    logging.info(f"Wrote meta/tasks.jsonl ({len(task_registry)} tasks)")

    # ── Write meta/episodes.jsonl ────────────────────────────────────────────
    with open(output_dir / "meta" / "episodes.jsonl", "w") as f:
        for meta in episodes_meta:
            f.write(json.dumps(meta) + "\n")
    logging.info(f"Wrote meta/episodes.jsonl ({total_ep} episodes)")

    # ── Write meta/stats.json ────────────────────────────────────────────────
    stats = {}
    if all_states:
        state_data = np.concatenate(all_states, axis=0)
        stats["observation.state"] = {
            "mean": state_data.mean(0).tolist(),
            "std": state_data.std(0).tolist(),
            "min": state_data.min(0).tolist(),
            "max": state_data.max(0).tolist(),
            "q01": np.quantile(state_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(state_data, 0.99, axis=0).tolist(),
        }
    if all_actions:
        action_data = np.concatenate(all_actions, axis=0)
        stats["action"] = {
            "mean": action_data.mean(0).tolist(),
            "std": action_data.std(0).tolist(),
            "min": action_data.min(0).tolist(),
            "max": action_data.max(0).tolist(),
            "q01": np.quantile(action_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(action_data, 0.99, axis=0).tolist(),
        }
    with open(output_dir / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logging.info("Wrote meta/stats.json")

    logging.info(
        f"\nConversion complete: {total_ep} episodes, {total_frames} frames, "
        f"{len(task_registry)} tasks -> {output_dir}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    convert(tyro.cli(Args))
