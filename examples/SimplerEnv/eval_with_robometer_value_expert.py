"""
SimplerEnv evaluation with Robometer-based action scoring + DINOv2 Value Expert.

During SDE rescue, samples N action candidates, generates DreamDojo rollout
videos, and uses Robometer (from /home/zhiqil/workspace/fxz/robometer) to
score each candidate.  The (video, score) pairs are saved for training a
DINOv2-based Value Expert that predicts the final outcome score from
multi-view video input.

Value Expert architecture:
  - Frozen DINOv2 backbone encodes each frame
  - Multi-view frames are spatially concatenated before encoding
    (if only 1 camera, the original single-view video is used as-is)
  - Temporal mean-pooling over frame features
  - MLP regressor outputs a scalar score in [0, 1]

Rescue detection uses Gemini (same as eval_with_gemini_rescue_dreamdojo_groot.py).
Robometer is only used during rescue to score the candidate action videos.

Phases:
  1. eval   – Run SimplerEnv rollouts, detect rescue with Gemini, score
              candidates with Robometer, save training data.
  2. train  – Train the DINOv2 Value Expert on collected data.

Prerequisites:
  - Robometer: /home/zhiqil/workspace/fxz/robometer (add to PYTHONPATH)
  - DreamDojo server(s) running on consecutive ports
  - GR00T checkpoint

Usage:
  # 1) Eval + collect training data (needs DreamDojo + Robometer)
  python examples/SimplerEnv/eval_with_robometer_value_expert.py eval \\
      --model_path nvidia/GR00T-N1.6-3B \\
      --robot_type google \\
      --dd_base_port 8020 \\
      --robometer_model_path aliangdw/qwen4b_pref_prog_succ_8_frames_all_part2 \\
      --output_dir data/robometer_value_expert

  # 2) Train value expert
  python examples/SimplerEnv/eval_with_robometer_value_expert.py train \\
      --data_dir data/robometer_value_expert \\
      --output_dir checkpoints/value_expert \\
      --epochs 30
"""

import argparse
import base64
import collections
import concurrent.futures
import glob
import json
import logging
import os
import pathlib
import shutil
import sys
import tempfile
import threading
import time

import cv2
import imageio
import numpy as np
import requests
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from pydantic import BaseModel
from google import genai
from google.genai import types
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb
except ImportError:
    wandb = None

# Robometer imports (add to PYTHONPATH if needed)
ROBOMETER_ROOT = "/home/zhiqil/workspace/fxz/robometer"
if ROBOMETER_ROOT not in sys.path:
    sys.path.insert(0, ROBOMETER_ROOT)

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.evals.eval_server import compute_batch_outputs
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator

# GR00T imports
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.gr00t_n1d7.gr00t_n1d6_sde import Gr00tN1d6SDEActionHead
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROLLOUT_FPS = 10

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

GEMINI_QUERY_INTERVAL_FRAMES = 20
GEMINI_HISTORY_FRAMES = 200
GEMINI_VALUE_MODEL = "gemini-3.1-flash-lite-preview"

RESCUE_SCORE_ABSOLUTE = 0.40
RESCUE_SCORE_DROP = 0.15


# ============================================================================
#  Gemini value scoring (rescue detection — same as original)
# ============================================================================

class ValueEvaluation(BaseModel):
    reasoning: str
    score: float
    status: str


_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(http_options={"api_version": "v1alpha"})
    return _gemini_client


def _query_gemini_value(frames: list, task_description: str, step_idx: int,
                        score_history: list, lock: threading.Lock) -> dict:
    """Query Gemini for a value score on a video clip. Thread-safe."""
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

    if latest_score <= RESCUE_SCORE_ABSOLUTE:
        logging.info(f"[Rescue] Triggered: score {latest_score:.2f} <= {RESCUE_SCORE_ABSOLUTE}")
        return True

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


def _write_gemini_results(gemini_results: list, rollout_dir: pathlib.Path,
                          task_description: str, episode_idx: int, suffix: str,
                          rescue_log: list | None = None,
                          rescue_selections: list | None = None):
    """Write Gemini value evaluation results to disk."""
    out_path = rollout_dir / "gemini_values.txt"
    with open(out_path, "w") as f:
        f.write(f"Task: {task_description}\n")
        f.write(f"Episode: {episode_idx}  Outcome: {suffix}\n")
        f.write("=" * 60 + "\n\n")
        for entry in sorted(gemini_results, key=lambda x: x.get("step", 0)):
            step = entry.get("step", "?")
            ts = f"{step / ROLLOUT_FPS:.1f}s" if isinstance(step, int) else "?"
            f.write(f"[Frame {step} / ~{ts}]\n")
            if "error" in entry:
                f.write(f"  ERROR: {entry['error']}\n")
            else:
                for item in entry.get("result", []):
                    f.write(f"  Status : {item.get('status', '')}\n")
                    f.write(f"  Score  : {item.get('score', '')}\n")
                    f.write(f"  Reason : {item.get('reasoning', '')}\n")
            f.write("\n")

    json_path = rollout_dir / "gemini_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "task": task_description,
            "episode_index": episode_idx,
            "outcome": suffix,
            "value_evaluations": sorted(gemini_results, key=lambda x: x.get("step", 0)),
            "rescue_activations": rescue_log or [],
            "rescue_selections": rescue_selections or [],
        }, f, indent=2)


# ============================================================================
#  DINOv2 Value Expert Model
# ============================================================================

class DINOv2ValueExpert(nn.Module):
    """
    DINOv2-based value expert that predicts a scalar outcome score from video.

    Input:  Multi-view video frames spatially concatenated into a single image
            per timestep.  For single-view, the original frames are used as-is.
    Output: Scalar score in [0, 1] (after sigmoid).

    Architecture:
      1. Frozen DINOv2 backbone encodes each frame → (B*T, feature_dim)
      2. Reshape to (B, T, feature_dim), temporal mean-pool → (B, feature_dim)
      3. MLP regressor → scalar logit
    """

    def __init__(
        self,
        dinov2_model: str = "dinov2_vitb14",
        hidden_dim: int = 256,
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

        self.regressor = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode video frames through DINOv2.

        Args:
            frames: (B, T, 3, H, W) float [0, 1]
                    For multi-view, views are already spatially concatenated
                    along the width dimension before calling this.

        Returns:
            (B, feature_dim) temporally-pooled features
        """
        B, T, C, H, W = frames.shape
        flat = frames.reshape(B * T, C, H, W)

        flat = F.interpolate(flat, size=(224, 224), mode="bilinear", align_corners=False)
        flat = (flat - self.img_mean) / self.img_std

        if self.freeze_backbone:
            with torch.no_grad():
                feat = self.backbone(flat)  # (B*T, feature_dim)
        else:
            feat = self.backbone(flat)

        feat = feat.view(B, T, -1)
        return feat.mean(dim=1)  # (B, feature_dim) temporal mean pool

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, 3, H, W) float [0, 1]
                    Multi-view frames should be spatially concatenated.

        Returns:
            logit: (B,) raw logit (apply sigmoid for score)
        """
        pooled = self._encode_frames(frames)
        return self.regressor(pooled).squeeze(-1)

    def predict_score(self, frames: torch.Tensor) -> torch.Tensor:
        """Returns predicted score in [0, 1]."""
        return torch.sigmoid(self.forward(frames))


# ============================================================================
#  Robometer scoring
# ============================================================================

class RobometerScorer:
    """
    Wraps the Robometer model for scoring candidate action videos.

    Loads the model once and scores videos by computing per-frame progress
    and success probabilities. Returns the final-frame progress as the score.
    """

    def __init__(self, model_path: str, device: torch.device | None = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model_path = model_path

        logging.info(f"[RobometerScorer] Loading model from {model_path} ...")
        self.exp_config, self.tokenizer, self.processor, self.reward_model = (
            load_model_from_hf(model_path=model_path, device=device)
        )
        self.reward_model.eval()
        self.batch_collator = setup_batch_collator(
            self.processor, self.tokenizer, self.exp_config, is_eval=True
        )

        loss_config = getattr(self.exp_config, "loss", None)
        self.is_discrete = (
            getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
            if loss_config else False
        )
        self.num_bins = (
            getattr(loss_config, "progress_discrete_bins", None)
            or getattr(self.exp_config.model, "progress_discrete_bins", 10)
        )
        logging.info("[RobometerScorer] Model loaded.")

    def score_video(
        self,
        video_frames: np.ndarray,
        task: str,
    ) -> dict:
        """
        Score a video clip.

        Args:
            video_frames: (T, H, W, C) uint8
            task: task description string

        Returns:
            dict with keys:
              - progress: np.ndarray of per-frame progress scores
              - success: np.ndarray of per-frame success probabilities
              - final_progress: float, last-frame progress score
              - final_success: float, last-frame success probability
        """
        T = int(video_frames.shape[0])
        traj = Trajectory(
            frames=video_frames,
            frames_shape=tuple(video_frames.shape),
            task=task,
            id="0",
            metadata={"subsequence_length": T},
            video_embeddings=None,
        )
        progress_sample = ProgressSample(trajectory=traj, sample_type="progress")
        batch = self.batch_collator([progress_sample])

        progress_inputs = batch["progress_inputs"]
        for key, value in progress_inputs.items():
            if hasattr(value, "to"):
                progress_inputs[key] = value.to(self.device)

        results = compute_batch_outputs(
            self.reward_model,
            self.tokenizer,
            progress_inputs,
            sample_type="progress",
            is_discrete_mode=self.is_discrete,
            num_bins=self.num_bins,
        )

        progress_pred = results.get("progress_pred", [])
        progress_array = (
            np.array(progress_pred[0], dtype=np.float32)
            if progress_pred and len(progress_pred) > 0
            else np.array([], dtype=np.float32)
        )

        outputs_success = results.get("outputs_success", {})
        success_probs = outputs_success.get("success_probs", []) if outputs_success else []
        success_array = (
            np.array(success_probs[0], dtype=np.float32)
            if success_probs and len(success_probs) > 0
            else np.array([], dtype=np.float32)
        )

        final_progress = float(progress_array[-1]) if progress_array.size > 0 else 0.0
        final_success = float(success_array[-1]) if success_array.size > 0 else 0.0

        return {
            "progress": progress_array,
            "success": success_array,
            "final_progress": final_progress,
            "final_success": final_success,
        }

    def score_candidates(
        self,
        history_frames: np.ndarray,
        candidate_video_paths: list[str],
        task: str,
    ) -> list[dict]:
        """
        Score multiple candidate continuation videos.

        Each candidate video is concatenated with the history to form
        the full context, then scored by Robometer.

        Args:
            history_frames: (T_hist, H, W, C) uint8, the current episode history
            candidate_video_paths: list of paths to DreamDojo-generated .mp4 files
            task: task description

        Returns:
            list of score dicts, one per candidate
        """
        scores = []
        for path in candidate_video_paths:
            try:
                reader = imageio.get_reader(path, "ffmpeg")
                cand_frames = np.array([f for f in reader])
                reader.close()

                # Concatenate history + candidate for full context
                combined = np.concatenate([history_frames, cand_frames], axis=0)
                result = self.score_video(combined, task)
                scores.append(result)
                logging.info(
                    f"[Robometer] {pathlib.Path(path).name}: "
                    f"progress={result['final_progress']:.3f} "
                    f"success={result['final_success']:.3f}"
                )
            except Exception as e:
                logging.error(f"[Robometer] Failed to score {path}: {e}")
                scores.append({
                    "progress": np.array([], dtype=np.float32),
                    "success": np.array([], dtype=np.float32),
                    "final_progress": 0.0,
                    "final_success": 0.0,
                    "error": str(e),
                })
        return scores


# ============================================================================
#  Multi-view spatial concatenation
# ============================================================================

def concat_multiview_frames(
    frames_list: list[np.ndarray],
) -> np.ndarray:
    """
    Spatially concatenate multi-view frames along the width axis.

    Args:
        frames_list: list of (T, H, W, C) uint8 arrays, one per camera view.
                     All must have the same T and H.
                     If only one view is provided, returns it as-is.

    Returns:
        (T, H, W_total, C) uint8 array
    """
    if len(frames_list) == 1:
        return frames_list[0]

    # Resize all views to have the same height
    target_h = frames_list[0].shape[1]
    resized = []
    for frames in frames_list:
        if frames.shape[1] != target_h:
            T, H, W, C = frames.shape
            new_w = int(W * target_h / H)
            out = np.empty((T, target_h, new_w, C), dtype=np.uint8)
            for t in range(T):
                out[t] = cv2.resize(frames[t], (new_w, target_h))
            resized.append(out)
        else:
            resized.append(frames)

    return np.concatenate(resized, axis=2)  # concat along width


# ============================================================================
#  DreamDojo generation
# ============================================================================

def _dreamdojo_generate(
    port: int,
    frame_np: np.ndarray,
    actions: np.ndarray,
    save_name: str,
    task_description: str = "",
    seed: int = 0,
) -> str | None:
    """POST a generation request to a dreamdojo_server.py instance."""
    url = f"http://127.0.0.1:{port}/generate"
    h, w = frame_np.shape[:2]
    frame_bytes = base64.b64encode(frame_np.tobytes()).decode()

    payload = {
        "frame": frame_bytes,
        "frame_height": h,
        "frame_width": w,
        "actions": actions.tolist(),
        "save_name": save_name,
        "prompt": task_description,
        "seed": seed,
    }
    try:
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()["save_path"]
    except Exception as e:
        logging.error(f"[DreamDojo port={port}] generation failed: {e}")
        return None


# ============================================================================
#  SimplerEnv helpers
# ============================================================================

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


def _actions_to_array(action_list: list[dict]) -> np.ndarray:
    rows = []
    for a in action_list:
        rows.append(np.concatenate([a[f"action.{k}"] for k in ACTION_KEYS]))
    return np.array(rows, dtype=np.float32)


# ============================================================================
#  Rescue action selection with Robometer scoring
# ============================================================================

def _rescue_select_action_robometer(
    obs: dict,
    replay_images: list,
    task_description: str,
    groot_policy,
    base_policy,
    sde_action_head,
    ode_action_head,
    build_groot_obs_fn,
    robometer_scorer: RobometerScorer,
    replan_steps: int,
    step_save_dir: pathlib.Path,
    dd_base_port: int,
    robot_type: str,
    num_samples: int = 5,
) -> tuple[list, dict]:
    """
    Sample action candidates via SDE, generate DreamDojo videos, score with
    Robometer, save training data, and return the best candidate.

    Returns:
        (best_actions, selection_record)
    """
    step_save_dir.mkdir(parents=True, exist_ok=True)
    img = _extract_image_from_obs(obs, robot_type)
    groot_obs = build_groot_obs_fn(obs)

    # Switch to SDE head for diverse sampling
    base_policy.model.action_head = sde_action_head

    action_chunks = []
    for i in range(num_samples):
        seed_i = int(time.time() * 1e6) % (2**31) + i
        torch.manual_seed(seed_i)
        torch.cuda.manual_seed(seed_i)
        action_dict, _ = groot_policy.get_action(groot_obs)
        action_list = _convert_to_simpler_actions(action_dict, replan_steps)
        action_chunks.append(action_list)

    # Restore ODE head
    base_policy.model.action_head = ode_action_head

    # Log action diversity
    for i in range(num_samples):
        arr = _actions_to_array(action_chunks[i][:replan_steps])
        logging.info(
            f"[Rescue] chunk_{i} actions mean={arr.mean():.6f} "
            f"std={arr.std():.6f} first={arr[0, :3]}"
        )

    # Parallel DreamDojo generation
    save_prefix = step_save_dir.name
    base_seed = int(time.time() * 1e6) % (2**31)
    tasks = [
        {
            "port": dd_base_port + i,
            "actions": _actions_to_array(action_chunks[i][:replan_steps]),
            "save_name": f"{save_prefix}/chunk_{i}",
            "seed": base_seed + i,
        }
        for i in range(num_samples)
    ]

    logging.info(f"[Rescue] Launching {num_samples} parallel DreamDojo requests...")

    def _submit(t):
        return _dreamdojo_generate(
            t["port"], img, t["actions"], t["save_name"], task_description,
            seed=t["seed"],
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_samples) as ex:
        futures = {ex.submit(_submit, t): i for i, t in enumerate(tasks)}
        save_paths = {}
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            save_paths[idx] = fut.result()

    valid = [
        (i, save_paths[i])
        for i in range(num_samples)
        if save_paths.get(i) and os.path.exists(save_paths[i])
    ]

    if not valid:
        logging.warning("[Rescue] All DreamDojo generations failed; using chunk 0.")
        return list(action_chunks[0][:replan_steps]), {
            "num_candidates": 0,
            "candidate_paths": [],
            "best_chunk_idx": 0,
            "error": "All DreamDojo generations failed",
            "robometer_scores": [],
        }

    # Copy candidate videos into step_save_dir
    local_valid = []
    for orig_i, orig_path in valid:
        dst = step_save_dir / f"candidate_{orig_i}.mp4"
        try:
            shutil.copy2(orig_path, dst)
            local_valid.append((orig_i, str(dst)))
        except Exception as e:
            logging.warning(f"[Rescue] Could not copy {orig_path} -> {dst}: {e}")
            local_valid.append((orig_i, orig_path))
    valid = local_valid

    valid_indices, valid_paths = zip(*valid)

    # Save history video
    history_video_path = str(step_save_dir / "history_video.mp4")
    imageio.mimwrite(
        history_video_path,
        [np.asarray(x) for x in replay_images],
        fps=ROLLOUT_FPS,
    )

    # Subsample history frames for Robometer (last 2 seconds = 20 frames)
    history_np = np.array([np.asarray(x) for x in replay_images[-20:]], dtype=np.uint8)

    # ---- Score with Robometer ----
    logging.info(f"[Rescue] Scoring {len(valid_paths)} candidates with Robometer...")
    robometer_scores = robometer_scorer.score_candidates(
        history_np, list(valid_paths), task_description,
    )

    # Select best candidate by Robometer final_progress
    best_idx_in_valid = max(
        range(len(robometer_scores)),
        key=lambda j: robometer_scores[j]["final_progress"],
    )
    best_chunk_idx = valid_indices[best_idx_in_valid]

    logging.info(
        f"[Rescue] Selected candidate {best_chunk_idx} "
        f"(progress={robometer_scores[best_idx_in_valid]['final_progress']:.3f})"
    )

    # ---- Save training data for value expert ----
    for j, (orig_i, path) in enumerate(valid):
        rm_result = robometer_scores[j]
        data_path = step_save_dir / f"value_train_candidate_{orig_i}.npz"
        save_dict = {
            "candidate_video_path": np.array(path),
            "task": np.array(task_description),
            "robometer_progress": rm_result["progress"],
            "robometer_success": rm_result["success"],
            "robometer_final_progress": np.float32(rm_result["final_progress"]),
            "robometer_final_success": np.float32(rm_result["final_success"]),
            "actions": _actions_to_array(action_chunks[orig_i][:replan_steps]),
            "is_selected": np.bool_(orig_i == best_chunk_idx),
        }
        np.savez_compressed(data_path, **save_dict)

    selection_record = {
        "num_candidates": len(valid),
        "candidate_paths": list(valid_paths),
        "best_chunk_idx": int(best_chunk_idx),
        "robometer_scores": [
            {
                "candidate_idx": int(valid_indices[j]),
                "final_progress": float(robometer_scores[j]["final_progress"]),
                "final_success": float(robometer_scores[j]["final_success"]),
            }
            for j in range(len(robometer_scores))
        ],
    }

    return list(action_chunks[best_chunk_idx][:replan_steps]), selection_record


# ============================================================================
#  Phase 1: Evaluation
# ============================================================================

def eval_simpler_env(args):
    """Run SimplerEnv evaluation with Gemini rescue detection + Robometer candidate scoring."""
    os.environ.setdefault("DISPLAY", "")

    import gymnasium as gym
    from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs

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
        raise ValueError(f"Unknown robot_type: {args.robot_type}")

    if args.skip_tasks > 0:
        task_names = task_names[args.skip_tasks:]

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load GR00T policy --
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

    # -- Load Robometer scorer (for candidate scoring during rescue) --
    robometer_device = torch.device(
        f"cuda:{args.robometer_device}" if args.robometer_device >= 0 else "cpu"
    )
    robometer_scorer = RobometerScorer(
        args.robometer_model_path, device=robometer_device,
    )

    # -- Rollout loop --
    total_episodes, total_successes = 0, 0

    for task_name in tqdm.tqdm(task_names, desc="tasks"):
        env_id = f"{env_prefix}/{task_name}"
        logging.info(f"\n=== Task: {env_id} ===")

        task_episodes, task_successes = 0, 0

        for trial_idx in tqdm.tqdm(
            range(args.num_trials_per_task), desc="episodes", leave=False,
        ):
            task_segment = task_name.replace(" ", "_")

            # Skip if already completed
            existing = [
                d for d in output_dir.glob(f"rollout_{task_segment}_ep{trial_idx}_*")
                if d.name.endswith("_success") or d.name.endswith("_failure")
            ]
            if existing:
                logging.info(f"Skip: {task_segment} ep {trial_idx}")
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
            logging.info(f"\nTask: {task_description}")

            action_plan = collections.deque()
            clean_images = []

            # Gemini scoring state (for rescue detection)
            score_history: list = []
            score_lock = threading.Lock()
            gemini_futures: list = []
            gemini_all_results: list = []
            gemini_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            rescue_log: list = []
            rescue_selections: list = []

            done, truncated = False, False

            logging.info(f"Starting episode {trial_idx + 1}...")
            for t in range(args.max_episode_steps):
                if done or truncated:
                    break

                try:
                    img = _extract_image_from_obs(obs, args.robot_type)
                    clean_images.append(img.copy())
                    num_frames = len(clean_images)

                    # -- Async Gemini value query every GEMINI_QUERY_INTERVAL_FRAMES --
                    if num_frames % GEMINI_QUERY_INTERVAL_FRAMES == 0:
                        clip = list(clean_images[-GEMINI_HISTORY_FRAMES:])
                        future = gemini_executor.submit(
                            _query_gemini_value,
                            clip, task_description, num_frames,
                            score_history, score_lock,
                        )
                        gemini_futures.append(future)
                        logging.info(f"[Gemini] Submitted value query at frame {num_frames}")

                    # -- Replan when action queue is empty --
                    if not action_plan:
                        rescue = _check_rescue_needed(score_history, score_lock)

                        if rescue:
                            logging.info(f"[Rescue] Activating at frame {num_frames}...")
                            rescue_log.append(num_frames)

                            from datetime import datetime as _dt
                            _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
                            step_save_dir = (
                                rollout_dir / "rescue_steps"
                                / f"{task_segment}_ep{trial_idx}_frame{num_frames}_{_ts}"
                            )

                            best_actions, sel_record = _rescue_select_action_robometer(
                                obs=obs,
                                replay_images=clean_images,
                                task_description=task_description,
                                groot_policy=groot_policy,
                                base_policy=base_policy,
                                sde_action_head=sde_action_head,
                                ode_action_head=ode_action_head,
                                build_groot_obs_fn=build_groot_obs_fn,
                                robometer_scorer=robometer_scorer,
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

            # Wait for all Gemini queries to finish
            gemini_executor.shutdown(wait=True)
            for future in gemini_futures:
                try:
                    gemini_all_results.append(future.result())
                except Exception as e:
                    gemini_all_results.append({"error": str(e)})

            success = done
            if success:
                task_successes += 1
                total_successes += 1
            task_episodes += 1
            total_episodes += 1

            suffix = "success" if success else "failure"
            final_dir = output_dir / f"rollout_{task_segment}_ep{trial_idx}_{suffix}"
            rollout_dir.rename(final_dir)
            rollout_dir = final_dir

            # Write Gemini results
            _write_gemini_results(
                gemini_all_results, rollout_dir, task_description, trial_idx, suffix,
                rescue_log=rescue_log, rescue_selections=rescue_selections,
            )

            if clean_images:
                imageio.mimwrite(
                    str(rollout_dir / "complete_video.mp4"),
                    [np.asarray(x) for x in clean_images],
                    fps=ROLLOUT_FPS,
                )

            # ---- Save final episode video + outcome for value expert training ----
            episode_train_path = rollout_dir / "value_train_episode.npz"
            clip_frames = np.array(
                [np.asarray(x) for x in clean_images[-20:]],
                dtype=np.uint8,
            )
            np.savez_compressed(
                episode_train_path,
                frames=clip_frames,
                task=np.array(task_description),
                success=np.bool_(success),
                outcome_score=np.float32(1.0 if success else 0.0),
                num_steps=np.int32(len(clean_images)),
                num_rescues=np.int32(len(rescue_log)),
            )

            env.close()

            logging.info(f"Success: {success}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )

        logging.info(f"Current task success rate: {float(task_successes) / float(max(task_episodes, 1))}")
        logging.info(f"Current total success rate: {float(total_successes) / float(max(total_episodes, 1))}")

    logging.info(f"Total success rate: {float(total_successes) / float(max(total_episodes, 1))}")
    logging.info(f"Total episodes: {total_episodes}")


# ============================================================================
#  Phase 2: Training the Value Expert
# ============================================================================

class ValueExpertDataset(Dataset):
    """
    Loads training data for the DINOv2 Value Expert.

    Supports two data sources:
      1. Candidate-level: value_train_candidate_*.npz from rescue steps
         (label = robometer_final_progress)
      2. Episode-level: value_train_episode.npz from full episodes
         (label = outcome_score: 1.0 for success, 0.0 for failure)

    Multi-view: if the video has multiple camera views, they should already
    be spatially concatenated in the saved frames.
    """

    def __init__(
        self,
        data_dir: str,
        max_frames: int = 16,
        use_episode_data: bool = True,
        use_candidate_data: bool = True,
    ):
        self.max_frames = max_frames
        self.samples = []  # list of (path, label_key, video_key)

        data_path = pathlib.Path(data_dir)

        # Candidate-level data
        if use_candidate_data:
            cand_files = sorted(
                data_path.rglob("value_train_candidate_*.npz")
            )
            for f in cand_files:
                self.samples.append((str(f), "robometer_final_progress", "candidate"))
            logging.info(f"Found {len(cand_files)} candidate training samples")

        # Episode-level data
        if use_episode_data:
            ep_files = sorted(
                data_path.rglob("value_train_episode.npz")
            )
            for f in ep_files:
                self.samples.append((str(f), "outcome_score", "episode"))
            logging.info(f"Found {len(ep_files)} episode training samples")

        if not self.samples:
            raise FileNotFoundError(
                f"No value_train_*.npz files found in {data_dir}"
            )
        logging.info(f"Total training samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_key, data_type = self.samples[idx]
        data = np.load(path, allow_pickle=True)

        if data_type == "candidate":
            # Load candidate video from saved path
            video_path = str(data["candidate_video_path"])
            try:
                reader = imageio.get_reader(video_path, "ffmpeg")
                frames = np.array([f for f in reader])
                reader.close()
            except Exception:
                # Fallback: return dummy
                frames = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        else:
            # Episode data has frames directly
            frames = data["frames"]

        label = float(data[label_key])

        # Subsample to max_frames
        T = frames.shape[0]
        if T > self.max_frames:
            indices = np.linspace(0, T - 1, self.max_frames, dtype=int)
            frames = frames[indices]
        elif T < self.max_frames:
            # Pad by repeating last frame
            pad = np.tile(frames[-1:], (self.max_frames - T, 1, 1, 1))
            frames = np.concatenate([frames, pad], axis=0)

        # (T, H, W, C) uint8 → (T, 3, H, W) float [0, 1]
        frames_t = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255.0
        label_t = torch.tensor(label, dtype=torch.float32)

        return frames_t, label_t


def _setup_distributed():
    if "RANK" not in os.environ:
        return 0, 0, 1
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def train_value_expert(args):
    """Train the DINOv2 Value Expert on collected data."""
    rank, local_rank, world_size = _setup_distributed()
    distributed = world_size > 1
    is_main = rank == 0

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main:
        logging.info(f"Device: {device}  (world_size={world_size})")

    train_dataset = ValueExpertDataset(
        args.data_dir,
        max_frames=args.max_frames,
        use_episode_data=args.use_episode_data,
        use_candidate_data=args.use_candidate_data,
    )

    val_dataset = None
    if args.val_dir:
        val_dataset = ValueExpertDataset(
            args.val_dir,
            max_frames=args.max_frames,
            use_episode_data=args.use_episode_data,
            use_candidate_data=args.use_candidate_data,
        )

    model = DINOv2ValueExpert(
        dinov2_model=args.dinov2_model,
        hidden_dim=args.hidden_dim,
        num_cameras=args.num_cameras,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                     find_unused_parameters=False)
    raw_model = model.module if distributed else model

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset:
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if is_main:
        logging.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    # MSE loss for continuous score regression
    criterion = nn.MSELoss()

    output_dir = pathlib.Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    best_val_loss = float("inf")

    # wandb
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
                "epochs": args.epochs,
                "max_frames": args.max_frames,
                "num_cameras": args.num_cameras,
                "num_samples": len(train_dataset),
            },
        )

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        if args.freeze_backbone:
            raw_model.backbone.eval()

        total_loss, num_batches = 0.0, 0

        for frames, labels in train_loader:
            frames = frames.to(device)  # (B, T, 3, H, W)
            labels = labels.to(device)  # (B,)

            logits = model(frames)
            preds = torch.sigmoid(logits)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / max(num_batches, 1)
        scheduler.step()

        # Validation
        val_str = "N/A"
        if val_loader is not None:
            model.eval()
            vl, vn = 0.0, 0
            val_mae = 0.0

            with torch.no_grad():
                for frames, labels in val_loader:
                    frames = frames.to(device)
                    labels = labels.to(device)
                    logits = model(frames)
                    preds = torch.sigmoid(logits)
                    vl += criterion(preds, labels).item()
                    val_mae += (preds - labels).abs().mean().item()
                    vn += 1

            avg_val_loss = vl / max(vn, 1)
            avg_val_mae = val_mae / max(vn, 1)
            val_str = f"loss={avg_val_loss:.4f} MAE={avg_val_mae:.4f}"

            if is_main and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(raw_model.state_dict(), output_dir / "best_model.pt")
                logging.info(f"  -> Saved best model (val_loss={avg_val_loss:.4f})")

        if is_main:
            logging.info(
                f"Epoch {epoch+1}/{args.epochs}  "
                f"train_loss={avg_train_loss:.4f}  "
                f"val=[{val_str}]  lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if use_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "lr": scheduler.get_last_lr()[0],
            }
            if val_loader is not None:
                log_dict["val/loss"] = avg_val_loss
                log_dict["val/mae"] = avg_val_mae
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
#  CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SimplerEnv eval with Robometer scoring + DINOv2 Value Expert"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- eval ----
    p_eval = subparsers.add_parser(
        "eval", help="Run SimplerEnv eval with Robometer scoring",
    )
    p_eval.add_argument("--model_path", type=str, default="nvidia/GR00T-N1.6-3B")
    p_eval.add_argument("--device", type=int, default=0)
    p_eval.add_argument("--noise_level", type=float, default=0.3)
    p_eval.add_argument("--action_horizon", type=int, default=1)
    p_eval.add_argument("--rescue_action_horizon", type=int, default=8)
    p_eval.add_argument("--robot_type", type=str, default="google",
                        choices=["google", "widowx"])
    p_eval.add_argument("--num_trials_per_task", type=int, default=20)
    p_eval.add_argument("--max_episode_steps", type=int, default=120)
    p_eval.add_argument("--seed", type=int, default=7)
    p_eval.add_argument("--skip_tasks", type=int, default=0)
    p_eval.add_argument("--num_rescue_samples", type=int, default=4)
    p_eval.add_argument("--dd_base_port", type=int, default=8020)
    p_eval.add_argument("--output_dir", type=str,
                        default="data/robometer_value_expert")
    # Robometer (for scoring candidates during rescue)
    p_eval.add_argument("--robometer_model_path", type=str, required=True,
                        help="Robometer HF model or local path")
    p_eval.add_argument("--robometer_device", type=int, default=-1,
                        help="GPU for Robometer (-1 = same as GR00T device)")

    # ---- train ----
    p_train = subparsers.add_parser(
        "train", help="Train DINOv2 Value Expert",
    )
    p_train.add_argument("--data_dir", type=str, required=True)
    p_train.add_argument("--val_dir", type=str, default=None)
    p_train.add_argument("--output_dir", type=str, default="checkpoints/value_expert")
    p_train.add_argument("--dinov2_model", type=str, default="dinov2_vitb14",
                         choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14",
                                  "dinov2_vitg14"])
    p_train.add_argument("--hidden_dim", type=int, default=256)
    p_train.add_argument("--num_cameras", type=int, default=1)
    p_train.add_argument("--freeze_backbone", action="store_true", default=True)
    p_train.add_argument("--no_freeze_backbone", dest="freeze_backbone",
                         action="store_false")
    p_train.add_argument("--batch_size", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--weight_decay", type=float, default=1e-4)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--num_workers", type=int, default=4)
    p_train.add_argument("--save_every", type=int, default=5)
    p_train.add_argument("--max_frames", type=int, default=16,
                         help="Max frames per video for training")
    p_train.add_argument("--use_episode_data", action="store_true", default=True)
    p_train.add_argument("--no_episode_data", dest="use_episode_data",
                         action="store_false")
    p_train.add_argument("--use_candidate_data", action="store_true", default=True)
    p_train.add_argument("--no_candidate_data", dest="use_candidate_data",
                         action="store_false")
    p_train.add_argument("--wandb_project", type=str, default=None)
    p_train.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.command == "eval":
        if args.robometer_device < 0:
            args.robometer_device = args.device
        eval_simpler_env(args)
    elif args.command == "train":
        train_value_expert(args)


if __name__ == "__main__":
    main()
