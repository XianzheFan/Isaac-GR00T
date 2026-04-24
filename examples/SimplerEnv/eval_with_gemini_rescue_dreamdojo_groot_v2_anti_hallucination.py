"""
SimplerEnv version of the Gemini-rescue + DreamDojo + GR00T evaluation.

Mirrors the LIBERO eval_with_gemini_rescue_dreamdojo_groot_sde.py flow but
uses SimplerEnv (Google Fractal / WidowX Bridge) environments.

Normal inference uses deterministic ODE sampling (matching the official eval
behavior via run_gr00t_server.py). Only during rescue does the script switch
to SDE (Euler-Maruyama) sampling for diverse action candidates.

Start one DreamDojo server before running this script:
    python ~/workspace/fxz/DreamDojo/examples/dreamdojo_server.py \
        --checkpoint outputs/dreamdojo/simpler_env/checkpoints/iter_000005000/model_ema_bf16.pt \
        --experiment dreamdojo_2b_480_640_simpler_env \
        --save-dir /tmp/dreamdojo_results \
        --port 8020

For N parallel candidates, start N server instances on consecutive ports:
    CUDA_VISIBLE_DEVICES=0 python dreamdojo_server.py --port 8020 ...
    CUDA_VISIBLE_DEVICES=1 python dreamdojo_server.py --port 8021 ...

Usage:
    python examples/SimplerEnv/eval_with_gemini_rescue_dreamdojo_groot_sde.py \
        --model_path /path/to/groot/checkpoint \
        --robot_type google \
        --dd_base_port 8020
"""

import base64
import collections
import concurrent.futures
import dataclasses
import json
import logging
import os
import pathlib
import shutil
import tempfile
import random
import threading
import time

import cv2
import imageio
import numpy as np
import requests
import torch
import tqdm
import tyro
from pydantic import BaseModel
from google import genai
from google.genai import types

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.gr00t_n1d6.gr00t_n1d6_sde import Gr00tN1d6SDEActionHead
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper


ROLLOUT_FPS = 10

GEMINI_QUERY_INTERVAL_FRAMES = 20
GEMINI_HISTORY_FRAMES = 200
GEMINI_VALUE_MODEL = "gemini-3.1-flash-lite-preview"
GEMINI_SELECT_MODEL = "gemini-3.1-flash-lite-preview"
# GEMINI_VALUE_MODEL = "gemini-3.1-pro-preview"
# GEMINI_SELECT_MODEL = "gemini-3.1-pro-preview"

RESCUE_SCORE_ABSOLUTE = 0.40
RESCUE_SCORE_DROP = 0.15

# Anti-hallucination: if score stays above this for N consecutive evals
# without env reporting done, trigger a verification re-query.
HALLUCINATION_HIGH_SCORE = 0.80
HALLUCINATION_CONSECUTIVE_COUNT = 3

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


class ValueEvaluation(BaseModel):
    reasoning: str
    score: float
    status: str


class BestIndex(BaseModel):
    best_index: int


_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(http_options={"api_version": "v1alpha"})
    return _gemini_client


def _dreamdojo_generate(port: int, frame_np: np.ndarray, actions: np.ndarray,
                        save_name: str, task_description: str = "",
                        seed: int = 0) -> str | None:
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


def _query_gemini_value(frames: list, task_description: str, step_idx: int,
                        score_history: list, lock: threading.Lock) -> dict:
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


def _query_gemini_verify(first_frame: np.ndarray, latest_frames: list,
                         task_description: str, reported_score: float) -> float | None:
    """Anti-hallucination verification (方案 A+B).

    Sends the first frame and the latest clip to Gemini with a stricter prompt
    that asks for explicit comparison of physical state change.  Returns the
    verified score, or *None* on error.
    """
    client = _get_gemini_client()
    tmp_first = tmp_clip = None
    first_file = clip_file = None
    try:
        # Encode first frame as a 1-frame video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_first = f.name
        imageio.mimwrite(tmp_first, [np.asarray(first_frame)], fps=1)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_clip = f.name
        imageio.mimwrite(tmp_clip, [np.asarray(x) for x in latest_frames], fps=ROLLOUT_FPS)

        first_file = client.files.upload(file=tmp_first)
        clip_file = client.files.upload(file=tmp_clip)
        for vf in (first_file, clip_file):
            info = client.files.get(name=vf.name)
            while info.state.name == "PROCESSING":
                time.sleep(2)
                info = client.files.get(name=vf.name)

        prompt = (
            f'You are a strict robot evaluation verifier.\n'
            f'Task: "{task_description}"\n'
            f'A previous evaluator scored the current state at {reported_score:.2f}, '
            f'claiming the task is nearly complete.\n\n'
            f'You are given:\n'
            f'1. [First Frame]: the very first frame of the episode (initial state)\n'
            f'2. [Recent Clip]: the most recent 2 seconds of the episode\n\n'
            f'Compare the two carefully:\n'
            f'- Has the target object\'s position PHYSICALLY changed between the first '
            f'frame and the last frame of the recent clip?\n'
            f'- For a "pick" task: is the object ACTUALLY lifted off the surface in '
            f'the FINAL frame of the recent clip? (not just earlier)\n'
            f'- Could the high score be a hallucination (object still on table, '
            f'gripper empty, arm moved away)?\n\n'
            f'Re-score strictly between 0.00 and 1.00. If the object has NOT '
            f'physically moved or is no longer held, score below 0.40.\n'
            f'Output JSON: [{{"reasoning": "...", "score": <float>, "status": "..."}}]'
        )

        response = client.models.generate_content(
            model=GEMINI_VALUE_MODEL,
            contents=[prompt, "\n[First Frame]:", first_file,
                      "\n[Recent Clip]:", clip_file],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[ValueEvaluation],
                temperature=0.0,
            ),
        )
        result = json.loads(response.text)
        if result:
            verified = float(result[0]["score"])
            logging.info(
                f"[Verify] reported={reported_score:.2f} -> verified={verified:.2f} "
                f"reason={result[0].get('reasoning', '')[:120]}"
            )
            return verified
        return None

    except Exception as e:
        logging.error(f"[Verify] error: {e}")
        return None

    finally:
        for vf in (first_file, clip_file):
            if vf is not None:
                try:
                    client.files.delete(name=vf.name)
                except Exception:
                    pass
        for p in (tmp_first, tmp_clip):
            if p is not None and os.path.exists(p):
                os.unlink(p)


def _check_rescue_needed(score_history: list, lock: threading.Lock,
                         first_frame: np.ndarray | None = None,
                         recent_frames: list | None = None,
                         task_description: str = "") -> bool:
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

    # --- Anti-hallucination check (方案 A + B) ---
    # If recent scores are consistently high but env hasn't reported done,
    # verify with a stricter first-frame-vs-current comparison.
    if first_frame is not None and recent_frames:
        recent_high = [s for _, s in sorted_scores if s >= HALLUCINATION_HIGH_SCORE]
        if len(recent_high) >= HALLUCINATION_CONSECUTIVE_COUNT:
            logging.info(
                f"[Verify] {len(recent_high)} consecutive scores >= {HALLUCINATION_HIGH_SCORE}, "
                f"running anti-hallucination verification..."
            )
            verified = _query_gemini_verify(
                first_frame, recent_frames[-int(2 * ROLLOUT_FPS):],
                task_description, latest_score,
            )
            if verified is not None and verified <= RESCUE_SCORE_ABSOLUTE:
                logging.info(
                    f"[Rescue] Triggered by verification: reported={latest_score:.2f} "
                    f"verified={verified:.2f}"
                )
                # Correct the score_history so future checks don't re-verify immediately
                with lock:
                    score_history.append((latest_frame + 1, verified))
                return True

    return False


def _gemini_select_best(current_video_path: str, candidate_paths: list,
                        task_description: str) -> int:
    """Select the best candidate video using Gemini.

    To mitigate VLM position bias, the candidate order is randomized before
    querying and the selected index is mapped back to the original order.
    """
    client = _get_gemini_client()

    # Shuffle candidate order to counteract position bias
    num_cands = len(candidate_paths)
    shuffled_order = list(range(num_cands))
    random.shuffle(shuffled_order)
    shuffled_paths = [candidate_paths[i] for i in shuffled_order]
    logging.info(f"[Gemini Select] presentation order: {shuffled_order}")

    current_file = client.files.upload(file=current_video_path)
    cand_files = [client.files.upload(file=p) for p in shuffled_paths]

    for f in [current_file] + cand_files:
        info = client.files.get(name=f.name)
        while info.state.name == "PROCESSING":
            time.sleep(2)
            info = client.files.get(name=f.name)
        if info.state.name == "FAILED":
            raise ValueError(f"Video processing failed: {f.name}")

    prompt = (
        f'You are an evaluation model in a robotic control system.\n'
        f'Based on the [Current Video] and the language command, select the most promising '
        f'candidate next video that best continues the task.\n\n'
        f'Language Command: "{task_description}"\n\n'
        f'Please evaluate the Current Video against the Candidate Next Videos '
        f'(Index 0 to {len(cand_files) - 1}).'
    )
    contents = [prompt, "\n[Current Video]:", current_file]
    for i, cf_file in enumerate(cand_files):
        contents += [f"\n[Candidate Next Video {i}]:", cf_file]

    response = client.models.generate_content(
        model=GEMINI_SELECT_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BestIndex,
            temperature=0.2,
        ),
    )
    result = json.loads(response.text)

    client.files.delete(name=current_file.name)
    for cf_file in cand_files:
        client.files.delete(name=cf_file.name)

    # Map shuffled index back to original index
    shuffled_best = int(result["best_index"])
    original_best = shuffled_order[min(shuffled_best, num_cands - 1)]
    logging.info(
        f"[Gemini Select] picked shuffled idx {shuffled_best} "
        f"-> original idx {original_best}"
    )
    return original_best


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


def _convert_to_simpler_action(action_chunk: dict, idx: int = 0) -> dict:
    """Convert GR00T action chunk to a single env-step action dict.
    Each value is a 1-element numpy array (required by SimplerEnv step).
    action_chunk values have shape (B, T, D) where B=1.
    """
    return {
        f"action.{key}": np.atleast_1d(action_chunk[f"action.{key}"][0, idx]).flatten()[:1]
        for key in ACTION_KEYS
    }


def _convert_to_simpler_actions(action_chunk: dict, action_horizon: int) -> list[dict]:
    """Convert GR00T action chunk to a list of env-step action dicts.
    action_chunk values have shape (B, T, D); action_horizon should be <= T.
    """
    first_key = f"action.{ACTION_KEYS[0]}"
    avail = action_chunk[first_key].shape[1]  # T dimension
    num_steps = min(action_horizon, avail)
    return [_convert_to_simpler_action(action_chunk, i) for i in range(num_steps)]


def _extract_image_from_obs(obs: dict, robot_type: str) -> np.ndarray:
    if robot_type == "google":
        return np.asarray(obs["video.image"])
    else:
        return np.asarray(obs["video.image_0"])


def _rescue_select_action(
    obs: dict,
    replay_images_for_history: list,
    task_description: str,
    groot_policy,
    base_policy,
    sde_action_head,
    ode_action_head,
    build_groot_obs_fn,
    replan_steps: int,
    step_save_dir: pathlib.Path,
    dd_base_port: int,
    robot_type: str,
    num_samples: int = 5,
) -> tuple[list, dict]:
    """
    Sample `num_samples` action chunks via GR00T SDE (stochastic, so each
    call gives different actions), send parallel DreamDojo generation requests,
    let Gemini pick the best, and return the chosen action chunk.
    """
    step_save_dir.mkdir(parents=True, exist_ok=True)
    img = _extract_image_from_obs(obs, robot_type)

    groot_obs = build_groot_obs_fn(obs)

    # Switch to SDE head for diverse sampling
    base_policy.model.action_head = sde_action_head

    # Sample multiple action chunks with explicit RNG re-seeding.
    # Without re-seeding, repeated calls can produce identical actions due to
    # CUDA RNG state interactions with the backbone forward pass.
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

    # Build flat action arrays for DreamDojo (each action is a dict, convert to array)
    def _actions_to_array(action_list: list[dict]) -> np.ndarray:
        rows = []
        for a in action_list:
            rows.append(np.concatenate([a[f"action.{k}"] for k in ACTION_KEYS]))
        return np.array(rows, dtype=np.float32)

    # Log action diversity for diagnostics
    for i in range(num_samples):
        arr = _actions_to_array(action_chunks[i][:replan_steps])
        logging.info(
            f"[Rescue] chunk_{i} actions mean={arr.mean():.6f} "
            f"std={arr.std():.6f} first={arr[0, :3]}"
        )

    # Parallel DreamDojo server requests
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

    logging.info(f"[Rescue] Launching {num_samples} parallel DreamDojo generation requests...")

    def _submit(t):
        return _dreamdojo_generate(t["port"], img, t["actions"], t["save_name"], task_description, seed=t["seed"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_samples) as ex:
        futures = {ex.submit(_submit, t): i for i, t in enumerate(tasks)}
        save_paths = {}
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            save_paths[idx] = fut.result()

    valid = [(i, save_paths[i]) for i in range(num_samples)
             if save_paths.get(i) and os.path.exists(save_paths[i])]

    if not valid:
        logging.warning("[Rescue] All DreamDojo generations failed; using chunk 0.")
        return list(action_chunks[0][:replan_steps]), {
            "num_candidates": 0, "candidate_paths": [], "raw_best": None,
            "best_chunk_idx": 0, "error": "All DreamDojo generations failed",
        }

    # Save history video for Gemini context
    current_video_path = str(step_save_dir / "current_actual_video.mp4")
    imageio.mimwrite(
        current_video_path,
        [np.asarray(x) for x in replay_images_for_history],
        fps=ROLLOUT_FPS,
    )

    # Copy DreamDojo candidate videos into step_save_dir
    local_valid_paths = []
    for orig_i, orig_path in valid:
        dst = step_save_dir / f"output_{orig_i}.mp4"
        try:
            shutil.copy2(orig_path, dst)
            local_valid_paths.append((orig_i, str(dst)))
        except Exception as e:
            logging.warning(f"[Rescue] Could not copy {orig_path} -> {dst}: {e}")
            local_valid_paths.append((orig_i, orig_path))
    valid = local_valid_paths

    valid_indices, valid_paths = zip(*valid)
    selection_record = {
        "num_candidates": len(valid),
        "candidate_paths": list(valid_paths),
        "raw_best": None,
        "best_chunk_idx": None,
        "error": None,
    }
    try:
        raw_best = _gemini_select_best(current_video_path, list(valid_paths), task_description)
        best_chunk_idx = valid_indices[min(raw_best, len(valid_indices) - 1)]
        selection_record["raw_best"] = raw_best
        selection_record["best_chunk_idx"] = int(best_chunk_idx)
        logging.info(f"[Rescue] Gemini selected candidate {raw_best} -> chunk {best_chunk_idx}")
    except Exception as e:
        logging.error(f"[Rescue] Gemini selection failed: {e}. Using first valid chunk.")
        best_chunk_idx = valid_indices[0]
        selection_record["error"] = str(e)

    return list(action_chunks[best_chunk_idx][:replan_steps]), selection_record


def _write_gemini_results(gemini_results: list, rollout_dir: pathlib.Path,
                          task_description: str, episode_idx: int, suffix: str,
                          rescue_log: list | None = None,
                          rescue_selections: list | None = None):
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
    logging.info(f"[Gemini] Results written to {out_path}")

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
    logging.info(f"[Gemini] JSON results written to {json_path}")


@dataclasses.dataclass
class Args:
    model_path: str = "nvidia/GR00T-N1.6-3B"
    device: int = 0
    noise_level: float = 0.3
    """SDE diffusion noise strength (only used during rescue sampling)."""
    num_inference_timesteps: int | None = None
    """Number of denoising steps for SDE (None uses model default, only during rescue)."""
    action_horizon: int = 1
    """Number of actions to execute per normal policy query (1 = query every step, matching official eval)."""
    rescue_action_horizon: int = 8
    """Number of actions in each rescue candidate chunk for DreamDojo generation."""

    robot_type: str = "google"  # google (Google Fractal) or widowx (WidowX Bridge)
    num_trials_per_task: int = 20
    max_episode_steps: int = 120

    video_out_path: str = ""  # auto-set based on robot_type
    seed: int = 7
    num_rescue_samples: int = 4
    dd_base_port: int = 8020

    skip_tasks: int = 0


def eval_simpler_env(args: Args) -> None:
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
        raise ValueError(f"Unknown robot_type: {args.robot_type}. Use 'google' or 'widowx'.")

    if args.skip_tasks > 0:
        logging.info(f"Skipping first {args.skip_tasks} tasks")
        task_names = task_names[args.skip_tasks:]

    if not args.video_out_path:
        args.video_out_path = f"data/simpler_env_rescue_{args.robot_type}"
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading GR00T policy from {args.model_path} ...")
    base_policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device=args.device,
        strict=True,
    )
    groot_policy = Gr00tSimPolicyWrapper(base_policy)
    ode_action_head = base_policy.model.action_head  # deterministic (normal inference)

    # Build SDE action head for rescue sampling (shares weights, stochastic sampling)
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

            # Create a fresh env each trial (SimplerEnv reset can be unreliable)
            env = gym.make(env_id)
            obs, info = env.reset(seed=args.seed + trial_idx)

            task_description = str(obs.get(
                "annotation.human.action.task_description",
                task_name,
            ))
            logging.info(f"\nTask: {task_description}")

            action_plan = collections.deque()
            clean_images = []

            score_history: list = []
            score_lock = threading.Lock()
            gemini_futures: list = []
            gemini_all_results: list = []
            gemini_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            rescue_log: list = []
            rescue_selections: list = []

            done, truncated = False, False

            logging.info(f"Starting episode {task_episodes + 1}...")
            for t in range(args.max_episode_steps):
                if done or truncated:
                    break

                try:
                    img = _extract_image_from_obs(obs, args.robot_type)
                    clean_images.append(img.copy())
                    num_frames = len(clean_images)

                    # Async Gemini value query every GEMINI_QUERY_INTERVAL_FRAMES
                    if num_frames % GEMINI_QUERY_INTERVAL_FRAMES == 0:
                        clip = list(clean_images[-GEMINI_HISTORY_FRAMES:])
                        future = gemini_executor.submit(
                            _query_gemini_value,
                            clip, task_description, num_frames,
                            score_history, score_lock,
                        )
                        gemini_futures.append(future)
                        logging.info(f"[Gemini] Submitted value query at frame {num_frames}")

                    # Replan when action queue is empty
                    if not action_plan:
                        rescue = _check_rescue_needed(
                            score_history, score_lock,
                            first_frame=clean_images[0] if clean_images else None,
                            recent_frames=clean_images,
                            task_description=task_description,
                        )

                        if rescue:
                            logging.info(f"[Rescue] Activating at frame {num_frames}...")
                            rescue_log.append(num_frames)
                            from datetime import datetime as _dt
                            _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
                            step_save_dir = rollout_dir / "rescue_steps" / f"{task_segment}_ep{trial_idx}_frame{num_frames}_{_ts}"
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
                            # Normal GR00T ODE inference (deterministic)
                            groot_obs = build_groot_obs_fn(obs)
                            action_dict, _ = groot_policy.get_action(groot_obs)
                            action_list = _convert_to_simpler_actions(
                                action_dict, args.action_horizon
                            )
                            action_plan.extend(action_list)

                    action = action_plan.popleft()
                    obs, reward, done, truncated, info = env.step(action)

                except Exception as e:
                    logging.error(f"Step exception: {e}")
                    break

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
            final_rollout_dir = (
                pathlib.Path(args.video_out_path)
                / f"rollout_{task_segment}_ep{trial_idx}_{suffix}"
            )
            rollout_dir.rename(final_rollout_dir)
            rollout_dir = final_rollout_dir

            _write_gemini_results(
                gemini_all_results, rollout_dir, task_description, trial_idx, suffix,
                rescue_log=rescue_log, rescue_selections=rescue_selections,
            )
            if rescue_log:
                with open(rollout_dir / "gemini_values.txt", "a") as f:
                    f.write("=" * 60 + "\n")
                    f.write(f"Rescue activations ({len(rescue_log)}): frames {rescue_log}\n")

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
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )

        logging.info(f"Current task success rate: {float(task_successes) / float(max(task_episodes, 1))}")
        logging.info(f"Current total success rate: {float(total_successes) / float(max(total_episodes, 1))}")

    logging.info(f"Total success rate: {float(total_successes) / float(max(total_episodes, 1))}")
    logging.info(f"Total episodes: {total_episodes}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_simpler_env(tyro.cli(Args))
