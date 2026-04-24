"""
Overlay Gemini scores and task info on episode videos.

Usage:
  python annotate_videos.py \
      --data_dir data/simpler_env_google_critical_labels_shifted \
      --output_dir data/annotated_critical \
      --max_episodes 10

  # Process all episodes:
  python annotate_videos.py \
      --data_dir data/simpler_env_google_switch_labels_shifted \
      --output_dir data/annotated_switch
"""

import argparse
import os
import glob

import cv2
import numpy as np


def annotate_episode(episode_dir, output_path, outcome="", task_stats=None):
    """Read video + npz files, overlay text, write annotated video."""
    video_path = os.path.join(episode_dir, "complete_video.mp4")
    if not os.path.exists(video_path):
        return False

    # Load per-frame metadata from npz files
    npz_files = sorted(glob.glob(os.path.join(episode_dir, "step_*.npz")))
    if not npz_files:
        return False

    frame_meta = {}
    task = ""
    prompt = ""
    for f in npz_files:
        try:
            d = np.load(f, allow_pickle=True)
        except Exception as e:
            print(f"    WARN: skipping corrupt file {f}: {e}")
            continue
        idx = int(d["frame_idx"])
        frame_meta[idx] = float(d["gemini_score"])
        if not task:
            task = str(d["task"])
        if not prompt:
            prompt = str(d["prompt"])

    # Read video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        score = frame_meta.get(frame_idx, None)

        # Draw task name
        cv2.putText(frame, task, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw prompt (may be long, truncate)
        disp_prompt = prompt if len(prompt) <= 45 else prompt[:42] + "..."
        cv2.putText(frame, disp_prompt, (5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

        # Draw score with color coding
        if score is not None:
            if score >= 0.6:
                color = (0, 200, 0)      # green - high
            elif score >= 0.3:
                color = (0, 200, 255)    # orange - medium
            else:
                color = (0, 0, 255)      # red - low
            cv2.putText(frame, f"score: {score:.2f}", (5, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Frame counter
        cv2.putText(frame, f"f:{frame_idx}", (w - 40, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

        # Bottom-right: outcome and task stats
        if outcome or task_stats:
            y_bottom = h - 10
            # Outcome label (success=green, failure=red)
            if outcome == "success":
                out_color = (0, 200, 0)
            elif outcome == "failure":
                out_color = (0, 0, 255)
            else:
                out_color = (200, 200, 200)
            if outcome:
                cv2.putText(frame, outcome.upper(), (w - 90, y_bottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, out_color, 1, cv2.LINE_AA)
            # Stats line above outcome
            if task_stats:
                s, f, total = task_stats["success"], task_stats["failure"], task_stats["total"]
                stats_text = f"S:{s} F:{f} #{total}"
                cv2.putText(frame, stats_text, (w - 120, y_bottom - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_episodes", type=int, default=0,
                        help="Max episodes to process (0 = all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    episodes = sorted([
        d for d in os.listdir(args.data_dir)
        if d.startswith("rollout_") and os.path.isdir(os.path.join(args.data_dir, d))
    ])

    if args.max_episodes > 0:
        episodes = episodes[:args.max_episodes]

    # Group by task for organized output and collect per-task stats
    task_dirs = {}
    task_stats = {}  # task -> {"success": int, "failure": int, "total": int}
    for ep in episodes:
        # e.g. rollout_google_robot_move_near_ep0_success
        parts = ep.split("_ep")
        task = parts[0].replace("rollout_", "")
        if task not in task_dirs:
            os.makedirs(os.path.join(args.output_dir, task), exist_ok=True)
            task_dirs[task] = True
        suffix = parts[1]  # e.g. "0_success"
        outcome = suffix.rsplit("_", 1)[1]
        if task not in task_stats:
            task_stats[task] = {"success": 0, "failure": 0, "total": 0}
        task_stats[task]["total"] += 1
        if outcome == "success":
            task_stats[task]["success"] += 1
        elif outcome == "failure":
            task_stats[task]["failure"] += 1

    print(f"Processing {len(episodes)} episodes -> {args.output_dir}")
    for task, st in sorted(task_stats.items()):
        print(f"  {task}: S={st['success']} F={st['failure']} #{st['total']}")

    for i, ep in enumerate(episodes):
        ep_path = os.path.join(args.data_dir, ep)
        # Extract task, ep number, outcome
        parts = ep.split("_ep")
        task = parts[0].replace("rollout_", "")
        suffix = parts[1]  # e.g. "0_success"
        ep_num = suffix.rsplit("_", 1)[0]
        outcome = suffix.rsplit("_", 1)[1]
        out_path = os.path.join(args.output_dir, task, f"{task}_ep{ep_num}_{outcome}.mp4")
        ok = annotate_episode(ep_path, out_path, outcome=outcome, task_stats=task_stats.get(task))
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(episodes)}] {ep} {'ok' if ok else 'SKIP'}")

    print("Done.")


if __name__ == "__main__":
    main()
