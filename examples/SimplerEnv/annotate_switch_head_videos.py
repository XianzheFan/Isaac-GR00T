"""
Overlay switch head scores on eval videos (post-processing).

Reads complete_video.mp4 + switch_head_log.json from each rollout dir,
writes annotated_video.mp4 alongside. Original video is untouched.

Usage:
  python annotate_switch_head_videos.py --data_dir data/switch_head_rescue_google
"""

import argparse
import json
import os

import cv2
import numpy as np


def annotate_episode(episode_dir):
    video_path = os.path.join(episode_dir, "complete_video.mp4")
    log_path = os.path.join(episode_dir, "switch_head_log.json")
    if not os.path.exists(video_path) or not os.path.exists(log_path):
        return False

    with open(log_path) as f:
        log = json.load(f)

    # Build frame → score lookup
    frame_scores = {}
    for entry in log.get("score_history", []):
        frame_scores[entry["frame"]] = entry["rescue_prob"]
    rescue_frames = set(log.get("rescue_activations", []))
    threshold = log.get("config", {}).get("rescue_threshold", 0.5)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(episode_dir, "annotated_video.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    task = log.get("task", "")
    outcome = log.get("outcome", "")

    frame_idx = 1
    last_score = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frame_scores:
            last_score = frame_scores[frame_idx]

        # Task name
        cv2.putText(frame, task, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Score
        if last_score is not None:
            if last_score >= threshold:
                color = (0, 0, 255)    # red = rescue
            elif last_score >= threshold * 0.6:
                color = (0, 200, 255)  # orange = warning
            else:
                color = (0, 200, 0)    # green = normal
            cv2.putText(frame, f"rescue_prob: {last_score:.3f} (thr={threshold:.2f})",
                        (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # Rescue marker
        if frame_idx in rescue_frames:
            cv2.putText(frame, "RESCUE", (w - 80, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # Frame counter + outcome
        cv2.putText(frame, f"f:{frame_idx}", (w - 50, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
        if outcome:
            out_color = (0, 200, 0) if outcome == "success" else (0, 0, 255)
            cv2.putText(frame, outcome.upper(), (5, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, out_color, 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    episodes = sorted([
        d for d in os.listdir(args.data_dir)
        if d.startswith("rollout_") and os.path.isdir(os.path.join(args.data_dir, d))
    ])

    print(f"Processing {len(episodes)} episodes in {args.data_dir}")
    for i, ep in enumerate(episodes):
        ok = annotate_episode(os.path.join(args.data_dir, ep))
        status = "ok" if ok else "SKIP"
        if (i + 1) % 20 == 0 or i == 0 or not ok:
            print(f"  [{i+1}/{len(episodes)}] {ep} {status}")

    print("Done.")


if __name__ == "__main__":
    main()
