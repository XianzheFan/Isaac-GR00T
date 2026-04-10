---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
- LeRobot
- fractal20220817_data
- rlds
- openx
- google_robot
configs:
- config_name: default
  data_files: data/*/*.parquet
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Description



- **Homepage:** [More Information Needed]
- **Paper:** [More Information Needed]
- **License:** apache-2.0

## Dataset Structure

[meta/info.json](meta/info.json):
```json
{
    "codebase_version": "v2.0",
    "robot_type": "google_robot",
    "total_episodes": 87212,
    "total_frames": 3786400,
    "total_tasks": 599,
    "total_videos": 87212,
    "total_chunks": 88,
    "chunks_size": 1000,
    "fps": 3,
    "splits": {
        "train": "0:87212"
    },
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": {
        "observation.images.image": {
            "dtype": "video",
            "shape": [
                256,
                320,
                3
            ],
            "names": [
                "height",
                "width",
                "rgb"
            ],
            "info": {
                "video.fps": 3.0,
                "video.height": 256,
                "video.width": 320,
                "video.channels": 3,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [
                8
            ],
            "names": {
                "motors": [
                    "x",
                    "y",
                    "z",
                    "rx",
                    "ry",
                    "rz",
                    "rw",
                    "gripper"
                ]
            }
        },
        "action": {
            "dtype": "float32",
            "shape": [
                7
            ],
            "names": {
                "motors": [
                    "x",
                    "y",
                    "z",
                    "roll",
                    "pitch",
                    "yaw",
                    "gripper"
                ]
            }
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": null
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        }
    }
}
```


## Citation

**BibTeX:**

```bibtex
[More Information Needed]
```