# Launch finetuning for all 4 LIBERO suites jointly (multi-task training).
# Usage:
#   NUM_GPUS=8
#   torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
#       examples/LIBERO/launch_finetune_libero_all.py

import os

import wandb
from gr00t.configs.base_config import get_default_config
from gr00t.experiment.experiment import run

if "LOGURU_LEVEL" not in os.environ:
    os.environ["LOGURU_LEVEL"] = "INFO"

# Wandb login with API key
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)

NUM_GPUS = int(os.environ.get("NUM_GPUS", 8))

config = get_default_config().load_dict(
    {
        "data": {
            "download_cache": False,
            "datasets": [
                {
                    "dataset_paths": ["examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/"],
                    "mix_ratio": 1.0,
                    "embodiment_tag": "libero_panda",
                },
                {
                    "dataset_paths": ["examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot/"],
                    "mix_ratio": 1.0,
                    "embodiment_tag": "libero_panda",
                },
                {
                    "dataset_paths": ["examples/LIBERO/libero_object_no_noops_1.0.0_lerobot/"],
                    "mix_ratio": 1.0,
                    "embodiment_tag": "libero_panda",
                },
                {
                    "dataset_paths": ["examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot/"],
                    "mix_ratio": 1.0,
                    "embodiment_tag": "libero_panda",
                },
            ],
        }
    }
)

config.load_config_path = None

# Model config
config.model.tune_llm = False
config.model.tune_visual = False
config.model.tune_projector = True
config.model.tune_diffusion_model = True
config.model.state_dropout_prob = 0.0
config.model.load_bf16 = False
config.model.reproject_vision = False
config.model.eagle_collator = True
config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
config.model.backbone_trainable_params_fp32 = True
config.model.use_relative_action = True
config.model.color_jitter_params = {
    "brightness": 0.3,
    "contrast": 0.4,
    "saturation": 0.5,
    "hue": 0.08,
}

# Training config
config.training.start_from_checkpoint = "nvidia/GR00T-N1.7-3B"
config.training.optim = "adamw_torch"
config.training.global_batch_size = 640
config.training.dataloader_num_workers = 4
config.training.learning_rate = 1e-4
config.training.gradient_accumulation_steps = 1
config.training.output_dir = "/tmp/libero_all"
config.training.save_steps = 2000
config.training.save_total_limit = 5
config.training.num_gpus = NUM_GPUS
config.training.use_wandb = True
config.training.max_steps = 40000
config.training.weight_decay = 1e-5
config.training.warmup_ratio = 0.05
config.training.wandb_project = "finetune-gr00t-n1d7"

# Data config
config.data.shard_size = 2**10
config.data.episode_sampling_rate = 0.1
config.data.num_shards_per_epoch = int(1e5)

run(config)
