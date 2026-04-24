# VLA Warmup + RL Token Training for GR00T-N1.7 on LIBERO.
#
# Phase 1 of the RLT pipeline (adapted from openpi/scripts/train_rlt.py):
#   1. Lightweight VLA fine-tune on perturbed_success data (OOD competence)
#   2. Train RL Token encoder-decoder:
#        encoder: z_{1:M} + e_rl -> Transformer -> z_rl  (compact bottleneck)
#        decoder: z_rl -> Transformer -> reconstruct z_{1:M}  (autoregressive)
#
# Usage:
#   NUM_GPUS=8
#   torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
#       examples/LIBERO/launch_warmup_rlt.py \
#       [--perturbed_data_dir <path>] \
#       [--output_dir <path>] \
#       [--rl_token_dim 2048] \
#       [--rlt_encoder_layers 2] \
#       [--rlt_decoder_layers 2] \
#       [--rlt_num_steps 5000] \
#       [--finetune_vla] \
#       [--vla_finetune_weight 0.1]

import argparse
import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from transformers.feature_extraction_utils import BatchFeature

from gr00t.configs.base_config import get_default_config
from gr00t.experiment.experiment import run

if "LOGURU_LEVEL" not in os.environ:
    os.environ["LOGURU_LEVEL"] = "INFO"

logger = logging.getLogger("warmup_rlt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# RL Token Encoder-Decoder (lightweight Transformer)
# ---------------------------------------------------------------------------


class RLTokenEncoder(nn.Module):
    """Compress VLA embeddings z_{1:M} into a single RL token z_rl.

    Architecture:
        Input: [z_1, z_2, ..., z_M, e_rl]  (M+1 tokens of dim ``embed_dim``)
        Process: Transformer encoder layers with self-attention
        Output: z_rl = output at position M+1 (the RL token position)

    The RL token acts as an information bottleneck: it must retain enough
    task-relevant information from the VLA's internal features to allow the
    decoder to reconstruct the original embeddings.
    """

    def __init__(
        self,
        embed_dim: int = 2048,
        rl_token_dim: int = 2048,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.rl_token_dim = rl_token_dim

        # Learnable RL token embedding (appended to the sequence)
        self.rl_token_embedding = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim or embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project to RL token dimension (if different from embed_dim)
        self.proj = (
            nn.Linear(embed_dim, rl_token_dim)
            if rl_token_dim != embed_dim
            else nn.Identity()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, M, embed_dim] — VLA internal embeddings (backbone_features).

        Returns:
            z_rl: [B, rl_token_dim] — compressed RL token representation.
        """
        B = z.shape[0]

        # Append learnable RL token to the sequence
        e_rl = self.rl_token_embedding.expand(B, -1, -1)  # [B, 1, embed_dim]
        augmented = torch.cat([z, e_rl], dim=1)  # [B, M+1, embed_dim]

        # Run through transformer encoder
        output = self.transformer(augmented)  # [B, M+1, embed_dim]

        # Extract the RL token (last position)
        z_rl = output[:, -1, :]  # [B, embed_dim]

        # Project to target dimension
        z_rl = self.proj(z_rl)  # [B, rl_token_dim]

        return z_rl


class RLTokenDecoder(nn.Module):
    """Autoregressively reconstruct VLA embeddings z_{1:M} from z_rl.

    Architecture (autoregressive):
        Step i: input = [z_rl, sg(z_1), ..., sg(z_{i-1})]
        Output: predicted z_i via decoder transformer + linear head

    Loss: L_ro = sum_{i=1}^{M} || h(decoder([z_rl, z_1:i-1])_i) - sg(z_i) ||^2

    During training we use teacher forcing with causal masking for efficiency.
    """

    def __init__(
        self,
        embed_dim: int = 2048,
        rl_token_dim: int = 2048,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.rl_token_dim = rl_token_dim

        # Project RL token back to embed_dim (if dimensions differ)
        self.rl_proj = (
            nn.Linear(rl_token_dim, embed_dim)
            if rl_token_dim != embed_dim
            else nn.Identity()
        )

        # Causal transformer decoder layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim or embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        # Output projection head: predict each embedding
        self.output_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self, z_rl: torch.Tensor, z_targets: torch.Tensor
    ) -> torch.Tensor:
        """Autoregressive reconstruction with teacher forcing.

        Args:
            z_rl: [B, rl_token_dim] — compressed RL token.
            z_targets: [B, M, embed_dim] — stop-gradient VLA embeddings (targets).

        Returns:
            z_pred: [B, M, embed_dim] — predicted embeddings for positions 1..M.
        """
        B, M, D = z_targets.shape

        # Project RL token to embed_dim
        z_rl_proj = self.rl_proj(z_rl).unsqueeze(1)  # [B, 1, embed_dim]

        # Build input sequence: [z_rl, sg(z_1), sg(z_2), ..., sg(z_{M-1})]
        # The decoder at position i predicts z_i using [z_rl, z_1, ..., z_{i-1}]
        shifted_targets = z_targets[:, :-1, :]  # [B, M-1, embed_dim]
        decoder_input = torch.cat([z_rl_proj, shifted_targets], dim=1)  # [B, M, embed_dim]

        # Causal mask: position i can only attend to positions <= i
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            M, device=z_rl.device, dtype=z_rl.dtype
        )

        # Run through causal decoder transformer
        output = self.transformer(decoder_input, mask=causal_mask)  # [B, M, embed_dim]

        # Project to predictions
        z_pred = self.output_head(output)  # [B, M, embed_dim]

        return z_pred


class RLTokenModule(nn.Module):
    """Combined RL Token encoder-decoder module.

    Wraps encoder + decoder and computes the reconstruction objective L_ro.
    """

    def __init__(
        self,
        embed_dim: int = 2048,
        rl_token_dim: int = 2048,
        encoder_num_layers: int = 2,
        decoder_num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = RLTokenEncoder(
            embed_dim=embed_dim,
            rl_token_dim=rl_token_dim,
            num_layers=encoder_num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.decoder = RLTokenDecoder(
            embed_dim=embed_dim,
            rl_token_dim=rl_token_dim,
            num_layers=decoder_num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute RL token and reconstruction loss.

        Args:
            z: [B, M, embed_dim] — VLA backbone features (will be stop-gradient'd).

        Returns:
            dict with:
                - z_rl: [B, rl_token_dim] — the RL token
                - loss_ro: scalar — reconstruction loss L_ro
                - z_pred: [B, M, embed_dim] — predicted embeddings
        """
        # Stop gradient on targets (VLA embeddings are frozen w.r.t. L_ro)
        z_sg = z.detach()

        # Encode -> z_rl
        z_rl = self.encoder(z_sg)  # [B, rl_token_dim]

        # Decode -> reconstruct z_{1:M}
        z_pred = self.decoder(z_rl, z_sg)  # [B, M, embed_dim]

        # Reconstruction loss: L_ro = mean || z_pred_i - sg(z_i) ||^2
        loss_ro = F.mse_loss(z_pred, z_sg)

        return {"z_rl": z_rl, "loss_ro": loss_ro, "z_pred": z_pred}


# ---------------------------------------------------------------------------
# Training logic
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="VLA Warmup + RL Token Training for GR00T-N1.7"
    )

    # Data
    parser.add_argument(
        "--perturbed_data_dir",
        type=str,
        default="examples/LIBERO/perturbed_success_lerobot/",
        help="Path to perturbed_success demonstration data (LeRobot format).",
    )
    parser.add_argument(
        "--dataset_paths",
        type=str,
        nargs="+",
        default=None,
        help="Override dataset paths for VLA fine-tuning. If not set, uses perturbed_data_dir.",
    )

    # RL Token architecture
    parser.add_argument("--rl_token_dim", type=int, default=2048)
    parser.add_argument("--rlt_encoder_layers", type=int, default=2)
    parser.add_argument("--rlt_decoder_layers", type=int, default=2)
    parser.add_argument("--rlt_num_heads", type=int, default=8)
    parser.add_argument("--rlt_dropout", type=float, default=0.1)

    # Training
    parser.add_argument(
        "--rlt_num_steps",
        type=int,
        default=5000,
        help="Number of RL token training steps (paper: 2000-10000).",
    )
    parser.add_argument("--rlt_learning_rate", type=float, default=1e-4)
    parser.add_argument("--rlt_batch_size", type=int, default=32)
    parser.add_argument(
        "--finetune_vla",
        action="store_true",
        default=True,
        help="Also fine-tune VLA backbone (projector + diffusion) during RL token training.",
    )
    parser.add_argument(
        "--no_finetune_vla",
        action="store_false",
        dest="finetune_vla",
    )
    parser.add_argument(
        "--vla_finetune_weight",
        type=float,
        default=0.1,
        help="L_total = L_ro + alpha * L_vla  (alpha = vla_finetune_weight).",
    )

    # VLA config overrides
    parser.add_argument("--vla_max_steps", type=int, default=40000)
    parser.add_argument("--vla_learning_rate", type=float, default=1e-4)
    parser.add_argument("--vla_global_batch_size", type=int, default=640)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="nvidia/GR00T-N1.7-3B",
        help="Base VLA checkpoint to start from.",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/libero_warmup_rlt",
    )
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--wandb_project", type=str, default="gr00t-rlt")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def run_phase1_rlt(args):
    """Phase 1: Train RL Token encoder-decoder on demonstration data.

    Jointly:
      - Train RL Token (encoder-decoder) via reconstruction loss L_ro
      - (Optionally) fine-tune VLA via flow matching loss L_vla

    Total loss: L_total = L_ro + alpha * L_vla

    The VLA backbone features z_{1:M} are:
      backbone_features: [B, seq_len, 2048]  (from Eagle VLM, layer 16)

    The RL token compresses these M tokens into a single z_rl of dim ``rl_token_dim``.
    """

    NUM_GPUS = int(os.environ.get("NUM_GPUS", 8))

    # Wandb login
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)

    # ----- Step 1: Set up VLA config for lightweight fine-tuning -----

    dataset_paths = args.dataset_paths or [args.perturbed_data_dir]
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": dataset_paths,
                        "mix_ratio": 1.0,
                        "embodiment_tag": "libero_panda",
                    },
                ],
            }
        }
    )

    config.load_config_path = None

    # Model config — lightweight fine-tuning (projector + diffusion only)
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
    config.training.start_from_checkpoint = args.checkpoint
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = args.vla_global_batch_size
    config.training.dataloader_num_workers = 4
    config.training.learning_rate = args.vla_learning_rate
    config.training.gradient_accumulation_steps = 1
    config.training.output_dir = os.path.join(args.output_dir, "vla_warmup")
    config.training.save_steps = args.save_steps
    config.training.save_total_limit = 5
    config.training.num_gpus = NUM_GPUS
    config.training.use_wandb = True
    config.training.max_steps = args.vla_max_steps
    config.training.weight_decay = 1e-5
    config.training.warmup_ratio = 0.05
    config.training.wandb_project = args.wandb_project

    # Data config
    config.data.shard_size = 2**10
    config.data.episode_sampling_rate = 0.1
    config.data.num_shards_per_epoch = int(1e5)

    if not args.finetune_vla:
        logger.info("Skipping VLA warmup fine-tuning (--no_finetune_vla).")
    else:
        logger.info("=" * 60)
        logger.info("Phase 1a: VLA Warmup Fine-Tuning on perturbed_success data")
        logger.info(f"  Checkpoint: {args.checkpoint}")
        logger.info(f"  Data:       {dataset_paths}")
        logger.info(f"  Max steps:  {args.vla_max_steps}")
        logger.info("=" * 60)
        run(config)

    # ----- Step 2: Train RL Token encoder-decoder -----

    logger.info("=" * 60)
    logger.info("Phase 1b: RL Token Encoder-Decoder Training")
    logger.info(f"  RL token dim:       {args.rl_token_dim}")
    logger.info(f"  Encoder layers:     {args.rlt_encoder_layers}")
    logger.info(f"  Decoder layers:     {args.rlt_decoder_layers}")
    logger.info(f"  Training steps:     {args.rlt_num_steps}")
    logger.info(f"  VLA finetune:       {args.finetune_vla} (weight={args.vla_finetune_weight})")
    logger.info("=" * 60)

    train_rl_token(args, config)


def train_rl_token(args, vla_config):
    """Train the RL Token encoder-decoder on VLA backbone features.

    For each batch:
      1. Forward VLA backbone to get z_{1:M} = backbone_features [B, M, 2048]
      2. Forward RL Token module:
           encoder(z_{1:M}) -> z_rl [B, rl_token_dim]
           decoder(z_rl, sg(z_{1:M})) -> z_pred [B, M, 2048]
           L_ro = MSE(z_pred, sg(z_{1:M}))
      3. (Optional) Compute VLA flow matching loss L_vla
      4. Total loss: L_total = L_ro + alpha * L_vla
    """
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    from gr00t.model import MODEL_REGISTRY

    # Determine device
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        local_rank = dist.get_rank()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = local_rank == 0

    if is_main:
        wandb.init(
            project=args.wandb_project,
            name=f"rl_token_training",
            config=vars(args),
        )

    # ----- Load the VLA model -----
    # Use the warmup checkpoint if VLA was fine-tuned, otherwise use base
    if args.finetune_vla:
        vla_checkpoint = os.path.join(args.output_dir, "vla_warmup")
        # Find latest checkpoint in the warmup dir
        import glob
        ckpt_dirs = sorted(glob.glob(os.path.join(vla_checkpoint, "checkpoint-*")))
        if ckpt_dirs:
            vla_checkpoint = ckpt_dirs[-1]
            logger.info(f"Using warmup checkpoint: {vla_checkpoint}")
        else:
            vla_checkpoint = args.checkpoint
            logger.info(f"No warmup checkpoint found, using base: {vla_checkpoint}")
    else:
        vla_checkpoint = args.checkpoint

    logger.info(f"Loading VLA model from {vla_checkpoint}...")

    # Load model via GR00T registry
    model_config = vla_config.model
    model_cls = MODEL_REGISTRY.get(model_config.model_type)
    if model_cls is None:
        # Fallback: load from pretrained
        from transformers import AutoModel, AutoConfig
        model_config_hf = AutoConfig.from_pretrained(vla_checkpoint, trust_remote_code=True)
        vla_model = AutoModel.from_pretrained(
            vla_checkpoint, config=model_config_hf, trust_remote_code=True
        )
    else:
        vla_model = model_cls(model_config)
        # Load checkpoint weights
        from gr00t.experiment.utils import load_checkpoint
        try:
            load_checkpoint(vla_model, vla_checkpoint)
        except Exception as e:
            logger.warning(f"Could not load checkpoint via load_checkpoint: {e}")
            logger.info("Attempting from_pretrained...")
            from transformers import AutoModel
            vla_model = AutoModel.from_pretrained(
                vla_checkpoint, trust_remote_code=True
            )

    vla_model = vla_model.to(device)

    # Freeze VLA for RL token training (only RL token module is trained,
    # plus optional lightweight VLA fine-tuning via L_vla)
    if not args.finetune_vla:
        for param in vla_model.parameters():
            param.requires_grad = False
    else:
        # Freeze backbone, keep action head trainable for L_vla
        vla_model.backbone.requires_grad_(False)
        # Action head projector + diffusion remain trainable
        vla_model.action_head.set_trainable_parameters(
            tune_projector=True,
            tune_diffusion_model=True,
            tune_vlln=False,
        )

    # ----- Create RL Token module -----
    # backbone_embedding_dim from GR00T config = 2048
    embed_dim = getattr(model_config, "backbone_embedding_dim", 2048)

    rlt_module = RLTokenModule(
        embed_dim=embed_dim,
        rl_token_dim=args.rl_token_dim,
        encoder_num_layers=args.rlt_encoder_layers,
        decoder_num_layers=args.rlt_decoder_layers,
        num_heads=args.rlt_num_heads,
        dropout=args.rlt_dropout,
    ).to(device)

    logger.info(
        f"RL Token module: {sum(p.numel() for p in rlt_module.parameters()):,} parameters"
    )

    # ----- Optimizer -----
    # Two parameter groups: RL Token (main) and VLA (optional, lower lr)
    param_groups = [
        {"params": rlt_module.parameters(), "lr": args.rlt_learning_rate},
    ]
    if args.finetune_vla:
        vla_trainable = [p for p in vla_model.parameters() if p.requires_grad]
        if vla_trainable:
            param_groups.append(
                {
                    "params": vla_trainable,
                    "lr": args.rlt_learning_rate * args.vla_finetune_weight,
                }
            )
            logger.info(
                f"VLA trainable params: {sum(p.numel() for p in vla_trainable):,}"
            )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-5)

    # Cosine LR schedule with warmup
    warmup_steps = int(args.rlt_num_steps * 0.05)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(args.rlt_num_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ----- DDP -----
    if is_distributed:
        rlt_module = DDP(rlt_module, device_ids=[local_rank])
        if args.finetune_vla and any(p.requires_grad for p in vla_model.parameters()):
            vla_model = DDP(vla_model, device_ids=[local_rank], find_unused_parameters=True)

    # ----- Data loader -----
    # Reuse the standard GR00T data pipeline
    from gr00t.data.dataset import ModalityConfig
    from gr00t.data.transform.base import ComposedModalityTransform
    from gr00t.experiment.data_config import GR00TDatasetConfig

    # Build dataset from the config
    from gr00t.experiment.runner import build_dataloaders

    # We use a simpler approach: create a minimal dataloader from the config
    # that yields batches compatible with vla_model.forward()
    logger.info("Setting up data loader...")

    # Use the experiment runner's data building infrastructure
    from gr00t.data.dataset import LeRobotSingleDataset

    dataset_path = args.perturbed_data_dir
    logger.info(f"Loading data from: {dataset_path}")

    # For simplicity, we iterate via the standard training pipeline's dataloader
    # but only run the RL token training loop (not HF Trainer)
    from torch.utils.data import DataLoader

    # We need to build the dataset + transforms matching the VLA's expected input
    # This is done by leveraging the model's collator
    vla_model_raw = vla_model.module if hasattr(vla_model, "module") else vla_model

    # Import the dataset loading utilities used by the experiment runner
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.data.embodiment import EmbodimentTag

    try:
        dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            embodiment_tag="libero_panda",
            model_config=model_config,
        )
    except Exception as e:
        logger.warning(f"Could not load dataset via LeRobotSingleDataset: {e}")
        logger.info("Falling back to generic HF dataset loading...")
        from datasets import load_dataset

        hf_dataset = load_dataset(dataset_path, split="train")
        dataset = hf_dataset

    # Collator from the VLA model handles tokenization + image processing
    collator = vla_model_raw.collator

    dataloader = DataLoader(
        dataset,
        batch_size=args.rlt_batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collator,
        drop_last=True,
        pin_memory=True,
    )

    # ----- Training loop -----
    output_dir = os.path.join(args.output_dir, "rl_token")
    os.makedirs(output_dir, exist_ok=True)

    best_loss = float("inf")
    global_step = 0
    vla_model.train()
    rlt_module.train()

    logger.info(f"Starting RL Token training for {args.rlt_num_steps} steps...")

    while global_step < args.rlt_num_steps:
        for batch in dataloader:
            if global_step >= args.rlt_num_steps:
                break

            optimizer.zero_grad()

            # ---- Forward through VLA backbone to get z_{1:M} ----
            with torch.set_grad_enabled(args.finetune_vla):
                backbone_inputs, action_inputs = vla_model_raw.prepare_input(batch)
                backbone_output = vla_model_raw.backbone(backbone_inputs)

            # backbone_features: [B, M, 2048]
            backbone_features = backbone_output.backbone_features

            # ---- RL Token forward: encode z_{1:M} -> z_rl, decode -> reconstruct ----
            rlt_output = rlt_module(backbone_features)
            loss_ro = rlt_output["loss_ro"]

            # ---- (Optional) VLA action loss for joint fine-tuning ----
            total_loss = loss_ro
            loss_vla = torch.tensor(0.0, device=device)

            if args.finetune_vla:
                action_output = vla_model_raw.action_head(backbone_output, action_inputs)
                loss_vla = action_output["loss"]
                total_loss = loss_ro + args.vla_finetune_weight * loss_vla

            # ---- Backward + step ----
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(rlt_module.parameters())
                + (
                    [p for p in vla_model.parameters() if p.requires_grad]
                    if args.finetune_vla
                    else []
                ),
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()

            # ---- Logging ----
            if global_step % 100 == 0 and is_main:
                loss_ro_val = loss_ro.item()
                loss_vla_val = loss_vla.item()
                total_val = total_loss.item()
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Step {global_step}/{args.rlt_num_steps} | "
                    f"L_ro: {loss_ro_val:.6f} | L_vla: {loss_vla_val:.6f} | "
                    f"L_total: {total_val:.6f} | lr: {lr:.2e}"
                )
                wandb.log(
                    {
                        "rlt/loss_ro": loss_ro_val,
                        "rlt/loss_vla": loss_vla_val,
                        "rlt/loss_total": total_val,
                        "rlt/lr": lr,
                        "rlt/step": global_step,
                    },
                    step=global_step,
                )
                if total_val < best_loss:
                    best_loss = total_val

            # ---- Checkpointing ----
            if (global_step + 1) % args.save_steps == 0 and is_main:
                ckpt_path = os.path.join(output_dir, f"step_{global_step + 1}")
                save_rlt_checkpoint(
                    rlt_module, vla_model, optimizer, global_step, ckpt_path
                )
                logger.info(f"Saved checkpoint to {ckpt_path}")

            global_step += 1

    # ---- Save final checkpoint ----
    if is_main:
        final_path = os.path.join(output_dir, "best")
        save_rlt_checkpoint(rlt_module, vla_model, optimizer, global_step, final_path)
        logger.info(f"Phase 1b complete. Best loss: {best_loss:.6f}")
        logger.info(f"Final checkpoint saved to {final_path}")
        wandb.log({"rlt/best_loss": best_loss})
        wandb.finish()


def save_rlt_checkpoint(rlt_module, vla_model, optimizer, step, path):
    """Save RL Token module + (optionally) VLA weights."""
    os.makedirs(path, exist_ok=True)
    rlt_state = (
        rlt_module.module.state_dict()
        if hasattr(rlt_module, "module")
        else rlt_module.state_dict()
    )
    torch.save(
        {
            "rlt_module": rlt_state,
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        os.path.join(path, "rlt_checkpoint.pt"),
    )
    # Also save VLA if it was fine-tuned
    vla_raw = vla_model.module if hasattr(vla_model, "module") else vla_model
    trainable_state = {
        k: v
        for k, v in vla_raw.state_dict().items()
        if any(
            p.requires_grad
            for n, p in vla_raw.named_parameters()
            if n == k
        )
    }
    if trainable_state:
        torch.save(trainable_state, os.path.join(path, "vla_trainable.pt"))


def load_rlt_checkpoint(rlt_module, path, device="cpu"):
    """Load a saved RL Token checkpoint."""
    ckpt = torch.load(os.path.join(path, "rlt_checkpoint.pt"), map_location=device)
    rlt_module.load_state_dict(ckpt["rlt_module"])
    return ckpt.get("step", 0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    args = parse_args()
    run_phase1_rlt(args)
