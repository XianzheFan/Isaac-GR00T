"""Gr00t SDE Policy implementation for inference.

Loads a standard Gr00t model checkpoint and replaces the action head's
ODE sampling with SDE (Euler-Maruyama) sampling at inference time.
The model weights are identical; only the denoising loop changes.
"""

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoProcessor

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.model.gr00t_n1d7.gr00t_n1d6_sde import Gr00tN1d6SDEActionHead
from gr00t.policy.gr00t_policy import Gr00tPolicy


class Gr00tSDEPolicy(Gr00tPolicy):
    """Policy that uses SDE sampling for action generation.

    Loads the same pretrained checkpoint as ``Gr00tPolicy``, then replaces
    the action head with ``Gr00tN1d6SDEActionHead`` which uses Euler-Maruyama
    integration instead of deterministic Euler.

    Args:
        embodiment_tag: The embodiment tag defining the robot/environment type.
        model_path: Path to the pretrained model checkpoint directory.
        device: Device to run the model on (e.g., 'cuda:0', 0, 'cpu').
        noise_level: SDE diffusion noise strength (default: 0.5).
        num_inference_timesteps: Number of denoising steps (default: None, uses model config).
        strict: Whether to enforce strict input validation (default: True).
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag,
        model_path: str,
        *,
        device: int | str,
        noise_level: float = 0.5,
        num_inference_timesteps: int | None = None,
        strict: bool = True,
    ):
        # Import to register all models.
        import gr00t.model  # noqa: F401

        # Call grandparent init (BasePolicy), skip Gr00tPolicy.__init__
        # to avoid loading the model twice.
        from gr00t.policy.policy import BasePolicy

        BasePolicy.__init__(self, strict=strict)
        model_dir = Path(model_path)

        # Load model normally.
        model = AutoModel.from_pretrained(model_dir)

        # Patch config with SDE parameters.
        model.config.noise_level = noise_level
        model.config.noise_method = "flow_sde"
        if num_inference_timesteps is not None:
            model.config.num_inference_timesteps = num_inference_timesteps

        # Replace the action head with the SDE variant, transferring all weights.
        sde_action_head = Gr00tN1d6SDEActionHead(model.config)
        sde_action_head.load_state_dict(model.action_head.state_dict())
        model.action_head = sde_action_head

        model.eval()
        model.to(device=device, dtype=torch.bfloat16)
        self.model = model

        # Load processor (same as base).
        self.processor: BaseProcessor = AutoProcessor.from_pretrained(model_dir)
        self.processor.eval()

        self.embodiment_tag = embodiment_tag
        self.modality_configs = self.processor.get_modality_configs()[self.embodiment_tag.value]
        self.collate_fn = self.processor.collator

        language_keys = self.modality_configs["language"].modality_keys
        language_delta_indices = self.modality_configs["language"].delta_indices
        assert len(language_keys) == 1, "Only one language key is supported"
        assert len(language_delta_indices) == 1, "Only one language delta index is supported"
        self.language_key = language_keys[0]
