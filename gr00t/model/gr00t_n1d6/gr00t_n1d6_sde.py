"""SDE (Stochastic Differential Equation) variant of the Gr00tN1d6 action head.

Replaces the deterministic Euler ODE solver with an Euler-Maruyama SDE solver
for flow matching inference. The training objective is identical; only the
sampling procedure changes.

Convention note:
  GR00T uses t in [0, 1] where t=0 is pure noise and t=1 is the clean target.
  openpi uses t in [1, 0] (reversed). The SDE formulas here are adapted to
  GR00T's forward-time convention.
"""

import math

import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.configs.model.gr00t_n1d6_sde import Gr00tN1d6SDEConfig
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead


class Gr00tN1d6SDEActionHead(Gr00tN1d6ActionHead):
    """Action head with SDE sampling for flow matching inference.

    Inherits all architecture and training from the base action head.
    Only overrides ``get_action_with_features`` to use Euler-Maruyama integration
    with time-dependent stochastic noise injection.
    """

    def __init__(self, config: Gr00tN1d6SDEConfig):
        super().__init__(config)
        self.noise_level = config.noise_level

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> BatchFeature:
        """Generate actions using the Flow SDE (Euler-Maruyama) diffusion process.

        Instead of the deterministic Euler step ``x += dt * v``, we use:
          1. Predict clean (x1) and noise (x0) endpoints from the velocity.
          2. Compute time-dependent diffusion coefficient sigma_i.
          3. Blend endpoints with SDE mixing weights.
          4. Inject Gaussian noise scaled by sigma_i (Euler-Maruyama).

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device

        # Start from pure noise (t=0 in GR00T convention).
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps

        for step in range(self.num_inference_timesteps):
            t = step * dt  # current time: 0, dt, 2*dt, ...
            t_discretized = int(t * self.num_timestep_buckets)

            # --- Forward pass (identical to ODE version) ---
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            sa_embs = torch.cat((state_features, action_features), dim=1)

            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # --- Flow SDE (Euler-Maruyama) ---
            # Adapted from openpi pi0_sde.py, converted to GR00T's convention
            # where t goes 0 -> 1 (noise -> clean).
            #
            # In openpi convention (tau = 1 - t, going 1 -> 0):
            #   delta = |d_tau|, sigma_i = noise_level * sqrt(tau / denom)
            #   x0_pred = x - v_openpi * tau   (clean target)
            #   x1_pred = x + v_openpi * (1-tau)  (noise)
            #
            # Mapping: tau = 1 - t, v_openpi = -pred_velocity
            #   clean_pred = x + pred_velocity * (1 - t)
            #   noise_pred = x - pred_velocity * t
            delta = dt

            # Time-dependent diffusion coefficient.
            # At t=0 (start, pure noise): denom=delta, sigma_i is large.
            # As t->1 (clean): sigma_i -> 0, recovering deterministic behavior.
            denom = delta if t < 1e-4 else t
            sigma_i = self.noise_level * math.sqrt((1.0 - t) / denom)

            # Predict both endpoints from current state and velocity.
            clean_pred = actions + pred_velocity * (1.0 - t)  # predicted clean action (at t=1)
            noise_pred = actions - pred_velocity * t  # predicted initial noise (at t=0)

            # SDE mixing weights (derived from openpi's flow SDE formulation).
            clean_weight = t + delta
            noise_weight = (1.0 - t - delta) - (sigma_i**2 * delta) / (2.0 * (1.0 - t))

            # Mean of the SDE transition kernel.
            x_mean = clean_pred * clean_weight + noise_pred * noise_weight
            # Standard deviation of the stochastic term.
            x_std = math.sqrt(delta) * sigma_i

            # Euler-Maruyama step: deterministic drift + stochastic diffusion.
            z = torch.randn_like(actions)
            actions = x_mean + x_std * z

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )
