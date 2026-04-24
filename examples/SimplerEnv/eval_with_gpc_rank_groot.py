"""
SimplerEnv evaluation with GPC-RANK: world-model-guided action selection.

Replaces the Gemini + DreamDojo rescue pipeline with the GPC-RANK algorithm
(Qi et al. 2025, "Inference-Time Enhancement of Generative Robot Policies
via Predictive World Modeling"):
  1. Every replan step, sample N candidate action chunks via GR00T ODE
  2. Autoregressively rollout each candidate with a learned world model
  3. Score predicted final frames with a learned reward predictor
  4. Execute the highest-ranked (lowest cost) action chunk

This avoids external VLM / video-generation dependencies — only local
neural network inference is needed.

Usage:
    python examples/SimplerEnv/eval_with_gpc_rank_groot.py \
        --model_path /path/to/groot/checkpoint \
        --robot_type google \
        --gpc_config /path/to/gpc_config.yml \
        --world_model_ckpt /path/to/world_model.pth \
        --reward_predictor_ckpt /path/to/reward_predictor.pth
"""

import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import time
from dataclasses import dataclass
from functools import partial as functools_partial
from typing import List, Tuple

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.v2 as v2
import tqdm
import tyro
import yaml

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper


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
ACTION_DIM = len(ACTION_KEYS)  # 7 for SimplerEnv robots


###############################################################################
#  World Model Building Blocks
###############################################################################

GN_GROUP_SIZE = 32
GN_EPS = 1e-5
ATTN_HEAD_DIM = 8

Conv1x1 = functools_partial(nn.Conv2d, kernel_size=1, stride=1, padding=0)
Conv3x3 = functools_partial(nn.Conv2d, kernel_size=3, stride=1, padding=1)


class WMGroupNorm(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.norm = nn.GroupNorm(num_groups, in_channels, eps=GN_EPS)

    def forward(self, x):
        return self.norm(x)


class AdaGroupNorm(nn.Module):
    def __init__(self, in_channels: int, cond_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.linear = nn.Linear(cond_channels, in_channels * 2)

    def forward(self, x, cond):
        x = F.group_norm(x, self.num_groups, eps=GN_EPS)
        scale, shift = self.linear(cond)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift


class SelfAttention2d(nn.Module):
    def __init__(self, in_channels: int, head_dim: int = ATTN_HEAD_DIM):
        super().__init__()
        self.n_head = max(1, in_channels // head_dim)
        self.norm = WMGroupNorm(in_channels)
        self.qkv_proj = Conv1x1(in_channels, in_channels * 3)
        self.out_proj = Conv1x1(in_channels, in_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        n, c, h, w = x.shape
        x_normed = self.norm(x)
        qkv = self.qkv_proj(x_normed).view(
            n, self.n_head * 3, c // self.n_head, h * w
        ).transpose(2, 3).contiguous()
        q, k, v = qkv.chunk(3, dim=1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(2, 3).reshape(n, c, h, w)
        return x + self.out_proj(y)


class FourierFeatures(nn.Module):
    def __init__(self, cond_channels: int):
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer("weight", torch.randn(1, cond_channels // 2))

    def forward(self, x):
        assert x.ndim == 1
        f = 2 * math.pi * x.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)


class Downsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=1)
        nn.init.orthogonal_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = Conv3x3(c, c)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_ch, attn):
        super().__init__()
        self.proj = Conv1x1(in_ch, out_ch) if in_ch != out_ch else nn.Identity()
        self.norm1 = AdaGroupNorm(in_ch, cond_ch)
        self.conv1 = Conv3x3(in_ch, out_ch)
        self.norm2 = AdaGroupNorm(out_ch, cond_ch)
        self.conv2 = Conv3x3(out_ch, out_ch)
        self.attn = SelfAttention2d(out_ch) if attn else nn.Identity()
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x, cond):
        r = self.proj(x)
        x = self.conv1(F.silu(self.norm1(x, cond)))
        x = self.conv2(F.silu(self.norm2(x, cond)))
        return self.attn(x + r)


class ResBlocks(nn.Module):
    def __init__(self, list_in, list_out, cond_ch, attn):
        super().__init__()
        self.resblocks = nn.ModuleList([
            ResBlock(i, o, cond_ch, attn) for i, o in zip(list_in, list_out)
        ])

    def forward(self, x, cond, to_cat=None):
        outputs = []
        for i, rb in enumerate(self.resblocks):
            if to_cat is not None:
                x = torch.cat((x, to_cat[i]), dim=1)
            x = rb(x, cond)
            outputs.append(x)
        return x, outputs


class UNet(nn.Module):
    def __init__(self, cond_ch, depths, channels, attn_depths):
        super().__init__()
        self._num_down = len(channels) - 1
        d_blocks, u_blocks = [], []
        for i, n in enumerate(depths):
            c1, c2 = channels[max(0, i - 1)], channels[i]
            d_blocks.append(
                ResBlocks([c1] + [c2] * (n - 1), [c2] * n, cond_ch, attn_depths[i])
            )
            u_blocks.append(
                ResBlocks(
                    [2 * c2] * n + [c1 + c2], [c2] * n + [c1], cond_ch, attn_depths[i]
                )
            )
        self.d_blocks = nn.ModuleList(d_blocks)
        self.u_blocks = nn.ModuleList(reversed(u_blocks))
        self.mid_blocks = ResBlocks(
            [channels[-1]] * 2, [channels[-1]] * 2, cond_ch, True
        )
        self.downsamples = nn.ModuleList(
            [nn.Identity()] + [Downsample(c) for c in channels[:-1]]
        )
        self.upsamples = nn.ModuleList(
            [nn.Identity()] + [Upsample(c) for c in reversed(channels[:-1])]
        )

    def forward(self, x, cond):
        *_, h, w = x.size()
        n = self._num_down
        ph = math.ceil(h / 2**n) * 2**n - h
        pw = math.ceil(w / 2**n) * 2**n - w
        x = F.pad(x, (0, pw, 0, ph))
        d_outputs = []
        for block, down in zip(self.d_blocks, self.downsamples):
            x_d = down(x)
            x, bo = block(x_d, cond)
            d_outputs.append((x_d, *bo))
        x, _ = self.mid_blocks(x, cond)
        u_outputs = []
        for block, up, skip in zip(self.u_blocks, self.upsamples, reversed(d_outputs)):
            x_u = up(x)
            x, bo = block(x_u, cond, skip[::-1])
            u_outputs.append((x_u, *bo))
        return x[..., :h, :w], d_outputs, u_outputs


class InnerModel(nn.Module):
    def __init__(
        self, img_channels, num_steps_conditioning, cond_channels,
        depths, channels, attn_depths, action_dim, is_upsampler=False,
    ):
        super().__init__()
        self.noise_emb = FourierFeatures(cond_channels)
        self.noise_cond_emb = FourierFeatures(cond_channels)
        self.act_emb = nn.Sequential(
            nn.Linear(action_dim, cond_channels // num_steps_conditioning),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(),
            nn.Linear(cond_channels, cond_channels),
        )
        n_in = (num_steps_conditioning + int(is_upsampler) + 1) * img_channels
        self.conv_in = Conv3x3(n_in, channels[0])
        self.unet = UNet(cond_channels, depths, channels, attn_depths)
        self.norm_out = WMGroupNorm(channels[0])
        self.conv_out = Conv3x3(channels[0], img_channels)
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, noisy_next_obs, c_noise, c_noise_cond, obs, act):
        act_emb = self.act_emb(act) if act is not None else 0
        cond = self.cond_proj(
            self.noise_emb(c_noise) + self.noise_cond_emb(c_noise_cond) + act_emb
        )
        x = self.conv_in(torch.cat((obs, noisy_next_obs), dim=1))
        x, _, _ = self.unet(x, cond)
        return self.conv_out(F.silu(self.norm_out(x)))


# Denoiser

def _add_dims(t, n):
    return t.reshape(t.shape + (1,) * (n - t.ndim))


@dataclass
class DenoiserConfig:
    img_channels: int = 3
    num_steps_conditioning: int = 4
    cond_channels: int = 256
    depths: list = None
    channels: list = None
    attn_depths: list = None
    action_dim: int = 7  # SimplerEnv: x,y,z,roll,pitch,yaw,gripper
    sigma_data: float = 0.5
    sigma_offset_noise: float = 0.1
    noise_previous_obs: bool = True

    def __post_init__(self):
        if self.depths is None:
            self.depths = [2, 2, 2, 2]
        if self.channels is None:
            self.channels = [96, 96, 96, 96]
        if self.attn_depths is None:
            self.attn_depths = [0, 0, 1, 1]


class Denoiser(nn.Module):
    def __init__(self, cfg: DenoiserConfig):
        super().__init__()
        self.cfg = cfg
        self.inner_model = InnerModel(
            img_channels=cfg.img_channels,
            num_steps_conditioning=cfg.num_steps_conditioning,
            cond_channels=cfg.cond_channels,
            depths=cfg.depths,
            channels=cfg.channels,
            attn_depths=cfg.attn_depths,
            action_dim=cfg.action_dim,
        )
        self.sample_sigma_training = None

    @property
    def _device(self):
        return self.inner_model.noise_emb.weight.device

    def setup_sigma_sampling(self, loc=-1.2, scale=1.2, sigma_min=2e-3, sigma_max=20):
        def _sample(n, dev):
            s = torch.randn(n, device=dev) * scale + loc
            return s.exp().clip(sigma_min, sigma_max)
        self.sample_sigma_training = _sample

    def apply_noise(self, x, sigma, sigma_offset):
        b, c, _, _ = x.shape
        offset = sigma_offset * torch.randn(b, c, 1, 1, device=self._device)
        return x + offset + torch.randn_like(x) * _add_dims(sigma, x.ndim)

    def _conditioners(self, sigma, sigma_cond=None):
        sd = self.cfg.sigma_data
        so = self.cfg.sigma_offset_noise
        sigma = (sigma**2 + so**2).sqrt()
        c_in = 1 / (sigma**2 + sd**2).sqrt()
        c_skip = sd**2 / (sigma**2 + sd**2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        c_noise_cond = (
            sigma_cond.log() / 4
            if sigma_cond is not None
            else torch.zeros_like(c_noise)
        )
        return tuple(
            _add_dims(c, n)
            for c, n in zip(
                (c_in, c_out, c_skip, c_noise, c_noise_cond), (4, 4, 4, 1, 1)
            )
        )

    @torch.no_grad()
    def denoise(self, noisy, sigma, sigma_cond, obs, act):
        c_in, c_out, c_skip, c_noise, c_noise_cond = self._conditioners(
            sigma, sigma_cond
        )
        rescaled_obs = obs / self.cfg.sigma_data
        model_out = self.inner_model(
            noisy * c_in, c_noise, c_noise_cond, rescaled_obs, act
        )
        d = c_skip * noisy + c_out * model_out
        return d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)


# DiffusionSampler

@dataclass
class SamplerConfig:
    num_steps: int = 3
    sigma_min: float = 2e-3
    sigma_max: float = 5
    rho: int = 7
    order: int = 1
    s_churn: float = 0
    s_tmin: float = 0
    s_tmax: float = float("inf")
    s_noise: float = 1
    s_cond: float = 0


def _build_sigmas(n, smin, smax, rho, dev):
    mi = smin ** (1 / rho)
    mx = smax ** (1 / rho)
    l = torch.linspace(0, 1, n, device=dev)
    sigmas = (mx + l * (mi - mx)) ** rho
    return torch.cat((sigmas, sigmas.new_zeros(1)))


class DiffusionSampler:
    def __init__(self, denoiser: Denoiser, cfg: SamplerConfig):
        self.denoiser = denoiser
        self.cfg = cfg
        self.sigmas = _build_sigmas(
            cfg.num_steps, cfg.sigma_min, cfg.sigma_max, cfg.rho, denoiser._device
        )

    @torch.no_grad()
    def sample(self, prev_obs, prev_act):
        """
        Args:
            prev_obs: (B, T, C, H, W) conditioning frames
            prev_act: (B, T, action_dim) conditioning actions
        Returns:
            predicted next frame: (B, C, H, W)
        """
        dev = prev_obs.device
        b, t, c, h, w = prev_obs.size()
        prev_obs_flat = prev_obs.reshape(b, t * c, h, w)
        s_in = torch.ones(b, device=dev)
        gamma_ = min(
            self.cfg.s_churn / max(len(self.sigmas) - 1, 1), 2**0.5 - 1
        )
        x = torch.randn(b, c, h, w, device=dev)
        for sigma, next_sigma in zip(self.sigmas[:-1], self.sigmas[1:]):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
            sigma_hat = sigma * (gamma + 1)
            if gamma > 0:
                x = x + torch.randn_like(x) * self.cfg.s_noise * (
                    sigma_hat**2 - sigma**2
                ) ** 0.5
            sigma_cond = None
            obs_input = prev_obs_flat
            if self.cfg.s_cond > 0:
                sigma_cond = torch.full((b,), self.cfg.s_cond, device=dev)
                obs_input = self.denoiser.apply_noise(
                    obs_input, sigma_cond, sigma_offset_noise=0
                )
            denoised = self.denoiser.denoise(
                x, sigma * s_in, sigma_cond, obs_input, prev_act
            )
            d = (x - denoised) / sigma_hat
            dt = next_sigma - sigma_hat
            if self.cfg.order == 1 or next_sigma == 0:
                x = x + d * dt
            else:
                x2 = x + d * dt
                denoised2 = self.denoiser.denoise(
                    x2, next_sigma * s_in, sigma_cond, obs_input, prev_act
                )
                d2 = (x2 - denoised2) / next_sigma
                x = x + (d + d2) / 2 * dt
        return x


###############################################################################
#  Reward Predictor (GPC-RANK: ResNet18 + MLP)
###############################################################################

class RewardPredictor(nn.Module):
    """Scalar task-completion score from a single image. Lower = closer to goal."""

    def __init__(self, output_dim: int = 1):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.mlp(self.resnet18(x))


class GPCRankSelector:
    """
    Core GPC-RANK for SimplerEnv:
      - Maintain observation & action history
      - Given N candidates, batch rollout via world model
      - Score predicted final frames, return best candidate
    """

    def __init__(
        self,
        sampler: DiffusionSampler,
        reward_predictor: RewardPredictor,
        num_candidates: int = 10,
        rollout_steps: int = 8,
        n_cond: int = 4,
        img_size: int = 96,
        action_dim: int = 7,
        action_stats: dict | None = None,
        spread_factor: float = 1.01,
        device: torch.device = None,
    ):
        self.sampler = sampler
        self.reward_predictor = reward_predictor
        self.num_candidates = num_candidates
        self.rollout_steps = rollout_steps
        self.n_cond = n_cond
        self.img_size = img_size
        self.action_dim = action_dim
        self.action_stats = action_stats
        self.spread_factor = spread_factor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(img_size),
            v2.ToDtype(torch.float32, scale=True),
        ])

        self.obs_history: List[np.ndarray] = []
        self.action_history: List[np.ndarray] = []

    def reset(self):
        self.obs_history.clear()
        self.action_history.clear()

    def update_obs(self, img_hwc_uint8: np.ndarray):
        t = self.transform(img_hwc_uint8).numpy()  # (3, H, W) [0,1]
        self.obs_history.append(t * 2.0 - 1.0)  # -> [-1, 1]

    def update_action(self, action: np.ndarray):
        self.action_history.append(self._norm_act(action))

    def _norm_act(self, a):
        if self.action_stats is None:
            return a.astype(np.float32)
        lo, hi = self.action_stats["min"], self.action_stats["max"]
        return (2.0 * (a - lo) / (hi - lo + 1e-8) - 1.0).astype(np.float32)

    @torch.no_grad()
    def rank(self, candidates: List[np.ndarray]) -> Tuple[int, List[float]]:
        """
        Rank candidates (each shape (T, action_dim)) via world-model rollout.
        Returns (best_idx, per-candidate scores).
        """
        N = len(candidates)
        n = self.n_cond

        if len(self.obs_history) < n:
            return 0, [0.0] * N

        # conditioning frames: (N, n, 3, H, W)
        frames = torch.tensor(
            np.stack(self.obs_history[-n:]),
            dtype=torch.float32,
            device=self.device,
        )
        frames = frames.unsqueeze(0).expand(N, -1, -1, -1, -1)

        # -- conditioning actions: (N, n, action_dim) --
        if len(self.action_history) >= n:
            act_hist = np.stack(self.action_history[-n:])
        else:
            pad = n - len(self.action_history)
            existing = (
                np.stack(self.action_history)
                if self.action_history
                else np.zeros((0, self.action_dim), np.float32)
            )
            act_hist = np.concatenate(
                [np.zeros((pad, self.action_dim), np.float32), existing]
            )
        act_t = (
            torch.tensor(act_hist, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .expand(N, -1, -1)
        )

        # candidate actions: (N, rollout, action_dim)
        rollout = min(self.rollout_steps, min(c.shape[0] for c in candidates))
        cand = torch.stack([
            torch.tensor(
                self._norm_act(c[:rollout]), dtype=torch.float32, device=self.device
            )
            for c in candidates
        ])  # (N, rollout, action_dim)

        # autoregressive rollout
        pred = frames.clone()  # (N, n, 3, H, W)
        acts = act_t.clone()  # (N, n, action_dim)

        for s in range(rollout):
            next_frame = self.sampler.sample(
                pred[:, -n:], acts[:, -n:]
            )  # (N, 3, H, W)
            pred = torch.cat([pred, next_frame.unsqueeze(1)], dim=1)
            acts = torch.cat([acts, cand[:, s : s + 1]], dim=1)

        # score final frames
        final = (pred[:, -1] + 1.0) / 2.0  # (N, 3, H, W) in [0,1]
        scores = self.reward_predictor(final).squeeze(-1).cpu().numpy()

        best = int(np.argmin(scores))
        return best, scores.tolist()


###############################################################################
#  SimplerEnv observation helpers
###############################################################################

def _build_groot_obs_google(obs: dict):
    img = np.asarray(obs["video.image"])
    return {
        "video.image": img[np.newaxis, np.newaxis].astype(np.uint8),
        "state.x": np.array([[[float(obs["state.x"][0])]]], dtype=np.float32),
        "state.y": np.array([[[float(obs["state.y"][0])]]], dtype=np.float32),
        "state.z": np.array([[[float(obs["state.z"][0])]]], dtype=np.float32),
        "state.rx": np.array([[[float(obs["state.rx"][0])]]], dtype=np.float32),
        "state.ry": np.array([[[float(obs["state.ry"][0])]]], dtype=np.float32),
        "state.rz": np.array([[[float(obs["state.rz"][0])]]], dtype=np.float32),
        "state.rw": np.array([[[float(obs["state.rw"][0])]]], dtype=np.float32),
        "state.gripper": np.array(
            [[[float(obs["state.gripper"][0])]]], dtype=np.float32
        ),
        "annotation.human.action.task_description": (
            str(obs["annotation.human.action.task_description"]),
        ),
    }


def _build_groot_obs_widowx(obs: dict):
    img = np.asarray(obs["video.image_0"])
    return {
        "video.image_0": img[np.newaxis, np.newaxis].astype(np.uint8),
        "state.x": np.array([[[float(obs["state.x"][0])]]], dtype=np.float32),
        "state.y": np.array([[[float(obs["state.y"][0])]]], dtype=np.float32),
        "state.z": np.array([[[float(obs["state.z"][0])]]], dtype=np.float32),
        "state.roll": np.array([[[float(obs["state.roll"][0])]]], dtype=np.float32),
        "state.pitch": np.array([[[float(obs["state.pitch"][0])]]], dtype=np.float32),
        "state.yaw": np.array([[[float(obs["state.yaw"][0])]]], dtype=np.float32),
        "state.pad": np.array([[[float(obs["state.pad"][0])]]], dtype=np.float32),
        "state.gripper": np.array(
            [[[float(obs["state.gripper"][0])]]], dtype=np.float32
        ),
        "annotation.human.action.task_description": (
            str(obs["annotation.human.action.task_description"]),
        ),
    }


def _convert_to_simpler_action(action_chunk: dict, idx: int = 0) -> dict:
    """Convert GR00T action chunk to a single env-step action dict."""
    return {
        f"action.{key}": np.atleast_1d(action_chunk[f"action.{key}"][0, idx]).flatten()[
            :1
        ]
        for key in ACTION_KEYS
    }


def _convert_to_simpler_actions(
    action_chunk: dict, action_horizon: int
) -> list[dict]:
    """Convert GR00T action chunk to a list of env-step action dicts."""
    first_key = f"action.{ACTION_KEYS[0]}"
    avail = action_chunk[first_key].shape[1]
    num_steps = min(action_horizon, avail)
    return [_convert_to_simpler_action(action_chunk, i) for i in range(num_steps)]


def _action_dicts_to_array(action_list: list[dict]) -> np.ndarray:
    """Convert list of action dicts to (T, action_dim) array."""
    rows = []
    for a in action_list:
        rows.append(np.concatenate([a[f"action.{k}"] for k in ACTION_KEYS]))
    return np.array(rows, dtype=np.float32)


def _extract_image_from_obs(obs: dict, robot_type: str) -> np.ndarray:
    if robot_type == "google":
        return np.asarray(obs["video.image"])
    else:
        return np.asarray(obs["video.image_0"])


###############################################################################
#  GPC-RANK action selection for SimplerEnv
###############################################################################

def _array_to_action_dicts(arr: np.ndarray) -> list[dict]:
    """Convert (T, action_dim) array back to a list of SimplerEnv action dicts."""
    result = []
    for t in range(arr.shape[0]):
        d = {}
        for i, k in enumerate(ACTION_KEYS):
            d[f"action.{k}"] = arr[t, i : i + 1].astype(np.float32)
        result.append(d)
    return result


def _gpc_rank_select_action(
    obs: dict,
    groot_policy,
    build_groot_obs_fn,
    ranker: GPCRankSelector,
    replan_steps: int,
    robot_type: str,
) -> tuple[list[dict], dict]:
    """
    Sample N candidate action chunks, rank with world model,
    return the best action chunk as a list of env-step dicts.

    Following the original GPC-RANK paper (Qi et al. 2026), diversity comes
    from the ODE action head's random initial noise (torch.randn) on each
    call — analogous to the paper's DDPM sampling from N different noise
    vectors.  No SDE is used.

    NOTE: Obs/action history updates are handled by the caller at every
    env step (matching the original GPC-RANK algorithm).
    """
    groot_obs = build_groot_obs_fn(obs)

    # Sample N candidates via the standard ODE action head.
    # Each call starts from a fresh torch.randn noise vector, giving a
    # different action — same diversity mechanism as the original DDPM
    # batch sampling: torch.randn((N, pred_horizon, action_dim)).
    candidate_action_lists = []
    for _i in range(ranker.num_candidates):
        action_dict, _ = groot_policy.get_action(groot_obs)
        action_list = _convert_to_simpler_actions(action_dict, replan_steps)
        candidate_action_lists.append(action_list)

    # Convert to arrays for ranking
    candidate_arrays = [
        _action_dicts_to_array(al[:replan_steps]) for al in candidate_action_lists
    ]

    # Spread candidates around the mean (GPC-RANK paper)
    arr = np.stack(candidate_arrays)  # (N, T, action_dim)
    mean = arr.mean(axis=0, keepdims=True)
    arr = mean + ranker.spread_factor * (arr - mean)
    candidate_arrays = [arr[i] for i in range(len(candidate_arrays))]

    # Rank via world model + reward predictor
    t0 = time.perf_counter()
    best_idx, scores = ranker.rank(candidate_arrays)
    rank_time = (time.perf_counter() - t0) * 1000

    logging.info(
        f"[GPC-RANK] best={best_idx}  score={scores[best_idx]:.4f}  "
        f"rank_time={rank_time:.0f}ms  scores={[f'{s:.4f}' for s in scores]}"
    )

    record = {
        "best_idx": best_idx,
        "scores": scores,
        "rank_time_ms": rank_time,
    }
    # Return the *spread* version of the best candidate (matching original
    # eval_baseline.py and gpc_rank_agilex_infer.py which both execute the
    # spread actions, not the pre-spread ones).
    return _array_to_action_dicts(candidate_arrays[best_idx]), record


###############################################################################
#  Load world model & reward predictor from config
###############################################################################

def _load_gpc_components(
    gpc_config_path: str,
    world_model_ckpt: str,
    reward_predictor_ckpt: str,
    action_stats_path: str | None,
    device: torch.device,
) -> tuple[GPCRankSelector, dict]:
    """Load world model, reward predictor, and build GPCRankSelector."""
    wm = yaml.safe_load(open(gpc_config_path))
    action_dim = wm.get("action_dim", ACTION_DIM)

    dcfg = DenoiserConfig(
        img_channels=3,
        num_steps_conditioning=wm.get("num_steps_conditioning", 4),
        cond_channels=wm.get("cond_channels", 256),
        depths=wm.get("depths", [2, 2, 2, 2]),
        channels=wm.get("channels", [96, 96, 96, 96]),
        attn_depths=wm.get("attn_depths", [0, 0, 1, 1]),
        action_dim=action_dim,
        sigma_data=wm.get("sigma_data", 0.5),
        sigma_offset_noise=wm.get("sigma_offset_noise", 0.1),
        noise_previous_obs=wm.get("noise_previous_obs", True),
    )
    denoiser = Denoiser(dcfg).to(device)
    denoiser.setup_sigma_sampling(
        loc=wm.get("sigma_loc", -1.2),
        scale=wm.get("sigma_scale", 1.2),
        sigma_min=wm.get("sigma_min", 2e-3),
        sigma_max=wm.get("sigma_max", 20),
    )
    denoiser.load_state_dict(torch.load(world_model_ckpt, map_location=device))
    denoiser.eval()
    logging.info(f"World model loaded: {world_model_ckpt}")

    scfg = SamplerConfig(
        num_steps=wm.get("num_steps_denoising", 3),
        sigma_min=wm.get("sampler_sigma_min", 2e-3),
        sigma_max=wm.get("sampler_sigma_max", 5),
    )
    wm_sampler = DiffusionSampler(denoiser, scfg)

    reward_pred = RewardPredictor(
        output_dim=wm.get("reward_output_dim", 1)
    ).to(device)
    reward_pred.load_state_dict(
        torch.load(reward_predictor_ckpt, map_location=device)
    )
    reward_pred.eval()
    logging.info(f"Reward predictor loaded: {reward_predictor_ckpt}")

    action_stats = None
    if action_stats_path and os.path.exists(action_stats_path):
        raw = np.load(action_stats_path, allow_pickle=True).item()
        action_stats = {"min": raw["action"]["min"], "max": raw["action"]["max"]}

    return wm, wm_sampler, reward_pred, action_stats


@dataclasses.dataclass
class Args:
    model_path: str = "nvidia/GR00T-N1.6-3B"
    device: int = 0
    action_horizon: int = 8
    """Number of actions to execute per GPC-RANK replan."""

    robot_type: str = "google"
    num_trials_per_task: int = 20
    max_episode_steps: int = 120

    video_out_path: str = ""
    seed: int = 7

    # GPC-RANK
    gpc_config: str = ""
    """YAML config for world model / reward predictor."""
    world_model_ckpt: str = ""
    """World model checkpoint (.pth)."""
    reward_predictor_ckpt: str = ""
    """Reward predictor checkpoint (.pth)."""
    action_stats_path: str = ""
    """Action normalization stats (.npy). Empty = no normalization."""
    num_candidates: int = 10
    """Number of action candidates to sample from GR00T policy."""
    rollout_steps: int = 8
    """World-model rollout horizon for scoring candidates."""

    skip_tasks: int = 0


def eval_simpler_env(args: Args) -> None:
    os.environ.setdefault("DISPLAY", "")

    import gymnasium as gym
    from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs

    np.random.seed(args.seed)
    register_simpler_envs()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

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
        args.video_out_path = f"data/simpler_env_gpc_rank_{args.robot_type}"
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading GR00T policy from {args.model_path} ...")
    base_policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device=args.device,
        strict=True,
    )
    groot_policy = Gr00tSimPolicyWrapper(base_policy)
    logging.info("GR00T policy loaded (ODE action head for GPC-RANK candidate sampling).")

    # -- Load GPC-RANK components --
    wm_cfg, wm_sampler, reward_pred, action_stats = _load_gpc_components(
        gpc_config_path=args.gpc_config,
        world_model_ckpt=args.world_model_ckpt,
        reward_predictor_ckpt=args.reward_predictor_ckpt,
        action_stats_path=args.action_stats_path or None,
        device=device,
    )

    ranker = GPCRankSelector(
        sampler=wm_sampler,
        reward_predictor=reward_pred,
        num_candidates=args.num_candidates,
        rollout_steps=args.rollout_steps,
        n_cond=wm_cfg.get("num_steps_conditioning", 4),
        img_size=wm_cfg.get("img_size", 96),
        action_dim=wm_cfg.get("action_dim", ACTION_DIM),
        action_stats=action_stats,
        spread_factor=wm_cfg.get("spread_factor", 1.01),
        device=device,
    )

    total_episodes, total_successes = 0, 0

    for task_name in tqdm.tqdm(task_names, desc="tasks"):
        env_id = f"{env_prefix}/{task_name}"
        logging.info(f"\n=== Task: {env_id} ===")

        task_episodes, task_successes = 0, 0

        for trial_idx in tqdm.tqdm(
            range(args.num_trials_per_task), desc="episodes", leave=False
        ):
            task_segment = task_name.replace(" ", "_")

            # Skip if already completed
            existing_dirs = [
                d
                for d in pathlib.Path(args.video_out_path).glob(
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

            # Create a fresh env each trial
            env = gym.make(env_id)
            obs, info = env.reset(seed=args.seed + trial_idx)

            task_description = str(
                obs.get("annotation.human.action.task_description", task_name)
            )
            logging.info(f"\nTask: {task_description}")

            action_plan = collections.deque()
            clean_images = []
            gpc_rank_log = []

            ranker.reset()

            done, truncated = False, False

            logging.info(f"Starting episode {task_episodes + 1}...")
            for t in range(args.max_episode_steps):
                if done or truncated:
                    break

                try:
                    img = _extract_image_from_obs(obs, args.robot_type)
                    clean_images.append(img.copy())

                    # Update observation history at every step (matches
                    # original GPC-RANK agilex_infer.py main loop).
                    ranker.update_obs(img)

                    # Replan when action queue is empty
                    if not action_plan:
                        best_actions, record = _gpc_rank_select_action(
                            obs=obs,
                            groot_policy=groot_policy,
                            build_groot_obs_fn=build_groot_obs_fn,
                            ranker=ranker,
                            replan_steps=args.action_horizon,
                            robot_type=args.robot_type,
                        )
                        record["frame"] = len(clean_images)
                        gpc_rank_log.append(record)
                        action_plan.extend(best_actions)

                    action = action_plan.popleft()
                    obs, reward, done, truncated, info = env.step(action)

                    # Update action history at every step (matches original
                    # GPC-RANK: ranker.update_action(act) after each execution).
                    act_array = np.concatenate(
                        [action[f"action.{k}"] for k in ACTION_KEYS]
                    )
                    ranker.update_action(act_array)

                except Exception as e:
                    logging.error(f"Step exception: {e}")
                    break

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

            # Save GPC-RANK log
            json_path = rollout_dir / "gpc_rank_results.json"
            with open(json_path, "w") as f:
                json.dump(
                    {
                        "task": task_description,
                        "episode_index": trial_idx,
                        "outcome": suffix,
                        "num_candidates": args.num_candidates,
                        "rollout_steps": args.rollout_steps,
                        "gpc_rank_decisions": gpc_rank_log,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            logging.info(f"[GPC-RANK] Results written to {json_path}")

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
                f"# successes: {total_successes} "
                f"({total_successes / total_episodes * 100:.1f}%)"
            )

        logging.info(
            f"Current task success rate: "
            f"{float(task_successes) / float(max(task_episodes, 1))}"
        )
        logging.info(
            f"Current total success rate: "
            f"{float(total_successes) / float(max(total_episodes, 1))}"
        )

    logging.info(
        f"Total success rate: {float(total_successes) / float(max(total_episodes, 1))}"
    )
    logging.info(f"Total episodes: {total_episodes}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_simpler_env(tyro.cli(Args))
