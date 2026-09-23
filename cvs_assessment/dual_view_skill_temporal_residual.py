"""Frozen-Skill-conditioned temporal residual over replaceable dual-view features."""
from __future__ import annotations

import torch
from torch import nn

from .text_conditioned_temporal_cvs import SharedTemporalCvsHead


def centered_probability_mean(logits: torch.Tensor, radius: int = 4) -> torch.Tensor:
    """Return logit of replicate-padded centered probability smoothing."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [time, criteria]")
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    probabilities = torch.sigmoid(logits.float()).transpose(0, 1)[None]
    if radius:
        probabilities = nn.functional.pad(probabilities, (radius, radius), mode="replicate")
        probabilities = nn.functional.avg_pool1d(probabilities, radius * 2 + 1, stride=1)
    probabilities = probabilities[0].transpose(0, 1).clamp(1e-5, 1 - 1e-5)
    return torch.logit(probabilities)


class DualViewSkillTemporalResidual(nn.Module):
    """Small shared temporal verifier; visual backbones and Skill text stay frozen."""

    def __init__(
        self, visual_dim: int = 2048, text_dim: int = 4096,
        hidden_dim: int = 96, dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        self.center = nn.Sequential(
            nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden_dim), nn.GELU(),
        )
        self.full = nn.Sequential(
            nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden_dim), nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4), nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.temporal = SharedTemporalCvsHead(
            frame_dim=hidden_dim, text_dim=text_dim, hidden_dim=hidden_dim,
            dilations=dilations, dropout=dropout,
        )

    def forward(
        self, center_features: torch.Tensor, full_features: torch.Tensor,
        baseline_logits: torch.Tensor, criterion_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if center_features.ndim != 2 or center_features.shape != full_features.shape:
            raise ValueError("visual features must align as [time, features]")
        if baseline_logits.ndim != 2 or baseline_logits.shape[0] != center_features.shape[0]:
            raise ValueError("baseline_logits must align as [time, criteria]")
        if criterion_embeddings.ndim != 2 or criterion_embeddings.shape[0] != baseline_logits.shape[1]:
            raise ValueError("one Skill embedding is required per criterion")
        center = self.center(center_features.float())
        full = self.full(full_features.float())
        frame = self.fusion(torch.cat([center, full, center * full, (center - full).abs()], dim=-1))
        time = frame.shape[0]
        progress = torch.linspace(0, 1, time, device=frame.device)
        output = self.temporal(
            frame[None].expand(baseline_logits.shape[1], -1, -1),
            baseline_logits.transpose(0, 1), criterion_embeddings,
            progress[None].expand(baseline_logits.shape[1], -1),
        )
        return {
            "logits": output["logit"].transpose(0, 1),
            "residual": output["residual"].transpose(0, 1),
            "frame_hidden": frame,
        }
