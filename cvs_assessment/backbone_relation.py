"""Shared text-conditioned temporal relations over frozen visual backbones."""
from __future__ import annotations

import torch
from torch import nn


class TextConditionedBackboneRelationVerifier(nn.Module):
    """Small shared verifier over a frozen visual backbone and frozen Skill text."""

    def __init__(
        self, visual_dim: int, text_dim: int = 4096, hidden_dim: int = 96,
        kernel_size: int = 5, dropout: float = 0.12,
    ) -> None:
        super().__init__()
        if visual_dim <= 0:
            raise ValueError("visual_dim must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        if hidden_dim % 8:
            raise ValueError("hidden_dim must be divisible by 8")
        padding = kernel_size // 2
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
        )
        self.condition = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.GELU(),
            nn.GroupNorm(8, hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.GELU(),
            nn.GroupNorm(8, hidden_dim),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, visual_windows: torch.Tensor, criterion_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if visual_windows.ndim != 3:
            raise ValueError("visual_windows must have shape [batch, time, features]")
        if criterion_embeddings.ndim != 2:
            raise ValueError("criterion_embeddings must have shape [batch, text_dim]")
        if len(visual_windows) != len(criterion_embeddings):
            raise ValueError("One criterion embedding is required per visual window")
        visual = self.visual_projection(visual_windows.float())
        text = self.text_projection(criterion_embeddings.float())[:, None, :]
        text = text.expand(-1, visual.shape[1], -1)
        conditioned = self.condition(torch.cat([
            visual, text, visual * text, (visual - text).abs(),
        ], dim=-1))
        encoded = self.temporal(conditioned.transpose(1, 2))
        center = encoded[:, :, encoded.shape[-1] // 2]
        pooled_mean = encoded.mean(dim=-1)
        pooled_max = encoded.amax(dim=-1)
        return self.classifier(torch.cat([
            center, pooled_mean, pooled_max,
        ], dim=-1)).squeeze(-1)
