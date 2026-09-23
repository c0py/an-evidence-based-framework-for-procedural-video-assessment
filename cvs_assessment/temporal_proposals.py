"""Generic local temporal proposal models for procedural interval evidence."""
from __future__ import annotations

import torch
from torch import nn


def centered_windows(sequence: torch.Tensor, radius: int) -> torch.Tensor:
    """Return edge-replicated local windows with shape [time, window, features]."""
    if sequence.ndim != 2:
        raise ValueError("sequence must have shape [time, features]")
    if len(sequence) == 0:
        raise ValueError("sequence must be non-empty")
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    if radius == 0:
        return sequence[:, None, :]
    padded = torch.cat([
        sequence[:1].expand(radius, -1), sequence,
        sequence[-1:].expand(radius, -1),
    ], dim=0)
    return padded.unfold(0, 2 * radius + 1, 1).permute(0, 2, 1).contiguous()


class TemporalProposalWindowVerifier(nn.Module):
    """Shared criterion-agnostic verifier over a local evidence-track window."""

    def __init__(
        self, feature_dim: int, hidden_dim: int = 48, kernel_size: int = 5,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        padding = kernel_size // 2
        self.encoder = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size, padding=padding),
            nn.GELU(), nn.GroupNorm(4, hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.GELU(), nn.GroupNorm(4, hidden_dim), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3:
            raise ValueError("windows must have shape [batch, time, features]")
        encoded = self.encoder(windows.float().transpose(1, 2))
        center = encoded[:, :, encoded.shape[-1] // 2]
        pooled_mean = encoded.mean(dim=-1)
        pooled_max = encoded.amax(dim=-1)
        return self.head(torch.cat([center, pooled_mean, pooled_max], dim=-1)).squeeze(-1)
