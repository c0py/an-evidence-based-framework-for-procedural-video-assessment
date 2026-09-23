"""Generic Skill-conditioned spatial-ROI and temporal relation verifier."""
from __future__ import annotations

import math

import torch
from torch import nn


class TextConditionedSpatialRoiRelationVerifier(nn.Module):
    """Attend over spatial/ROI tokens per frame, then model their temporal state."""

    def __init__(
        self, token_dim: int, geometry_dim: int = 9, text_dim: int = 4096,
        hidden_dim: int = 64, attention_heads: int = 4,
        temporal_kernel_size: int = 5, dropout: float = 0.12,
    ) -> None:
        super().__init__()
        if token_dim <= 0 or geometry_dim <= 0:
            raise ValueError("token_dim and geometry_dim must be positive")
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if temporal_kernel_size <= 0 or temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be a positive odd number")
        self.hidden_dim = hidden_dim
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim), nn.GELU(),
        )
        self.geometry_projection = nn.Sequential(
            nn.LayerNorm(geometry_dim), nn.Linear(geometry_dim, hidden_dim), nn.GELU(),
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim), nn.GELU(),
        )
        self.condition = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(),
        )
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=attention_heads,
            dim_feedforward=hidden_dim * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, num_layers=1)
        self.frame_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(),
        )
        padding = temporal_kernel_size // 2
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, temporal_kernel_size, padding=padding),
            nn.GELU(), nn.GroupNorm(8, hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, temporal_kernel_size, padding=padding),
            nn.GELU(), nn.GroupNorm(8, hidden_dim), nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, token_windows: torch.Tensor, geometry_windows: torch.Tensor,
        criterion_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if token_windows.ndim != 4:
            raise ValueError("token_windows must have shape [batch, time, tokens, features]")
        if geometry_windows.ndim != 4:
            raise ValueError("geometry_windows must have shape [batch, time, tokens, features]")
        if token_windows.shape[:3] != geometry_windows.shape[:3]:
            raise ValueError("Token and geometry windows must align")
        if criterion_embeddings.ndim != 2 or len(criterion_embeddings) != len(token_windows):
            raise ValueError("One criterion embedding is required per token window")
        batch, time, token_count, _ = token_windows.shape
        visual = self.visual_projection(token_windows.float())
        visual = visual + self.geometry_projection(geometry_windows.float())
        text = self.text_projection(criterion_embeddings.float())
        expanded_text = text[:, None, None, :].expand(-1, time, token_count, -1)
        conditioned = self.condition(torch.cat([
            visual, expanded_text, visual * expanded_text,
            (visual - expanded_text).abs(),
        ], dim=-1))
        flat = conditioned.reshape(batch * time, token_count, self.hidden_dim)
        presence = geometry_windows[..., 6].reshape(batch * time, token_count) > 0.5
        encoded = self.spatial_encoder(flat, src_key_padding_mask=~presence)
        flat_text = text[:, None, :].expand(-1, time, -1).reshape(batch * time, self.hidden_dim)
        attention_logits = (encoded * flat_text[:, None, :]).sum(dim=-1) / math.sqrt(self.hidden_dim)
        attention_logits = attention_logits.masked_fill(~presence, -torch.inf)
        attention = torch.softmax(attention_logits, dim=-1)
        attended = (encoded * attention[..., None]).sum(dim=1)
        divisor = presence.sum(dim=1, keepdim=True).clamp_min(1)
        mean = (encoded * presence[..., None]).sum(dim=1) / divisor
        maximum = encoded.masked_fill(~presence[..., None], -torch.inf).amax(dim=1)
        frame = self.frame_projection(torch.cat([attended, mean, maximum], dim=-1))
        frame = frame.reshape(batch, time, self.hidden_dim)
        encoded_time = self.temporal(frame.transpose(1, 2))
        center = encoded_time[:, :, encoded_time.shape[-1] // 2]
        temporal_mean = encoded_time.mean(dim=-1)
        temporal_max = encoded_time.amax(dim=-1)
        return self.classifier(torch.cat([
            center, temporal_mean, temporal_max,
        ], dim=-1)).squeeze(-1)


class _DilatedTemporalResidualBlock(nn.Module):
    """Small task-neutral residual TCN block used by the direct interval head."""

    def __init__(self, hidden_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                hidden_dim, hidden_dim, kernel_size=3,
                padding=dilation, dilation=dilation,
            ),
            nn.GroupNorm(8, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class TextConditionedDirectIntervalModel(nn.Module):
    """Predict procedural-state intervals from frozen visual tokens and Skill text.

    The model is deliberately criterion-ID-free.  It spatially grounds frozen
    visual/ROI tokens with a frozen criterion text embedding, applies a
    full-sequence dilated TCN, and emits frame residuals, onset/offset logits,
    and a video-pair presence logit.  The frame residual is initialized to zero,
    so the untrained model exactly preserves the supplied baseline probability.
    """

    def __init__(
        self, token_dim: int, geometry_dim: int = 9, text_dim: int = 4096,
        hidden_dim: int = 64, attention_heads: int = 4,
        temporal_dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        if token_dim <= 0 or geometry_dim <= 6 or text_dim <= 0:
            raise ValueError("Token, text, and geometry dimensions are invalid")
        if hidden_dim % attention_heads or hidden_dim % 8:
            raise ValueError("hidden_dim must be divisible by attention_heads and 8")
        if not temporal_dilations or any(value <= 0 for value in temporal_dilations):
            raise ValueError("temporal_dilations must contain positive values")
        self.hidden_dim = int(hidden_dim)
        self.temporal_dilations = tuple(map(int, temporal_dilations))
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim), nn.GELU(),
        )
        self.geometry_projection = nn.Sequential(
            nn.LayerNorm(geometry_dim), nn.Linear(geometry_dim, hidden_dim), nn.GELU(),
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim), nn.GELU(),
        )
        self.condition = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(),
        )
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=attention_heads,
            dim_feedforward=hidden_dim * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, num_layers=1)
        self.frame_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(),
        )
        # Frame representation + criterion text + baseline probability/logit,
        # criterion threshold, and normalized video progress.
        self.temporal_input = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + 4),
            nn.Linear(hidden_dim * 2 + 4, hidden_dim), nn.GELU(),
        )
        self.temporal_blocks = nn.ModuleList([
            _DilatedTemporalResidualBlock(hidden_dim, dilation, dropout)
            for dilation in self.temporal_dilations
        ])
        self.frame_residual = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.boundary = nn.Conv1d(hidden_dim, 2, kernel_size=1)
        self.presence = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.frame_residual.weight)
        nn.init.zeros_(self.frame_residual.bias)

    @property
    def receptive_field_steps(self) -> int:
        return 1 + 2 * sum(self.temporal_dilations)

    def forward(
        self, tokens: torch.Tensor, geometry: torch.Tensor,
        criterion_embeddings: torch.Tensor, baseline_probabilities: torch.Tensor,
        baseline_thresholds: torch.Tensor, progress: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if tokens.ndim != 4 or geometry.ndim != 4:
            raise ValueError("tokens and geometry must have [batch, time, tokens, features]")
        if tokens.shape[:3] != geometry.shape[:3]:
            raise ValueError("Token and geometry sequences must align")
        batch, time, token_count, _ = tokens.shape
        if criterion_embeddings.ndim != 2 or len(criterion_embeddings) != batch:
            raise ValueError("One criterion embedding is required per sequence")
        if baseline_probabilities.shape != (batch, time):
            raise ValueError("baseline_probabilities must have shape [batch, time]")
        if progress.shape != (batch, time):
            raise ValueError("progress must have shape [batch, time]")
        if baseline_thresholds.ndim == 1:
            if baseline_thresholds.shape != (batch,):
                raise ValueError("baseline_thresholds must have shape [batch] or [batch, time]")
            baseline_thresholds = baseline_thresholds[:, None].expand(-1, time)
        elif baseline_thresholds.shape != (batch, time):
            raise ValueError("baseline_thresholds must have shape [batch] or [batch, time]")
        if valid_mask is None:
            valid_mask = torch.ones(
                (batch, time), dtype=torch.bool, device=tokens.device,
            )
        if valid_mask.shape != (batch, time):
            raise ValueError("valid_mask must have shape [batch, time]")

        visual = self.visual_projection(tokens.float())
        visual = visual + self.geometry_projection(geometry.float())
        text = self.text_projection(criterion_embeddings.float())
        expanded_text = text[:, None, None, :].expand(-1, time, token_count, -1)
        conditioned = self.condition(torch.cat([
            visual, expanded_text, visual * expanded_text,
            (visual - expanded_text).abs(),
        ], dim=-1))
        flat = conditioned.reshape(batch * time, token_count, self.hidden_dim)
        presence = geometry[..., 6].reshape(batch * time, token_count) > 0.5
        # Padded frames can contain no present token.  Keeping one zero-valued
        # placeholder unmasked avoids all-masked attention NaNs; valid_mask
        # removes that frame from every loss and pooled prediction.
        no_token = ~presence.any(dim=1)
        if no_token.any():
            presence = presence.clone()
            presence[no_token, 0] = True
            flat = flat.clone()
            flat[no_token, 0] = 0.0
        encoded = self.spatial_encoder(flat, src_key_padding_mask=~presence)
        flat_text = text[:, None, :].expand(-1, time, -1).reshape(
            batch * time, self.hidden_dim,
        )
        attention_logits = (
            encoded * flat_text[:, None, :]
        ).sum(dim=-1) / math.sqrt(self.hidden_dim)
        attention_logits = attention_logits.masked_fill(~presence, -torch.inf)
        attention = torch.softmax(attention_logits, dim=-1)
        attended = (encoded * attention[..., None]).sum(dim=1)
        divisor = presence.sum(dim=1, keepdim=True).clamp_min(1)
        mean = (encoded * presence[..., None]).sum(dim=1) / divisor
        maximum = encoded.masked_fill(~presence[..., None], -torch.inf).amax(dim=1)
        frame = self.frame_projection(torch.cat([attended, mean, maximum], dim=-1))
        frame = frame.reshape(batch, time, self.hidden_dim)

        probabilities = baseline_probabilities.float().clamp(1e-5, 1.0 - 1e-5)
        base_logits = torch.logit(probabilities)
        scalar_features = torch.stack([
            probabilities, base_logits / 6.0,
            baseline_thresholds.float(), progress.float(),
        ], dim=-1)
        temporal_input = self.temporal_input(torch.cat([
            frame, text[:, None, :].expand(-1, time, -1), scalar_features,
        ], dim=-1))
        temporal_input = temporal_input * valid_mask[..., None]
        hidden = temporal_input.transpose(1, 2)
        for block in self.temporal_blocks:
            hidden = block(hidden)
        hidden = hidden * valid_mask[:, None, :]
        residual = self.frame_residual(hidden).squeeze(1)
        corrected = base_logits + residual
        boundary = self.boundary(hidden).transpose(1, 2)

        valid = valid_mask[:, None, :]
        divisor = valid.sum(dim=-1).clamp_min(1)
        temporal_mean = (hidden * valid).sum(dim=-1) / divisor
        temporal_max = hidden.masked_fill(~valid, -torch.inf).amax(dim=-1)
        empty = ~valid_mask.any(dim=1)
        if empty.any():
            temporal_max = temporal_max.clone()
            temporal_max[empty] = 0.0
        presence_logit = self.presence(torch.cat([
            temporal_mean, temporal_max,
        ], dim=-1)).squeeze(-1)
        return {
            "frame_logits": corrected,
            "frame_residual": residual,
            "boundary_logits": boundary,
            "presence_logits": presence_logit,
        }
