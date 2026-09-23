"""Small criterion-ID-free fusion heads over frozen visual and Skill features."""
from __future__ import annotations

import math

import torch
from torch import nn


class TextConditionedSpatialFrameHead(nn.Module):
    """Fuse frozen PeskaVLP, YOLO token, geometry, and Skill text features."""

    def __init__(
        self, peska_dim: int = 768, token_dim: int = 1280,
        geometry_dim: int = 9, text_dim: int = 4096,
        hidden_dim: int = 128, attention_heads: int = 4, dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.hidden_dim = hidden_dim
        self.peska = nn.Sequential(nn.LayerNorm(peska_dim), nn.Linear(peska_dim, hidden_dim), nn.GELU())
        self.visual = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim), nn.GELU())
        self.geometry = nn.Sequential(nn.LayerNorm(geometry_dim), nn.Linear(geometry_dim, hidden_dim), nn.GELU())
        self.text = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim), nn.GELU())
        self.condition = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4), nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            hidden_dim, attention_heads, hidden_dim * 2, dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.spatial = nn.TransformerEncoder(layer, num_layers=1)
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4), nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self, peska_features: torch.Tensor, tokens: torch.Tensor,
        geometry: torch.Tensor, criterion_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if tokens.ndim != 3 or geometry.ndim != 3 or tokens.shape[:2] != geometry.shape[:2]:
            raise ValueError("tokens and geometry must align as [batch, tokens, features]")
        visual = self.visual(tokens.float()) + self.geometry(geometry.float())
        text = self.text(criterion_embeddings.float())
        expanded = text[:, None, :].expand(-1, tokens.shape[1], -1)
        conditioned = self.condition(torch.cat([
            visual, expanded, visual * expanded, (visual - expanded).abs(),
        ], dim=-1))
        present = geometry[..., 6] > 0.5
        no_token = ~present.any(dim=1)
        if no_token.any():
            present = present.clone(); conditioned = conditioned.clone()
            present[no_token, 0] = True; conditioned[no_token, 0] = 0
        encoded = self.spatial(conditioned, src_key_padding_mask=~present)
        attention_logits = (encoded * text[:, None, :]).sum(-1) / math.sqrt(self.hidden_dim)
        attention_logits = attention_logits.masked_fill(~present, -torch.inf)
        attention = torch.softmax(attention_logits, dim=-1)
        attended = (encoded * attention[..., None]).sum(1)
        mean = (encoded * present[..., None]).sum(1) / present.sum(1, keepdim=True).clamp_min(1)
        peska = self.peska(peska_features.float())
        hidden = self.fusion(torch.cat([
            attended, mean, peska, attended * peska,
        ], dim=-1))
        return {"logit": self.classifier(hidden).squeeze(-1), "hidden": hidden}


class _TemporalBlock(nn.Module):
    def __init__(self, hidden_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, 1), nn.GroupNorm(8, hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.network(value)


class SharedTemporalCvsHead(nn.Module):
    """Shared temporal correction over text-conditioned frozen-frame evidence."""

    def __init__(
        self, frame_dim: int = 128, text_dim: int = 4096,
        hidden_dim: int = 128, dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.text = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim), nn.GELU())
        self.input = nn.Sequential(
            nn.LayerNorm(frame_dim + hidden_dim + 3),
            nn.Linear(frame_dim + hidden_dim + 3, hidden_dim), nn.GELU(),
        )
        self.blocks = nn.ModuleList([_TemporalBlock(hidden_dim, value, dropout) for value in dilations])
        self.residual = nn.Conv1d(hidden_dim, 1, 1)
        nn.init.zeros_(self.residual.weight); nn.init.zeros_(self.residual.bias)

    def forward(
        self, frame_hidden: torch.Tensor, frame_logits: torch.Tensor,
        criterion_embeddings: torch.Tensor, progress: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, time, _ = frame_hidden.shape
        if valid_mask is None:
            valid_mask = torch.ones(batch, time, dtype=torch.bool, device=frame_hidden.device)
        text = self.text(criterion_embeddings.float())[:, None, :].expand(-1, time, -1)
        probability = torch.sigmoid(frame_logits.float())
        scalar = torch.stack([probability, frame_logits.float() / 6.0, progress.float()], dim=-1)
        hidden = self.input(torch.cat([frame_hidden.float(), text, scalar], dim=-1))
        hidden = (hidden * valid_mask[..., None]).transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden)
        residual = self.residual(hidden).squeeze(1) * valid_mask
        return {"logit": frame_logits.float() + residual, "residual": residual}
