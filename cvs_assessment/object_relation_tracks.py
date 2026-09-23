"""Text-conditioned small models over structured object-relation tracks."""
from __future__ import annotations

import torch
from torch import nn


def criterion_relation_feature_mask(
    feature_names: list[str], relevant_semantic_types: list[str],
) -> torch.Tensor:
    """Build a Skill-driven mask retaining only relevant tracks and relation pairs."""
    relevant = set(relevant_semantic_types)
    values = []
    for name in feature_names:
        if name.startswith("track_"):
            semantic_type = name[len("track_"):].split("_adjacent_")[0]
            if "_adjacent_" not in name:
                semantic_type = name[len("track_"):].rsplit("_", 3)[0]
                semantic_type = next((item for item in relevant if name.startswith(f"track_{item}_")), semantic_type)
            keep = semantic_type in relevant or any(
                name.startswith(f"track_{item}_") for item in relevant
            )
        elif name.startswith("relation_") and "__" in name:
            pair = name[len("relation_"):].split("_copresence_ratio")[0]
            pair = pair.split("_left_mask_contact")[0]
            pair = pair.split("_mean_bbox_iou")[0]
            pair = pair.split("_mean_center_distance")[0]
            pair = pair.split("_right_mask_contact")[0]
            left, right = pair.split("__", 1)
            keep = left in relevant and right in relevant
        else:
            keep = False
        values.append(float(keep))
    return torch.tensor(values, dtype=torch.float32)


class TextConditionedObjectRelationTrackVerifier(nn.Module):
    """Shared temporal verifier over predicted masks, objects, and relations."""

    def __init__(
        self, relation_feature_dim: int, text_dim: int = 4096,
        hidden_dim: int = 64, kernel_size: int = 5, dropout: float = 0.12,
    ) -> None:
        super().__init__()
        if relation_feature_dim <= 0:
            raise ValueError("relation_feature_dim must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        padding = kernel_size // 2
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(relation_feature_dim),
            nn.Linear(relation_feature_dim, hidden_dim), nn.GELU(),
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim), nn.GELU(),
        )
        self.condition = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.GELU(), nn.GroupNorm(8, hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.GELU(), nn.GroupNorm(8, hidden_dim), nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, relation_windows: torch.Tensor, criterion_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if relation_windows.ndim != 3:
            raise ValueError("relation_windows must have shape [batch, time, features]")
        if criterion_embeddings.ndim != 2:
            raise ValueError("criterion_embeddings must have shape [batch, text_dim]")
        if len(relation_windows) != len(criterion_embeddings):
            raise ValueError("One criterion embedding is required per relation window")
        visual = self.visual_projection(relation_windows.float())
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
