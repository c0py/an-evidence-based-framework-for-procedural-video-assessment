"""Controlled upper-stage adaptation of the official CVS-AdaptNet visual tower."""
from __future__ import annotations

import torch
from torch import nn


def _keep_batch_norm_statistics_frozen(module: nn.Module) -> None:
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()


class CVSAdaptLayer4Fusion(nn.Module):
    """Fuse dual-view layer4 embeddings with frozen generic visual context."""

    def __init__(
        self,
        layer4: nn.Module,
        projection: nn.Module,
        context_dim: int = 9234,
        hidden_dim: int = 512,
        dropout: float = .20,
        criterion_count: int = 3,
        layer4_trainable: bool = True,
        projection_trainable: bool = True,
    ) -> None:
        super().__init__()
        self.layer4 = layer4
        self.projection = projection
        self.context_dim = int(context_dim)
        self.layer4_trainable = bool(layer4_trainable)
        self.projection_trainable = bool(projection_trainable)
        self.layer4.requires_grad_(self.layer4_trainable)
        self.projection.requires_grad_(self.projection_trainable)
        visual_dim = self.context_dim + 768 * 2
        self.visual_head = nn.Sequential(
            nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(hidden_dim, criterion_count)
        self.layer4.apply(_keep_batch_norm_statistics_frozen)

    def train(self, mode: bool = True):
        super().train(mode)
        # SAGES batches are not allowed to rewrite pretrained BatchNorm running statistics.
        self.layer4.apply(_keep_batch_norm_statistics_frozen)
        return self

    def upper_visual_parameters(self) -> list[nn.Parameter]:
        return [
            parameter for parameter in list(self.layer4.parameters()) + list(self.projection.parameters())
            if parameter.requires_grad
        ]

    def head_parameters(self) -> list[nn.Parameter]:
        upper_ids = {id(parameter) for parameter in self.upper_visual_parameters()}
        return [
            parameter for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in upper_ids
        ]

    def encode_visual(self, layer3: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if layer3.ndim != 5 or tuple(layer3.shape[1:]) != (2, 1024, 19, 19):
            raise ValueError("layer3 must be [batch,2,1024,19,19]")
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError("Frozen context shape mismatch")
        batch = layer3.shape[0]
        values = self.layer4(layer3.float().reshape(batch * 2, 1024, 19, 19))
        values = torch.nn.functional.adaptive_avg_pool2d(values, 1).flatten(1)
        values = self.projection(values).reshape(batch, 2, 768)
        return self.visual_head(torch.cat([values[:, 0], values[:, 1], context.float()], dim=-1))

    def forward(
        self,
        layer3: torch.Tensor,
        context: torch.Tensor,
        criterion_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.output(self.encode_visual(layer3, context))


class SkillConditionedCVSAdaptLayer4Fusion(CVSAdaptLayer4Fusion):
    """Shared frozen-Skill output over the controlled adapted visual representation."""

    def __init__(self, layer4: nn.Module, projection: nn.Module, text_dim: int = 4096, **kwargs) -> None:
        super().__init__(layer4, projection, criterion_count=1, **kwargs)
        hidden_dim = int(kwargs.get("hidden_dim", 512)); dropout = float(kwargs.get("dropout", .20))
        self.output = nn.Identity()
        self.text = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim), nn.GELU())
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4), nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, layer3: torch.Tensor, context: torch.Tensor, criterion_embeddings: torch.Tensor) -> torch.Tensor:
        visual = self.encode_visual(layer3, context)[:, None, :]
        text = self.text(criterion_embeddings.float())[None, :, :]
        visual = visual.expand(-1, text.shape[1], -1); text = text.expand(visual.shape[0], -1, -1)
        return self.shared(torch.cat([visual, text, visual * text, (visual - text).abs()], dim=-1)).squeeze(-1)
