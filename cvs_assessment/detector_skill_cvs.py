"""Generic small heads over frozen detector and frozen Skill representations."""
from __future__ import annotations

import torch
from torch import nn


class DetectorMultiLabelProbe(nn.Module):
    """Criterion-count-configurable detector feature probe."""

    def __init__(self, input_dim: int, criterion_count: int, hidden_dim: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, criterion_count),
        )

    def forward(self, features: torch.Tensor, criterion_embeddings: torch.Tensor | None = None) -> torch.Tensor:
        return self.network(features.float())


class GroupedFeatureFusionProbe(nn.Module):
    """Fuse heterogeneous frozen visual feature groups after group-wise normalization."""

    def __init__(
        self,
        group_dims: tuple[int, ...],
        criterion_count: int,
        branch_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if not group_dims or any(int(value) <= 0 for value in group_dims):
            raise ValueError("group_dims must contain positive dimensions")
        self.group_dims = tuple(map(int, group_dims))
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, branch_dim), nn.GELU(), nn.Dropout(dropout)
            )
            for dim in self.group_dims
        ])
        fused_dim = len(self.group_dims) * branch_dim
        self.output = nn.Sequential(
            nn.LayerNorm(fused_dim), nn.Linear(fused_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, criterion_count),
        )

    def encode_visual(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != sum(self.group_dims):
            raise ValueError("Feature dimension does not match group_dims")
        groups = torch.split(features.float(), self.group_dims, dim=-1)
        return torch.cat([branch(values) for branch, values in zip(self.branches, groups)], dim=-1)

    def forward(self, features: torch.Tensor, criterion_embeddings: torch.Tensor | None = None) -> torch.Tensor:
        return self.output(self.encode_visual(features))


class SkillConditionedGroupedFeatureHead(nn.Module):
    """Shared Skill-conditioned decision head over independently normalized visual groups."""

    def __init__(
        self,
        group_dims: tuple[int, ...],
        text_dim: int = 4096,
        branch_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if not group_dims or any(int(value) <= 0 for value in group_dims):
            raise ValueError("group_dims must contain positive dimensions")
        self.group_dims = tuple(map(int, group_dims))
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, branch_dim), nn.GELU(), nn.Dropout(dropout)
            )
            for dim in self.group_dims
        ])
        fused_dim = len(self.group_dims) * branch_dim
        self.visual = nn.Sequential(nn.LayerNorm(fused_dim), nn.Linear(fused_dim, hidden_dim), nn.GELU())
        self.text = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim), nn.GELU())
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4), nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, criterion_embeddings: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or criterion_embeddings.ndim != 2:
            raise ValueError("features and criterion_embeddings must both be matrices")
        if features.shape[-1] != sum(self.group_dims):
            raise ValueError("Feature dimension does not match group_dims")
        groups = torch.split(features.float(), self.group_dims, dim=-1)
        visual = self.visual(torch.cat([
            branch(values) for branch, values in zip(self.branches, groups)
        ], dim=-1))[:, None, :]
        text = self.text(criterion_embeddings.float())[None, :, :]
        visual = visual.expand(-1, text.shape[1], -1)
        text = text.expand(visual.shape[0], -1, -1)
        return self.shared(torch.cat([
            visual, text, visual * text, (visual - text).abs()
        ], dim=-1)).squeeze(-1)


class SpatialTokenFusionProbe(nn.Module):
    """Attend over compact dual-view spatial tokens and optional frozen context."""

    def __init__(
        self,
        context_dim: int,
        criterion_count: int,
        token_dim: int = 384,
        token_count: int = 34,
        hidden_dim: int = 128,
        attention_layers: int = 1,
        attention_heads: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.token_dim = int(token_dim)
        self.token_count = int(token_count)
        self.token_norm = nn.LayerNorm(token_dim)
        self.token_projection = nn.Linear(token_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, token_count, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=attention_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.attention = nn.TransformerEncoder(layer, num_layers=attention_layers, enable_nested_tensor=False)
        self.context = None if context_dim == 0 else nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        summary_dim = hidden_dim * (2 if context_dim == 0 else 3)
        self.visual_output = nn.Sequential(
            nn.LayerNorm(summary_dim), nn.Linear(summary_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.output = nn.Linear(hidden_dim, criterion_count)
        nn.init.normal_(self.position, std=.02)

    def encode_visual(self, features: torch.Tensor) -> torch.Tensor:
        expected = self.context_dim + self.token_count * self.token_dim
        if features.shape[-1] != expected:
            raise ValueError(f"Expected {expected} features, received {features.shape[-1]}")
        context = features[..., :self.context_dim] if self.context_dim else None
        tokens = features[..., self.context_dim:].float().reshape(-1, self.token_count, self.token_dim)
        tokens = self.attention(self.token_projection(self.token_norm(tokens)) + self.position)
        # The two DINO CLS tokens are at positions 0 and 17; global mean preserves all local patches.
        summary = [tokens[:, (0, 17)].mean(dim=1), tokens.mean(dim=1)]
        if context is not None:
            summary.append(self.context(context.float()))
        return self.visual_output(torch.cat(summary, dim=-1))

    def forward(self, features: torch.Tensor, criterion_embeddings: torch.Tensor | None = None) -> torch.Tensor:
        return self.output(self.encode_visual(features))


class SkillConditionedSpatialTokenHead(SpatialTokenFusionProbe):
    """Skill-conditioned shared output over compact dual-view spatial attention."""

    def __init__(self, context_dim: int, text_dim: int = 4096, **kwargs) -> None:
        super().__init__(context_dim=context_dim, criterion_count=1, **kwargs)
        hidden_dim = int(kwargs.get("hidden_dim", 128)); dropout = float(kwargs.get("dropout", .15))
        self.output = nn.Identity()
        self.text = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim), nn.GELU())
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4), nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, criterion_embeddings: torch.Tensor) -> torch.Tensor:
        visual = self.encode_visual(features)[:, None, :]
        text = self.text(criterion_embeddings.float())[None, :, :]
        visual = visual.expand(-1, text.shape[1], -1); text = text.expand(visual.shape[0], -1, -1)
        return self.shared(torch.cat([visual, text, visual * text, (visual - text).abs()], dim=-1)).squeeze(-1)


class PretrainedDinoLastBlockAdapter(nn.Module):
    """Adapt a pretrained DINO final block over compact pre-last dual-view tokens."""

    def __init__(
        self,
        pretrained_block: nn.Module,
        pretrained_norm: nn.Module,
        context_dim: int,
        criterion_count: int,
        hidden_dim: int = 256,
        dropout: float = .15,
        block_trainable: bool = True,
    ) -> None:
        super().__init__()
        self.block = pretrained_block
        self.norm = pretrained_norm
        self.context_dim = int(context_dim)
        self.block_trainable = bool(block_trainable)
        for parameter in list(self.block.parameters()) + list(self.norm.parameters()):
            parameter.requires_grad_(self.block_trainable)
        self.context = None if context_dim == 0 else nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        summary_dim = 384 * 4 + (hidden_dim if context_dim else 0)
        self.visual = nn.Sequential(
            nn.LayerNorm(summary_dim), nn.Linear(summary_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.output = nn.Linear(hidden_dim, criterion_count)

    def backbone_parameters(self):
        return list(self.block.parameters()) + list(self.norm.parameters())

    def head_parameters(self):
        backbone_ids = {id(value) for value in self.backbone_parameters()}
        return [value for value in self.parameters() if id(value) not in backbone_ids]

    def encode_visual(self, features: torch.Tensor) -> torch.Tensor:
        expected = self.context_dim + 34 * 384
        if features.shape[-1] != expected:
            raise ValueError(f"Expected {expected} features, received {features.shape[-1]}")
        context = features[..., :self.context_dim] if self.context_dim else None
        tokens = features[..., self.context_dim:].float().reshape(-1, 34, 384)
        center = self.norm(self.block(tokens[:, :17]))
        full = self.norm(self.block(tokens[:, 17:]))
        summary = [center[:, 0], center.mean(dim=1), full[:, 0], full.mean(dim=1)]
        if context is not None:
            summary.append(self.context(context.float()))
        return self.visual(torch.cat(summary, dim=-1))

    def forward(self, features: torch.Tensor, criterion_embeddings: torch.Tensor | None = None) -> torch.Tensor:
        return self.output(self.encode_visual(features))


class SkillConditionedDinoLastBlockAdapter(PretrainedDinoLastBlockAdapter):
    """Shared Skill-conditioned output on the adapted pretrained DINO final block."""

    def __init__(self, pretrained_block: nn.Module, pretrained_norm: nn.Module, context_dim: int, text_dim: int = 4096, **kwargs) -> None:
        super().__init__(pretrained_block, pretrained_norm, context_dim, criterion_count=1, **kwargs)
        hidden_dim = int(kwargs.get("hidden_dim", 256)); dropout = float(kwargs.get("dropout", .15))
        self.output = nn.Identity()
        self.text = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim), nn.GELU())
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4), nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, criterion_embeddings: torch.Tensor) -> torch.Tensor:
        visual = self.encode_visual(features)[:, None, :]
        text = self.text(criterion_embeddings.float())[None, :, :]
        visual = visual.expand(-1, text.shape[1], -1); text = text.expand(visual.shape[0], -1, -1)
        return self.shared(torch.cat([visual, text, visual * text, (visual - text).abs()], dim=-1)).squeeze(-1)


class SkillConditionedDetectorHead(nn.Module):
    """One shared decision function conditioned by replaceable Skill embeddings."""

    def __init__(self, input_dim: int, text_dim: int = 4096, hidden_dim: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        self.visual = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.text = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim), nn.GELU())
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4), nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, criterion_embeddings: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or criterion_embeddings.ndim != 2:
            raise ValueError("features and criterion_embeddings must both be matrices")
        visual = self.visual(features.float())[:, None, :]
        text = self.text(criterion_embeddings.float())[None, :, :]
        visual = visual.expand(-1, text.shape[1], -1); text = text.expand(visual.shape[0], -1, -1)
        return self.shared(torch.cat([visual, text, visual * text, (visual - text).abs()], dim=-1)).squeeze(-1)


class _SharedTemporalResidualBlock(nn.Module):
    """Criterion-agnostic residual block used after Skill/visual interaction."""

    def __init__(self, hidden_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=dilation, dilation=dilation),
            nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, 1), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm((values + self.network(values)).transpose(1, 2)).transpose(1, 2)


class SkillConditionedTemporalHead(nn.Module):
    """Small shared temporal head over frozen visual features and frozen Skills.

    The temporal filters and output function are shared by every criterion. New
    criteria can therefore be supplied through embeddings without introducing a
    criterion-specific classifier. Inputs are ``[video, time, feature]`` and the
    returned logits are ``[video, time, criterion]``.
    """

    def __init__(
        self,
        input_dim: int,
        text_dim: int = 4096,
        hidden_dim: int = 128,
        dropout: float = 0.15,
        dilations: tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__()
        self.visual = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.text = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim), nn.GELU())
        self.interaction = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4), nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.temporal = nn.ModuleList(
            [_SharedTemporalResidualBlock(hidden_dim, int(dilation), dropout) for dilation in dilations]
        )
        self.output = nn.Linear(hidden_dim, 1)
        self.dilations = tuple(map(int, dilations))

    def forward(self, features: torch.Tensor, criterion_embeddings: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or criterion_embeddings.ndim != 2:
            raise ValueError("features must be [video,time,feature] and criterion_embeddings a matrix")
        visual = self.visual(features.float())[:, :, None, :]
        text = self.text(criterion_embeddings.float())[None, None, :, :]
        visual = visual.expand(-1, -1, text.shape[2], -1)
        text = text.expand(visual.shape[0], visual.shape[1], -1, -1)
        values = self.interaction(torch.cat([visual, text, visual * text, (visual - text).abs()], dim=-1))
        videos, steps, criteria, hidden = values.shape
        values = values.permute(0, 2, 3, 1).reshape(videos * criteria, hidden, steps)
        for block in self.temporal:
            values = block(values)
        values = values.reshape(videos, criteria, hidden, steps).permute(0, 3, 1, 2)
        return self.output(values).squeeze(-1)
