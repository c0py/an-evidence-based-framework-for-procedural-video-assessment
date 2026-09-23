"""Task-conditioned visual evidence adapters for interchangeable SOP criteria."""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


class LowRankResidualLinear(nn.Module):
    """Freeze an existing linear layer and add a trainable low-rank residual."""

    def __init__(
        self, base: nn.Linear, rank: int = 8, alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False, dtype=torch.float32)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.base.requires_grad_(False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        frozen = self.base(value)
        residual = self.lora_b(self.lora_a(self.dropout(value.float())))
        return frozen + residual.to(frozen.dtype) * self.scale


def add_qwen_vision_lora(
    visual: nn.Module, *, last_blocks: int = 6, rank: int = 8,
    alpha: float = 16.0, dropout: float = 0.05,
    module_names: Iterable[str] = ("qkv", "proj", "linear_fc1", "linear_fc2"),
) -> list[str]:
    """Attach LoRA only to the final Qwen vision blocks.

    The adapter is criterion-agnostic: criterion semantics enter through the
    shared state head rather than through criterion-specific output weights.
    """
    blocks = visual.blocks
    start = max(0, len(blocks) - int(last_blocks))
    selected = set(module_names)
    replaced: list[str] = []
    for block_index in range(start, len(blocks)):
        block = blocks[block_index]
        parents = (("attn", block.attn), ("mlp", block.mlp))
        for parent_name, parent in parents:
            for name, child in list(parent.named_children()):
                if name not in selected or not isinstance(child, nn.Linear):
                    continue
                setattr(
                    parent, name,
                    LowRankResidualLinear(child, rank=rank, alpha=alpha, dropout=dropout),
                )
                replaced.append(f"blocks.{block_index}.{parent_name}.{name}")
    if not replaced:
        raise RuntimeError("No Qwen vision linear layers were selected for LoRA")
    return replaced


def trainable_adapter_state(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return only low-rank parameters, excluding the frozen foundation model."""
    return {
        name: value.detach().cpu()
        for name, value in module.state_dict().items()
        if "lora_a." in name or "lora_b." in name
    }


class CriterionConditionedVisualStateHead(nn.Module):
    """Predict absent/partial/full from visual tokens and arbitrary criterion text.

    There are no criterion IDs or criterion-specific classifier branches. A new
    task can supply a different text embedding while keeping this contract.
    """

    def __init__(
        self, visual_dim: int = 4096, text_dim: int = 4096,
        hidden_dim: int = 384, dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden_dim), nn.GELU(),
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim), nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self, visual_tokens: list[torch.Tensor] | tuple[torch.Tensor, ...],
        criterion_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if len(visual_tokens) != len(criterion_embeddings):
            raise ValueError("One criterion embedding is required per image")
        rows = []
        for tokens, criterion in zip(visual_tokens, criterion_embeddings):
            visual = self.visual_projection(tokens.float())
            query = self.text_projection(criterion.float())
            attention = torch.softmax(
                visual @ query / math.sqrt(float(visual.shape[-1])), dim=0,
            )
            attended = (attention[:, None] * visual).sum(0)
            global_mean = visual.mean(0)
            rows.append(torch.cat([
                attended, global_mean, query, attended * query,
                (attended - query).abs(),
            ]))
        return self.classifier(torch.stack(rows))


class MultiFrameCriterionVerifier(nn.Module):
    """Shared text-conditioned verifier over time-aligned raw/overlay/ROI panels."""

    def __init__(
        self, visual_dim: int = 4096, text_dim: int = 4096,
        hidden_dim: int = 256, max_frames: int = 17, dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden_dim), nn.GELU(),
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim), nn.GELU(),
        )
        self.frame_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5), nn.Linear(hidden_dim * 5, hidden_dim), nn.GELU(),
        )
        self.position = nn.Parameter(torch.zeros(max_frames, hidden_dim))
        nn.init.normal_(self.position, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 3,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5), nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 3),
        )

    def encode(
        self, panel_features: torch.Tensor, criterion_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if panel_features.ndim != 4:
            raise ValueError("panel_features must have shape [batch, time, panels, visual_dim]")
        batch, frames, panels, _ = panel_features.shape
        if panels < 2:
            raise ValueError("At least two complementary visual panels are required")
        if frames > self.position.shape[0]:
            raise ValueError("Input sequence exceeds max_frames")
        if criterion_embeddings.shape[0] != batch:
            raise ValueError("One criterion embedding is required per sequence")
        visual = self.visual_projection(panel_features.float())
        query = self.text_projection(criterion_embeddings.float())
        panel_attention = torch.softmax(
            (visual * query[:, None, None]).sum(-1) / math.sqrt(float(visual.shape[-1])),
            dim=2,
        )
        attended_panel = (panel_attention[..., None] * visual).sum(2)
        panel_mean = visual.mean(2)
        repeated_query = query[:, None].expand(-1, frames, -1)
        frame_rows = torch.cat([
            attended_panel, panel_mean, repeated_query,
            attended_panel * repeated_query,
            (attended_panel - repeated_query).abs(),
        ], dim=-1)
        temporal = self.temporal(
            self.frame_fusion(frame_rows) + self.position[:frames][None],
        )
        time_attention = torch.softmax(
            (temporal * query[:, None]).sum(-1) / math.sqrt(float(temporal.shape[-1])),
            dim=1,
        )
        attended_time = (time_attention[..., None] * temporal).sum(1)
        global_time = temporal.mean(1)
        return torch.cat([
            attended_time, global_time, query, attended_time * query,
            (attended_time - query).abs(),
        ], dim=-1)

    def forward(
        self, panel_features: torch.Tensor, criterion_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(self.encode(panel_features, criterion_embeddings))


class MultiFrameCriterionResidualFusion(nn.Module):
    """Correct a temporal small-model logit using text-conditioned visual evidence.

    With the zero-initialized residual head this module exactly preserves the
    small-model probability.  It can therefore learn only evidence-supported
    corrections without discarding the stronger temporal prior.
    """

    def __init__(
        self, visual_dim: int = 4096, text_dim: int = 4096,
        hidden_dim: int = 256, max_frames: int = 7, dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.max_frames = int(max_frames)
        self.evidence_encoder = MultiFrameCriterionVerifier(
            visual_dim=visual_dim, text_dim=text_dim, hidden_dim=hidden_dim,
            max_frames=max_frames, dropout=dropout,
        )
        self.score_projection = nn.Sequential(
            nn.LayerNorm(max_frames), nn.Linear(max_frames, hidden_dim), nn.GELU(),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 6),
            nn.Linear(hidden_dim * 6, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self, panel_features: torch.Tensor, criterion_embeddings: torch.Tensor,
        base_score_sequence: torch.Tensor, base_center_probability: torch.Tensor,
        base_threshold: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if base_score_sequence.ndim != 2:
            raise ValueError("base_score_sequence must have shape [batch, time]")
        if base_score_sequence.shape[1] != self.max_frames:
            raise ValueError("base_score_sequence length must equal max_frames")
        evidence = self.evidence_encoder.encode(panel_features, criterion_embeddings)
        scores = base_score_sequence.float().clamp(1e-6, 1.0 - 1e-6)
        threshold = base_threshold.float().clamp(1e-6, 1.0 - 1e-6)
        score_logits = torch.logit(scores) - torch.logit(threshold)[:, None]
        score_context = self.score_projection(score_logits.clamp(-12.0, 12.0))
        residual = self.residual_head(torch.cat([evidence, score_context], dim=-1)).squeeze(-1)
        base_logit = torch.logit(
            base_center_probability.float().clamp(1e-6, 1.0 - 1e-6)
        )
        return base_logit + residual, residual


class TextConditionedMaskTrackVerifier(nn.Module):
    """Shared criterion-text verifier over structured mask-track evidence.

    This is a replaceable specialized evidence tool.  The foundation-model text
    embedding is frozen and the module has no criterion-ID-specific branch.
    """

    def __init__(
        self, mask_feature_dim: int, text_dim: int = 4096,
        hidden_dim: int = 128, dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if mask_feature_dim <= 0:
            raise ValueError("mask_feature_dim must be positive")
        self.mask_projection = nn.Sequential(
            nn.LayerNorm(mask_feature_dim), nn.Linear(mask_feature_dim, hidden_dim), nn.GELU(),
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim), nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, mask_track_features: torch.Tensor, criterion_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if mask_track_features.ndim != 2:
            raise ValueError("mask_track_features must have shape [batch, features]")
        if criterion_embeddings.ndim != 2:
            raise ValueError("criterion_embeddings must have shape [batch, text_dim]")
        if len(mask_track_features) != len(criterion_embeddings):
            raise ValueError("One criterion embedding is required per mask-track row")
        visual = self.mask_projection(mask_track_features.float())
        text = self.text_projection(criterion_embeddings.float())
        return self.classifier(torch.cat([
            visual, text, visual * text, (visual - text).abs(),
        ], dim=-1)).squeeze(-1)
