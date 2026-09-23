from __future__ import annotations

import torch
from torch import nn
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torchvision.models import ResNet18_Weights, resnet18, resnet50


class SharedFrameEncoder(nn.Module):
    """Small replaceable visual backbone for frame-level CVS experiments."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.output_dim = 512

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images).flatten(1)


class CriterionConditionedHead(nn.Module):
    """Queries the shared representation with one learned query per CVS criterion."""

    def __init__(self, feature_dim: int = 512, criteria: tuple[str, ...] = ("two_structures", "cystic_plate", "hepatocystic_triangle")) -> None:
        super().__init__()
        self.criteria = criteria
        self.query = nn.Parameter(torch.randn(len(criteria), feature_dim) * 0.02)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.classifiers = nn.ModuleList([nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 1)) for _ in criteria])

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        keys, values = self.key(features), self.value(features)
        # A single frame is treated as a one-token visual region; a region-token backbone can replace it unchanged.
        weights = torch.softmax(keys[:, None, :] * self.query[None, :, :], dim=-1)
        grounded = weights * values[:, None, :]
        return {name: self.classifiers[i](grounded[:, i]).squeeze(-1) for i, name in enumerate(self.criteria)}


class UnifiedThreeLabelHead(nn.Module):
    """Ablation baseline: same encoder, no criterion-specific query grounding."""

    def __init__(self, feature_dim: int = 512, num_labels: int = 3) -> None:
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class FeatureFusionCalibrator(nn.Module):
    """Small task-neutral head for fusing frozen evidence-tool features."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class CriterionConditionedCvsModel(nn.Module):
    def __init__(self, pretrained_backbone: bool = False) -> None:
        super().__init__()
        self.encoder = SharedFrameEncoder(pretrained_backbone)
        self.head = CriterionConditionedHead(self.encoder.output_dim)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.encoder(images))


class PeskaVLPVisualEncoder(nn.Module):
    """PeskaVLP's ResNet-50 image tower and 768-D visual projection.

    This intentionally reimplements only the official image tower.  It avoids
    importing PeskaVLP's old training stack (mmengine/transformers pins) while
    retaining parameter-compatible module names and numerical operations.
    """

    output_dim = 768

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet50(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Linear(2048, self.output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(images))

    def freeze(self) -> "PeskaVLPVisualEncoder":
        self.requires_grad_(False)
        self.eval()
        return self

    def train(self, mode: bool = True) -> "PeskaVLPVisualEncoder":
        # A frozen pretrained tool must keep BatchNorm statistics fixed even
        # while its downstream head is being trained.
        super().train(False)
        return self

    def load_official_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = _find_tensor_state_dict(payload)
        mapped: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            key = key.removeprefix("module.")
            if key.startswith("backbone_img.model."):
                mapped["backbone." + key.removeprefix("backbone_img.model.")] = value
            elif key.startswith("backbone_img.global_embedder."):
                mapped["projection." + key.removeprefix("backbone_img.global_embedder.")] = value
        if not mapped:
            raise ValueError("Checkpoint contains no PeskaVLP backbone_img parameters")
        incompatible = self.load_state_dict(mapped, strict=False)
        missing = [key for key in incompatible.missing_keys if not key.startswith("backbone.fc.")]
        unexpected = list(incompatible.unexpected_keys)
        if missing or unexpected:
            raise ValueError(f"Incomplete PeskaVLP image tower: missing={missing}, unexpected={unexpected}")
        return {
            "source_parameter_count": len(state),
            "loaded_visual_parameter_count": len(mapped),
        }


def _find_tensor_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    """Find the model state in the common raw/mmengine checkpoint layouts."""
    if isinstance(payload, Mapping):
        if payload and all(isinstance(key, str) and torch.is_tensor(value) for key, value in payload.items()):
            return payload
        for key in ("state_dict", "model_state_dict", "model", "model_state"):
            if key in payload:
                try:
                    return _find_tensor_state_dict(payload[key])
                except (TypeError, ValueError):
                    pass
    raise ValueError("Could not locate a tensor state_dict in the PeskaVLP checkpoint")


class PeskaVLPCvsModel(nn.Module):
    """Frozen PeskaVLP image tool plus a trainable criterion-conditioned head."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = PeskaVLPVisualEncoder()
        self.head = CriterionConditionedHead(feature_dim=self.encoder.output_dim)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            features = self.encoder(images)
        return self.head(features)

    def forward_features(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(features)


class OrdinalCriterionHead(nn.Module):
    """Criterion-conditioned ordinal head for partial/full CVS evidence.

    Each criterion emits two logits: evidence is at least partial, and evidence
    is fully satisfied. A training-time consistency loss enforces that full
    evidence cannot be more probable than partial-or-full evidence.
    """

    def __init__(
        self, feature_dim: int = 768,
        criteria: tuple[str, ...] = ("two_structures", "cystic_plate", "hepatocystic_triangle"),
        monotonic: bool = False,
    ) -> None:
        super().__init__()
        self.criteria = criteria
        self.monotonic = monotonic
        self.query = nn.Parameter(torch.randn(len(criteria), feature_dim) * 0.02)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.classifiers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 2)) for _ in criteria
        ])

    def forward(self, features: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        keys, values = self.key(features), self.value(features)
        weights = torch.softmax(keys[:, None, :] * self.query[None, :, :], dim=-1)
        grounded = weights * values[:, None, :]
        output = {}
        for index, criterion in enumerate(self.criteria):
            logits = self.classifiers[index](grounded[:, index])
            support_logit = logits[:, 0]
            full_logit = logits[:, 1]
            if self.monotonic:
                # Interpret the second output as P(full | support).  The
                # resulting full probability is guaranteed to be no greater
                # than P(support), while the state_dict remains compatible
                # with historical independent-logit checkpoints.
                full_probability = torch.sigmoid(support_logit) * torch.sigmoid(full_logit)
                full_logit = torch.logit(full_probability.clamp(1e-6, 1.0 - 1e-6))
            output[criterion] = {
                "support_or_full": support_logit,
                "full_only": full_logit,
            }
        return output


class PeskaVLPOrdinalCvsModel(nn.Module):
    """Frozen PeskaVLP image encoder plus the ordinal CVS head."""

    def __init__(self, monotonic: bool = False) -> None:
        super().__init__()
        self.encoder = PeskaVLPVisualEncoder()
        self.head = OrdinalCriterionHead(
            feature_dim=self.encoder.output_dim, monotonic=monotonic,
        )

    def forward(self, images: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        with torch.no_grad():
            features = self.encoder(images)
        return self.head(features)

    def forward_features(self, features: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        return self.head(features)


class TemporalResidualBlock(nn.Module):
    """A compact dilated residual block for frozen surgical-video features."""

    def __init__(self, hidden_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                hidden_dim, hidden_dim, kernel_size=3,
                padding=dilation, dilation=dilation,
            ),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class TemporalOrdinalBoundaryHead(nn.Module):
    """Temporal correction head over frozen PeskaVLP and frame-head outputs.

    The frame logits provide a strong initialization. The TCN learns residual
    ordinal corrections and an auxiliary onset/offset task for each CVS
    criterion. Input and output sequences use ``[batch, time, ...]`` layout.
    """

    def __init__(
        self, feature_dim: int = 768, hidden_dim: int = 128,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.15,
        criteria: tuple[str, ...] = (
            "two_structures", "cystic_plate", "hepatocystic_triangle",
        ),
    ) -> None:
        super().__init__()
        if hidden_dim % 8:
            raise ValueError("hidden_dim must be divisible by 8 for GroupNorm")
        if not criteria:
            raise ValueError("TemporalOrdinalBoundaryHead requires at least one criterion")
        self.criteria = tuple(criteria)
        self.num_criteria = len(criteria)
        # Visual features + two ordinal logits per criterion + normalized progress.
        self.input_dim = feature_dim + 2 * self.num_criteria + 1
        self.hidden_dim = hidden_dim
        self.dilations = dilations
        self.input_projection = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            TemporalResidualBlock(hidden_dim, dilation, dropout)
            for dilation in dilations
        ])
        self.ordinal_residual = nn.Conv1d(
            hidden_dim, 2 * self.num_criteria, kernel_size=1,
        )
        self.boundary = nn.Conv1d(
            hidden_dim, 2 * self.num_criteria, kernel_size=1,
        )
        # Start from the already trained frame model. Temporal corrections are
        # learned only when supported by sequence supervision.
        nn.init.zeros_(self.ordinal_residual.weight)
        nn.init.zeros_(self.ordinal_residual.bias)

    @property
    def receptive_field_steps(self) -> int:
        return 1 + 2 * sum(self.dilations)

    def forward(
        self, features: torch.Tensor, frame_logits: torch.Tensor,
        phase_progress: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        values = torch.cat(
            [features, frame_logits, phase_progress.unsqueeze(-1)], dim=-1,
        )
        hidden = self.input_projection(values).transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden)
        residual = self.ordinal_residual(hidden).transpose(1, 2)
        corrected = frame_logits + residual
        boundary = self.boundary(hidden).transpose(1, 2).reshape(
            features.shape[0], features.shape[1], self.num_criteria, 2,
        )
        return {
            "support_or_full": corrected[..., :self.num_criteria],
            "full_only": corrected[..., self.num_criteria:],
            "boundary": boundary,
            "ordinal_residual": residual,
        }


class DualScaleTemporalFusionHead(nn.Module):
    """Task-neutral TCN with dense event and video-presence predictions.

    The caller supplies aligned multi-cadence evidence features.  Keeping the
    cadence alignment outside the model makes this head reusable for arbitrary
    procedural tasks and criterion counts.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int = 96,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.15,
        criteria: tuple[str, ...] = (
            "two_structures", "cystic_plate", "hepatocystic_triangle",
        ),
    ) -> None:
        super().__init__()
        if hidden_dim % 8:
            raise ValueError("hidden_dim must be divisible by 8 for GroupNorm")
        if not criteria:
            raise ValueError("DualScaleTemporalFusionHead requires criteria")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.criteria = tuple(criteria)
        self.dilations = tuple(dilations)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            TemporalResidualBlock(hidden_dim, dilation, dropout)
            for dilation in self.dilations
        ])
        self.frame_classifier = nn.Conv1d(hidden_dim, len(criteria), kernel_size=1)
        self.presence_classifier = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(criteria)),
        )

    @property
    def receptive_field_steps(self) -> int:
        return 1 + 2 * sum(self.dilations)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.input_projection(features).transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden)
        frame_logits = self.frame_classifier(hidden).transpose(1, 2)
        pooled = torch.cat([hidden.mean(dim=-1), hidden.amax(dim=-1)], dim=-1)
        return {
            "frame_logits": frame_logits,
            "presence_logits": self.presence_classifier(pooled),
        }
