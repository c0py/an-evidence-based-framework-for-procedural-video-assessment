"""Strong supervised dual-view CVS model with safely gated frozen Skills."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .models import PeskaVLPVisualEncoder


class SafeResidualDualViewCvs(nn.Module):
    """Visual primary classifier plus a zero-initialized Skill residual."""

    def __init__(
        self,
        positive_prototypes: torch.Tensor,
        negative_prototypes: torch.Tensor,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if positive_prototypes.shape != negative_prototypes.shape:
            raise ValueError("Skill prototypes must align")
        self.encoder = PeskaVLPVisualEncoder()
        self.register_buffer(
            "positive_prototypes", F.normalize(positive_prototypes.float(), dim=-1),
        )
        self.register_buffer(
            "negative_prototypes", F.normalize(negative_prototypes.float(), dim=-1),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(1536), nn.Linear(1536, 768), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(768, 768),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)
        self.visual_classifier = nn.Sequential(
            nn.LayerNorm(768), nn.Dropout(dropout),
            nn.Linear(768, len(positive_prototypes)),
        )
        self.skill_logit_scale = nn.Parameter(torch.tensor(2.0))
        self.skill_gate = nn.Parameter(torch.zeros(len(positive_prototypes)))

    @property
    def num_criteria(self) -> int:
        return int(self.positive_prototypes.shape[0])

    def set_visual_trainability(self, stage: str) -> None:
        self.encoder.requires_grad_(False)
        if stage == "head_only":
            return
        if stage == "layer4_projection":
            self.encoder.backbone.layer4.requires_grad_(True)
            self.encoder.projection.requires_grad_(True)
            return
        if stage == "all_visual":
            self.encoder.requires_grad_(True)
            return
        raise ValueError(stage)

    def train(self, mode: bool = True) -> "SafeResidualDualViewCvs":
        super().train(mode)
        self.encoder.eval()
        return self

    def score(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        visual_logits = self.visual_classifier(features.float())
        normalized = F.normalize(features.float(), dim=-1)
        positive = normalized @ self.positive_prototypes.transpose(0, 1)
        negative = normalized @ self.negative_prototypes.transpose(0, 1)
        skill_logits = self.skill_logit_scale.clamp(0.0, 4.5).exp() * (positive - negative)
        gate = torch.tanh(self.skill_gate)
        return {
            "logits": visual_logits + gate[None, :] * skill_logits,
            "visual_logits": visual_logits,
            "skill_logits": skill_logits,
            "skill_gate": gate,
        }

    def forward(self, center_images: torch.Tensor, global_images: torch.Tensor) -> dict[str, torch.Tensor]:
        center = self.encoder(center_images).float()
        global_value = self.encoder(global_images).float()
        fused = 0.5 * (center + global_value) + self.fusion(
            torch.cat([center, global_value], dim=-1),
        )
        output = self.score(fused)
        center_output = self.score(center)
        global_output = self.score(global_value)
        output.update({
            "center_visual_logits": center_output["visual_logits"],
            "global_visual_logits": global_output["visual_logits"],
            "fused_features": fused,
        })
        return output
