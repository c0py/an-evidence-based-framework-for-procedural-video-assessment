"""PeskaVLP-aligned frozen Skill prototypes with visual-only adaptation."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .models import PeskaVLPVisualEncoder


class AlignedSkillVisualCvs(nn.Module):
    """Shared image-to-Skill scoring with no trainable language parameters."""

    def __init__(
        self,
        positive_prototypes: torch.Tensor,
        negative_prototypes: torch.Tensor,
        positive_prompts: torch.Tensor | None = None,
        negative_prompts: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if positive_prototypes.shape != negative_prototypes.shape:
            raise ValueError("Positive and negative prototypes must align")
        if positive_prototypes.ndim != 2 or positive_prototypes.shape[1] != 768:
            raise ValueError("Skill prototypes must be [criteria, 768]")
        self.encoder = PeskaVLPVisualEncoder()
        self.register_buffer("positive_prototypes", F.normalize(positive_prototypes.float(), dim=-1))
        self.register_buffer("negative_prototypes", F.normalize(negative_prototypes.float(), dim=-1))
        if positive_prompts is None:
            positive_prompts = positive_prototypes[:, None, :]
        if negative_prompts is None:
            negative_prompts = negative_prototypes[:, None, :]
        self.register_buffer("positive_prompts", F.normalize(positive_prompts.float(), dim=-1))
        self.register_buffer("negative_prompts", F.normalize(negative_prompts.float(), dim=-1))
        self.logit_scale = nn.Parameter(torch.tensor(2.0))
        self.bias = nn.Parameter(torch.zeros(len(positive_prototypes)))

    @property
    def num_criteria(self) -> int:
        return int(self.positive_prototypes.shape[0])

    def set_visual_trainability(self, stage: str) -> None:
        self.encoder.requires_grad_(False)
        if stage == "frozen":
            return
        if stage == "layer4_projection":
            self.encoder.backbone.layer4.requires_grad_(True)
            self.encoder.projection.requires_grad_(True)
            return
        raise ValueError(stage)

    def train(self, mode: bool = True) -> "AlignedSkillVisualCvs":
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(images).float()
        normalized = F.normalize(features, dim=-1)
        positive = normalized @ self.positive_prototypes.transpose(0, 1)
        negative = normalized @ self.negative_prototypes.transpose(0, 1)
        scale = self.logit_scale.clamp(0.0, 4.5).exp()
        logits = scale * (positive - negative) + self.bias[None, :]
        return {
            "logits": logits,
            "features": features,
            "normalized_features": normalized,
            "positive_similarity": positive,
            "negative_similarity": negative,
        }

    def prompt_contrastive_loss(self, normalized_features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Frozen-text version of CVS-AdaptNet's per-criterion symmetric KL."""
        batch = len(labels)
        if batch < 2:
            return normalized_features.sum() * 0.0
        scale = self.logit_scale.clamp(0.0, 4.5).exp()
        total = normalized_features.new_tensor(0.0)
        prompt_count = self.positive_prompts.shape[1]
        prompt_index = torch.arange(batch, device=labels.device) % prompt_count
        for criterion in range(self.num_criteria):
            positive = self.positive_prompts[criterion, prompt_index]
            negative = self.negative_prompts[criterion, prompt_index]
            selected = torch.where(labels[:, criterion, None] > 0.5, positive, negative)
            selected = F.normalize(selected, dim=-1)
            image_logits = scale * normalized_features @ selected.transpose(0, 1)
            text_logits = image_logits.transpose(0, 1)
            same_label = (labels[:, criterion, None] == labels[None, :, criterion]).float()
            target = F.softmax(same_label * 10.0, dim=1)
            image_loss = F.kl_div(F.log_softmax(image_logits, dim=1), target, reduction="batchmean")
            text_loss = F.kl_div(F.log_softmax(text_logits, dim=1), target, reduction="batchmean")
            total = total + 0.5 * (image_loss + text_loss)
        return total / self.num_criteria


class DualViewSkillInitializedCvs(nn.Module):
    """Global/center visual classifier initialized and regularized by Skills."""

    def __init__(
        self,
        positive_prototypes: torch.Tensor,
        negative_prototypes: torch.Tensor,
        positive_prompts: torch.Tensor,
        negative_prompts: torch.Tensor,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        direction = F.normalize(
            positive_prototypes.float() - negative_prototypes.float(), dim=-1,
        )
        self.encoder = PeskaVLPVisualEncoder()
        self.register_buffer("skill_direction", direction)
        self.register_buffer("positive_prompts", F.normalize(positive_prompts.float(), dim=-1))
        self.register_buffer("negative_prompts", F.normalize(negative_prompts.float(), dim=-1))
        self.fusion = nn.Sequential(
            nn.LayerNorm(1536), nn.Linear(1536, 768), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(768, 768),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)
        self.classifier_weight = nn.Parameter(direction.clone())
        self.bias = nn.Parameter(torch.zeros(len(direction)))
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

    @property
    def num_criteria(self) -> int:
        return int(self.skill_direction.shape[0])

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

    def train(self, mode: bool = True) -> "DualViewSkillInitializedCvs":
        super().train(mode)
        self.encoder.eval()
        return self

    def score_features(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features.float(), dim=-1)
        weight = F.normalize(self.classifier_weight.float(), dim=-1)
        scale = self.logit_scale.clamp(0.0, 4.5).exp()
        return scale * features @ weight.transpose(0, 1) + self.bias[None, :]

    def forward(self, center_images: torch.Tensor, global_images: torch.Tensor) -> dict[str, torch.Tensor]:
        center = self.encoder(center_images).float()
        global_value = self.encoder(global_images).float()
        residual = self.fusion(torch.cat([center, global_value], dim=-1))
        fused = 0.5 * (center + global_value) + residual
        return {
            "logits": self.score_features(fused),
            "center_logits": self.score_features(center),
            "global_logits": self.score_features(global_value),
            "normalized_features": F.normalize(fused, dim=-1),
            "fused_features": fused,
        }

    def skill_anchor_loss(self) -> torch.Tensor:
        weight = F.normalize(self.classifier_weight.float(), dim=-1)
        return (1.0 - (weight * self.skill_direction).sum(-1)).mean()

    def prompt_contrastive_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch = len(labels)
        if batch < 2:
            return features.sum() * 0.0
        scale = self.logit_scale.clamp(0.0, 4.5).exp()
        prompt_count = self.positive_prompts.shape[1]
        prompt_index = torch.arange(batch, device=labels.device) % prompt_count
        total = features.new_tensor(0.0)
        for criterion in range(self.num_criteria):
            positive = self.positive_prompts[criterion, prompt_index]
            negative = self.negative_prompts[criterion, prompt_index]
            selected = torch.where(labels[:, criterion, None] > 0.5, positive, negative)
            logits = scale * features @ F.normalize(selected, dim=-1).transpose(0, 1)
            same = (labels[:, criterion, None] == labels[None, :, criterion]).float()
            target = F.softmax(same * 10.0, dim=1)
            total = total + F.kl_div(
                F.log_softmax(logits, dim=1), target, reduction="batchmean",
            )
        return total / self.num_criteria
