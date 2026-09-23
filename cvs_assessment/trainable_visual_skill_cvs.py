"""Criterion-text-conditioned CVS model with a trainable small visual tower."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .models import PeskaVLPVisualEncoder
from .text_conditioned_temporal_cvs import TextConditionedSpatialFrameHead


class TrainableVisualSkillSpatialCvs(nn.Module):
    """Adapt the visual tower while keeping Skill text and YOLO inputs frozen.

    The same image/text scoring functions are shared by all criteria. Criterion
    identity is supplied only through its frozen natural-language embedding.
    """

    def __init__(
        self,
        criterion_embeddings: torch.Tensor,
        *,
        token_dim: int = 1280,
        geometry_dim: int = 9,
        alignment_dim: int = 256,
        spatial_hidden_dim: int = 128,
        attention_heads: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if criterion_embeddings.ndim != 2:
            raise ValueError("criterion_embeddings must be [criteria, text_dim]")
        self.encoder = PeskaVLPVisualEncoder()
        text_dim = int(criterion_embeddings.shape[1])
        self.register_buffer(
            "criterion_embeddings", criterion_embeddings.detach().float().clone(),
            persistent=True,
        )
        self.visual_alignment = nn.Sequential(
            nn.LayerNorm(self.encoder.output_dim),
            nn.Linear(self.encoder.output_dim, alignment_dim),
            nn.GELU(),
            nn.Linear(alignment_dim, alignment_dim),
        )
        self.text_alignment = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, alignment_dim),
            nn.GELU(),
            nn.Linear(alignment_dim, alignment_dim),
        )
        self.text_bias = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, 1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.0))
        self.spatial_head = TextConditionedSpatialFrameHead(
            peska_dim=self.encoder.output_dim,
            token_dim=token_dim,
            geometry_dim=geometry_dim,
            text_dim=text_dim,
            hidden_dim=spatial_hidden_dim,
            attention_heads=attention_heads,
            dropout=dropout,
        )
        # Alignment is the stable primary branch. Spatial evidence earns its
        # influence from zero during training instead of perturbing it at init.
        self.spatial_scale = nn.Parameter(torch.tensor(0.0))

    @property
    def num_criteria(self) -> int:
        return int(self.criterion_embeddings.shape[0])

    def load_encoder_state(self, state: dict[str, torch.Tensor]) -> None:
        self.encoder.load_state_dict(state, strict=True)

    def set_visual_trainability(self, stage: str) -> None:
        """Select a predeclared visual adaptation stage."""
        self.encoder.requires_grad_(False)
        if stage == "head_only":
            return
        if stage == "layer4_projection":
            self.encoder.backbone.layer4.requires_grad_(True)
            self.encoder.projection.requires_grad_(True)
            return
        raise ValueError(f"Unknown visual adaptation stage: {stage}")

    def train(self, mode: bool = True) -> "TrainableVisualSkillSpatialCvs":
        super().train(mode)
        # The source tower's BatchNorm statistics are kept frozen. Trainable
        # convolution/projection weights still receive gradients in eval mode.
        self.encoder.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        geometry: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score all criteria.

        ``tokens`` is [B,T,D] and ``geometry`` is [B,C,T,G], where the final
        geometry channel is criterion-specific ROI relevance.
        """
        if tokens.ndim != 3:
            raise ValueError("tokens must be [batch, tokens, features]")
        if geometry.ndim != 4:
            raise ValueError("geometry must be [batch, criteria, tokens, features]")
        batch, criteria, token_count, _ = geometry.shape
        if batch != len(images) or criteria != self.num_criteria:
            raise ValueError("image, criterion, and geometry dimensions do not align")
        if tokens.shape[:2] != (batch, token_count):
            raise ValueError("token and geometry dimensions do not align")

        image_features = self.encoder(images)
        visual = F.normalize(self.visual_alignment(image_features.float()), dim=-1)
        text = F.normalize(self.text_alignment(self.criterion_embeddings), dim=-1)
        scale = self.logit_scale.clamp(0.0, 4.5).exp()
        alignment_logits = scale * visual @ text.transpose(0, 1)
        alignment_logits = alignment_logits + self.text_bias(
            self.criterion_embeddings,
        ).flatten()[None, :]

        flat_image = image_features[:, None, :].expand(-1, criteria, -1).reshape(
            batch * criteria, -1,
        )
        flat_tokens = tokens[:, None, :, :].expand(
            -1, criteria, -1, -1,
        ).reshape(batch * criteria, token_count, -1)
        flat_geometry = geometry.reshape(batch * criteria, token_count, -1)
        flat_text = self.criterion_embeddings[None, :, :].expand(
            batch, -1, -1,
        ).reshape(batch * criteria, -1)
        spatial = self.spatial_head(
            flat_image, flat_tokens, flat_geometry, flat_text,
        )
        spatial_logits = spatial["logit"].reshape(batch, criteria)
        spatial_hidden = spatial["hidden"].reshape(batch, criteria, -1)
        logits = alignment_logits + torch.tanh(self.spatial_scale) * spatial_logits
        return {
            "logits": logits,
            "alignment_logits": alignment_logits,
            "spatial_logits": spatial_logits,
            "spatial_hidden": spatial_hidden,
            "image_features": image_features,
        }
