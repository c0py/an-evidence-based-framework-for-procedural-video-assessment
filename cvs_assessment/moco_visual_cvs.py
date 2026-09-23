"""MoCo-initialized ResNet-50 models for frame-level CVS recognition."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import resnet50


class MocoResNet50Cvs(nn.Module):
    """A small vision-only CVS classifier with a ResNet-50 backbone.

    The language/Skill path is intentionally absent from this baseline.  It is
    the safe visual anchor to which later zero-gated Skill and temporal
    residuals can be attached.
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        self.backbone = resnet50(weights=None)
        feature_dim = int(self.backbone.fc.in_features)
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(feature_dim, int(num_classes))

    @property
    def feature_dim(self) -> int:
        return int(self.classifier.in_features)

    def train(self, mode: bool = True) -> "MocoResNet50Cvs":
        super().train(mode)
        # Match the official Endoscapes simple-classifier configuration:
        # convolutional weights are trainable but BatchNorm statistics stay
        # frozen (norm_eval=True in mmdetection).
        if mode:
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        logits = self.classifier(self.dropout(features))
        return {"logits": logits, "features": features}


def _tensor_mappings(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Mapping[str, torch.Tensor]]]:
    found: list[tuple[tuple[str, ...], Mapping[str, torch.Tensor]]] = []
    if isinstance(value, Mapping):
        tensors = {str(key): item for key, item in value.items() if torch.is_tensor(item)}
        if tensors:
            found.append((path, tensors))
        for key, item in value.items():
            if isinstance(item, Mapping):
                found.extend(_tensor_mappings(item, path + (str(key),)))
    return found


def _normalise_meta_moco_key(key: str) -> str | None:
    prefixes = ("module.encoder_q.", "encoder_q.")
    for prefix in prefixes:
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    else:
        return None
    if key.startswith("fc."):
        return None
    return key


def _normalise_vissl_key(key: str) -> str | None:
    for prefix in (
        "module.trunk.", "trunk.", "_feature_blocks.",
        "module._feature_blocks.",
    ):
        if key.startswith(prefix):
            key = key[len(prefix):]
    if key.startswith("_feature_blocks."):
        key = key[len("_feature_blocks."):]

    replacements = (
        ("conv1.0.", "conv1."),
        ("conv1.1.", "bn1."),
        ("res2.", "layer1."),
        ("res3.", "layer2."),
        ("res4.", "layer3."),
        ("res5.", "layer4."),
    )
    for source, target in replacements:
        if key.startswith(source):
            key = target + key[len(source):]
            break
    if key.startswith(("fc.", "avgpool.", "flatten.")):
        return None
    return key


def load_pretrained_backbone(
    model: MocoResNet50Cvs,
    checkpoint_path: Path,
    source_format: str,
) -> dict[str, Any]:
    """Load a Meta-MoCo or VISSL-MoCo checkpoint into torchvision ResNet-50."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    candidates = _tensor_mappings(payload)
    if not candidates:
        raise ValueError(f"No tensor state dictionaries found in {checkpoint_path}")

    if source_format == "meta_moco_v2":
        normalise = _normalise_meta_moco_key
    elif source_format == "vissl_moco_v2":
        normalise = _normalise_vissl_key
    elif source_format == "torchvision_resnet50":
        normalise = lambda key: key if not key.startswith("fc.") else None
    else:
        raise ValueError(f"Unsupported source format: {source_format}")

    target = model.backbone.state_dict()
    best: tuple[int, tuple[str, ...], dict[str, torch.Tensor]] | None = None
    for path, state in candidates:
        converted: dict[str, torch.Tensor] = {}
        for source_key, tensor in state.items():
            target_key = normalise(source_key)
            if target_key in target and target[target_key].shape == tensor.shape:
                converted[target_key] = tensor
        score = len(converted)
        if best is None or score > best[0]:
            best = (score, path, converted)

    assert best is not None
    score, source_path, converted = best
    expected = {key for key in target if not key.startswith("fc.")}
    missing = sorted(expected - set(converted))
    if missing:
        raise ValueError(
            f"Checkpoint mapping is incomplete ({score}/{len(expected)} tensors); "
            f"first missing keys: {missing[:12]}"
        )
    result = model.backbone.load_state_dict(converted, strict=False)
    unexpected = sorted(result.unexpected_keys)
    missing_after = sorted(key for key in result.missing_keys if not key.startswith("fc."))
    if unexpected or missing_after:
        raise ValueError(
            f"Backbone load mismatch: missing={missing_after[:12]}, unexpected={unexpected[:12]}"
        )
    return {
        "source_format": source_format,
        "source_mapping_path": list(source_path),
        "loaded_tensor_count": score,
        "target_tensor_count": len(expected),
        "all_backbone_tensors_loaded": True,
    }
