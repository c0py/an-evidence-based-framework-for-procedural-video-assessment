"""Minimal frozen PeskaVLP BioClinicalBERT text tower."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class FrozenPeskaVLPTextEncoder(nn.Module):
    """Load only the official PeskaVLP text tower and reproduce its pooling."""

    output_dim = 768

    def __init__(self, config_tokenizer_path: str | Path) -> None:
        super().__init__()
        from transformers import AutoTokenizer, BertConfig, BertModel

        path = str(Path(config_tokenizer_path).resolve())
        config = BertConfig.from_pretrained(path, local_files_only=True)
        config.output_hidden_states = True
        self.model = BertModel(config)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenPeskaVLPTextEncoder":
        super().train(False)
        return self

    def load_official_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("Expected a tensor state dictionary")
        prefix = "backbone_text.model."
        state = {
            key.removeprefix(prefix): value
            for key, value in payload.items()
            if key.startswith(prefix)
        }
        if not state:
            raise ValueError("No PeskaVLP text parameters found")
        incompatible = self.model.load_state_dict(state, strict=False)
        missing = [key for key in incompatible.missing_keys if key != "embeddings.position_ids"]
        unexpected = [key for key in incompatible.unexpected_keys if key != "embeddings.position_ids"]
        if missing or unexpected:
            raise ValueError(f"Incomplete text tower: missing={missing}, unexpected={unexpected}")
        return {
            "loaded_text_parameter_tensors": len(state),
            "ignored_position_id_buffer": "embeddings.position_ids" in incompatible.unexpected_keys,
        }

    @torch.inference_mode()
    def encode(self, texts: list[str], device: torch.device, batch_size: int = 16) -> torch.Tensor:
        self.to(device).eval()
        output = []
        for start in range(0, len(texts), batch_size):
            current = texts[start:start + batch_size]
            tokens = self.tokenizer(
                current, return_tensors="pt", truncation=True,
                padding="max_length", max_length=77,
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            values = self.model(**tokens).hidden_states[-4:]
            stacked = torch.stack(values, dim=1)
            mask = tokens["attention_mask"][:, None, :, None].float()
            # Official PeskaVLP aggregates word pieces by summation, pads the
            # resulting sequence back to 77, averages over 77, then sums the
            # last four BERT layers. This is algebraically the same operation.
            pooled = (stacked * mask).sum(dim=2).sum(dim=1) / 77.0
            output.append(pooled.cpu())
        return torch.cat(output)
