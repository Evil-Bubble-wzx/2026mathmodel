"""Missing-aware temporal encoders and reliability-gated multimodal fusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    text_dim: int = 768
    audio_dim: int = 74
    vision_dim: int = 35
    hidden_dim: int = 96
    num_heads: int = 4
    num_layers: int = 1
    dropout: float = 0.20
    max_length: int = 50

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MaskedTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        max_length: int,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.missing_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.missing_token, std=0.02)
        self.position = nn.Parameter(torch.zeros(1, max_length, hidden_dim))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.attention_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        values: torch.Tensor,
        timeline: torch.Tensor,
        observed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = self.projection(values)
        missing = self.missing_token.expand_as(projected)
        hidden = torch.where(observed.unsqueeze(-1), projected, missing)
        hidden = hidden + self.position[:, : hidden.shape[1], :]
        hidden = self.transformer(hidden, src_key_padding_mask=~timeline)
        scores = self.attention_score(hidden).squeeze(-1)
        scores = scores.masked_fill(~timeline, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        reliability = observed.sum(dim=1).float() / timeline.sum(dim=1).clamp_min(1).float()
        return self.output_norm(pooled), weights, reliability


class RobustFusionModel(nn.Module):
    MODALITIES = ("text", "audio", "vision")

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        dimensions = {
            "text": config.text_dim,
            "audio": config.audio_dim,
            "vision": config.vision_dim,
        }
        self.encoders = nn.ModuleDict(
            {
                name: MaskedTemporalEncoder(
                    input_dim=dimensions[name],
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    num_layers=config.num_layers,
                    dropout=config.dropout,
                    max_length=config.max_length,
                )
                for name in self.MODALITIES
            }
        )
        self.gate_networks = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(config.hidden_dim + 1, config.hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim // 2, 1),
                )
                for name in self.MODALITIES
            }
        )
        self.shared = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.classifier = nn.Linear(config.hidden_dim, 3)
        self.regressor = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        timeline: torch.Tensor,
        text_observed: torch.Tensor,
        audio_observed: torch.Tensor,
        vision_observed: torch.Tensor,
        gate_mode: str = "dynamic",
    ) -> dict[str, torch.Tensor]:
        values = {"text": text, "audio": audio, "vision": vision}
        observed = {
            "text": text_observed,
            "audio": audio_observed,
            "vision": vision_observed,
        }
        pooled: dict[str, torch.Tensor] = {}
        temporal_weights: dict[str, torch.Tensor] = {}
        reliabilities: list[torch.Tensor] = []
        gate_logits: list[torch.Tensor] = []
        for name in self.MODALITIES:
            representation, weights, reliability = self.encoders[name](
                values[name], timeline, observed[name]
            )
            pooled[name] = representation
            temporal_weights[name] = weights
            reliabilities.append(reliability)
            gate_input = torch.cat([representation, reliability.unsqueeze(-1)], dim=-1)
            # The logarithmic prior makes a wholly absent modality unattractive
            # while allowing its learned missing token to express uncertainty.
            gate_logits.append(
                self.gate_networks[name](gate_input).squeeze(-1)
                + torch.log(reliability + 0.05)
            )
        reliability_tensor = torch.stack(reliabilities, dim=1)
        if gate_mode == "dynamic":
            gates = torch.softmax(torch.stack(gate_logits, dim=1), dim=1)
        elif gate_mode == "uniform":
            gates = torch.full_like(reliability_tensor, 1.0 / len(self.MODALITIES))
        else:
            raise ValueError(f"Unknown gate mode {gate_mode!r}")
        representation_tensor = torch.stack([pooled[name] for name in self.MODALITIES], dim=1)
        fused = torch.sum(representation_tensor * gates.unsqueeze(-1), dim=1)
        shared = self.shared(fused)
        logits = self.classifier(shared)
        regression = 3.0 * torch.tanh(self.regressor(shared).squeeze(-1))
        return {
            "logits": logits,
            "regression": regression,
            "gates": gates,
            "reliability": reliability_tensor,
            "text_attention": temporal_weights["text"],
            "audio_attention": temporal_weights["audio"],
            "vision_attention": temporal_weights["vision"],
        }


def augment_local_blocks(
    observed: dict[str, torch.Tensor],
    timeline: torch.Tensor,
    probability: float = 0.45,
    min_fraction: float = 0.10,
    max_fraction: float = 0.40,
) -> dict[str, torch.Tensor]:
    """Mask random contiguous local intervals without deleting whole samples."""

    augmented = {name: value.clone() for name, value in observed.items()}
    batch_size = timeline.shape[0]
    for batch_index in range(batch_size):
        valid_positions = torch.nonzero(timeline[batch_index], as_tuple=False).flatten()
        length = int(valid_positions.numel())
        if length == 0:
            continue
        for name in RobustFusionModel.MODALITIES:
            if torch.rand((), device=timeline.device).item() >= probability:
                continue
            fraction = min_fraction + (max_fraction - min_fraction) * torch.rand(
                (), device=timeline.device
            ).item()
            block_length = max(1, min(length - 1 if length > 1 else 1, round(length * fraction)))
            max_start = max(0, length - block_length)
            start = int(
                torch.randint(max_start + 1, (1,), device=timeline.device).item()
            )
            positions = valid_positions[start : start + block_length]
            augmented[name][batch_index, positions] = False
    return augmented


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor,
        regression_weight: float = 0.60,
        correlation_weight: float = 0.20,
        consistency_weight: float = 0.10,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.regression_weight = regression_weight
        self.correlation_weight = correlation_weight
        self.consistency_weight = consistency_weight

    @staticmethod
    def correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction - prediction.mean()
        target = target - target.mean()
        denominator = torch.sqrt(
            torch.sum(prediction.square()) * torch.sum(target.square()) + 1e-8
        )
        correlation = torch.sum(prediction * target) / denominator
        return 1.0 - correlation

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        classification: torch.Tensor,
        regression: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        classification_loss = F.cross_entropy(
            outputs["logits"], classification, weight=self.class_weights
        )
        regression_loss = F.smooth_l1_loss(outputs["regression"], regression, beta=0.5)
        correlation_loss = self.correlation_loss(outputs["regression"], regression)
        probabilities = torch.softmax(outputs["logits"], dim=-1)
        polarity_axis = torch.tensor(
            [-1.0, 0.0, 1.0], device=probabilities.device, dtype=probabilities.dtype
        )
        expected_polarity = probabilities @ polarity_axis
        consistency_loss = F.mse_loss(expected_polarity, outputs["regression"] / 3.0)
        total = (
            classification_loss
            + self.regression_weight * regression_loss
            + self.correlation_weight * correlation_loss
            + self.consistency_weight * consistency_loss
        )
        parts = {
            "classification": float(classification_loss.detach()),
            "regression": float(regression_loss.detach()),
            "correlation": float(correlation_loss.detach()),
            "consistency": float(consistency_loss.detach()),
        }
        return total, parts
