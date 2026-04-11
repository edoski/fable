"""Baseline temporal models."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import NamedTuple, cast

import torch
from torch import nn

from ..core.config import ModelConfig, ModelFamily


class ModelOutputs(NamedTuple):
    logits: torch.Tensor
    fee_hat: torch.Tensor


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_length, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        positional_encoding = cast(torch.Tensor, self.get_buffer("pe"))
        return inputs + positional_encoding[:, : inputs.size(1)]


class TemporalModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> ModelOutputs:
        raise NotImplementedError


class TemporalOutputHead(nn.Module):
    def __init__(self, hidden_dim: int, action_count: int, head_hidden_dim: int) -> None:
        super().__init__()
        self.classifier = MLPHead(hidden_dim, head_hidden_dim, action_count)
        self.regressor = MLPHead(hidden_dim, head_hidden_dim, 1)

    def forward(self, encoded: torch.Tensor) -> ModelOutputs:
        return ModelOutputs(
            logits=self.classifier(encoded),
            fee_hat=self.regressor(encoded).squeeze(-1),
        )


class LSTMBaseline(TemporalModel):
    def __init__(self, n_features: int, action_count: int, config: ModelConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(n_features, config.input_projection_dim)
        self.backbone = nn.LSTM(
            input_size=config.input_projection_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_head = TemporalOutputHead(
            config.hidden_size,
            action_count,
            config.head_hidden_dim,
        )

    def forward(self, inputs: torch.Tensor) -> ModelOutputs:
        projected = self.input_projection(inputs)
        outputs, _ = self.backbone(projected)
        last_state = outputs[:, -1, :]
        return self.output_head(last_state)


class TransformerBaseline(TemporalModel):
    def __init__(self, n_features: int, action_count: int, config: ModelConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(n_features, config.d_model)
        self.position_encoding = SinusoidalPositionalEncoding(config.d_model)
        self.encoder = build_transformer_encoder(config)
        self.output_head = TemporalOutputHead(
            config.d_model,
            action_count,
            config.head_hidden_dim,
        )

    def forward(self, inputs: torch.Tensor) -> ModelOutputs:
        projected = self.input_projection(inputs)
        encoded = self.encoder(self.position_encoding(projected))
        last_state = encoded[:, -1, :]
        return self.output_head(last_state)


class TransformerLSTMBaseline(TemporalModel):
    def __init__(self, n_features: int, action_count: int, config: ModelConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(n_features, config.d_model)
        self.position_encoding = SinusoidalPositionalEncoding(config.d_model)
        self.encoder = build_transformer_encoder(config)
        self.lstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.hidden_size,
            num_layers=max(1, config.num_layers - 1),
            dropout=config.dropout if config.num_layers > 2 else 0.0,
            batch_first=True,
        )
        self.output_head = TemporalOutputHead(
            config.hidden_size,
            action_count,
            config.head_hidden_dim,
        )

    def forward(self, inputs: torch.Tensor) -> ModelOutputs:
        projected = self.input_projection(inputs)
        encoded = self.encoder(self.position_encoding(projected))
        recurrent, _ = self.lstm(encoded)
        last_state = recurrent[:, -1, :]
        return self.output_head(last_state)


def build_model(n_features: int, action_count: int, config: ModelConfig) -> TemporalModel:
    if config.family is ModelFamily.LSTM:
        return LSTMBaseline(n_features, action_count, config)
    if config.family is ModelFamily.TRANSFORMER:
        return TransformerBaseline(n_features, action_count, config)
    if config.family is ModelFamily.TRANSFORMER_LSTM:
        return TransformerLSTMBaseline(n_features, action_count, config)
    raise ValueError(f"Unsupported model family: {config.family}")


def build_transformer_encoder(config: ModelConfig) -> nn.TransformerEncoder:
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=config.d_model,
        nhead=config.nhead,
        dim_feedforward=config.feedforward_dim,
        dropout=config.dropout,
        activation="gelu",
        batch_first=True,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)
