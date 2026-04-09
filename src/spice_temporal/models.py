"""Baseline temporal models."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn

from spice_temporal.config import ModelConfig
from spice_temporal.contracts import ModelOutputs


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
        pe = cast(torch.Tensor, self.pe)
        return inputs + pe[:, : inputs.size(1)]


class LSTMBaseline(nn.Module):
    def __init__(self, n_features: int, n_classes: int, config: ModelConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(n_features, config.input_projection_dim)
        self.backbone = nn.LSTM(
            input_size=config.input_projection_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = MLPHead(config.hidden_size, config.head_hidden_dim, n_classes)
        self.regressor = MLPHead(config.hidden_size, config.head_hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> ModelOutputs:
        projected = self.input_projection(inputs)
        outputs, _ = self.backbone(projected)
        last_state = outputs[:, -1, :]
        return ModelOutputs(
            logits=self.classifier(last_state),
            fee_hat=self.regressor(last_state).squeeze(-1),
        )


class TransformerBaseline(nn.Module):
    def __init__(self, n_features: int, n_classes: int, config: ModelConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(n_features, config.d_model)
        self.position_encoding = SinusoidalPositionalEncoding(config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)
        self.classifier = MLPHead(config.d_model, config.head_hidden_dim, n_classes)
        self.regressor = MLPHead(config.d_model, config.head_hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> ModelOutputs:
        projected = self.input_projection(inputs)
        encoded = self.encoder(self.position_encoding(projected))
        last_state = encoded[:, -1, :]
        return ModelOutputs(
            logits=self.classifier(last_state),
            fee_hat=self.regressor(last_state).squeeze(-1),
        )


class TransformerLSTMBaseline(nn.Module):
    def __init__(self, n_features: int, n_classes: int, config: ModelConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(n_features, config.d_model)
        self.position_encoding = SinusoidalPositionalEncoding(config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)
        self.lstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.hidden_size,
            num_layers=max(1, config.num_layers - 1),
            dropout=config.dropout if config.num_layers > 2 else 0.0,
            batch_first=True,
        )
        self.classifier = MLPHead(config.hidden_size, config.head_hidden_dim, n_classes)
        self.regressor = MLPHead(config.hidden_size, config.head_hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> ModelOutputs:
        projected = self.input_projection(inputs)
        encoded = self.encoder(self.position_encoding(projected))
        recurrent, _ = self.lstm(encoded)
        last_state = recurrent[:, -1, :]
        return ModelOutputs(
            logits=self.classifier(last_state),
            fee_hat=self.regressor(last_state).squeeze(-1),
        )


def build_model(n_features: int, n_classes: int, config: ModelConfig) -> nn.Module:
    family = config.family
    if family == "lstm":
        return LSTMBaseline(n_features, n_classes, config)
    if family == "transformer":
        return TransformerBaseline(n_features, n_classes, config)
    if family == "transformer_lstm":
        return TransformerLSTMBaseline(n_features, n_classes, config)
    raise ValueError(f"Unsupported model family: {family}")
