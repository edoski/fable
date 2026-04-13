"""Baseline temporal models."""

from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from ..prediction import PredictionOutputSpec

if TYPE_CHECKING:
    from .families.lstm import LstmModelConfig
    from .families.transformer import TransformerModelConfig
    from .families.transformer_lstm import TransformerLstmModelConfig


@dataclass(frozen=True, slots=True)
class ModelOutputs:
    heads: dict[str, torch.Tensor]

    def head(self, head_id: str) -> torch.Tensor:
        try:
            return self.heads[head_id]
        except KeyError as exc:
            known = ", ".join(sorted(self.heads)) or "<none>"
            raise ValueError(f"Unknown output head: {head_id}. Known heads: {known}") from exc


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


def sequence_lengths_from_mask(input_mask: torch.Tensor) -> torch.Tensor:
    lengths = input_mask.to(dtype=torch.int64).sum(dim=1)
    if torch.any(lengths <= 0):
        raise ValueError("input_mask must contain at least one valid timestep per sample")
    return lengths


def take_last_valid(encoded: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
    lengths = sequence_lengths_from_mask(input_mask)
    last_positions = (lengths - 1).to(device=encoded.device)
    batch_indices = torch.arange(encoded.size(0), device=encoded.device)
    return encoded[batch_indices, last_positions]


def packed_lstm_last_state_reference(
    backbone: nn.LSTM,
    inputs: torch.Tensor,
    input_mask: torch.Tensor,
) -> torch.Tensor:
    lengths = sequence_lengths_from_mask(input_mask)
    packed = pack_padded_sequence(
        inputs,
        lengths.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    _, (hidden_state, _) = backbone(packed)
    return hidden_state[-1]


class TemporalModel(nn.Module, ABC):
    """Base class for models that solve the temporal candidate-choice problem."""


class TemporalOutputHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_spec: PredictionOutputSpec,
        head_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                head.id: MLPHead(hidden_dim, head_hidden_dim, head.size)
                for head in output_spec.heads
            }
        )

    def forward(self, encoded: torch.Tensor) -> ModelOutputs:
        return ModelOutputs(
            heads={head_id: head(encoded) for head_id, head in self.heads.items()}
        )


class LSTMBaseline(TemporalModel):
    def __init__(
        self,
        n_features: int,
        output_spec: PredictionOutputSpec,
        config: LstmModelConfig,
    ) -> None:
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
            output_spec,
            config.head_hidden_dim,
        )

    def forward(self, inputs: torch.Tensor, input_mask: torch.Tensor) -> ModelOutputs:
        projected = self.input_projection(inputs)
        recurrent, _ = self.backbone(projected)
        last_state = take_last_valid(recurrent, input_mask)
        return self.output_head(last_state)


class TransformerBaseline(TemporalModel):
    def __init__(
        self,
        n_features: int,
        output_spec: PredictionOutputSpec,
        config: TransformerModelConfig,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(n_features, config.d_model)
        self.position_encoding = SinusoidalPositionalEncoding(config.d_model)
        self.encoder = build_transformer_encoder(config)
        self.output_head = TemporalOutputHead(
            config.d_model,
            output_spec,
            config.head_hidden_dim,
        )

    def forward(self, inputs: torch.Tensor, input_mask: torch.Tensor) -> ModelOutputs:
        projected = self.input_projection(inputs)
        encoded = self.encoder(
            self.position_encoding(projected),
            src_key_padding_mask=~input_mask.bool(),
        )
        return self.output_head(take_last_valid(encoded, input_mask))


class TransformerLSTMBaseline(TemporalModel):
    def __init__(
        self,
        n_features: int,
        output_spec: PredictionOutputSpec,
        config: TransformerLstmModelConfig,
    ) -> None:
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
            output_spec,
            config.head_hidden_dim,
        )

    def forward(self, inputs: torch.Tensor, input_mask: torch.Tensor) -> ModelOutputs:
        projected = self.input_projection(inputs)
        encoded = self.encoder(
            self.position_encoding(projected),
            src_key_padding_mask=~input_mask.bool(),
        )
        recurrent, _ = self.lstm(encoded)
        return self.output_head(take_last_valid(recurrent, input_mask))


class TransformerEncoderConfig(Protocol):
    d_model: int
    nhead: int
    feedforward_dim: int
    dropout: float
    transformer_layers: int


def build_transformer_encoder(config: TransformerEncoderConfig) -> nn.TransformerEncoder:
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=config.d_model,
        nhead=config.nhead,
        dim_feedforward=config.feedforward_dim,
        dropout=config.dropout,
        activation="gelu",
        batch_first=True,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)
