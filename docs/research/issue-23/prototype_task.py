"""DISPOSABLE PROTOTYPE: shared Min-Block-Fee task and three concrete models.

Question: can one architecture-independent two-head task interface serve three closed
model definitions, loss, scoring, evaluation, and scheduling through plain tensors
without a registry, plugin, adapter, abstract family base, generic head map, target-batch
protocol, decoded-result ABI, or generic accumulator?

This file is not production code. It uses synthetic tensors only.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal, NamedTuple, TypeAlias, TypedDict, assert_never, cast

import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.classification import MulticlassF1Score, MulticlassStatScores

ClassificationLossMode = Literal["unweighted", "corrected_inverse_frequency"]


class HistoricalBatch(TypedDict):
    inputs: torch.Tensor
    label: torch.Tensor
    target: torch.Tensor
    base_fees: torch.Tensor
    origin_block: torch.Tensor


class MinBlockFeeOutput(NamedTuple):
    action_logits: torch.Tensor
    minimum_fee_z: torch.Tensor


@dataclass(frozen=True, slots=True)
class LstmDefinition:
    projection_width: int
    hidden_width: int
    layers: int
    dropout: float
    head_width: int
    family: Literal["lstm"] = "lstm"


@dataclass(frozen=True, slots=True)
class TransformerDefinition:
    model_width: int
    attention_heads: int
    transformer_layers: int
    feedforward_width: int
    dropout: float
    head_width: int
    family: Literal["transformer"] = "transformer"


@dataclass(frozen=True, slots=True)
class TransformerLstmDefinition:
    model_width: int
    attention_heads: int
    transformer_layers: int
    feedforward_width: int
    lstm_hidden_width: int
    lstm_layers: int
    dropout: float
    head_width: int
    family: Literal["transformer_lstm"] = "transformer_lstm"


ModelDefinition: TypeAlias = LstmDefinition | TransformerDefinition | TransformerLstmDefinition


@dataclass(frozen=True, slots=True)
class TargetState:
    mean: float
    scale: float

    @classmethod
    def fit(cls, raw_minima: torch.Tensor) -> TargetState:
        if raw_minima.ndim != 1 or raw_minima.numel() == 0:
            raise ValueError("raw_minima must be one nonempty vector")
        if raw_minima.dtype != torch.int64 or bool(torch.any(raw_minima <= 0)):
            raise ValueError("raw_minima must contain positive int64 fees")
        natural_log = torch.log(raw_minima.to(torch.float64))
        mean = natural_log.mean()
        scale = natural_log.std(correction=0)
        if not bool(torch.isfinite(mean)) or not bool(torch.isfinite(scale)):
            raise ValueError("target state must be finite")
        if float(scale) <= 0.0:
            raise ValueError("target scale must be positive")
        return cls(mean=float(mean), scale=float(scale))

    def standardize(self, raw_minima: torch.Tensor) -> torch.Tensor:
        natural_log = torch.log(raw_minima.to(torch.float64))
        standardized = (natural_log - self.mean) / self.scale
        return standardized.to(torch.float32)

    def natural_log(self, standardized: torch.Tensor) -> torch.Tensor:
        return self.mean + self.scale * standardized.to(torch.float64)


@dataclass(frozen=True, slots=True)
class ClassificationLossState:
    mode: ClassificationLossMode
    sample_count: int
    support: tuple[int, ...]

    @classmethod
    def fit(
        cls,
        labels: torch.Tensor,
        *,
        horizon: int,
        mode: ClassificationLossMode,
    ) -> ClassificationLossState:
        if labels.ndim != 1 or labels.numel() == 0:
            raise ValueError("labels must be one nonempty vector")
        if labels.dtype != torch.int64:
            raise ValueError("labels must be int64")
        if horizon <= 0 or bool(torch.any((labels < 0) | (labels >= horizon))):
            raise ValueError("labels must lie in 0...K-1")
        counts = torch.bincount(labels.cpu(), minlength=horizon)
        if mode == "corrected_inverse_frequency" and bool(torch.any(counts == 0)):
            raise ValueError("corrected inverse frequency requires every class in training")
        if mode not in {"unweighted", "corrected_inverse_frequency"}:
            raise ValueError(f"unknown classification loss mode: {mode}")
        return cls(
            mode=mode,
            sample_count=int(labels.numel()),
            support=tuple(int(value) for value in counts.tolist()),
        )

    def weights(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.mode == "unweighted":
            return None
        counts = torch.tensor(self.support, device=device, dtype=dtype)
        return self.sample_count / (len(self.support) * counts)


@dataclass(frozen=True, slots=True)
class MinBlockFeeLoss:
    total: torch.Tensor
    classification: torch.Tensor
    regression: torch.Tensor


@dataclass(frozen=True, slots=True)
class PredictiveResult:
    sample_count: int
    total_loss: float
    classification_loss: float
    regression_loss: float
    earliest_accuracy: float
    earliest_macro_f1: float
    natural_log_mae: float
    natural_log_mse: float
    target_support: tuple[int, ...]
    prediction_count: tuple[int, ...]
    true_positive: tuple[int, ...]


class _Head(nn.Module):
    def __init__(
        self,
        input_width: int,
        hidden_width: int,
        output_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, output_width),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class _TaskHeads(nn.Module):
    def __init__(
        self,
        input_width: int,
        head_width: int,
        horizon: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.action = _Head(input_width, head_width, horizon, dropout)
        self.minimum_fee = _Head(input_width, head_width, 1, dropout)

    def forward(self, encoded: torch.Tensor) -> MinBlockFeeOutput:
        return MinBlockFeeOutput(
            action_logits=self.action(encoded),
            minimum_fee_z=self.minimum_fee(encoded).squeeze(-1),
        )


class _SinusoidalPosition(nn.Module):
    def __init__(self, model_width: int, context_blocks: int) -> None:
        super().__init__()
        if model_width % 2 != 0:
            raise ValueError("model_width must be even for sinusoidal positions")
        position = torch.arange(context_blocks, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_width, 2, dtype=torch.float32)
            * (-math.log(10000.0) / model_width)
        )
        values = torch.zeros(context_blocks, model_width, dtype=torch.float32)
        values[:, 0::2] = torch.sin(position * div_term)
        values[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("values", values.unsqueeze(0), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = cast(torch.Tensor, self.get_buffer("values"))
        return inputs + values[:, : inputs.shape[1]]


def _require_inputs(
    inputs: torch.Tensor,
    *,
    context_blocks: int,
    input_width: int,
) -> None:
    if inputs.ndim != 3 or tuple(inputs.shape[1:]) != (context_blocks, input_width):
        raise ValueError(f"inputs must have shape [B,{context_blocks},{input_width}]")


def _transformer_encoder(
    *,
    model_width: int,
    attention_heads: int,
    layers: int,
    feedforward_width: int,
    dropout: float,
) -> nn.TransformerEncoder:
    if model_width % attention_heads != 0:
        raise ValueError("model_width must be divisible by attention_heads")
    layer = nn.TransformerEncoderLayer(
        d_model=model_width,
        nhead=attention_heads,
        dim_feedforward=feedforward_width,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers)


class MinBlockFeeLSTM(nn.Module):
    def __init__(
        self,
        *,
        input_width: int,
        context_blocks: int,
        horizon: int,
        definition: LstmDefinition,
    ) -> None:
        super().__init__()
        self.input_width = input_width
        self.context_blocks = context_blocks
        self.input_projection = nn.Linear(input_width, definition.projection_width)
        self.backbone = nn.LSTM(
            input_size=definition.projection_width,
            hidden_size=definition.hidden_width,
            num_layers=definition.layers,
            dropout=definition.dropout if definition.layers > 1 else 0.0,
            batch_first=True,
        )
        self.heads = _TaskHeads(
            definition.hidden_width,
            definition.head_width,
            horizon,
            definition.dropout,
        )

    def forward(self, inputs: torch.Tensor) -> MinBlockFeeOutput:
        _require_inputs(
            inputs,
            context_blocks=self.context_blocks,
            input_width=self.input_width,
        )
        recurrent, _ = self.backbone(self.input_projection(inputs))
        return self.heads(recurrent[:, -1])


class MinBlockFeeTransformer(nn.Module):
    def __init__(
        self,
        *,
        input_width: int,
        context_blocks: int,
        horizon: int,
        definition: TransformerDefinition,
    ) -> None:
        super().__init__()
        self.input_width = input_width
        self.context_blocks = context_blocks
        self.input_projection = nn.Linear(input_width, definition.model_width)
        self.position = _SinusoidalPosition(definition.model_width, context_blocks)
        self.encoder = _transformer_encoder(
            model_width=definition.model_width,
            attention_heads=definition.attention_heads,
            layers=definition.transformer_layers,
            feedforward_width=definition.feedforward_width,
            dropout=definition.dropout,
        )
        self.heads = _TaskHeads(
            definition.model_width,
            definition.head_width,
            horizon,
            definition.dropout,
        )

    def forward(self, inputs: torch.Tensor) -> MinBlockFeeOutput:
        _require_inputs(
            inputs,
            context_blocks=self.context_blocks,
            input_width=self.input_width,
        )
        encoded = self.encoder(self.position(self.input_projection(inputs)))
        return self.heads(encoded[:, -1])


class MinBlockFeeTransformerLSTM(nn.Module):
    def __init__(
        self,
        *,
        input_width: int,
        context_blocks: int,
        horizon: int,
        definition: TransformerLstmDefinition,
    ) -> None:
        super().__init__()
        self.input_width = input_width
        self.context_blocks = context_blocks
        self.input_projection = nn.Linear(input_width, definition.model_width)
        self.position = _SinusoidalPosition(definition.model_width, context_blocks)
        self.encoder = _transformer_encoder(
            model_width=definition.model_width,
            attention_heads=definition.attention_heads,
            layers=definition.transformer_layers,
            feedforward_width=definition.feedforward_width,
            dropout=definition.dropout,
        )
        self.lstm = nn.LSTM(
            input_size=definition.model_width,
            hidden_size=definition.lstm_hidden_width,
            num_layers=definition.lstm_layers,
            dropout=definition.dropout if definition.lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.heads = _TaskHeads(
            definition.lstm_hidden_width,
            definition.head_width,
            horizon,
            definition.dropout,
        )

    def forward(self, inputs: torch.Tensor) -> MinBlockFeeOutput:
        _require_inputs(
            inputs,
            context_blocks=self.context_blocks,
            input_width=self.input_width,
        )
        encoded = self.encoder(self.position(self.input_projection(inputs)))
        recurrent, _ = self.lstm(encoded)
        return self.heads(recurrent[:, -1])


ConcreteModel: TypeAlias = MinBlockFeeLSTM | MinBlockFeeTransformer | MinBlockFeeTransformerLSTM


def build_model(
    *,
    input_width: int,
    context_blocks: int,
    horizon: int,
    definition: ModelDefinition,
) -> ConcreteModel:
    match definition:
        case LstmDefinition():
            return MinBlockFeeLSTM(
                input_width=input_width,
                context_blocks=context_blocks,
                horizon=horizon,
                definition=definition,
            )
        case TransformerDefinition():
            return MinBlockFeeTransformer(
                input_width=input_width,
                context_blocks=context_blocks,
                horizon=horizon,
                definition=definition,
            )
        case TransformerLstmDefinition():
            return MinBlockFeeTransformerLSTM(
                input_width=input_width,
                context_blocks=context_blocks,
                horizon=horizon,
                definition=definition,
            )
        case _:
            assert_never(definition)


def _validate_output(output: MinBlockFeeOutput) -> tuple[int, int]:
    if output.action_logits.ndim != 2 or output.minimum_fee_z.ndim != 1:
        raise ValueError("task output must have logits [B,K] and minimum_fee_z [B]")
    sample_count, horizon = output.action_logits.shape
    if horizon <= 0 or output.minimum_fee_z.shape != (sample_count,):
        raise ValueError("task output batch shapes must match")
    if not bool(torch.isfinite(output.action_logits).all()) or not bool(
        torch.isfinite(output.minimum_fee_z).all()
    ):
        raise ValueError("both task outputs must be finite")
    return sample_count, horizon


def _loss_terms(
    output: MinBlockFeeOutput,
    *,
    label: torch.Tensor,
    target: torch.Tensor,
    classification: ClassificationLossState,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_count, horizon = _validate_output(output)
    if label.shape != (sample_count,) or target.shape != (sample_count,):
        raise ValueError("label and target must match the task output batch")
    if len(classification.support) != horizon:
        raise ValueError("classification state width must equal K")
    weights = classification.weights(
        device=output.action_logits.device,
        dtype=output.action_logits.dtype,
    )
    classification_terms = F.cross_entropy(
        output.action_logits,
        label,
        weight=weights,
        reduction="none",
    )
    regression_terms = F.smooth_l1_loss(
        output.minimum_fee_z,
        target,
        reduction="none",
    )
    return classification_terms, regression_terms


def min_block_fee_loss(
    output: MinBlockFeeOutput,
    *,
    label: torch.Tensor,
    target: torch.Tensor,
    classification: ClassificationLossState,
) -> MinBlockFeeLoss:
    classification_terms, regression_terms = _loss_terms(
        output,
        label=label,
        target=target,
        classification=classification,
    )
    if classification_terms.numel() == 0:
        raise ValueError("loss requires a nonempty batch")
    classification_loss = classification_terms.mean()
    regression_loss = regression_terms.mean()
    return MinBlockFeeLoss(
        total=classification_loss + regression_loss,
        classification=classification_loss,
        regression=regression_loss,
    )


def decode_action(output: MinBlockFeeOutput) -> torch.Tensor:
    _validate_output(output)
    return output.action_logits.argmax(dim=-1)


class PredictiveScorer:
    """Concrete one-phase scorer shared by every closed architecture."""

    def __init__(
        self,
        *,
        horizon: int,
        target_state: TargetState,
        classification: ClassificationLossState,
    ) -> None:
        self.target_state = target_state
        self.classification = classification
        self.sample_count = 0
        self.classification_sum = 0.0
        self.regression_sum = 0.0
        self.absolute_error_sum = 0.0
        self.squared_error_sum = 0.0
        self.f1 = MulticlassF1Score(
            num_classes=horizon,
            average="macro",
            multidim_average="global",
            zero_division=0,
        )
        self.stats = MulticlassStatScores(
            num_classes=horizon,
            average=None,
            multidim_average="global",
        )

    def update(self, output: MinBlockFeeOutput, batch: HistoricalBatch) -> None:
        classification_terms, regression_terms = _loss_terms(
            output,
            label=batch["label"],
            target=batch["target"],
            classification=self.classification,
        )
        decoded = decode_action(output)
        predicted_log = self.target_state.natural_log(output.minimum_fee_z)
        target_log = self.target_state.natural_log(batch["target"])
        errors = predicted_log - target_log
        count = int(batch["label"].numel())
        self.sample_count += count
        self.classification_sum += float(classification_terms.to(torch.float64).sum())
        self.regression_sum += float(regression_terms.to(torch.float64).sum())
        self.absolute_error_sum += float(errors.to(torch.float64).abs().sum())
        self.squared_error_sum += float(errors.to(torch.float64).square().sum())
        self.f1.update(decoded, batch["label"])
        self.stats.update(decoded, batch["label"])

    def compute(self) -> PredictiveResult:
        if self.sample_count <= 0:
            raise ValueError("cannot score an empty phase")
        stats = self.stats.compute().to(device="cpu", dtype=torch.int64)
        true_positive = tuple(int(value) for value in stats[:, 0].tolist())
        prediction_count = tuple(
            int(tp + fp) for tp, fp in zip(stats[:, 0].tolist(), stats[:, 1].tolist(), strict=True)
        )
        target_support = tuple(int(value) for value in stats[:, 4].tolist())
        return PredictiveResult(
            sample_count=self.sample_count,
            total_loss=(self.classification_sum + self.regression_sum) / self.sample_count,
            classification_loss=self.classification_sum / self.sample_count,
            regression_loss=self.regression_sum / self.sample_count,
            earliest_accuracy=sum(true_positive) / self.sample_count,
            earliest_macro_f1=float(self.f1.compute()),
            natural_log_mae=self.absolute_error_sum / self.sample_count,
            natural_log_mse=self.squared_error_sum / self.sample_count,
            target_support=target_support,
            prediction_count=prediction_count,
            true_positive=true_positive,
        )


def _targets(base_fees: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, TargetState]:
    if base_fees.ndim != 2 or base_fees.dtype != torch.int64:
        raise ValueError("base_fees must have shape [B,K] and dtype int64")
    raw_minima, labels = base_fees.min(dim=1)
    state = TargetState.fit(raw_minima)
    return labels, raw_minima, state


def _slice_batch(batch: HistoricalBatch, start: int, stop: int) -> HistoricalBatch:
    return {key: value[start:stop] for key, value in batch.items()}  # type: ignore[return-value]


def _slice_output(output: MinBlockFeeOutput, start: int, stop: int) -> MinBlockFeeOutput:
    return MinBlockFeeOutput(output.action_logits[start:stop], output.minimum_fee_z[start:stop])


def _probe_architectures(
    definitions: tuple[ModelDefinition, ...],
    *,
    batch: HistoricalBatch,
    target_state: TargetState,
    classification: ClassificationLossState,
) -> dict[str, object]:
    observations: dict[str, object] = {}
    input_width = int(batch["inputs"].shape[2])
    context_blocks = int(batch["inputs"].shape[1])
    horizon = int(batch["base_fees"].shape[1])
    for definition in definitions:
        model = build_model(
            input_width=input_width,
            context_blocks=context_blocks,
            horizon=horizon,
            definition=definition,
        ).eval()
        with torch.inference_mode():
            whole = model(batch["inputs"])
            full = model(batch["inputs"][:3])
            tail = model(batch["inputs"][3:])
        if not isinstance(whole, MinBlockFeeOutput):
            raise TypeError("every concrete model must return MinBlockFeeOutput")
        torch.testing.assert_close(
            torch.cat((full.action_logits, tail.action_logits)),
            whole.action_logits,
        )
        torch.testing.assert_close(
            torch.cat((full.minimum_fee_z, tail.minimum_fee_z)),
            whole.minimum_fee_z,
        )
        full_batch = _slice_batch(batch, 0, 3)
        tail_batch = _slice_batch(batch, 3, 4)
        full_loss = min_block_fee_loss(
            full,
            label=full_batch["label"],
            target=full_batch["target"],
            classification=classification,
        )
        tail_loss = min_block_fee_loss(
            tail,
            label=tail_batch["label"],
            target=tail_batch["target"],
            classification=classification,
        )
        scorer = PredictiveScorer(
            horizon=horizon,
            target_state=target_state,
            classification=classification,
        )
        scorer.update(full, full_batch)
        scorer.update(tail, tail_batch)
        result = scorer.compute()
        observations[definition.family] = {
            "model_class": type(model).__name__,
            "output_type": type(full).__name__,
            "full_action_logits_shape": tuple(full.action_logits.shape),
            "full_minimum_fee_z_shape": tuple(full.minimum_fee_z.shape),
            "tail_action_logits_shape": tuple(tail.action_logits.shape),
            "tail_minimum_fee_z_shape": tuple(tail.minimum_fee_z.shape),
            "full_decoded_shape": tuple(decode_action(full).shape),
            "tail_decoded_shape": tuple(decode_action(tail).shape),
            "full_loss_finite": bool(torch.isfinite(full_loss.total)),
            "tail_loss_finite": bool(torch.isfinite(tail_loss.total)),
            "scored_samples": result.sample_count,
            "whole_equals_full_plus_tail": True,
        }
    return observations


def run_observations() -> dict[str, object]:
    torch.manual_seed(2026)
    context_blocks = 4
    feature_names = (
        "log_base_fee_per_gas",
        "gas_utilization",
        "log_exact_forming_base_fee_per_gas",
    )
    base_fees = torch.tensor(
        [[100, 90, 95], [50, 50, 60], [80, 70, 60], [30, 40, 35]],
        dtype=torch.int64,
    )
    labels, raw_minima, target_state = _targets(base_fees)
    target = target_state.standardize(raw_minima)
    batch: HistoricalBatch = {
        "inputs": torch.randn(4, context_blocks, len(feature_names), dtype=torch.float32),
        "label": labels,
        "target": target,
        "base_fees": base_fees,
        "origin_block": torch.arange(1000, 1004, dtype=torch.int64),
    }
    definitions: tuple[ModelDefinition, ...] = (
        LstmDefinition(5, 7, 1, 0.0, 6),
        TransformerDefinition(8, 2, 1, 16, 0.0, 6),
        TransformerLstmDefinition(8, 2, 1, 16, 7, 1, 0.0, 6),
    )
    unweighted = ClassificationLossState.fit(labels, horizon=3, mode="unweighted")
    corrected = ClassificationLossState.fit(
        labels,
        horizon=3,
        mode="corrected_inverse_frequency",
    )
    architectures = _probe_architectures(
        definitions,
        batch=batch,
        target_state=target_state,
        classification=corrected,
    )

    output = MinBlockFeeOutput(
        action_logits=torch.tensor(
            [[0.1, 3.0, 0.0], [2.0, 2.0, 0.0], [1.0, 2.0, 3.0], [0.0, 2.0, 1.0]],
            dtype=torch.float32,
        ),
        minimum_fee_z=target + torch.tensor([0.1, -0.2, 0.3, -0.4]),
    )
    unweighted_loss = min_block_fee_loss(
        output,
        label=labels,
        target=target,
        classification=unweighted,
    )
    corrected_loss = min_block_fee_loss(
        output,
        label=labels,
        target=target,
        classification=corrected,
    )
    corrected_weights = corrected.weights(
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    if corrected_weights is None:
        raise AssertionError("corrected inverse-frequency mode must produce weights")

    full_scorer = PredictiveScorer(
        horizon=3,
        target_state=target_state,
        classification=corrected,
    )
    full_scorer.update(output, batch)
    full_result = full_scorer.compute()
    chunked_scorer = PredictiveScorer(
        horizon=3,
        target_state=target_state,
        classification=corrected,
    )
    for start, stop in ((0, 3), (3, 4)):
        chunked_scorer.update(_slice_output(output, start, stop), _slice_batch(batch, start, stop))
    if asdict(full_result) != asdict(chunked_scorer.compute()):
        raise AssertionError("scoring changed with batch partition")

    serving_output = _slice_output(output, 0, 1)
    serving_action = int(decode_action(serving_output).item())

    return {
        "task_seam": {
            "output": "MinBlockFeeOutput",
            "output_fields": ("action_logits", "minimum_fee_z"),
            "target_state": "TargetState",
            "loss": "min_block_fee_loss",
            "decode": "decode_action",
            "scorer": "PredictiveScorer",
            "architecture_independent": True,
        },
        "architectures": architectures,
        "model_construction": {
            "definition_tags": tuple(definition.family for definition in definitions),
            "direct_exhaustive_match": True,
            "registry": False,
            "plugin": False,
            "adapter": False,
            "abstract_family_base": False,
            "generic_head_map": False,
        },
        "targets": {
            "raw_base_fees": base_fees.tolist(),
            "earliest_labels": labels.tolist(),
            "raw_minima": raw_minima.tolist(),
            "tie_row_label": int(labels[1]),
            "target_mean_float64": target_state.mean,
            "target_scale_float64": target_state.scale,
            "standardized_shape": tuple(target.shape),
        },
        "loss": {
            "unweighted": {
                "total": float(unweighted_loss.total),
                "classification": float(unweighted_loss.classification),
                "regression": float(unweighted_loss.regression),
                "weights": None,
            },
            "corrected_inverse_frequency": {
                "total": float(corrected_loss.total),
                "classification": float(corrected_loss.classification),
                "regression": float(corrected_loss.regression),
                "support": corrected.support,
                "weights": corrected_weights.tolist(),
            },
            "regression_coefficient": 1.0,
            "sample_denominator": 4,
        },
        "decode": {
            "actions": decode_action(output).tolist(),
            "first_argmax_on_tie": int(decode_action(output)[1]),
            "uses_action_mask": False,
        },
        "scorer": {
            **asdict(full_result),
            "batch_partition_invariant": True,
            "torchmetrics_objects": ("MulticlassF1Score", "MulticlassStatScores"),
        },
        "evaluator_input": {
            "inputs": {"shape": tuple(batch["inputs"].shape), "dtype": str(batch["inputs"].dtype)},
            "label": {"shape": tuple(batch["label"].shape), "dtype": str(batch["label"].dtype)},
            "target": {"shape": tuple(batch["target"].shape), "dtype": str(batch["target"].dtype)},
            "base_fees": {
                "shape": tuple(batch["base_fees"].shape),
                "dtype": str(batch["base_fees"].dtype),
            },
            "origin_block": {
                "shape": tuple(batch["origin_block"].shape),
                "dtype": str(batch["origin_block"].dtype),
            },
        },
        "serving": {
            "model_output_shapes": {
                "action_logits": tuple(serving_output.action_logits.shape),
                "minimum_fee_z": tuple(serving_output.minimum_fee_z.shape),
            },
            "decoded_k_shape": (),
            "decoded_k": serving_action,
            "scheduling_input": "decoded_k_only",
            "auxiliary_role": "retained prediction; not an authoritative fee quote or action",
        },
        "probe_scope": {
            "synthetic_full_tail_only": True,
            "quality_comparison": False,
            "ranking": False,
            "model_selection": False,
            "experiment_counts_inferred": False,
        },
    }
