"""Typed models for reference reconstruction analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

TieBreak = Literal["earliest", "latest"]
TailPolicy = Literal["drop", "clip"]
WarmupMode = Literal["per_split", "global_before_split"]
GasRatioMode = Literal["derived_percent", "derived_unit", "csv_raw"]
TimeSinceStartMode = Literal["elapsed_seconds", "elapsed_blocks"]
BaseFeeTrendMode = Literal[
    "binary_slope_sign_200",
    "raw_slope_200",
    "binary_mean_delta_200",
    "binary_prev_delta_sign",
]
AuditFindingStatus = Literal["aligned", "inferred", "mismatch", "gap"]


@dataclass(frozen=True, slots=True)
class ReferenceMetricRow:
    chain: str
    delay_seconds: int
    model_type: str
    mae_block: float
    mae_block_rounded: float
    acc_block_rounded: float
    mae_min_fee: float
    unique_min_block_classes: int | None = None

    @property
    def spice_model_id(self) -> str:
        mapping = {
            "LSTM": "lstm",
            "Transformer": "transformer",
            "TransformerLSTM": "transformer_lstm",
        }
        return mapping.get(self.model_type, self.model_type.lower())

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AuditFinding:
    area: str
    status: AuditFindingStatus
    detail: str

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReferenceRawDataset:
    chain: str
    csv_path: str
    row_count: int
    first_timestamp: int
    last_timestamp: int

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LabelCandidate:
    include_current_block: bool
    deadline_inclusive: bool
    tie_break: TieBreak
    tail_policy: TailPolicy

    @property
    def candidate_id(self) -> str:
        return "-".join(
            [
                "inc" if self.include_current_block else "exc",
                "inclusive" if self.deadline_inclusive else "exclusive",
                self.tie_break,
                self.tail_policy,
            ]
        )

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SplitCandidate:
    train_fraction: float
    validation_fraction: float
    warmup_mode: WarmupMode

    @property
    def candidate_id(self) -> str:
        train = int(round(self.train_fraction * 1000))
        validation = int(round(self.validation_fraction * 1000))
        return f"train{train}_val{validation}_{self.warmup_mode}"

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureCandidate:
    gas_ratio_mode: GasRatioMode
    time_since_start_mode: TimeSinceStartMode
    base_fee_trend_mode: BaseFeeTrendMode

    @property
    def candidate_id(self) -> str:
        return "-".join(
            [
                self.gas_ratio_mode,
                self.time_since_start_mode,
                self.base_fee_trend_mode,
            ]
        )

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LabelSearchResult:
    chain: str
    delay_seconds: int
    label_candidate: LabelCandidate
    split_candidate: SplitCandidate
    expected_unique_classes: int | None
    train_unique_classes: int
    validation_unique_classes: int
    test_unique_classes: int
    rows_total: int
    rows_kept: int
    rows_dropped: int
    invalid_ratio: float
    score: float

    def payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["label_candidate"] = self.label_candidate.payload()
        payload["split_candidate"] = self.split_candidate.payload()
        return payload


@dataclass(frozen=True, slots=True)
class FeatureCandidateSummary:
    chain: str
    delay_seconds: int
    feature_candidate: FeatureCandidate
    gas_ratio_min: float
    gas_ratio_max: float
    time_since_start_last: float
    base_fee_trend_unique: tuple[float, ...]
    score: float
    notes: tuple[str, ...]

    def payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["feature_candidate"] = self.feature_candidate.payload()
        payload["base_fee_trend_unique"] = list(self.base_fee_trend_unique)
        payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True, slots=True)
class CurrentParityAudit:
    preset: str
    dataset: dict[str, object]
    chain: dict[str, object]
    problem: dict[str, object]
    feature_set: dict[str, object]
    dataset_builder: dict[str, object]
    prediction: dict[str, object]
    model: dict[str, object]
    evaluation: dict[str, object] | None
    compiler_runtime: dict[str, object]
    realization_policy: dict[str, object]
    compiled_prediction: dict[str, object]
    compiled_evaluator: dict[str, object] | None
    local_corpora: list[dict[str, object]]
    reference_raw: list[dict[str, object]]
    reference_metrics: list[dict[str, object]]
    findings: list[dict[str, object]]

    def payload(self) -> dict[str, object]:
        return asdict(self)
