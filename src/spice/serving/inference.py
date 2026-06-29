"""Online Sepolia prediction service."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import polars as pl

from ..modeling.scoring import predict_decoded_result
from ..prediction.decoded_offsets import require_decoded_offsets
from ..temporal.execution_policy import PreparedActionSpace
from ..temporal.input_normalization import transform_problem_store_features
from ..temporal.problem_store import CompiledProblemStore
from .analytics import ServingAnalyticsStore
from .live_blocks import LiveBlockWindow, LiveSepoliaClient
from .runtime import ServingRuntime
from .schemas import (
    AnalyticsResponse,
    ModelInfoResponse,
    ObserveTransactionRequest,
    ObserveTransactionResponse,
    PredictionRequest,
    PredictionResponse,
)


@dataclass(frozen=True, slots=True)
class _OnlinePreparedSample:
    store: CompiledProblemStore
    action_space: PreparedActionSpace
    observed_block: int
    observed_timestamp: int


class OnlinePredictionService:
    def __init__(
        self,
        *,
        runtime: ServingRuntime,
        live_blocks: LiveSepoliaClient,
        analytics: ServingAnalyticsStore,
    ) -> None:
        self._runtime = runtime
        self._live_blocks = live_blocks
        self._analytics = analytics

    def model_info(self) -> ModelInfoResponse:
        manifest = self._runtime.artifact.manifest
        return ModelInfoResponse(
            chain_name=self._runtime.chain.name,
            chain_id=self._runtime.chain.runtime.chain_id,
            artifact_id=manifest.artifact_id,
            model_family=manifest.model.id,
            max_delay_seconds=manifest.temporal_capability.max_delay_seconds,
            slot_spacing_seconds=self._runtime.slot_spacing_seconds,
            demo_contract_address=self._runtime.config.demo_contract_address,
        )

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        max_delay = self._runtime.artifact.manifest.temporal_capability.max_delay_seconds
        if request.max_wait_seconds > max_delay:
            raise ValueError(
                f"max_wait_seconds exceeds artifact capability: {request.max_wait_seconds} > "
                f"{max_delay}"
            )
        window = await self._live_blocks.fetch_confirmed_window(
            support_block_count=self._runtime.support_block_count,
        )
        prepared = self._prepare_online_sample(window, max_wait_seconds=request.max_wait_seconds)
        decoded = predict_decoded_result(
            self._runtime.artifact.model,
            prediction_contract=self._runtime.prediction_contract,
            execution_policy=self._runtime.problem_contract.execution_policy,
            store=prepared.store,
            action_space=prepared.action_space,
            runtime_plan=self._runtime.runtime_plan,
        )
        selected_offset = int(
            require_decoded_offsets(decoded).select(np.array([0], dtype=np.int64))[0]
        )
        slot_seconds = self._runtime.slot_spacing_seconds
        recommended_wait_seconds = int(round(selected_offset * slot_seconds))
        observed_block = prepared.observed_block
        broadcast_after_block = observed_block + selected_offset
        target_block = broadcast_after_block + 1
        prediction = PredictionResponse(
            request_id=f"req_{uuid4().hex}",
            chain_name=self._runtime.chain.name,
            chain_id=self._runtime.chain.runtime.chain_id,
            artifact_id=self._runtime.artifact.manifest.artifact_id,
            observed_block=observed_block,
            observed_timestamp=prepared.observed_timestamp,
            baseline_block=observed_block + 1,
            broadcast_after_block=broadcast_after_block,
            target_block=target_block,
            target_timestamp_estimate=int(
                round(prepared.observed_timestamp + (selected_offset + 1) * slot_seconds)
            ),
            selected_offset=selected_offset,
            recommended_wait_seconds=recommended_wait_seconds,
            expires_at_unix=_unix_now() + self._runtime.config.prediction_ttl_seconds,
            support_start_block=window.support_start_block,
            support_end_block=window.support_end_block,
        )
        self._analytics.record_prediction(prediction)
        return prediction

    async def observe_transaction(
        self,
        request_id: str,
        request: ObserveTransactionRequest,
    ) -> ObserveTransactionResponse:
        prediction = self._analytics.get_prediction(request_id)
        receipt = await self._live_blocks.transaction_receipt(request.tx_hash)
        if receipt is None:
            raise ValueError(f"transaction receipt is not available: {request.tx_hash}")
        baseline_base_fee = await self._live_blocks.base_fee_per_gas(prediction.baseline_block)
        included_base_fee = await self._live_blocks.base_fee_per_gas(receipt.block_number)
        baseline_fee = baseline_base_fee * receipt.gas_used
        model_fee = included_base_fee * receipt.gas_used
        savings = baseline_fee - model_fee
        savings_percent = 0.0 if baseline_fee <= 0 else float(savings / baseline_fee * 100.0)
        response = ObserveTransactionResponse(
            request_id=request_id,
            tx_hash=request.tx_hash,
            included_block=receipt.block_number,
            gas_used=str(receipt.gas_used),
            baseline_block=prediction.baseline_block,
            baseline_fee_wei=str(baseline_fee),
            model_fee_wei=str(model_fee),
            savings_wei=str(savings),
            savings_percent=savings_percent,
        )
        self._analytics.record_observation(response)
        return response

    def analytics(self) -> AnalyticsResponse:
        return self._analytics.analytics()

    def _prepare_online_sample(
        self,
        window: LiveBlockWindow,
        *,
        max_wait_seconds: int,
    ) -> _OnlinePreparedSample:
        feature_table = self._runtime.feature_contract.build_table(_prepare_blocks(window.blocks))
        n_rows = int(feature_table.feature_matrix.shape[0])
        if n_rows == 0:
            raise ValueError("live block window produced no feature rows")
        anchor_row = n_rows - 1
        sequence_length = self._runtime.sequence_length
        context_start = anchor_row - sequence_length + 1
        prerequisites = self._runtime.problem_contract.feature_prerequisites
        if context_start < 0:
            raise ValueError("live block window is too short for serving sequence length")
        if context_start < prerequisites.warmup_rows:
            raise ValueError("live block window is too short for feature warmup")
        timestamps = feature_table.series.timestamps.astype(np.int64, copy=False)
        if int(timestamps[context_start] - timestamps[0]) < prerequisites.history_seconds:
            raise ValueError("live block window is too short for feature history")

        store = CompiledProblemStore(
            feature_matrix=feature_table.feature_matrix,
            log_base_fees=feature_table.series.log_base_fees,
            timestamps=timestamps,
            block_numbers=feature_table.series.block_numbers,
            anchor_rows=np.array([anchor_row], dtype=np.int64),
            context_start_rows=np.array([context_start], dtype=np.int64),
            candidate_start_rows=np.array([anchor_row], dtype=np.int64),
            candidate_end_rows=np.array([anchor_row + 1], dtype=np.int64),
            max_candidate_slots=self._runtime.artifact.manifest.temporal_capability.action_width,
        )
        scaled_store = transform_problem_store_features(
            store,
            self._runtime.artifact.manifest.scaler,
        )
        action_mask = np.zeros((1, scaled_store.max_candidate_slots), dtype=np.bool_)
        max_allowed_offset = _max_allowed_offset(
            max_wait_seconds=max_wait_seconds,
            slot_spacing_seconds=self._runtime.slot_spacing_seconds,
            action_width=scaled_store.max_candidate_slots,
        )
        action_mask[0, : max_allowed_offset + 1] = True
        action_space = PreparedActionSpace(
            sample_indices=np.array([0], dtype=np.int64),
            max_candidate_slots=scaled_store.max_candidate_slots,
            action_mask=action_mask,
        )
        return _OnlinePreparedSample(
            store=scaled_store,
            action_space=action_space,
            observed_block=int(feature_table.series.block_numbers[anchor_row]),
            observed_timestamp=int(feature_table.series.timestamps[anchor_row]),
        )


def _prepare_blocks(blocks: pl.DataFrame) -> pl.DataFrame:
    if blocks.height == 0:
        raise ValueError("serving received an empty block frame")
    return (
        blocks.sort("timestamp")
        .unique(subset=["block_number"], keep="first", maintain_order=True)
        .sort("block_number")
    )


def _max_allowed_offset(
    *,
    max_wait_seconds: int,
    slot_spacing_seconds: float,
    action_width: int,
) -> int:
    if slot_spacing_seconds <= 0:
        raise ValueError("slot_spacing_seconds must be positive")
    if max_wait_seconds < 0:
        raise ValueError("max_wait_seconds must be non-negative")
    return min(action_width - 1, max(0, math.floor(max_wait_seconds / slot_spacing_seconds)))


def _unix_now() -> int:
    return int(datetime.now(tz=UTC).timestamp())
