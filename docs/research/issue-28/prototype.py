"""Interactive disposable probe for Issue 28's fixed-context data contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from historical_dataset import (
    FeatureFitProvenance,
    FeatureState,
    HistoricalArrays,
    HistoricalDataset,
    TargetFitProvenance,
    TargetState,
    earliest_minimum,
    fit_feature_state,
    fit_target_state,
    prepare_arrays,
    prepare_live_input,
)
from torch.utils.data import DataLoader

C = 200
K = 5
FEATURE_NAMES = ("log_base_fee", "gas_utilization", "hour_sin")


@dataclass(frozen=True, slots=True)
class Fixture:
    raw_rows: np.ndarray
    base_fees: np.ndarray
    block_numbers: np.ndarray
    feature_provenance: FeatureFitProvenance
    target_provenance: TargetFitProvenance
    feature_state: FeatureState
    target_state: TargetState
    feature_support: np.ndarray
    arrays: HistoricalArrays
    training: HistoricalDataset
    validation: HistoricalDataset
    testing: HistoricalDataset
    origins: dict[str, np.ndarray]


def build_fixture() -> Fixture:
    row_count = 260
    positions = np.arange(row_count, dtype=np.float64)
    raw_rows = np.column_stack(
        (
            np.log(1_000_000.0 + 97.0 * positions),
            0.35 + (positions % 23.0) / 100.0,
            np.sin(positions / 7.0),
        )
    ).astype(np.float64)
    block_numbers = np.arange(10_000, 10_000 + row_count, dtype=np.int64)
    base_fees = (1_000_000 + ((np.arange(row_count, dtype=np.int64) * 137) % 20_003)).astype(
        np.int64
    )
    origins = {
        "training": np.arange(210, 219, dtype=np.int64),
        "validation": np.arange(225, 229, dtype=np.int64),
        "testing": np.arange(235, 238, dtype=np.int64),
    }
    first = int(origins["training"][0])
    base_fees[first + 1 : first + 1 + K] = np.array(
        [1_100_000, 900_000, 900_000, 1_200_000, 1_300_000],
        dtype=np.int64,
    )

    feature_support = np.arange(
        int(origins["training"][0]) - C + 1,
        int(origins["training"][-1]) + 1,
        dtype=np.int64,
    )
    feature_provenance = FeatureFitProvenance(
        corpus_id="synthetic:issue-28",
        chain_id=1,
        regime="synthetic-fixed-context",
        first_block=int(block_numbers[feature_support[0]]),
        last_block=int(block_numbers[feature_support[-1]]),
        count=int(feature_support.size),
    )
    target_provenance = TargetFitProvenance(
        corpus_id="synthetic:issue-28",
        chain_id=1,
        regime="synthetic-fixed-context",
        first_origin_block=int(block_numbers[origins["training"][0]]),
        last_origin_block=int(block_numbers[origins["training"][-1]]),
        count=int(origins["training"].size),
        k=K,
    )
    feature_state = fit_feature_state(
        raw_rows,
        feature_support,
        block_numbers,
        names=FEATURE_NAMES,
        provenance=feature_provenance,
    )
    target_state = fit_target_state(
        base_fees,
        origins["training"],
        block_numbers,
        k=K,
        provenance=target_provenance,
        chunk_size=3,
    )
    arrays = prepare_arrays(
        raw_rows,
        base_fees,
        block_numbers,
        names=FEATURE_NAMES,
        feature_state=feature_state,
        target_state=target_state,
        corpus_id=feature_provenance.corpus_id,
        chain_id=feature_provenance.chain_id,
        regime=feature_provenance.regime,
        feature_support_positions=feature_support,
        training_origins=origins["training"],
        k=K,
    )
    return Fixture(
        raw_rows=raw_rows,
        base_fees=base_fees,
        block_numbers=block_numbers,
        feature_provenance=feature_provenance,
        target_provenance=target_provenance,
        feature_state=feature_state,
        target_state=target_state,
        feature_support=feature_support,
        arrays=arrays,
        training=HistoricalDataset(arrays, origins["training"], c=C, k=K, chunk_size=3),
        validation=HistoricalDataset(arrays, origins["validation"], c=C, k=K, chunk_size=3),
        testing=HistoricalDataset(arrays, origins["testing"], c=C, k=K, chunk_size=3),
        origins=origins,
    )


def audit_contract(fixture: Fixture) -> dict[str, Any]:
    item = fixture.training[0]
    before = fixture.training[0]["inputs"]
    item["inputs"][0, 0] += 1.0
    mutation_isolated = torch.equal(before, fixture.training[0]["inputs"])

    first_origin = int(fixture.origins["training"][0])
    raw_context = fixture.raw_rows[first_origin - C + 1 : first_origin + 1]
    live = prepare_live_input(
        raw_context,
        names=FEATURE_NAMES,
        state=fixture.feature_state,
        c=C,
    )
    offline_live_equal = torch.equal(live[0], fixture.training[0]["inputs"])

    duplicate_support = np.concatenate(
        [
            np.arange(origin - C + 1, origin + 1, dtype=np.int64)
            for origin in fixture.origins["training"]
        ]
    )
    direct_population = fixture.raw_rows[fixture.feature_support]
    direct_means = direct_population.mean(axis=0, dtype=np.float64)
    direct_scales = direct_population.std(axis=0, ddof=0, dtype=np.float64)
    repeated_population = fixture.raw_rows[duplicate_support]
    repeated_means = repeated_population.mean(axis=0, dtype=np.float64)
    repeated_scales = repeated_population.std(axis=0, ddof=0, dtype=np.float64)
    direct_fit_equal = np.array_equal(direct_means, fixture.feature_state.means) and np.array_equal(
        direct_scales,
        fixture.feature_state.scales,
    )
    multiplicity_fit_differs = not (
        np.array_equal(repeated_means, fixture.feature_state.means)
        and np.array_equal(repeated_scales, fixture.feature_state.scales)
    )

    labels, _ = earliest_minimum(
        fixture.base_fees,
        fixture.origins["training"][:1],
        k=K,
        chunk_size=1,
    )
    total_origins = sum(len(value) for value in fixture.origins.values())
    shared_bytes = fixture.arrays.shared_bytes
    hypothetical_context_bytes = total_origins * C * len(FEATURE_NAMES) * 4
    hypothetical_outcome_bytes = total_origins * K * 8

    result = {
        "shared_storage": {
            "same_object_for_three_roles": (
                fixture.training._arrays is fixture.validation._arrays is fixture.testing._arrays
            ),
            "inputs_shape": list(fixture.arrays.input_shape),
            "base_fee_shape": list(fixture.arrays.base_fee_shape),
            "persistent_shared_bytes": shared_bytes,
            "not_materialized_context_bytes": hypothetical_context_bytes,
            "not_materialized_outcome_bytes": hypothetical_outcome_bytes,
            "item_mutation_isolated": mutation_isolated,
        },
        "item": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
            for key, value in fixture.training[0].items()
        },
        "first_exact_tie_label": int(labels[0]),
        "direct_unique_physical_row_fit": direct_fit_equal,
        "origin_multiplicity_is_excluded": multiplicity_fit_differs,
        "offline_live_feature_parity": offline_live_equal,
        "item_inputs_contiguous": fixture.training[0]["inputs"].is_contiguous(),
    }
    _require(all(_flatten_bools(result)), "contract audit failed")
    _require(result["first_exact_tie_label"] == 1, "earliest raw tie did not select first index")
    return result


def show_batches(fixture: Fixture) -> dict[str, Any]:
    loader = DataLoader(
        fixture.training,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    batches = list(loader)
    result = {
        "batch_sizes": [int(batch["label"].shape[0]) for batch in batches],
        "first_batch_shapes": {key: list(value.shape) for key, value in batches[0].items()},
        "default_mapping_collation": isinstance(batches[0], dict),
        "collated_inputs_contiguous": batches[0]["inputs"].is_contiguous(),
        "all_cpu": all(value.device.type == "cpu" for batch in batches for value in batch.values()),
    }
    _require(result["batch_sizes"] == [4, 4, 1], "full/tail batch behavior changed")
    _require(
        result["first_batch_shapes"]["inputs"] == [4, C, len(FEATURE_NAMES)],
        "bad input shape",
    )
    _require(result["first_batch_shapes"]["base_fees"] == [4, K], "bad outcome shape")
    return result


def show_shuffle_and_resume(fixture: Fixture) -> dict[str, Any]:
    def new_loader() -> DataLoader:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(2026)
        return DataLoader(
            fixture.training,
            batch_size=4,
            shuffle=True,
            generator=generator,
            num_workers=0,
            drop_last=False,
        )

    loader = new_loader()
    epoch_1 = _origin_order(loader)
    epoch_2 = _origin_order(loader)
    restarted_epoch = _origin_order(new_loader())
    result = {
        "epoch_1": epoch_1,
        "epoch_2_same_loader": epoch_2,
        "fresh_seeded_loader": restarted_epoch,
        "same_seed_repeats_first_epoch": epoch_1 == restarted_epoch,
        "continued_generator_advances": epoch_1 != epoch_2,
        "resume_implication": (
            "Reconstructing and reseeding restarts the permutation sequence; Issue 16 permits this "
            "non-exact completed-validation-boundary continuation. Persist no loader state."
        ),
    }
    _require(result["same_seed_repeats_first_epoch"], "seeded first epoch was not deterministic")
    _require(result["continued_generator_advances"], "generator did not advance across epochs")
    return result


def show_workers(fixture: Fixture) -> dict[str, Any]:
    loader = DataLoader(
        fixture.validation,
        batch_size=3,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=False,
        drop_last=False,
    )
    batches = list(loader)
    observed_order = [int(block) for batch in batches for block in batch["origin_block"].tolist()]
    expected_order = fixture.block_numbers[fixture.origins["validation"]].tolist()
    result = {
        "num_workers": loader.num_workers,
        "prefetch_factor": loader.prefetch_factor,
        "persistent_workers": loader.persistent_workers,
        "batch_sizes": [int(batch["label"].shape[0]) for batch in batches],
        "sequential_origin_order": observed_order == expected_order,
        "worker_output_is_cpu": all(
            value.device.type == "cpu" for batch in batches for value in batch.values()
        ),
        "ownership": "DataLoader settings are direct ephemeral host inputs, not dataset state.",
    }
    _require(result["batch_sizes"] == [3, 1], "worker loader lost its tail batch")
    _require(result["sequential_origin_order"], "worker loader changed validation order")
    _require(result["worker_output_is_cpu"], "dataset worker produced a device tensor")
    del loader
    return result


def show_mps_boundary(fixture: Fixture) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        return {
            "available": False,
            "ownership": "No local device observation; target host remains #55/#26.",
        }
    batch = next(iter(DataLoader(fixture.validation, batch_size=2, shuffle=False)))
    moved = {key: batch[key].to("mps") for key in ("inputs", "label", "target")}
    torch.mps.synchronize()
    pin_result: str
    try:
        pinned = next(
            iter(
                DataLoader(
                    fixture.validation,
                    batch_size=2,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                )
            )
        )
        pin_result = f"is_pinned={pinned['inputs'].is_pinned()}"
    except RuntimeError as error:
        pin_result = f"RuntimeError: {error}"
    result = {
        "available": True,
        "direct_pytorch_smoke_moved_fields": {
            key: str(value.device) for key, value in moved.items()
        },
        "raw_outcomes_remained": str(batch["base_fees"].device),
        "origin_blocks_remained": str(batch["origin_block"].device),
        "pin_memory_local_observation": pin_result,
        "ownership": (
            "This MPS smoke does not prescribe field-selective transfer. #26/#55 choose native "
            "host transfer and CUDA pinning, including Lightning recursive transfer if selected."
        ),
    }
    del moved
    return result


def show_failures(fixture: Fixture) -> dict[str, str]:
    failures: dict[str, str] = {}

    def capture(name: str, action: Callable[[], object]) -> None:
        try:
            action()
        except (TypeError, ValueError) as error:
            failures[name] = f"{type(error).__name__}: {error}"
        else:
            raise RuntimeError(f"{name} did not fail closed")

    nonfinite = fixture.raw_rows.copy()
    nonfinite[0, 0] = np.nan
    support = np.arange(11, 219, dtype=np.int64)
    capture(
        "nonfinite_feature_fit",
        lambda: fit_feature_state(
            nonfinite,
            support,
            fixture.block_numbers,
            names=FEATURE_NAMES,
            provenance=fixture.feature_provenance,
        ),
    )
    capture(
        "duplicate_feature_support",
        lambda: fit_feature_state(
            fixture.raw_rows,
            np.concatenate((support, support[-1:])),
            fixture.block_numbers,
            names=FEATURE_NAMES,
            provenance=fixture.feature_provenance,
        ),
    )
    constant = fixture.raw_rows.copy()
    constant[:, 0] = 1.0
    capture(
        "zero_feature_scale",
        lambda: fit_feature_state(
            constant,
            support,
            fixture.block_numbers,
            names=FEATURE_NAMES,
            provenance=fixture.feature_provenance,
        ),
    )
    constant_fees = np.full(fixture.base_fees.shape, 1_000_000, dtype=np.int64)
    capture(
        "zero_target_scale",
        lambda: fit_target_state(
            constant_fees,
            fixture.origins["training"],
            fixture.block_numbers,
            k=K,
            provenance=fixture.target_provenance,
        ),
    )
    capture(
        "bad_feature_provenance",
        lambda: prepare_arrays(
            fixture.raw_rows,
            fixture.base_fees,
            fixture.block_numbers,
            names=FEATURE_NAMES,
            feature_state=fixture.feature_state,
            target_state=fixture.target_state,
            corpus_id="wrong",
            chain_id=fixture.feature_provenance.chain_id,
            regime=fixture.feature_provenance.regime,
            feature_support_positions=fixture.feature_support,
            training_origins=fixture.origins["training"],
            k=K,
        ),
    )
    capture(
        "bad_target_provenance",
        lambda: prepare_arrays(
            fixture.raw_rows,
            fixture.base_fees,
            fixture.block_numbers,
            names=FEATURE_NAMES,
            feature_state=fixture.feature_state,
            target_state=TargetState(
                mean=fixture.target_state.mean,
                scale=fixture.target_state.scale,
                provenance=replace(
                    fixture.target_provenance,
                    count=fixture.target_provenance.count + 1,
                ),
            ),
            corpus_id=fixture.feature_provenance.corpus_id,
            chain_id=fixture.feature_provenance.chain_id,
            regime=fixture.feature_provenance.regime,
            feature_support_positions=fixture.feature_support,
            training_origins=fixture.origins["training"],
            k=K,
        ),
    )
    capture(
        "cross_state_identity",
        lambda: prepare_arrays(
            fixture.raw_rows,
            fixture.base_fees,
            fixture.block_numbers,
            names=FEATURE_NAMES,
            feature_state=fixture.feature_state,
            target_state=TargetState(
                mean=fixture.target_state.mean,
                scale=fixture.target_state.scale,
                provenance=replace(fixture.target_provenance, chain_id=137),
            ),
            corpus_id=fixture.feature_provenance.corpus_id,
            chain_id=fixture.feature_provenance.chain_id,
            regime=fixture.feature_provenance.regime,
            feature_support_positions=fixture.feature_support,
            training_origins=fixture.origins["training"],
            k=K,
        ),
    )
    capture(
        "wrong_feature_state_dtype",
        lambda: FeatureState(
            names=FEATURE_NAMES,
            means=fixture.feature_state.means.astype(np.float32),
            scales=fixture.feature_state.scales,
            provenance=fixture.feature_provenance,
        ),
    )
    capture(
        "wrong_target_state_dtype",
        lambda: TargetState(
            mean=np.float32(fixture.target_state.mean),
            scale=fixture.target_state.scale,
            provenance=fixture.target_provenance,
        ),
    )
    capture(
        "nonfinite_float32_target",
        lambda: TargetState(
            mean=np.float64(-1e308),
            scale=np.float64(1.0),
            provenance=fixture.target_provenance,
        ).transform(np.array([1], dtype=np.int64)),
    )
    no_outcome = np.array([fixture.base_fees.size - K], dtype=np.int64)
    capture(
        "incomplete_outcome",
        lambda: earliest_minimum(fixture.base_fees, no_outcome, k=K),
    )
    nonpositive = fixture.base_fees.copy()
    nonpositive[0] = 0
    capture(
        "nonpositive_raw_fee",
        lambda: earliest_minimum(nonpositive, fixture.origins["training"], k=K),
    )
    return failures


def run_all(fixture: Fixture) -> dict[str, Any]:
    return {
        "contract": audit_contract(fixture),
        "batches": show_batches(fixture),
        "shuffle_resume": show_shuffle_and_resume(fixture),
        "workers": show_workers(fixture),
        "mps": show_mps_boundary(fixture),
        "failures": show_failures(fixture),
    }


def interactive(fixture: Fixture) -> None:
    actions: dict[str, tuple[str, Callable[[Fixture], dict[str, Any]]]] = {
        "a": ("audit storage/state/parity", audit_contract),
        "b": ("show default full/tail batches", show_batches),
        "s": ("show seeded shuffle/resume", show_shuffle_and_resume),
        "w": ("exercise worker/prefetch/persistence", show_workers),
        "m": ("exercise local MPS host boundary", show_mps_boundary),
        "f": ("show fail-closed cases", show_failures),
    }
    last: dict[str, Any] = {
        "question": (
            "Does one lazy fixed-C map-style HistoricalDataset plus ordinary DataLoader cover the "
            "approved tensor, state, ordering, worker, and transfer contract?"
        ),
        "fixture": {
            "C": C,
            "K": K,
            "roles": {key: len(value) for key, value in fixture.origins.items()},
        },
    }
    while True:
        print("\033[2J\033[H", end="")
        print("\033[1mIssue 28 disposable prototype\033[0m")
        print(json.dumps(last, indent=2))
        print("\n\033[1m[a]\033[0m audit  \033[1m[b]\033[0m batches  \033[1m[s]\033[0m shuffle")
        print(
            "\033[1m[w]\033[0m workers  \033[1m[m]\033[0m MPS  "
            "\033[1m[f]\033[0m failures  \033[1m[q]\033[0m quit"
        )
        choice = input("> ").strip().lower()
        if choice == "q":
            return
        selected = actions.get(choice)
        if selected is None:
            last = {"error": f"unknown action {choice!r}"}
            continue
        label, action = selected
        try:
            last = {"action": label, "result": action(fixture)}
        except Exception as error:  # Prototype shell: surface failures without hiding state.
            last = {"action": label, "error": f"{type(error).__name__}: {error}"}


def _origin_order(loader: DataLoader) -> list[int]:
    return [int(block) for batch in loader for block in batch["origin_block"].tolist()]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _flatten_bools(value: object):
    if isinstance(value, bool):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _flatten_bools(child)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="run every bounded probe and print JSON")
    args = parser.parse_args()
    fixture = build_fixture()
    if args.all:
        print(json.dumps(run_all(fixture), indent=2))
        return
    interactive(fixture)


if __name__ == "__main__":
    main()
