from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from web3.exceptions import Web3RPCError

from spice.acquisition.datasets import (
    ensure_evaluation_dataset,
    ensure_history_dataset,
)
from spice.acquisition.metadata import (
    build_dataset_metadata,
    load_dataset_metadata,
    provider_metadata,
)
from spice.acquisition.rpc import (
    AcquisitionRuntimeSnapshot,
    BlockPullPlan,
    BlockRange,
    RpcController,
    TimestampRange,
    Web3BlockClient,
)
from spice.acquisition.windowing import required_history_block_count
from spice.core.console import NullReporter
from spice.core.json import write_json
from spice.data.io import load_block_frame
from spice.data.validation import validate_contiguous_block_frame
from spice.workflows.acquire import run as run_acquire
from tests.support import (
    base_overrides,
    compose_experiment,
    make_block_rows,
    write_dataset_dir,
)


def _window_for_rows(
    rows: list[dict[str, int | None]],
    *,
    block_time_seconds: int = 12,
) -> TimestampRange:
    start_timestamp = int(rows[0]["timestamp"])
    end_timestamp = int(rows[-1]["timestamp"]) + block_time_seconds
    return TimestampRange(start=start_timestamp, end=end_timestamp)


def _rpc_controller(
    *,
    batch_size: int = 8,
    min_batch_size: int = 2,
    concurrency: int = 2,
    concurrency_rungs: tuple[int, ...] = (1, 2),
) -> RpcController:
    return RpcController(
        configured_batch_size=batch_size,
        min_batch_size=min_batch_size,
        concurrency_rungs=concurrency_rungs,
        configured_concurrency=concurrency,
    )


def _runtime_snapshot() -> AcquisitionRuntimeSnapshot:
    return AcquisitionRuntimeSnapshot(
        configured_batch_size=8,
        final_batch_size=8,
        min_batch_size=2,
        configured_concurrency=2,
        final_concurrency=2,
        concurrency_rungs=(1, 2),
        oversize_error_count=0,
        transient_error_count=0,
        oversize_backoffs=0,
        transient_backoffs=0,
        concurrency_recoveries=0,
    )


def _write_metadata(config, *, providers) -> None:
    history_validation = validate_contiguous_block_frame(
        load_block_frame(config.paths.history_dir),
        dataset_path=config.paths.history_dir,
        expected_chain_id=config.chain.chain_id,
    )
    evaluation_validation = validate_contiguous_block_frame(
        load_block_frame(config.paths.evaluation_dir),
        dataset_path=config.paths.evaluation_dir,
        expected_chain_id=config.chain.chain_id,
    )
    metadata = build_dataset_metadata(
        config=config,
        history_dir=config.paths.history_dir,
        evaluation_dir=config.paths.evaluation_dir,
        providers=list(providers),
        history_validation=history_validation,
        evaluation_validation=evaluation_validation,
        acquisition_runtime=_runtime_snapshot(),
    )
    write_json(config.paths.dataset_metadata_path, metadata)


def test_ensure_history_dataset_extends_missing_prefix_only(tmp_path) -> None:
    config = compose_experiment(
        "acquire",
        overrides=base_overrides(tmp_path)
        + [
            "acquisition.chunk_size=2",
            "dataset.temporal.lookback_seconds=24",
            "dataset.temporal.max_delay_seconds=12",
            "dataset.sampling.sample_count=4",
            "acquisition.history_sample_budget=4",
        ],
    )
    output_dir = tmp_path / "history"
    working_dir = tmp_path / "work"
    existing_rows = make_block_rows(
        2,
        start_block=102,
        start_timestamp=1_700_000_024,
        include_gas_limit=True,
    )
    write_dataset_dir(output_dir, existing_rows)
    history_plan = BlockPullPlan(
        window=TimestampRange(start=1_700_000_000, end=1_700_000_048),
        block_range=BlockRange(start=100, end=104),
        expected_rows=4,
        expected_files=2,
    )

    class PrefixClient:
        def __init__(self) -> None:
            self.pulled_ranges: list[BlockRange] = []

        def plan_block_range(
            self,
            block_range: BlockRange,
            *,
            window: TimestampRange,
            chunk_size: int,
        ) -> BlockPullPlan:
            return BlockPullPlan(
                window=window,
                block_range=block_range,
                expected_rows=block_range.count,
                expected_files=max(1, (block_range.count + chunk_size - 1) // chunk_size),
            )

        async def pull_block_range(
            self,
            output_dir: Path,
            *,
            plan: BlockPullPlan,
            chunk_size: int,
            rpc_controller: RpcController,
            reporter,
        ) -> BlockPullPlan:
            del chunk_size, rpc_controller, reporter
            self.pulled_ranges.append(plan.block_range)
            rows = make_block_rows(
                plan.expected_rows,
                start_block=plan.block_range.start,
                start_timestamp=1_700_000_000,
                include_gas_limit=True,
            )
            write_dataset_dir(output_dir, rows)
            return plan

    client = PrefixClient()
    result = asyncio.run(
        ensure_history_dataset(
            config=config,
            block_client=client,
            output_dir=output_dir,
            working_dir=working_dir,
            history_plan=history_plan,
            required_history_blocks=4,
            rpc_controller=_rpc_controller(batch_size=8),
            reporter=NullReporter(),
        )
    )

    assert client.pulled_ranges == [BlockRange(start=100, end=102)]
    assert result.pulled_blocks is True
    assert result.reused is False
    assert load_block_frame(result.path)["block_number"].to_list() == [100, 101, 102, 103]


def test_ensure_evaluation_dataset_reuses_overlap_and_fetches_missing_edges_only(tmp_path) -> None:
    config = compose_experiment(
        "acquire",
        overrides=base_overrides(tmp_path) + ["acquisition.chunk_size=2"],
    )
    output_dir = tmp_path / "evaluation"
    working_dir = tmp_path / "work"
    existing_rows = make_block_rows(
        3,
        start_block=101,
        start_timestamp=1_700_000_012,
        include_gas_limit=True,
    )
    write_dataset_dir(output_dir, existing_rows)
    evaluation_plan = BlockPullPlan(
        window=TimestampRange(start=1_700_000_000, end=1_700_000_060),
        block_range=BlockRange(start=100, end=105),
        expected_rows=5,
        expected_files=3,
    )

    class OverlapClient:
        def __init__(self) -> None:
            self.pulled_ranges: list[BlockRange] = []

        def plan_block_range(
            self,
            block_range: BlockRange,
            *,
            window: TimestampRange,
            chunk_size: int,
        ) -> BlockPullPlan:
            return BlockPullPlan(
                window=window,
                block_range=block_range,
                expected_rows=block_range.count,
                expected_files=max(1, (block_range.count + chunk_size - 1) // chunk_size),
            )

        async def pull_block_range(
            self,
            output_dir: Path,
            *,
            plan: BlockPullPlan,
            chunk_size: int,
            rpc_controller: RpcController,
            reporter,
        ) -> BlockPullPlan:
            del chunk_size, rpc_controller, reporter
            self.pulled_ranges.append(plan.block_range)
            start_timestamp = 1_700_000_000 if plan.block_range.start == 100 else 1_700_000_048
            rows = make_block_rows(
                plan.expected_rows,
                start_block=plan.block_range.start,
                start_timestamp=start_timestamp,
                include_gas_limit=True,
            )
            write_dataset_dir(output_dir, rows)
            return plan

    client = OverlapClient()
    result = asyncio.run(
        ensure_evaluation_dataset(
            config=config,
            block_client=client,
            output_dir=output_dir,
            working_dir=working_dir,
            evaluation_plan=evaluation_plan,
            rpc_controller=_rpc_controller(batch_size=8),
            reporter=NullReporter(),
        )
    )

    assert client.pulled_ranges == [BlockRange(start=100, end=101), BlockRange(start=104, end=105)]
    assert result.pulled_blocks is True
    assert load_block_frame(result.path)["block_number"].to_list() == [100, 101, 102, 103, 104]


def test_acquire_reuses_valid_canonical_blocks_across_provider_change(
    tmp_path,
    monkeypatch,
) -> None:
    config = compose_experiment(
        "acquire",
        overrides=base_overrides(tmp_path)
        + [
            "dataset.temporal.lookback_seconds=24",
            "dataset.temporal.max_delay_seconds=12",
            "dataset.sampling.sample_count=4",
            "acquisition.history_sample_budget=4",
        ],
    )
    previous_provider_config = compose_experiment(
        "acquire",
        overrides=base_overrides(tmp_path)
        + [
            "presets.provider=alchemy",
            "dataset.temporal.lookback_seconds=24",
            "dataset.temporal.max_delay_seconds=12",
            "dataset.sampling.sample_count=4",
            "acquisition.history_sample_budget=4",
        ],
    )
    required_history_blocks = required_history_block_count(config)
    history_rows = make_block_rows(
        required_history_blocks,
        start_block=100,
        start_timestamp=(
            config.evaluation_window_start_timestamp - required_history_blocks * 12
        ),
        include_gas_limit=True,
    )
    evaluation_rows = make_block_rows(
        5,
        start_block=10_001,
        start_timestamp=config.evaluation_window_start_timestamp,
        include_gas_limit=True,
    )
    write_dataset_dir(config.paths.history_dir, history_rows)
    write_dataset_dir(config.paths.evaluation_dir, evaluation_rows)
    _write_metadata(
        config,
        providers=[provider_metadata(previous_provider_config)],
    )

    history_plan = BlockPullPlan(
        window=_window_for_rows(history_rows),
        block_range=BlockRange(start=100, end=100 + required_history_blocks),
        expected_rows=required_history_blocks,
        expected_files=1,
    )
    evaluation_plan = BlockPullPlan(
        window=TimestampRange(
            start=config.evaluation_window_start_timestamp,
            end=config.evaluation_window_end_timestamp,
        ),
        block_range=BlockRange(start=10_001, end=10_006),
        expected_rows=5,
        expected_files=1,
    )

    class NoPullAcquireClient:
        def __init__(self, provider, chain) -> None:
            del provider
            self.chain = chain

        async def close(self) -> None:
            return None

        async def plan_history_window(
            self,
            *,
            end_timestamp: int,
            required_history_blocks: int,
            chunk_size: int,
        ) -> BlockPullPlan:
            del end_timestamp, required_history_blocks, chunk_size
            return history_plan

        async def plan_window(self, window: TimestampRange, *, chunk_size: int) -> BlockPullPlan:
            del chunk_size
            if window == evaluation_plan.window:
                return evaluation_plan
            return history_plan

        def plan_block_range(
            self,
            block_range: BlockRange,
            *,
            window: TimestampRange,
            chunk_size: int,
        ) -> BlockPullPlan:
            return BlockPullPlan(
                window=window,
                block_range=block_range,
                expected_rows=block_range.count,
                expected_files=max(1, (block_range.count + chunk_size - 1) // chunk_size),
            )

        async def pull_block_range(self, *_args, **_kwargs):
            raise AssertionError("provider change alone should not trigger a pull")

    monkeypatch.setattr("spice.workflows.acquire.Web3BlockClient", NoPullAcquireClient)

    run_acquire(config, reporter=NullReporter())

    metadata = load_dataset_metadata(config.paths.dataset_metadata_path)
    assert metadata is not None
    assert [provider.name for provider in metadata.providers] == ["alchemy"]


def test_acquire_extension_appends_new_provider_once(tmp_path, monkeypatch) -> None:
    config = compose_experiment(
        "acquire",
        overrides=base_overrides(tmp_path)
        + [
            "dataset.temporal.lookback_seconds=24",
            "dataset.temporal.max_delay_seconds=12",
            "dataset.sampling.sample_count=4",
            "acquisition.history_sample_budget=4",
        ],
    )
    previous_provider_config = compose_experiment(
        "acquire",
        overrides=base_overrides(tmp_path)
        + [
            "presets.provider=alchemy",
            "dataset.temporal.lookback_seconds=24",
            "dataset.temporal.max_delay_seconds=12",
            "dataset.sampling.sample_count=4",
            "acquisition.history_sample_budget=4",
        ],
    )
    required_history_blocks = required_history_block_count(config)
    history_rows = make_block_rows(
        required_history_blocks - 2,
        start_block=102,
        start_timestamp=(
            config.evaluation_window_start_timestamp - (required_history_blocks - 2) * 12
        ),
        include_gas_limit=True,
    )
    evaluation_rows = make_block_rows(
        5,
        start_block=10_001,
        start_timestamp=config.evaluation_window_start_timestamp,
        include_gas_limit=True,
    )
    write_dataset_dir(config.paths.history_dir, history_rows)
    write_dataset_dir(config.paths.evaluation_dir, evaluation_rows)
    _write_metadata(
        config,
        providers=[provider_metadata(previous_provider_config)],
    )

    history_plan = BlockPullPlan(
        window=TimestampRange(
            start=config.evaluation_window_start_timestamp - required_history_blocks * 12,
            end=config.evaluation_window_start_timestamp,
        ),
        block_range=BlockRange(start=100, end=100 + required_history_blocks),
        expected_rows=required_history_blocks,
        expected_files=max(1, (required_history_blocks + 8191) // 8192),
    )
    evaluation_plan = BlockPullPlan(
        window=TimestampRange(
            start=config.evaluation_window_start_timestamp,
            end=config.evaluation_window_end_timestamp,
        ),
        block_range=BlockRange(start=10_001, end=10_006),
        expected_rows=5,
        expected_files=1,
    )

    class PrefixAcquireClient:
        def __init__(self, provider, chain) -> None:
            del provider
            self.chain = chain

        async def close(self) -> None:
            return None

        async def plan_history_window(
            self,
            *,
            end_timestamp: int,
            required_history_blocks: int,
            chunk_size: int,
        ) -> BlockPullPlan:
            del end_timestamp, required_history_blocks, chunk_size
            return history_plan

        async def plan_window(self, window: TimestampRange, *, chunk_size: int) -> BlockPullPlan:
            del chunk_size
            if window == evaluation_plan.window:
                return evaluation_plan
            return history_plan

        def plan_block_range(
            self,
            block_range: BlockRange,
            *,
            window: TimestampRange,
            chunk_size: int,
        ) -> BlockPullPlan:
            return BlockPullPlan(
                window=window,
                block_range=block_range,
                expected_rows=block_range.count,
                expected_files=max(1, (block_range.count + chunk_size - 1) // chunk_size),
            )

        async def pull_block_range(
            self,
            output_dir: Path,
            *,
            plan: BlockPullPlan,
            chunk_size: int,
            rpc_controller: RpcController,
            reporter,
        ) -> BlockPullPlan:
            del chunk_size, rpc_controller, reporter
            rows = make_block_rows(
                plan.expected_rows,
                start_block=plan.block_range.start,
                start_timestamp=(
                    config.evaluation_window_start_timestamp - required_history_blocks * 12
                ),
                include_gas_limit=True,
            )
            write_dataset_dir(output_dir, rows)
            return plan

    monkeypatch.setattr("spice.workflows.acquire.Web3BlockClient", PrefixAcquireClient)

    run_acquire(config, reporter=NullReporter())

    metadata = load_dataset_metadata(config.paths.dataset_metadata_path)
    assert metadata is not None
    assert [provider.name for provider in metadata.providers] == ["alchemy", "publicnode"]
    history_frame = load_block_frame(config.paths.history_dir)
    assert history_frame["block_number"][0] == 100
    assert history_frame["block_number"][-1] == 100 + required_history_blocks - 1
    assert history_frame.height == required_history_blocks


def test_acquire_failure_preserves_last_good_canonical_dataset(tmp_path, monkeypatch) -> None:
    config = compose_experiment(
        "acquire",
        overrides=base_overrides(tmp_path)
        + [
            "dataset.temporal.lookback_seconds=24",
            "dataset.temporal.max_delay_seconds=12",
            "dataset.sampling.sample_count=4",
            "acquisition.history_sample_budget=4",
        ],
    )
    required_history_blocks = required_history_block_count(config)
    history_rows = make_block_rows(
        required_history_blocks - 2,
        start_block=102,
        start_timestamp=(
            config.evaluation_window_start_timestamp - (required_history_blocks - 2) * 12
        ),
        include_gas_limit=True,
    )
    evaluation_rows = make_block_rows(
        5,
        start_block=10_001,
        start_timestamp=config.evaluation_window_start_timestamp,
        include_gas_limit=True,
    )
    write_dataset_dir(config.paths.history_dir, history_rows)
    write_dataset_dir(config.paths.evaluation_dir, evaluation_rows)

    history_plan = BlockPullPlan(
        window=TimestampRange(
            start=config.evaluation_window_start_timestamp - required_history_blocks * 12,
            end=config.evaluation_window_start_timestamp,
        ),
        block_range=BlockRange(start=100, end=100 + required_history_blocks),
        expected_rows=required_history_blocks,
        expected_files=max(1, (required_history_blocks + 8191) // 8192),
    )
    evaluation_plan = BlockPullPlan(
        window=TimestampRange(
            start=config.evaluation_window_start_timestamp,
            end=config.evaluation_window_end_timestamp,
        ),
        block_range=BlockRange(start=10_001, end=10_006),
        expected_rows=5,
        expected_files=1,
    )

    class FailingAcquireClient:
        def __init__(self, provider, chain) -> None:
            del provider
            self.chain = chain

        async def close(self) -> None:
            return None

        async def plan_history_window(
            self,
            *,
            end_timestamp: int,
            required_history_blocks: int,
            chunk_size: int,
        ) -> BlockPullPlan:
            del end_timestamp, required_history_blocks, chunk_size
            return history_plan

        async def plan_window(self, window: TimestampRange, *, chunk_size: int) -> BlockPullPlan:
            del chunk_size
            if window == evaluation_plan.window:
                return evaluation_plan
            return history_plan

        def plan_block_range(
            self,
            block_range: BlockRange,
            *,
            window: TimestampRange,
            chunk_size: int,
        ) -> BlockPullPlan:
            return BlockPullPlan(
                window=window,
                block_range=block_range,
                expected_rows=block_range.count,
                expected_files=max(1, (block_range.count + chunk_size - 1) // chunk_size),
            )

        async def pull_block_range(
            self,
            output_dir: Path,
            *,
            plan: BlockPullPlan,
            chunk_size: int,
            rpc_controller: RpcController,
            reporter,
        ) -> BlockPullPlan:
            del chunk_size, rpc_controller, reporter
            write_dataset_dir(
                output_dir,
                make_block_rows(
                    plan.expected_rows,
                    start_block=plan.block_range.start,
                    start_timestamp=(
                        config.evaluation_window_start_timestamp - required_history_blocks * 12
                    ),
                    include_gas_limit=True,
                ),
            )
            raise RuntimeError("rpc exploded")

    monkeypatch.setattr("spice.workflows.acquire.Web3BlockClient", FailingAcquireClient)

    with pytest.raises(RuntimeError, match="rpc exploded"):
        run_acquire(config, reporter=NullReporter())

    history_frame = load_block_frame(config.paths.history_dir)
    assert history_frame["block_number"][0] == 102
    assert history_frame["block_number"][-1] == 100 + required_history_blocks - 1
    assert history_frame.height == required_history_blocks - 2
    assert load_block_frame(config.paths.evaluation_dir)["block_number"].to_list() == [
        10_001,
        10_002,
        10_003,
        10_004,
        10_005,
    ]


def test_web3_block_client_pull_block_range_retries_oversized_batches_and_preserves_order(
    tmp_path,
    monkeypatch,
) -> None:
    timestamps = {block: 100 + block * 12 for block in range(6)}

    class FakeBatch:
        def __init__(self) -> None:
            self.requests: list[object] = []

        async def __aenter__(self) -> FakeBatch:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def add(self, block: object) -> None:
            self.requests.append(block)

        async def async_execute(self) -> list[dict[str, int]]:
            blocks = [await request for request in self.requests]
            if len(blocks) > 2:
                raise Web3RPCError(
                    "response too large",
                    rpc_response={"error": {"code": -32003, "message": "response too large"}},
                )
            return blocks

    class FakeEth:
        async def get_block(
            self,
            block_number: int | str,
            _full_transactions: bool = False,
        ) -> dict[str, int]:
            number = int(block_number)
            return {
                "number": number,
                "timestamp": timestamps[number],
                "baseFeePerGas": 1_000_000_000 + number,
                "gasUsed": 20_000_000 + number,
                "gasLimit": 30_000_000 + number,
            }

    class FakeWeb3:
        eth = FakeEth()

        def batch_requests(self) -> FakeBatch:
            return FakeBatch()

    monkeypatch.setattr(
        "spice.acquisition.rpc.build_async_web3",
        lambda _provider, _chain: FakeWeb3(),
    )

    client = Web3BlockClient(
        provider=compose_experiment("acquire", overrides=base_overrides(tmp_path)).provider,
        chain=compose_experiment("acquire", overrides=base_overrides(tmp_path)).chain,
    )
    plan = BlockPullPlan(
        window=TimestampRange(start=112, end=172),
        block_range=BlockRange(start=1, end=6),
        expected_rows=5,
        expected_files=2,
    )
    controller = _rpc_controller(batch_size=4, min_batch_size=2, concurrency=2)

    pulled_plan = asyncio.run(
        client.pull_block_range(
            tmp_path / "history",
            plan=plan,
            chunk_size=3,
            rpc_controller=controller,
            reporter=NullReporter(),
        )
    )

    frame = load_block_frame(tmp_path / "history")

    assert pulled_plan == plan
    assert frame["block_number"].to_list() == [1, 2, 3, 4, 5]
    assert controller.current_batch_size == 2
    assert controller.oversize_error_count == 1
    assert controller.oversize_backoffs == 1


def test_web3_block_client_retries_single_transient_failure_without_concurrency_backoff(
    tmp_path,
    monkeypatch,
) -> None:
    attempts: dict[tuple[int, ...], int] = {}

    class FakeBatch:
        def __init__(self) -> None:
            self.requests: list[object] = []

        async def __aenter__(self) -> FakeBatch:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def add(self, block: object) -> None:
            self.requests.append(block)

        async def async_execute(self) -> list[dict[str, int]]:
            blocks = [await request for request in self.requests]
            key = tuple(int(block["number"]) for block in blocks)
            attempts[key] = attempts.get(key, 0) + 1
            if key == (0, 1) and attempts[key] == 1:
                raise TimeoutError("transient timeout")
            return blocks

    class FakeEth:
        async def get_block(
            self,
            block_number: int | str,
            _full_transactions: bool = False,
        ) -> dict[str, int]:
            number = int(block_number)
            return {
                "number": number,
                "timestamp": 100 + number * 12,
                "baseFeePerGas": 1_000_000_000 + number,
                "gasUsed": 20_000_000 + number,
                "gasLimit": 30_000_000 + number,
            }

    class FakeWeb3:
        eth = FakeEth()

        def batch_requests(self) -> FakeBatch:
            return FakeBatch()

    monkeypatch.setattr(
        "spice.acquisition.rpc.build_async_web3",
        lambda _provider, _chain: FakeWeb3(),
    )

    client = Web3BlockClient(
        provider=compose_experiment("acquire", overrides=base_overrides(tmp_path)).provider,
        chain=compose_experiment("acquire", overrides=base_overrides(tmp_path)).chain,
    )
    plan = BlockPullPlan(
        window=TimestampRange(start=100, end=148),
        block_range=BlockRange(start=0, end=4),
        expected_rows=4,
        expected_files=2,
    )
    controller = _rpc_controller(batch_size=2, min_batch_size=1, concurrency=2)

    asyncio.run(
        client.pull_block_range(
            tmp_path / "history",
            plan=plan,
            chunk_size=2,
            rpc_controller=controller,
            reporter=NullReporter(),
        )
    )

    assert controller.current_concurrency == 2
    assert controller.transient_error_count == 1
    assert controller.transient_backoffs == 0
    assert load_block_frame(tmp_path / "history")["block_number"].to_list() == [0, 1, 2, 3]
