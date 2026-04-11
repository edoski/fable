from __future__ import annotations

from pathlib import Path

from spice.acquisition.datasets import ensure_block_dataset, ensure_history_dataset
from spice.acquisition.metadata import load_dataset_metadata
from spice.acquisition.rpc import BlockPullPlan, BlockRange, TimestampRange, Web3BlockClient
from spice.acquisition.windowing import history_range_from_metadata, required_history_block_count
from spice.core.console import NullReporter
from spice.data.io import load_block_frame
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


def test_ensure_block_dataset_reuses_clean_output_without_pull(tmp_path) -> None:
    output_dir = tmp_path / "history"
    rows = make_block_rows(
        4,
        start_block=100,
        start_timestamp=1_700_000_000,
        include_gas_limit=True,
    )
    write_dataset_dir(output_dir, rows)
    plan = BlockPullPlan(
        window=_window_for_rows(rows),
        block_range=BlockRange(start=100, end=104),
        expected_rows=4,
        expected_files=2,
    )

    class NoPullClient:
        def pull_block_range(self, *_args, **_kwargs):
            raise AssertionError("clean dataset should be reused")

    pulled_plan, validation = ensure_block_dataset(
        block_client=NoPullClient(),
        output_dir=output_dir,
        plan=plan,
        expected_chain_id=1,
        chunk_size=2,
        rpc_batch_size=8,
        overwrite=False,
        reporter=NullReporter(),
    )

    assert pulled_plan is None
    assert validation.status == "clean"
    assert validation.row_count == 4


def test_ensure_block_dataset_rebuilds_invalid_existing_output(tmp_path) -> None:
    output_dir = tmp_path / "history"
    invalid_rows = make_block_rows(
        4,
        start_block=100,
        start_timestamp=1_700_000_000,
        include_gas_limit=True,
    )
    write_dataset_dir(output_dir, invalid_rows[:3] + [invalid_rows[2]])

    rebuilt_rows = make_block_rows(
        4,
        start_block=200,
        start_timestamp=1_700_000_100,
        include_gas_limit=True,
    )
    plan = BlockPullPlan(
        window=_window_for_rows(rebuilt_rows),
        block_range=BlockRange(start=200, end=204),
        expected_rows=4,
        expected_files=2,
    )

    class RebuildingClient:
        def __init__(self) -> None:
            self.calls = 0

        def pull_block_range(
            self,
            output_dir: Path,
            *,
            plan: BlockPullPlan,
            chunk_size: int,
            rpc_batch_size: int,
            reporter,
        ) -> BlockPullPlan:
            del reporter
            assert chunk_size == 2
            assert rpc_batch_size == 8
            assert plan.block_range == BlockRange(start=200, end=204)
            self.calls += 1
            write_dataset_dir(output_dir, rebuilt_rows)
            return plan

    client = RebuildingClient()
    pulled_plan, validation = ensure_block_dataset(
        block_client=client,
        output_dir=output_dir,
        plan=plan,
        expected_chain_id=1,
        chunk_size=2,
        rpc_batch_size=8,
        overwrite=False,
        reporter=NullReporter(),
    )

    assert client.calls == 1
    assert pulled_plan == plan
    assert validation.status == "clean"
    assert load_block_frame(output_dir)["block_number"].to_list() == [200, 201, 202, 203]


def test_ensure_history_dataset_expands_by_block_count_until_requirement_met(tmp_path) -> None:
    config = compose_experiment(
        "acquire",
        overrides=base_overrides(tmp_path) + ["acquisition.chunk_size=2"],
    )
    output_dir = tmp_path / "history"
    initial_plan = BlockPullPlan(
        window=TimestampRange(start=1_700_000_000, end=1_700_000_120),
        block_range=BlockRange(start=10, end=16),
        expected_rows=6,
        expected_files=3,
    )

    class ExpandingHistoryClient:
        def __init__(self) -> None:
            self.plans: list[BlockPullPlan] = []

        def pull_block_range(
            self,
            output_dir: Path,
            *,
            plan: BlockPullPlan,
            chunk_size: int,
            rpc_batch_size: int,
            reporter,
        ) -> BlockPullPlan:
            del chunk_size, rpc_batch_size, reporter
            self.plans.append(plan)
            row_count = 4 if len(self.plans) == 1 else 6
            rows = make_block_rows(
                row_count,
                start_block=plan.block_range.start,
                start_timestamp=plan.window.start,
                include_gas_limit=True,
            )
            write_dataset_dir(output_dir, rows)
            return plan

        def expand_history_plan(
            self,
            current: BlockPullPlan,
            *,
            observed_row_count: int,
            required_history_blocks: int,
            chunk_size: int,
        ) -> BlockPullPlan:
            assert observed_row_count == 4
            assert required_history_blocks == 6
            assert chunk_size == 2
            return BlockPullPlan(
                window=TimestampRange(start=current.window.start - 48, end=current.window.end),
                block_range=BlockRange(
                    start=current.block_range.start - 4,
                    end=current.block_range.end,
                ),
                expected_rows=10,
                expected_files=5,
            )

    client = ExpandingHistoryClient()
    pulled_plan, validation, resolved_plan = ensure_history_dataset(
        config=config,
        block_client=client,
        output_dir=output_dir,
        history_plan=initial_plan,
        required_history_blocks=6,
        reporter=NullReporter(),
    )

    assert pulled_plan == client.plans[1]
    assert validation.status == "clean"
    assert validation.row_count == 6
    assert len(client.plans) == 2
    assert client.plans[1].block_range.start == 6
    assert resolved_plan == client.plans[1]
    assert load_block_frame(output_dir).height == 6


def test_web3_block_client_pull_block_range_writes_chunked_dataset(
    tmp_path,
    monkeypatch,
) -> None:
    timestamps = {
        0: 100,
        1: 112,
        2: 124,
        3: 136,
        4: 148,
        5: 160,
    }

    class FakeBatch:
        def __init__(self) -> None:
            self.requests: list[dict[str, int]] = []

        def __enter__(self) -> FakeBatch:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def add(self, block: dict[str, int]) -> None:
            self.requests.append(block)

        def execute(self) -> list[dict[str, int]]:
            return list(self.requests)

    class FakeEth:
        def get_block(
            self,
            block_number: int | str,
            _full_transactions: bool = False,
        ) -> dict[str, int]:
            if block_number == "latest":
                block_number = 5
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
        "spice.acquisition.rpc.build_web3",
        lambda _provider, _chain: FakeWeb3(),
    )

    config = compose_experiment("acquire", overrides=base_overrides(tmp_path))
    client = Web3BlockClient(provider=config.provider, chain=config.chain)
    plan = client.plan_window(
        TimestampRange(start=112, end=160),
        chunk_size=2,
    )
    pulled_plan = client.pull_block_range(
        tmp_path / "history",
        plan=plan,
        chunk_size=2,
        rpc_batch_size=3,
        reporter=NullReporter(),
    )

    frame = load_block_frame(tmp_path / "history")

    assert pulled_plan == plan
    assert pulled_plan.expected_rows == 4
    assert pulled_plan.expected_files == 2
    assert frame["block_number"].to_list() == [1, 2, 3, 4]
    assert len(list((tmp_path / "history").glob("*.parquet"))) == 2


def test_acquire_workflow_writes_block_planned_datasets_and_actual_history_metadata(
    tmp_path,
    monkeypatch,
) -> None:
    config = compose_experiment(
        "acquire",
        overrides=base_overrides(tmp_path)
        + [
            "dataset.temporal.lookback_seconds=24",
            "dataset.temporal.max_delay_seconds=12",
            "dataset.sampling.anchor_count=4",
        ],
    )
    required_history_blocks = required_history_block_count(config)
    block_time_seconds = int(config.chain.block_time_seconds)
    expected_history_start = (
        config.dataset.window.start_timestamp - required_history_blocks * block_time_seconds
    )

    class FakeWorkflowBlockClient:
        history_requests: list[tuple[int, int, int]] = []

        def __init__(self, provider, chain) -> None:
            del provider
            self.chain = chain

        def plan_history_window(
            self,
            *,
            end_timestamp: int,
            required_history_blocks: int,
            chunk_size: int,
        ) -> BlockPullPlan:
            self.__class__.history_requests.append(
                (end_timestamp, required_history_blocks, chunk_size)
            )
            return BlockPullPlan(
                window=TimestampRange(start=expected_history_start, end=end_timestamp),
                block_range=BlockRange(
                    start=100,
                    end=100 + required_history_blocks,
                ),
                expected_rows=required_history_blocks,
                expected_files=1,
            )

        def plan_window(self, window: TimestampRange, *, chunk_size: int) -> BlockPullPlan:
            del chunk_size
            expected_rows = 32
            return BlockPullPlan(
                window=window,
                block_range=BlockRange(start=10_001, end=10_001 + expected_rows),
                expected_rows=expected_rows,
                expected_files=1,
            )

        def pull_block_range(
            self,
            output_dir: Path,
            *,
            plan: BlockPullPlan,
            chunk_size: int,
            rpc_batch_size: int,
            reporter,
        ) -> BlockPullPlan:
            del chunk_size, rpc_batch_size, reporter
            rows = make_block_rows(
                plan.expected_rows,
                start_block=plan.block_range.start,
                start_timestamp=plan.window.start,
                block_time_seconds=block_time_seconds,
                include_gas_limit=True,
            )
            assert int(rows[-1]["timestamp"]) < plan.window.end
            write_dataset_dir(output_dir, rows)
            return plan

    monkeypatch.setattr(
        "spice.workflows.acquire.Web3BlockClient",
        FakeWorkflowBlockClient,
    )

    run_acquire(config, reporter=NullReporter())

    metadata_path = config.paths.dataset_metadata_path
    history_dir = config.paths.history_dir
    evaluation_dir = config.paths.evaluation_dir
    metadata = load_dataset_metadata(metadata_path)

    assert metadata is not None
    assert FakeWorkflowBlockClient.history_requests == [
        (
            config.dataset.window.start_timestamp,
            required_history_blocks,
            config.acquisition.chunk_size,
        )
    ]
    assert metadata.paths.history == history_dir.as_posix()
    assert metadata.paths.evaluation == evaluation_dir.as_posix()
    assert metadata.validation.history.status == "clean"
    assert metadata.validation.evaluation.status == "clean"
    assert history_range_from_metadata(metadata).start == expected_history_start
    assert history_range_from_metadata(metadata).end == config.dataset.window.start_timestamp
    assert load_block_frame(history_dir).height == required_history_blocks
    assert load_block_frame(evaluation_dir).height == 32
