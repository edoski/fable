from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

import polars as pl
import pytest
from web3.middleware import ExtraDataToPOAMiddleware

import fable.acquisition as acquisition
from fable.acquisition import acquire_corpus
from fable.config import CorpusDefinition, CorpusRequest
from fable.corpus.io import load_corpus


def _request(*, first: int = 100, last: int = 104) -> CorpusRequest:
    return CorpusRequest(
        corpus_id=UUID("11111111-1111-4111-8111-111111111111"),
        definition=CorpusDefinition(chain_id=1, first_block=first, last_block=last),
    )


def _hash(number: int) -> str:
    return f"{number + 1:064x}"


class _FakeEth:
    def __init__(
        self,
        *,
        chain_id: int = 1,
        finalized_number: int = 106,
        fail_once: set[int] | None = None,
        block_changes: dict[int, dict[str, object]] | None = None,
        tagged_changes: dict[str, object] | None = None,
        numbered_anchor_changes: dict[str, object] | None = None,
    ) -> None:
        self.endpoint_chain_id = chain_id
        self.finalized_number = finalized_number
        self.fail_once = set() if fail_once is None else set(fail_once)
        self.block_changes = {} if block_changes is None else block_changes
        self.tagged_changes = {} if tagged_changes is None else tagged_changes
        self.numbered_anchor_changes = (
            {} if numbered_anchor_changes is None else numbered_anchor_changes
        )
        self.calls: list[int | str] = []
        self.tagged_anchor_read = False

    @property
    def chain_id(self):
        async def get_chain_id() -> int:
            return self.endpoint_chain_id

        return get_chain_id()

    async def get_block(self, number: int | str, full_transactions: bool) -> dict[str, object]:
        assert full_transactions is False
        self.calls.append(number)
        if number == "finalized":
            self.tagged_anchor_read = True
            return {**self._block(self.finalized_number), **self.tagged_changes}
        assert isinstance(number, int)
        if number in self.fail_once:
            self.fail_once.remove(number)
            raise OSError(f"provider failed for block {number}")
        changes = self.block_changes.get(number, {})
        if number == self.finalized_number and self.tagged_anchor_read:
            changes = {**changes, **self.numbered_anchor_changes}
        return {**self._block(number), **changes}

    @staticmethod
    def _block(number: int) -> dict[str, object]:
        return {
            "number": number,
            "hash": "0x" + _hash(number),
            "parentHash": bytes.fromhex(_hash(number - 1)),
            "timestamp": 1_700_000_000 + number,
            "baseFeePerGas": 1_000_000_000 + number,
            "gasUsed": 15_000_000 + number,
            "gasLimit": 30_000_000,
            "transactions": [f"tx-{number}"] * (number % 3),
        }


class _FakeProvider:
    def __init__(self, rpc_url: str, canonical_path: Path) -> None:
        self.rpc_url = rpc_url
        self.canonical_path = canonical_path
        self.closed = False

    async def disconnect(self) -> None:
        assert not self.canonical_path.exists()
        self.closed = True


class _FakeMiddlewareOnion:
    def __init__(self) -> None:
        self.injections: list[tuple[object, int]] = []

    def inject(self, middleware: object, *, layer: int) -> None:
        self.injections.append((middleware, layer))


class _FakeWeb3:
    def __init__(self, provider: _FakeProvider, eth: _FakeEth) -> None:
        self.provider = provider
        self.eth = eth
        self.middleware_onion = _FakeMiddlewareOnion()


class _Web3Harness:
    def __init__(self, eths: list[_FakeEth], canonical_path: Path) -> None:
        self.eths = list(eths)
        self.canonical_path = canonical_path
        self.providers: list[_FakeProvider] = []
        self.web3s: list[_FakeWeb3] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def provider_factory(rpc_url: str) -> _FakeProvider:
            provider = _FakeProvider(rpc_url, self.canonical_path)
            self.providers.append(provider)
            return provider

        def web3_factory(provider: _FakeProvider) -> _FakeWeb3:
            web3 = _FakeWeb3(provider, self.eths.pop(0))
            self.web3s.append(web3)
            return web3

        monkeypatch.setattr(acquisition, "AsyncHTTPProvider", provider_factory)
        monkeypatch.setattr(acquisition, "AsyncWeb3", web3_factory)


def _canonical_path(root: Path, request: CorpusRequest) -> Path:
    return root / "corpora" / str(request.corpus_id)


def _hidden_path(root: Path, request: CorpusRequest) -> Path:
    return root / "corpora" / f".{request.corpus_id}"


def test_acquire_resumes_complete_chunks_and_publishes_one_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(last=4_201)
    canonical = _canonical_path(tmp_path, request)
    failed = _FakeEth(finalized_number=4_203, fail_once={4_198})
    resumed = _FakeEth(finalized_number=4_203)
    harness = _Web3Harness([failed, resumed], canonical)
    harness.install(monkeypatch)

    with pytest.raises(ExceptionGroup):
        asyncio.run(
            acquire_corpus(
                request,
                storage_root=tmp_path,
                rpc_url="https://rpc.example",
                poa=False,
            )
        )

    hidden = _hidden_path(tmp_path, request)
    chunks = sorted((hidden / "chunks").glob("*.parquet"))
    assert not canonical.exists()
    assert (hidden / "request.json").exists()
    assert len(chunks) == 1
    assert pl.read_parquet(chunks[0])["block_number"].to_list() == list(range(100, 4_196))

    asyncio.run(
        acquire_corpus(
            request,
            storage_root=tmp_path,
            rpc_url="https://rpc.example",
            poa=True,
        )
    )

    corpus = load_corpus(tmp_path, request.corpus_id)
    assert corpus.request == request
    assert corpus.finalized_anchor.block_number == 4_203
    assert corpus.blocks["block_number"].to_list() == list(range(100, 4_202))
    assert corpus.blocks.columns == [
        "block_number",
        "timestamp",
        "chain_id",
        "base_fee_per_gas",
        "gas_used",
        "gas_limit",
        "tx_count",
    ]
    assert not hidden.exists()
    assert all(provider.closed for provider in harness.providers)
    assert all(provider.rpc_url == "https://rpc.example" for provider in harness.providers)
    assert harness.web3s[0].middleware_onion.injections == []
    assert harness.web3s[1].middleware_onion.injections == [(ExtraDataToPOAMiddleware, 0)]
    assert min(number for number in resumed.calls if isinstance(number, int)) == 4_196


FailureSetup = Callable[[Path, CorpusRequest], _FakeEth]


def _ordinary_failure(_: Path, request: CorpusRequest) -> _FakeEth:
    return _FakeEth(finalized_number=request.definition.last_block + 2)


def _stored_hash_failure(root: Path, request: CorpusRequest) -> _FakeEth:
    hidden = _hidden_path(root, request)
    chunks = hidden / "chunks"
    chunks.mkdir(parents=True)
    (hidden / "request.json").write_text(request.model_dump_json(), encoding="utf-8")

    first = request.definition.first_block
    last = request.definition.last_block
    rows = [
        {
            "block_number": number,
            "block_hash": "A" * 64 if number == first else _hash(number),
            "parent_hash": ("a" * 64 if number == first + 1 else _hash(number - 1)),
            "timestamp": 1_700_000_000 + number,
            "chain_id": request.definition.chain_id,
            "base_fee_per_gas": 1_000_000_000 + number,
            "gas_used": 15_000_000 + number,
            "gas_limit": 30_000_000,
            "tx_count": number % 3,
        }
        for number in range(first, last + 1)
    ]
    pl.DataFrame(rows).write_parquet(chunks / f"{first:020d}-{last:020d}.parquet")
    return _ordinary_failure(root, request)


@pytest.mark.parametrize(
    ("case", "setup", "error"),
    [
        ("destination", _ordinary_failure, FileExistsError),
        ("request", _ordinary_failure, ValueError),
        ("chunk", _ordinary_failure, ValueError),
        ("stored_hash", _stored_hash_failure, ValueError),
        ("chain", lambda _root, request: _FakeEth(chain_id=2), ValueError),
        (
            "number",
            lambda _root, request: _FakeEth(
                block_changes={102: {"number": 103}},
                finalized_number=request.definition.last_block + 2,
            ),
            ValueError,
        ),
        (
            "parent",
            lambda _root, request: _FakeEth(
                block_changes={104: {"parentHash": b"\xff" * 32}},
                finalized_number=request.definition.last_block + 2,
            ),
            ValueError,
        ),
        (
            "timestamp",
            lambda _root, request: _FakeEth(
                block_changes={104: {"timestamp": 1}},
                finalized_number=request.definition.last_block + 2,
            ),
            ValueError,
        ),
        (
            "ancestry",
            lambda _root, request: _FakeEth(
                block_changes={105: {"parentHash": b"\xff" * 32}},
                finalized_number=request.definition.last_block + 2,
            ),
            ValueError,
        ),
        (
            "anchor",
            lambda _root, request: _FakeEth(
                finalized_number=request.definition.last_block + 2,
                numbered_anchor_changes={"hash": "0x" + "f" * 64},
            ),
            ValueError,
        ),
    ],
)
def test_acquire_rejects_mismatch_finality_or_existing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    setup: FailureSetup,
    error: type[BaseException],
) -> None:
    request = _request()
    canonical = _canonical_path(tmp_path, request)
    hidden = _hidden_path(tmp_path, request)
    if case == "destination":
        canonical.mkdir(parents=True)
    elif case == "request":
        hidden.mkdir(parents=True)
        (hidden / "chunks").mkdir()
        other = _request(last=105)
        (hidden / "request.json").write_text(other.model_dump_json(), encoding="utf-8")
    elif case == "chunk":
        hidden.mkdir(parents=True)
        chunks = hidden / "chunks"
        chunks.mkdir()
        (hidden / "request.json").write_text(request.model_dump_json(), encoding="utf-8")
        pl.DataFrame({"wrong": [1]}).write_parquet(chunks / "bad.parquet")

    eth = setup(tmp_path, request)
    harness = _Web3Harness([eth], canonical)
    harness.install(monkeypatch)

    with pytest.raises(error):
        asyncio.run(
            acquire_corpus(
                request,
                storage_root=tmp_path,
                rpc_url="https://rpc.example",
                poa=False,
            )
        )

    if case == "destination":
        assert canonical.exists()
        assert not hidden.exists()
        assert harness.providers == []
        return
    assert not canonical.exists()
    assert (hidden / "request.json").exists()
    if case in {"request", "chunk", "stored_hash"}:
        assert harness.providers == []
    else:
        assert harness.providers[0].closed
    if case in {"ancestry", "anchor"}:
        assert len(list((hidden / "chunks").glob("*.parquet"))) == 1
