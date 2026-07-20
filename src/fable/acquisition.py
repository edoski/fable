"""Native Corpus acquisition, finalization, and publication."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

import polars as pl
from web3 import AsyncWeb3
from web3.middleware import ExtraDataToPOAMiddleware
from web3.providers.rpc import AsyncHTTPProvider

from .addresses import corpus_blocks_path, corpus_json_path
from .config import CorpusDefinition, CorpusRequest
from .corpus.blocks import BlockFrame
from .corpus.contract import Corpus, FinalizedAnchor

_CHECKPOINT_SIZE = 4096
_CONCURRENCY = 4

_CHUNK_NAME = re.compile(r"(\d+)-(\d+)\.parquet\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")


class _BlockRow(TypedDict):
    block_number: int
    block_hash: str
    parent_hash: str
    timestamp: int
    chain_id: int
    base_fee_per_gas: int
    gas_used: int
    gas_limit: int
    tx_count: int


async def acquire_corpus(
    request: CorpusRequest,
    *,
    storage_root: Path,
    rpc_url: str,
    poa: bool,
) -> None:
    """Acquire and publish the exact native Corpus requested."""

    destination = corpus_json_path(storage_root, request.corpus_id).parent
    if destination.exists():
        raise FileExistsError(destination)

    hidden = destination.parent / f".{request.corpus_id}"
    request_path, chunks_path = _prepare_hidden(hidden, request)
    chunk_paths, next_block, previous_hash, previous_timestamp = _load_chunks(
        chunks_path,
        request,
    )

    provider = AsyncHTTPProvider(rpc_url)
    web3 = AsyncWeb3(provider)
    try:
        if poa:
            web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        endpoint_chain_id = _integer("endpoint chain_id", await web3.eth.chain_id, minimum=0)
        if endpoint_chain_id != request.definition.chain_id:
            raise ValueError(
                "Endpoint chain_id does not match the Corpus request: "
                f"expected {request.definition.chain_id}, got {endpoint_chain_id}"
            )

        while next_block <= request.definition.last_block:
            last_block = min(
                next_block + _CHECKPOINT_SIZE - 1,
                request.definition.last_block,
            )
            rows, previous_hash, previous_timestamp = await _acquire_chunk(
                web3,
                request,
                first_block=next_block,
                last_block=last_block,
                previous_hash=previous_hash,
                previous_timestamp=previous_timestamp,
            )
            chunk_path = chunks_path / _chunk_filename(next_block, last_block)
            _write_chunk(chunk_path, rows)
            chunk_paths.append(chunk_path)
            next_block = last_block + 1

        last_hash, last_parent_hash = _last_link(chunk_paths)
        anchor = await _finalized_anchor(
            web3,
            request,
            last_hash=last_hash,
            last_parent_hash=last_parent_hash,
        )
    finally:
        await provider.disconnect()

    blocks_path = hidden / corpus_blocks_path(storage_root, request.corpus_id).name
    _stream_blocks(chunk_paths, blocks_path)
    corpus_path = hidden / corpus_json_path(storage_root, request.corpus_id).name
    corpus_path.write_text(
        json.dumps(
            {
                "request": request.model_dump(mode="json"),
                "finalized_anchor": anchor.model_dump(mode="json"),
            },
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    Corpus(
        request=request,
        finalized_anchor=anchor,
        blocks=BlockFrame(pl.read_parquet(blocks_path), request.definition),
    )

    for chunk_path in chunk_paths:
        chunk_path.unlink()
    chunks_path.rmdir()
    request_path.unlink()
    hidden.rename(destination)


def _prepare_hidden(hidden: Path, request: CorpusRequest) -> tuple[Path, Path]:
    request_path = hidden / "request.json"
    chunks_path = hidden / "chunks"
    if hidden.exists():
        persisted = CorpusRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
        if persisted != request:
            raise ValueError("Hidden Corpus request does not match the requested Corpus")
        return request_path, chunks_path

    chunks_path.mkdir(parents=True)
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    return request_path, chunks_path


def _load_chunks(
    chunks_path: Path,
    request: CorpusRequest,
) -> tuple[list[Path], int, str | None, int | None]:
    parsed: list[tuple[int, int, Path]] = []
    for path in chunks_path.iterdir():
        match = _CHUNK_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected checkpoint chunk: {path.name}")
        parsed.append((int(match.group(1)), int(match.group(2)), path))
    parsed.sort(key=lambda item: item[0])

    expected = request.definition.first_block
    previous_hash: str | None = None
    previous_timestamp: int | None = None
    paths: list[Path] = []
    for first_block, last_block, path in parsed:
        expected_last = min(
            expected + _CHECKPOINT_SIZE - 1,
            request.definition.last_block,
        )
        if (first_block, last_block) != (expected, expected_last):
            raise ValueError(
                "Checkpoint chunks must be the deterministic complete prefix: "
                f"expected {expected}..{expected_last}, got {first_block}..{last_block}"
            )
        frame = pl.read_parquet(path)
        previous_hash, previous_timestamp = _validate_chunk(
            frame,
            request,
            first_block=first_block,
            last_block=last_block,
            previous_hash=previous_hash,
            previous_timestamp=previous_timestamp,
        )
        paths.append(path)
        expected = last_block + 1
    return paths, expected, previous_hash, previous_timestamp


async def _acquire_chunk(
    web3: AsyncWeb3,
    request: CorpusRequest,
    *,
    first_block: int,
    last_block: int,
    previous_hash: str | None,
    previous_timestamp: int | None,
) -> tuple[list[_BlockRow], str, int]:
    rows: list[_BlockRow] = []
    for batch_start in range(first_block, last_block + 1, _CONCURRENCY):
        numbers = list(range(batch_start, min(batch_start + _CONCURRENCY, last_block + 1)))
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(web3.eth.get_block(number, False)) for number in numbers]
        for number, task in zip(numbers, tasks, strict=True):
            row = _block_row(task.result(), request, expected_number=number)
            block_hash = row["block_hash"]
            parent_hash = row["parent_hash"]
            timestamp = row["timestamp"]
            if previous_hash is not None and parent_hash != previous_hash:
                raise ValueError(f"Parent link mismatch at block {number}")
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise ValueError(f"Timestamp decreases at block {number}")
            rows.append(row)
            previous_hash = block_hash
            previous_timestamp = timestamp
    last_row = rows[-1]
    return rows, last_row["block_hash"], last_row["timestamp"]


def _block_row(
    raw_block: object,
    request: CorpusRequest,
    *,
    expected_number: int,
) -> _BlockRow:
    if not isinstance(raw_block, Mapping):
        raise TypeError("RPC block response must be a mapping")
    number = _integer("block number", raw_block["number"], minimum=0)
    if number != expected_number:
        raise ValueError(f"RPC block number mismatch: expected {expected_number}, got {number}")
    timestamp = _integer("timestamp", raw_block["timestamp"], minimum=0)
    base_fee = _integer("base_fee_per_gas", raw_block["baseFeePerGas"], minimum=1)
    gas_used = _integer("gas_used", raw_block["gasUsed"], minimum=0)
    gas_limit = _integer("gas_limit", raw_block["gasLimit"], minimum=1)
    if gas_used > gas_limit:
        raise ValueError(f"gas_used exceeds gas_limit at block {number}")
    transactions = raw_block["transactions"]
    if not isinstance(transactions, list | tuple):
        raise TypeError("RPC block transactions must be a sequence")
    return {
        "block_number": number,
        "block_hash": _normalized_hash("block hash", raw_block["hash"]),
        "parent_hash": _normalized_hash("parent hash", raw_block["parentHash"]),
        "timestamp": timestamp,
        "chain_id": request.definition.chain_id,
        "base_fee_per_gas": base_fee,
        "gas_used": gas_used,
        "gas_limit": gas_limit,
        "tx_count": len(transactions),
    }


def _validate_chunk(
    frame: pl.DataFrame,
    request: CorpusRequest,
    *,
    first_block: int,
    last_block: int,
    previous_hash: str | None,
    previous_timestamp: int | None,
) -> tuple[str, int]:
    if frame.columns[1:3] != ["block_hash", "parent_hash"] or (
        frame.schema.get("block_hash") != pl.String or frame.schema.get("parent_hash") != pl.String
    ):
        raise ValueError("Checkpoint proof columns must be ordered non-null strings")
    if frame["block_hash"].null_count() or frame["parent_hash"].null_count():
        raise ValueError("Checkpoint proof columns must be ordered non-null strings")

    definition = CorpusDefinition(
        chain_id=request.definition.chain_id,
        first_block=first_block,
        last_block=last_block,
    )
    canonical = BlockFrame(frame.drop("block_hash", "parent_hash"), definition).to_polars()
    block_hashes = frame["block_hash"]
    parent_hashes = frame["parent_hash"]
    if (
        not block_hashes.str.contains(r"^[0-9a-f]{64}$").all()
        or not parent_hashes.str.contains(r"^[0-9a-f]{64}$").all()
    ):
        raise ValueError("Checkpoint hashes must be exactly 64 lowercase hexadecimal characters")
    if previous_hash is not None and parent_hashes[0] != previous_hash:
        raise ValueError(f"Checkpoint parent link mismatch at block {first_block}")
    if (
        frame.height > 1
        and not (parent_hashes.slice(1) == block_hashes.slice(0, frame.height - 1)).all()
    ):
        raise ValueError("Checkpoint parent links must match within the chunk")
    first_timestamp = int(canonical["timestamp"][0])
    if previous_timestamp is not None and first_timestamp < previous_timestamp:
        raise ValueError(f"Checkpoint timestamp decreases at block {first_block}")
    return str(block_hashes[-1]), int(canonical["timestamp"][-1])


def _write_chunk(path: Path, rows: list[_BlockRow]) -> None:
    frame = pl.DataFrame(rows, orient="row")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        frame.write_parquet(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


async def _finalized_anchor(
    web3: AsyncWeb3,
    request: CorpusRequest,
    *,
    last_hash: str,
    last_parent_hash: str,
) -> FinalizedAnchor:
    tagged_number, tagged_hash, tagged_parent_hash = _header(
        await web3.eth.get_block("finalized", False),
        label="finalized anchor",
    )
    if tagged_number < request.definition.last_block:
        raise ValueError("Finalized anchor precedes the requested last block")

    previous_hash = last_hash
    if tagged_number == request.definition.last_block:
        if tagged_hash != last_hash or tagged_parent_hash != last_parent_hash:
            raise ValueError("Finalized anchor does not match the staged last block")
    else:
        for number in range(request.definition.last_block + 1, tagged_number):
            ancestry_number, ancestry_hash, ancestry_parent_hash = _header(
                await web3.eth.get_block(number, False),
                label=f"finalized ancestry block {number}",
                expected_number=number,
            )
            if ancestry_parent_hash != previous_hash:
                raise ValueError(f"Finalized ancestry breaks at block {ancestry_number}")
            previous_hash = ancestry_hash
        if tagged_parent_hash != previous_hash:
            raise ValueError("Requested range is not an ancestor of the finalized anchor")

    numbered_number, numbered_hash, numbered_parent_hash = _header(
        await web3.eth.get_block(tagged_number, False),
        label="numbered finalized anchor",
        expected_number=tagged_number,
    )
    if (numbered_number, numbered_hash, numbered_parent_hash) != (
        tagged_number,
        tagged_hash,
        tagged_parent_hash,
    ):
        raise ValueError("Numbered finalized anchor differs from the tagged anchor")
    return FinalizedAnchor(block_number=tagged_number, block_hash=tagged_hash)


def _header(
    raw_block: object,
    *,
    label: str,
    expected_number: int | None = None,
) -> tuple[int, str, str]:
    if not isinstance(raw_block, Mapping):
        raise TypeError(f"{label} must be a mapping")
    number = _integer(f"{label} number", raw_block["number"], minimum=0)
    if expected_number is not None and number != expected_number:
        raise ValueError(f"{label} number mismatch: expected {expected_number}, got {number}")
    return (
        number,
        _normalized_hash(f"{label} hash", raw_block["hash"]),
        _normalized_hash(f"{label} parent hash", raw_block["parentHash"]),
    )


def _integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _normalized_hash(name: str, value: object) -> str:
    if isinstance(value, str):
        raw = value[2:] if value.startswith(("0x", "0X")) else value
    elif isinstance(value, bytes | bytearray):
        raw = bytes(value).hex()
    else:
        raise TypeError(f"{name} must be bytes or a hexadecimal string")
    normalized = raw.lower()
    if _HASH.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be 64 hexadecimal characters")
    return normalized


def _stored_hash(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Stored {name} must be a string")
    if _HASH.fullmatch(value) is None:
        raise ValueError(f"Stored {name} must be exactly 64 lowercase hexadecimal characters")
    return value


def _chunk_filename(first_block: int, last_block: int) -> str:
    return f"{first_block:020d}-{last_block:020d}.parquet"


def _last_link(chunk_paths: list[Path]) -> tuple[str, str]:
    frame = pl.read_parquet(chunk_paths[-1], columns=["block_hash", "parent_hash"])
    block_hash, parent_hash = frame.row(-1)
    return (
        _stored_hash("block hash", block_hash),
        _stored_hash("parent hash", parent_hash),
    )


def _stream_blocks(chunk_paths: list[Path], destination: Path) -> None:
    pl.concat([pl.scan_parquet(path) for path in chunk_paths]).drop(
        "block_hash", "parent_hash"
    ).sink_parquet(destination, maintain_order=True)


__all__ = ["acquire_corpus"]
