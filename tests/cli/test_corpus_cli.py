from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
import typer
from typer.testing import CliRunner

import fable.cli.commands.corpus as cli
from fable.config import CorpusDefinition, CorpusRequest


def _app() -> typer.Typer:
    app = typer.Typer()
    app.add_typer(cli.app, name="corpus")
    return app


def _request() -> CorpusRequest:
    return CorpusRequest(
        corpus_id=UUID("11111111-1111-4111-8111-111111111111"),
        definition=CorpusDefinition(chain_id=1, first_block=100, last_block=104),
    )


def test_corpus_acquire_forwards_request_deployment_and_stays_silent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    calls: list[tuple[CorpusRequest, Path, str, bool]] = []

    async def fake_acquire_corpus(
        received: CorpusRequest,
        *,
        storage_root: Path,
        rpc_url: str,
        poa: bool,
    ) -> None:
        calls.append((received, storage_root, rpc_url, poa))

    monkeypatch.setattr(cli, "acquire_corpus", fake_acquire_corpus)

    result = CliRunner().invoke(
        _app(),
        [
            "corpus",
            "acquire",
            str(request_path),
            "--rpc-url",
            "https://rpc.example",
            "--no-poa",
        ],
    )

    assert result.exit_code == 0
    assert result.output == ""
    assert calls == [(request, tmp_path, "https://rpc.example", False)]


class _OwnerFailure(RuntimeError):
    pass


@pytest.mark.parametrize("case", ["relative_root", "owner_failure"])
def test_corpus_acquire_propagates_native_failures(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(_request().model_dump_json(), encoding="utf-8")
    failure = _OwnerFailure("owner failed")
    calls = 0

    async def fake_acquire_corpus(
        request: CorpusRequest,
        *,
        storage_root: Path,
        rpc_url: str,
        poa: bool,
    ) -> None:
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(cli, "acquire_corpus", fake_acquire_corpus)
    monkeypatch.setenv("STORAGE_ROOT", "relative" if case == "relative_root" else str(tmp_path))

    result = CliRunner().invoke(
        _app(),
        [
            "corpus",
            "acquire",
            str(request_path),
            "--rpc-url",
            "https://rpc.example",
            "--poa",
        ],
    )

    assert result.exit_code == 1
    assert result.output == ""
    if case == "relative_root":
        assert isinstance(result.exception, ValueError)
        assert str(result.exception) == "STORAGE_ROOT must be an absolute path"
        assert calls == 0
    else:
        assert result.exception is failure
        assert calls == 1
