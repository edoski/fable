"""CLI transport wrapper for remote storage sync operations."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast

from ..core.errors import StateConflictError
from ..core.files import promote_paths_atomic
from .roots import (
    ArtifactSelector,
    StudySelector,
    reindex_root,
    resolve_artifact_record,
    resolve_study_record,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m spice.storage.sync_cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-stage")
    prepare.add_argument("--destination-root", required=True)
    prepare.add_argument("--staged-root", required=True)
    prepare.add_argument("--replace", action="store_true")
    prepare.set_defaults(func=_prepare_stage_command)

    finalize = subparsers.add_parser("finalize-stage")
    finalize.add_argument("--storage-root", required=True)
    finalize.add_argument("--destination-root", required=True)
    finalize.add_argument("--staged-root", required=True)
    finalize.add_argument("--replace", action="store_true")
    finalize.set_defaults(func=_finalize_stage_command)

    resolve_study = subparsers.add_parser("resolve-study-record")
    resolve_study.add_argument("--storage-root", required=True)
    resolve_study.add_argument("--selector-json", required=True)
    resolve_study.set_defaults(func=_resolve_study_record_command)

    resolve_artifact = subparsers.add_parser("resolve-artifact-record")
    resolve_artifact.add_argument("--storage-root", required=True)
    resolve_artifact.add_argument("--selector-json", required=True)
    resolve_artifact.set_defaults(func=_resolve_artifact_record_command)

    namespace = parser.parse_args(argv)
    namespace.func(namespace)


def _prepare_stage_command(args: argparse.Namespace) -> None:
    destination_root = Path(args.destination_root)
    staged_root = Path(args.staged_root)
    if destination_root.exists() and not args.replace:
        raise StateConflictError(f"Destination already exists: {destination_root}")
    staged_root.parent.mkdir(parents=True, exist_ok=True)
    if staged_root.exists():
        shutil.rmtree(staged_root)
    staged_root.mkdir(parents=True, exist_ok=True)


def _finalize_stage_command(args: argparse.Namespace) -> None:
    storage_root = Path(args.storage_root)
    destination_root = Path(args.destination_root)
    staged_root = Path(args.staged_root)
    if destination_root.exists() and not args.replace:
        raise StateConflictError(f"Destination already exists: {destination_root}")
    promote_paths_atomic([(destination_root, staged_root)])
    reindex_root(storage_root, root_path=destination_root)


def _resolve_study_record_command(args: argparse.Namespace) -> None:
    selector_payload = json.loads(args.selector_json)
    record = resolve_study_record(
        Path(args.storage_root),
        selector=StudySelector(**selector_payload),
    )
    _emit_record_json(record)


def _resolve_artifact_record_command(args: argparse.Namespace) -> None:
    selector_payload = json.loads(args.selector_json)
    record = resolve_artifact_record(
        Path(args.storage_root),
        selector=ArtifactSelector(**selector_payload),
    )
    _emit_record_json(record)


def _emit_record_json(record: object) -> None:
    if not is_dataclass(record) or isinstance(record, type):
        raise TypeError("record must be a dataclass instance")
    payload = asdict(cast(Any, record))
    print(
        json.dumps(
            {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in payload.items()
            }
        )
    )


if __name__ == "__main__":
    main()
