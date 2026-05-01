"""Machine-facing helper for remote storage transfer operations."""

from __future__ import annotations

import argparse
from pathlib import Path

from .catalog.codecs import encode_remote_catalog_record
from .catalog.root_kind_specs import spec_for_root_kind
from .engine import RootKind
from .lifecycle import cleanup_root_stage, prepare_root_stage, promote_root_stage


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
    finalize.add_argument("--root-kind", required=True)
    finalize.add_argument("--replace", action="store_true")
    finalize.set_defaults(func=_finalize_stage_command)

    cleanup = subparsers.add_parser("cleanup-stage")
    cleanup.add_argument("--staged-root", required=True)
    cleanup.set_defaults(func=_cleanup_stage_command)

    resolve = subparsers.add_parser("resolve-record")
    resolve.add_argument("--storage-root", required=True)
    resolve.add_argument("--root-kind", required=True)
    resolve.add_argument("--root-id", required=True)
    resolve.set_defaults(func=_resolve_record_command)

    namespace = parser.parse_args(argv)
    namespace.func(namespace)


def _prepare_stage_command(args: argparse.Namespace) -> None:
    prepare_root_stage(
        destination_root=Path(args.destination_root),
        staged_root=Path(args.staged_root),
        replace=args.replace,
    )


def _finalize_stage_command(args: argparse.Namespace) -> None:
    storage_root = Path(args.storage_root)
    destination_root = Path(args.destination_root)
    staged_root = Path(args.staged_root)
    promote_root_stage(
        storage_root=storage_root,
        destination_root=destination_root,
        staged_root=staged_root,
        expected_root_kind=RootKind(args.root_kind),
        replace=args.replace,
    )


def _cleanup_stage_command(args: argparse.Namespace) -> None:
    cleanup_root_stage(Path(args.staged_root))


def _resolve_record_command(args: argparse.Namespace) -> None:
    root_kind = RootKind(args.root_kind)
    spec = spec_for_root_kind(root_kind)
    record = spec.resolve_record(Path(args.storage_root), args.root_id)
    print(encode_remote_catalog_record(record))


if __name__ == "__main__":
    main()
