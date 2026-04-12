"""Atomic file persistence helpers."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4


def write_path_atomic(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        writer(tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def write_text_atomic(path: Path, payload: str, *, encoding: str = "utf-8") -> None:
    def _write(tmp_path: Path) -> None:
        tmp_path.write_text(payload, encoding=encoding)

    write_path_atomic(path, _write)


def make_temp_dir(parent: Path, *, prefix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=parent, prefix=prefix))


def promote_paths_atomic(promotions: list[tuple[Path, Path]]) -> None:
    backups: list[tuple[Path, Path]] = []
    promoted: list[Path] = []
    try:
        for target, _source in promotions:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = target.parent / f".{target.name}.backup.{uuid4().hex}"
                os.replace(target, backup)
                backups.append((target, backup))
        for target, source in promotions:
            os.replace(source, target)
            promoted.append(target)
    except Exception:
        for target in promoted:
            remove_path(target)
        for target, backup in reversed(backups):
            if backup.exists():
                os.replace(backup, target)
        raise
    else:
        for _target, backup in backups:
            remove_path(backup)


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
        return
    if path.exists():
        path.unlink()


def prune_empty_directories(path: Path, *, stop_at: Path | None = None) -> None:
    current = path
    boundary = stop_at
    while True:
        if boundary is not None and current == boundary:
            return
        try:
            entries = list(current.iterdir())
        except FileNotFoundError:
            entries = []
        except NotADirectoryError:
            return
        except OSError:
            return
        substantive_entries = [entry for entry in entries if entry.name != ".DS_Store"]
        if substantive_entries:
            return
        for entry in entries:
            if entry.name == ".DS_Store":
                entry.unlink(missing_ok=True)
        try:
            current.rmdir()
        except FileNotFoundError:
            pass
        except OSError:
            return
        parent = current.parent
        if parent == current:
            return
        current = parent
