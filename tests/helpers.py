from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from click.testing import Result
from typer import Typer
from typer.testing import CliRunner

from fable.config import BlockWindow

REMOTE_YAML = """ssh: research-alias
image: /opt/fable image.sif
storage_root: /remote/storage root
log_root: /remote/logs
resources:
  partition: thesis-partition
  gres: gpu:a100:1
  cpus_per_task: 8
  memory_gb: 48
  time_limit: "17:23:45"
"""


def run_script(script: Path, *arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *(str(argument) for argument in arguments)],
        check=True,
        capture_output=True,
        text=True,
    )


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def write_remote(path: Path, contents: str = REMOTE_YAML) -> None:
    path.write_text(contents, encoding="utf-8")


def dispatch(app: Typer, *arguments: str, input: str | None = None) -> Result:
    return CliRunner().invoke(app, list(arguments), input=input)


def window(first: int) -> BlockWindow:
    return BlockWindow(first_parent_block=first, last_parent_block=first + 9)
