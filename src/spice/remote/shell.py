"""SSH and rsync transport helpers for the single remote cluster."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..config import ExecutionSpec
from ..config.registry import load_named_group
from ..core.errors import SpiceOperatorError

DEFAULT_REMOTE_EXECUTION_NAME = "disi_l40"


@dataclass(frozen=True, slots=True)
class RemoteExecutionTarget:
    name: str
    spec: ExecutionSpec

    @property
    def ssh_destination(self) -> str:
        return f"{self.spec.ssh.user}@{self.spec.ssh.host}"


def build_remote_shell_argv(target: RemoteExecutionTarget, command: str) -> list[str]:
    return [
        "ssh",
        target.ssh_destination,
        "bash",
        "-lc",
        shlex.quote(command),
    ]


def resolve_remote_target(name: str | None = None) -> RemoteExecutionTarget:
    resolved_name = DEFAULT_REMOTE_EXECUTION_NAME if name is None else name
    payload = load_named_group(resolved_name, "execution")
    return RemoteExecutionTarget(
        name=resolved_name,
        spec=ExecutionSpec.model_validate(payload),
    )


def run_remote_cli(
    target: RemoteExecutionTarget,
    args: list[str],
    *,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    spice_path = shlex.quote(str(target.spec.paths.spice_path))
    repo_root = shlex.quote(str(target.spec.paths.repo_root))
    argv = shlex.join(args)
    return run_remote_command(
        target,
        f"cd {repo_root} && {spice_path} {argv}",
        capture_output=capture_output,
    )


def run_remote_module(
    target: RemoteExecutionTarget,
    module: str,
    args: list[str],
    *,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    python_path = shlex.quote(str(target.spec.paths.python_path))
    repo_root = shlex.quote(str(target.spec.paths.repo_root))
    module_name = shlex.quote(module)
    argv = shlex.join(args)
    return run_remote_command(
        target,
        f"cd {repo_root} && {python_path} -m {module_name} {argv}",
        capture_output=capture_output,
    )


def run_remote_command(
    target: RemoteExecutionTarget,
    command: str,
    *,
    input_text: str | None = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        build_remote_shell_argv(target, command),
        input=input_text,
        text=True,
        capture_output=capture_output,
        check=False,
    )


def run_rsync_to_remote(
    target: RemoteExecutionTarget,
    *,
    source_root: Path,
    destination_root: Path,
) -> None:
    result = subprocess.run(
        [
            "rsync",
            "-a",
            f"{source_root.as_posix()}/",
            f"{target.ssh_destination}:{destination_root.as_posix()}/",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    ensure_remote_success(result, action=f"rsync to {destination_root}")


def run_rsync_from_remote(
    target: RemoteExecutionTarget,
    *,
    source_root: Path,
    destination_root: Path,
) -> None:
    result = subprocess.run(
        [
            "rsync",
            "-a",
            f"{target.ssh_destination}:{source_root.as_posix()}/",
            f"{destination_root.as_posix()}/",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    ensure_remote_success(result, action=f"rsync from {source_root}")


def ensure_remote_success(
    result: subprocess.CompletedProcess[str],
    *,
    action: str,
) -> subprocess.CompletedProcess[str]:
    if result.returncode == 0:
        return result
    message = (result.stderr or result.stdout).strip()
    suffix = f": {message}" if message else ""
    raise SpiceOperatorError(f"{action} failed{suffix}")
