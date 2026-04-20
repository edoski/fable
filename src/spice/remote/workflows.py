"""Remote workflow submission and follow helpers."""

from __future__ import annotations

import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import ExecutionWorkflowSpec, WorkflowTask
from ..core.errors import SpiceOperatorError
from .shell import (
    RemoteExecutionTarget,
    build_remote_shell_argv,
    ensure_remote_success,
    resolve_remote_target,
    run_remote_command,
)

_SBATCH_JOB_ID_PATTERN = re.compile(r"Submitted batch job (?P<job_id>\d+)")
_FINAL_JOB_STATES = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
        "NODE_FAIL",
    }
)
_WORKFLOW_SPEC_ATTRS = {
    WorkflowTask.TRAIN: "train",
    WorkflowTask.TUNE: "tune",
    WorkflowTask.EVALUATE: "evaluate",
}


@dataclass(frozen=True, slots=True)
class RemoteJobSubmission:
    task: WorkflowTask
    execution_name: str
    target: RemoteExecutionTarget
    job_id: str
    log_path: Path


def submit_remote_workflow(
    task: WorkflowTask,
    *,
    cli_args: list[str],
    execution_name: str | None = None,
    dependency: str | None = None,
) -> RemoteJobSubmission:
    target = resolve_remote_target(execution_name)
    workflow_spec = _workflow_spec(target, task)
    log_path_template = target.spec.paths.log_root / f"spice-{task.value}-%j.out"
    script = _render_sbatch_script(
        target=target,
        task=task,
        workflow_spec=workflow_spec,
        cli_args=cli_args,
        log_path_template=log_path_template,
    )
    submit_command = " && ".join(
        [
            f"mkdir -p {shlex.quote(str(target.spec.paths.log_root))}",
            f"mkdir -p {shlex.quote(str(target.spec.paths.storage_root))}",
            _build_sbatch_submit_command(dependency=dependency),
        ]
    )
    result = ensure_remote_success(
        run_remote_command(target, submit_command, input_text=script),
        action=f"submit remote {task.value}",
    )
    match = _SBATCH_JOB_ID_PATTERN.search(result.stdout)
    if match is None:
        raise SpiceOperatorError(
            f"submit remote {task.value} failed: could not parse job id from sbatch output"
        )
    job_id = match.group("job_id")
    return RemoteJobSubmission(
        task=task,
        execution_name=target.name,
        target=target,
        job_id=job_id,
        log_path=Path(str(log_path_template).replace("%j", job_id)),
    )


def follow_remote_job(submission: RemoteJobSubmission) -> str | None:
    tail_process = subprocess.Popen(
        build_remote_shell_argv(
            submission.target,
            (
                "while [ ! -f "
                f"{shlex.quote(str(submission.log_path))}"
                " ]; do sleep 2; done; "
                f"tail -n +1 -F {shlex.quote(str(submission.log_path))}"
            ),
        ),
        text=True,
    )
    try:
        while True:
            state = read_remote_job_state(submission)
            if state in _FINAL_JOB_STATES:
                time.sleep(2)
                return state
            if state is None:
                return read_remote_job_final_state(submission)
            time.sleep(5)
    finally:
        if tail_process.poll() is None:
            tail_process.terminate()
            try:
                tail_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tail_process.kill()
                tail_process.wait(timeout=5)


def read_remote_job_state(submission: RemoteJobSubmission) -> str | None:
    squeue_result = ensure_remote_success(
        run_remote_command(
            submission.target,
            f"squeue -h -j {shlex.quote(submission.job_id)} -o %T",
        ),
        action=f"query remote job {submission.job_id}",
    )
    state = _first_output_line(squeue_result.stdout)
    if state:
        return state
    return read_remote_job_final_state(submission)


def read_remote_job_final_state(submission: RemoteJobSubmission) -> str | None:
    result = run_remote_command(
        submission.target,
        " ".join(
            [
                "sacct",
                "-n",
                "-X",
                f"-j {shlex.quote(submission.job_id)}",
                "--format=State",
            ]
        ),
    )
    if result.returncode != 0:
        return None
    state = _first_output_line(result.stdout)
    if state is None:
        return None
    return state.split()[0]


def _workflow_spec(
    target: RemoteExecutionTarget,
    task: WorkflowTask,
) -> ExecutionWorkflowSpec:
    workflow_attr = _WORKFLOW_SPEC_ATTRS.get(task)
    if workflow_attr is not None:
        return getattr(target.spec.workflows, workflow_attr)
    raise SpiceOperatorError(f"Remote execution is not supported for workflow: {task.value}")


def _render_sbatch_script(
    *,
    target: RemoteExecutionTarget,
    task: WorkflowTask,
    workflow_spec: ExecutionWorkflowSpec,
    cli_args: list[str],
    log_path_template: Path,
) -> str:
    repo_root = shlex.quote(str(target.spec.paths.repo_root))
    venv_activate_path = shlex.quote(str(target.spec.paths.venv_activate_path))
    storage_root = str(target.spec.paths.storage_root)
    cli_command = shlex.join(
        [
            str(target.spec.paths.spice_path),
            task.value,
            *cli_args,
            "--storage-root",
            storage_root,
        ]
    )
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=spice-{task.value}",
        f"#SBATCH --partition={workflow_spec.partition}",
        f"#SBATCH --gres=gpu:{workflow_spec.gpus}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={workflow_spec.cpus_per_task}",
        f"#SBATCH --mem={workflow_spec.memory_gb}G",
        f"#SBATCH --time={workflow_spec.time_limit}",
        f"#SBATCH --output={log_path_template}",
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(str(target.spec.paths.log_root))}",
        f"mkdir -p {shlex.quote(storage_root)}",
        f"cd {repo_root}",
        f"source {venv_activate_path}",
        "export PYTHONUNBUFFERED=1",
        f"exec {cli_command}",
    ]
    return "\n".join(lines) + "\n"


def _build_sbatch_submit_command(*, dependency: str | None) -> str:
    command = ["cat", "|", "sbatch"]
    if dependency is not None:
        command.append(f"--dependency={shlex.quote(dependency)}")
    return " ".join(command)


def _first_output_line(payload: str) -> str | None:
    for line in payload.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None
