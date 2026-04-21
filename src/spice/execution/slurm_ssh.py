"""SLURM-over-SSH execution backend."""

from __future__ import annotations

import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from ..config.resolution import WorkflowRequest
from ..config.models import WorkflowTask
from ..config.registry import load_named_group
from ..core.errors import SpiceOperatorError
from .models import ExecutionSpec, ExecutionWorkflowSpec

_EXECUTION_SPEC_NAME = "disi_l40"
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
class ExecutionTarget:
    name: str
    spec: ExecutionSpec

    @property
    def ssh_destination(self) -> str:
        return f"{self.spec.ssh.user}@{self.spec.ssh.host}"


@dataclass(frozen=True, slots=True)
class ExecutionJobSubmission:
    task: WorkflowTask
    target: ExecutionTarget
    job_id: str
    log_path: Path


def build_execution_shell_argv(target: ExecutionTarget, command: str) -> list[str]:
    return [
        "ssh",
        target.ssh_destination,
        "bash",
        "-lc",
        shlex.quote(command),
    ]


def load_execution_target() -> ExecutionTarget:
    payload = load_named_group(_EXECUTION_SPEC_NAME, "execution")
    return ExecutionTarget(
        name=_EXECUTION_SPEC_NAME,
        spec=ExecutionSpec.model_validate(payload),
    )


def run_execution_module(
    target: ExecutionTarget,
    module: str,
    args: list[str],
    *,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    python_path = shlex.quote(str(target.spec.paths.python_path))
    repo_root = shlex.quote(str(target.spec.paths.repo_root))
    module_name = shlex.quote(module)
    argv = shlex.join(args)
    return run_execution_command(
        target,
        f"cd {repo_root} && {python_path} -m {module_name} {argv}",
        capture_output=capture_output,
    )


def run_execution_command(
    target: ExecutionTarget,
    command: str,
    *,
    input_text: str | None = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        build_execution_shell_argv(target, command),
        input=input_text,
        text=True,
        capture_output=capture_output,
        check=False,
    )


def run_rsync_to_execution_target(
    target: ExecutionTarget,
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
    ensure_execution_success(result, action=f"rsync to {destination_root}")


def run_rsync_from_execution_target(
    target: ExecutionTarget,
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
    ensure_execution_success(result, action=f"rsync from {source_root}")


def ensure_execution_success(
    result: subprocess.CompletedProcess[str],
    *,
    action: str,
) -> subprocess.CompletedProcess[str]:
    if result.returncode == 0:
        return result
    message = (result.stderr or result.stdout).strip()
    suffix = f": {message}" if message else ""
    raise SpiceOperatorError(f"{action} failed{suffix}")


def submit_execution_workflow(
    task: WorkflowTask,
    *,
    request: WorkflowRequest,
    dependency: str | None = None,
) -> ExecutionJobSubmission:
    target = load_execution_target()
    workflow_spec = _workflow_spec(target, task)
    log_path_template = target.spec.paths.log_root / f"spice-{task.value}-%j.out"
    script = _render_sbatch_script(
        target=target,
        task=task,
        workflow_spec=workflow_spec,
        request=request,
        log_path_template=log_path_template,
    )
    submit_command = " && ".join(
        [
            f"mkdir -p {shlex.quote(str(target.spec.paths.log_root))}",
            f"mkdir -p {shlex.quote(str(target.spec.paths.storage_root))}",
            _build_sbatch_submit_command(dependency=dependency),
        ]
    )
    result = ensure_execution_success(
        run_execution_command(target, submit_command, input_text=script),
        action=f"submit {task.value}",
    )
    match = _SBATCH_JOB_ID_PATTERN.search(result.stdout)
    if match is None:
        raise SpiceOperatorError(
            f"submit {task.value} failed: could not parse job id from sbatch output"
        )
    job_id = match.group("job_id")
    return ExecutionJobSubmission(
        task=task,
        target=target,
        job_id=job_id,
        log_path=Path(str(log_path_template).replace("%j", job_id)),
    )


def follow_execution_job(submission: ExecutionJobSubmission) -> str | None:
    tail_process = subprocess.Popen(
        build_execution_shell_argv(
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
            state = read_execution_job_state(submission)
            if state in _FINAL_JOB_STATES:
                time.sleep(2)
                return state
            if state is None:
                return read_execution_job_final_state(submission)
            time.sleep(5)
    finally:
        if tail_process.poll() is None:
            tail_process.terminate()
            try:
                tail_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tail_process.kill()
                tail_process.wait(timeout=5)


def read_execution_job_state(submission: ExecutionJobSubmission) -> str | None:
    squeue_result = ensure_execution_success(
        run_execution_command(
            submission.target,
            f"squeue -h -j {shlex.quote(submission.job_id)} -o %T",
        ),
        action=f"query job {submission.job_id}",
    )
    state = _first_output_line(squeue_result.stdout)
    if state:
        return state
    return read_execution_job_final_state(submission)


def read_execution_job_final_state(submission: ExecutionJobSubmission) -> str | None:
    result = run_execution_command(
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
    target: ExecutionTarget,
    task: WorkflowTask,
) -> ExecutionWorkflowSpec:
    workflow_attr = _WORKFLOW_SPEC_ATTRS.get(task)
    if workflow_attr is None:
        raise SpiceOperatorError(f"Execution backend does not support workflow: {task.value}")
    return getattr(target.spec.workflows, workflow_attr)


def _render_sbatch_script(
    *,
    target: ExecutionTarget,
    task: WorkflowTask,
    workflow_spec: ExecutionWorkflowSpec,
    request: WorkflowRequest,
    log_path_template: Path,
) -> str:
    repo_root = shlex.quote(str(target.spec.paths.repo_root))
    venv_activate_path = shlex.quote(str(target.spec.paths.venv_activate_path))
    storage_root = str(target.spec.paths.storage_root)
    python_path = str(target.spec.paths.python_path)
    remote_request = request.model_copy(update={"storage_root": target.spec.paths.storage_root})
    request_json = remote_request.model_dump_json(exclude_none=True)
    remote_command = shlex.join(
        [
            python_path,
            "-m",
            "spice.execution.remote_runner",
            task.value,
            request_json,
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
        f"exec {remote_command}",
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
