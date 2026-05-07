"""Target-bound SSH/SLURM execution session."""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import typed_groups as typed
from ..config.models import WorkflowTask
from ..config.resolved_workflows import ResolvedWorkflowConfig
from ..config.workflow_snapshots import workflow_config_snapshot_json
from ..core.errors import SpiceOperatorError
from .models import ExecutionSpec, ExecutionWorkflowSpec

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
    job_id: str
    log_path: Path


def open_execution_session(target_name: str) -> ExecutionSession:
    return ExecutionSession(_load_execution_target(target_name))


def _load_execution_target(name: str) -> ExecutionTarget:
    return ExecutionTarget(
        name=name,
        spec=typed.load(typed.EXECUTION, name),
    )


def _ensure_execution_success(
    result: subprocess.CompletedProcess[str],
    *,
    action: str,
) -> subprocess.CompletedProcess[str]:
    if result.returncode == 0:
        return result
    message = (result.stderr or result.stdout).strip()
    suffix = f": {message}" if message else ""
    raise SpiceOperatorError(f"{action} failed{suffix}")


@dataclass(frozen=True, slots=True)
class ExecutionSession:
    target: ExecutionTarget

    @property
    def follow_by_default(self) -> bool:
        return self.target.spec.follow_by_default

    def build_shell_argv(self, command: str) -> list[str]:
        return [
            "ssh",
            self.target.ssh_destination,
            "bash",
            "-lc",
            shlex.quote(command),
        ]

    def run_command(
        self,
        command: str,
        *,
        input_text: str | None = None,
        capture_output: bool = True,
        check_action: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            self.build_shell_argv(command),
            input=input_text,
            text=True,
            capture_output=capture_output,
            check=False,
        )
        if check_action is not None:
            return _ensure_execution_success(result, action=check_action)
        return result

    def run_module(
        self,
        module: str,
        args: list[str],
        *,
        capture_output: bool = True,
        check_action: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        python_path = shlex.quote(str(self.target.spec.paths.python_path))
        repo_root = shlex.quote(str(self.target.spec.paths.repo_root))
        module_name = shlex.quote(module)
        argv = shlex.join(args)
        return self.run_command(
            f"cd {repo_root} && {python_path} -m {module_name} {argv}",
            capture_output=capture_output,
            check_action=check_action,
        )

    def rsync_to(self, *, source_root: Path, destination_root: Path) -> None:
        result = subprocess.run(
            [
                "rsync",
                "-a",
                f"{source_root.as_posix()}/",
                f"{self.target.ssh_destination}:{destination_root.as_posix()}/",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        _ensure_execution_success(result, action=f"rsync to {destination_root}")

    def rsync_from(self, *, source_root: Path, destination_root: Path) -> None:
        result = subprocess.run(
            [
                "rsync",
                "-a",
                f"{self.target.ssh_destination}:{source_root.as_posix()}/",
                f"{destination_root.as_posix()}/",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        _ensure_execution_success(result, action=f"rsync from {source_root}")

    def submit_workflow(
        self,
        task: WorkflowTask,
        *,
        config: ResolvedWorkflowConfig,
        dependency: str | None = None,
    ) -> ExecutionJobSubmission:
        workflow_spec = _workflow_spec(self.target, task)
        log_path_template = self.target.spec.paths.log_root / f"spice-{task.value}-%j.out"
        script = _render_sbatch_script(
            target=self.target,
            task=task,
            workflow_spec=workflow_spec,
            config=config,
            log_path_template=log_path_template,
        )
        submit_command = " && ".join(
            [
                f"mkdir -p {shlex.quote(str(self.target.spec.paths.log_root))}",
                f"mkdir -p {shlex.quote(str(self.target.spec.paths.storage_root))}",
                _build_sbatch_submit_command(dependency=dependency),
            ]
        )
        result = self.run_command(
            submit_command,
            input_text=script,
            check_action=f"submit {task.value}",
        )
        match = _SBATCH_JOB_ID_PATTERN.search(result.stdout)
        if match is None:
            raise SpiceOperatorError(
                f"submit {task.value} failed: could not parse job id from sbatch output"
            )
        job_id = match.group("job_id")
        return ExecutionJobSubmission(
            task=task,
            job_id=job_id,
            log_path=Path(str(log_path_template).replace("%j", job_id)),
        )

    def follow_job(self, submission: ExecutionJobSubmission) -> str | None:
        tail_process = subprocess.Popen(
            self.build_shell_argv(
                (
                    "while [ ! -f "
                    f"{shlex.quote(str(submission.log_path))}"
                    " ]; do sleep 2; done; "
                    f"tail -n +1 -F {shlex.quote(str(submission.log_path))}"
                ),
            ),
            text=True,
            stdout=sys.stderr,
        )
        try:
            while True:
                state = self.read_job_state(submission)
                if state in _FINAL_JOB_STATES:
                    time.sleep(2)
                    return state
                if state is None:
                    return self.read_job_final_state(submission)
                time.sleep(5)
        finally:
            if tail_process.poll() is None:
                tail_process.terminate()
                try:
                    tail_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    tail_process.kill()
                    tail_process.wait(timeout=5)

    def read_job_state(self, submission: ExecutionJobSubmission) -> str | None:
        squeue_result = self.run_command(
            f"squeue -h -j {shlex.quote(submission.job_id)} -o %T",
            check_action=f"query job {submission.job_id}",
        )
        state = _first_output_line(squeue_result.stdout)
        if state:
            return state
        return self.read_job_final_state(submission)

    def read_job_final_state(self, submission: ExecutionJobSubmission) -> str | None:
        result = self.run_command(
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

    def remote_git_commit(self) -> str:
        repo_root = shlex.quote(str(self.target.spec.paths.repo_root))
        result = self.run_command(
            f"cd {repo_root} && git rev-parse HEAD",
            check_action=f"read remote git commit for {self.target.name}",
        )
        commit = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        if not commit:
            raise SpiceOperatorError(
                f"read remote git commit for {self.target.name} returned no output"
            )
        return commit


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
    config: ResolvedWorkflowConfig,
    log_path_template: Path,
) -> str:
    repo_root = shlex.quote(str(target.spec.paths.repo_root))
    venv_activate_path = shlex.quote(str(target.spec.paths.venv_activate_path))
    storage_root = str(target.spec.paths.storage_root)
    python_path = str(target.spec.paths.python_path)
    config_json = workflow_config_snapshot_json(
        config,
        storage_root_override=target.spec.paths.storage_root,
    )
    remote_command = shlex.join(
        [
            python_path,
            "-m",
            "spice.execution.remote_runner",
            task.value,
            config_json,
        ]
    )
    log_path_expression = str(log_path_template).replace("%j", "${SLURM_JOB_ID:-unknown}")
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
        f"export SPICE_EXECUTION_TARGET={shlex.quote(target.name)}",
        f"export SPICE_WORKFLOW_TASK={shlex.quote(task.value)}",
        'export SPICE_EXECUTION_JOB_ID="${SLURM_JOB_ID:-}"',
        'export SPICE_EXECUTION_REF="slurm:${SLURM_JOB_ID:-}"',
        f'export SPICE_EXECUTION_LOG_PATH="{log_path_expression}"',
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
