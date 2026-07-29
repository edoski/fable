"""Submit one typed workflow through SSH and Slurm."""

from __future__ import annotations

import re
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import UUID

import yaml
from pydantic import Field, ValidationInfo, field_validator, model_validator

from .config import EvaluateRequest, TrainRequest, TuneRequest, WorkflowRequest
from .records import StrictFrozenRecord

_NonEmptyString = Annotated[str, Field(min_length=1)]
_NonNegativeInt = Annotated[int, Field(ge=0)]
_PositiveInt = Annotated[int, Field(gt=0)]
_JOB_ID_PATTERN = re.compile(r"([0-9]+)(?:;[^;\r\n]+)?\n?")
MAX_PACKED_PROCESS_COUNT = 3


class _Resources(StrictFrozenRecord):
    partition: _NonEmptyString
    gres: str
    cpus_per_task: _PositiveInt
    memory_gb: _PositiveInt
    time_limit: _NonEmptyString


class _Remote(StrictFrozenRecord):
    ssh: _NonEmptyString
    image: _NonEmptyString
    storage_root: _NonEmptyString
    log_root: _NonEmptyString
    resources: _Resources

    @field_validator("image", "storage_root", "log_root")
    @classmethod
    def validate_absolute_path(cls, value: str, info: ValidationInfo) -> str:  # noqa: V107
        if not Path(value).is_absolute():
            raise ValueError(f"{info.field_name} must be an absolute path")
        return value


class CandidateProcessInput(StrictFrozenRecord):
    request: TuneRequest
    method_index: _NonNegativeInt

    @model_validator(mode="after")
    def validate_method_index(self) -> Self:
        self.request.method_at(self.method_index)
        return self


def submit(request: WorkflowRequest) -> int:
    """Submit one Train or Evaluate request and return its positive Slurm ID."""

    remote = _load_remote()
    return _invoke_sbatch(
        remote,
        _render_allocation_script(
            remote,
            (request.model_dump_json(),),
            "workflow",
        ),
    )


def submit_workflow_batch(requests: Sequence[WorkflowRequest]) -> int:
    """Submit independent workflows as isolated one-GPU steps in one Slurm job."""

    requests = tuple(requests)
    _require_packed_count(requests)
    identities = tuple(_workflow_identity(request) for request in requests)
    if len(set(identities)) != len(identities):
        raise ValueError("packed workflow identities must be unique")
    remote = _load_remote()
    return _invoke_sbatch(
        remote,
        _render_allocation_script(
            remote,
            tuple(request.model_dump_json() for request in requests),
            "workflow",
        ),
    )


def submit_candidate(request: TuneRequest, method_index: int) -> int:
    remote = _load_remote()
    candidate_json = CandidateProcessInput(
        request=request,
        method_index=method_index,
    ).model_dump_json()
    return _invoke_sbatch(
        remote,
        _render_allocation_script(remote, (candidate_json,), "candidate"),
    )


def submit_candidate_batch(candidates: Sequence[CandidateProcessInput]) -> int:
    """Submit independent candidates as isolated one-GPU steps in one Slurm job."""

    candidates = tuple(candidates)
    _require_packed_count(candidates)
    slots = tuple(
        (candidate.request.study_id, candidate.method_index) for candidate in candidates
    )
    if len(set(slots)) != len(slots):
        raise ValueError("packed candidate slots must be unique")
    remote = _load_remote()
    return _invoke_sbatch(
        remote,
        _render_allocation_script(
            remote,
            tuple(candidate.model_dump_json() for candidate in candidates),
            "candidate",
        ),
    )


def _load_remote() -> _Remote:
    return _Remote.model_validate(yaml.safe_load(Path("REMOTE.yaml").read_bytes()))


def _invoke_sbatch(remote: _Remote, script: str) -> int:
    result = subprocess.run(
        [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            remote.ssh,
            "sbatch",
            "--parsable",
        ],
        input=script,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    return _parse_job_id(result.stdout)


def _render_allocation_script(
    remote: _Remote,
    process_inputs_json: tuple[str, ...],
    leaf: Literal["workflow", "candidate"],
) -> str:
    resources = remote.resources
    task_count = len(process_inputs_json)
    lines = [
        "#!/bin/bash",
        f"#SBATCH --partition={resources.partition}",
        "#SBATCH --nodes=1",
        f"#SBATCH --ntasks={task_count}",
        f"#SBATCH --gres={_scaled_gres(resources.gres, task_count)}",
        f"#SBATCH --cpus-per-task={resources.cpus_per_task}",
        f"#SBATCH --mem={resources.memory_gb * task_count}G",
        f"#SBATCH --time={resources.time_limit}",
        f"#SBATCH --output={remote.log_root}/%j.out",
        f"#SBATCH --chdir={shlex.quote(remote.storage_root)}",
        f"export STORAGE_ROOT={shlex.quote(remote.storage_root)}",
        "pids=()",
    ]
    for slot, process_input_json in enumerate(process_inputs_json):
        step_output = (
            ""
            if task_count == 1
            else (
                f"--output={shlex.quote(remote.log_root)}/${{SLURM_JOB_ID}}-{slot}.out "
                f"--error={shlex.quote(remote.log_root)}/${{SLURM_JOB_ID}}-{slot}.out "
            )
        )
        lines.extend(
            (
                (
                    "srun --exclusive --exact --nodes=1 --ntasks=1 "
                    f"--gres={resources.gres} "
                    f"--cpus-per-task={resources.cpus_per_task} "
                    f"--mem={resources.memory_gb}G "
                    f"{step_output}"
                    f"apptainer run --nv --bind {shlex.quote(remote.storage_root)} "
                    f"{shlex.quote(remote.image)} remote {leaf} "
                    f"<<'FABLE_REQUEST_{slot}' &"
                ),
                process_input_json,
                f"FABLE_REQUEST_{slot}",
                'pids+=("$!")',
            )
        )
    lines.extend(
        (
            "status=0",
            'for pid in "${pids[@]}"; do',
            '    if ! wait "$pid"; then status=1; fi',
            "done",
            'exit "$status"',
            "",
        )
    )
    return "\n".join(lines)


def _require_packed_count(inputs: Sequence[object]) -> None:
    if not 2 <= len(inputs) <= MAX_PACKED_PROCESS_COUNT:
        raise ValueError("a packed job requires two or three process inputs")


def _workflow_identity(request: WorkflowRequest) -> tuple[str, UUID]:
    match request:
        case TrainRequest():
            return request.workflow, request.artifact_id
        case EvaluateRequest():
            return request.workflow, request.evaluation_id


def _scaled_gres(gres: str, count: int) -> str:
    resource, separator, configured_count = gres.rpartition(":")
    if not separator or configured_count != "1":
        raise ValueError("packed execution requires a one-GPU GRES")
    return f"{resource}:{count}"


def _parse_job_id(output: str) -> int:
    match = _JOB_ID_PATTERN.fullmatch(output)
    if match is None or (job_id := int(match.group(1))) <= 0:
        raise ValueError(f"invalid sbatch --parsable output: {output!r}")
    return job_id
