"""Submit one typed workflow through SSH and Slurm."""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import Field, ValidationInfo, field_validator, model_validator

from .config import TuneRequest, WorkflowRequest
from .records import StrictFrozenRecord

_NonEmptyString = Annotated[str, Field(min_length=1)]
_NonNegativeInt = Annotated[int, Field(ge=0)]
_PositiveInt = Annotated[int, Field(gt=0)]
MAX_ALLOCATION_PROCESS_COUNT = 3


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


def submit_workflows(requests: Sequence[WorkflowRequest]) -> int:
    """Submit independent workflows as isolated one-GPU steps in one Slurm job."""

    return _submit_allocation(requests, "workflow")


def submit_candidates(candidates: Sequence[CandidateProcessInput]) -> int:
    """Submit independent candidates as isolated one-GPU steps in one Slurm job."""

    return _submit_allocation(candidates, "candidate")


def _submit_allocation(
    inputs: Sequence[StrictFrozenRecord], leaf: Literal["workflow", "candidate"]
) -> int:
    if not 1 <= len(inputs) <= MAX_ALLOCATION_PROCESS_COUNT:
        raise ValueError("an allocation requires one to three process inputs")
    remote = _Remote.model_validate(yaml.safe_load(Path("REMOTE.yaml").read_bytes()))
    return _invoke_sbatch(
        remote,
        _render_allocation_script(
            remote, tuple(process_input.model_dump_json() for process_input in inputs), leaf
        ),
    )


def _invoke_sbatch(remote: _Remote, script: str) -> int:
    result = subprocess.run(
        ["ssh", "-T", "-o", "BatchMode=yes", remote.ssh, "sbatch", "--parsable"],
        input=script,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    return _parse_job_id(result.stdout)


def _render_allocation_script(
    remote: _Remote, process_inputs_json: tuple[str, ...], leaf: Literal["workflow", "candidate"]
) -> str:
    resources = remote.resources
    task_count = len(process_inputs_json)

    def render_step(slot: int, process_input_json: str) -> str:
        command = (
            "srun --exclusive --exact --nodes=1 --ntasks=1 "
            f"--gres={resources.gres} "
            f"--cpus-per-task={resources.cpus_per_task} "
            f"--mem={resources.memory_gb}G "
            f"--output={shlex.quote(remote.log_root)}/${{SLURM_JOB_ID}}-{slot}.out "
            f"--error={shlex.quote(remote.log_root)}/${{SLURM_JOB_ID}}-{slot}.out "
            f"apptainer run --nv --bind {shlex.quote(remote.storage_root)} "
            f"{shlex.quote(remote.image)} remote {leaf}"
        )
        return f"""\
{command} <<'FABLE_REQUEST_{slot}' &
{process_input_json}
FABLE_REQUEST_{slot}
pids+=("$!")"""

    steps = "\n".join(
        render_step(slot, process_input_json)
        for slot, process_input_json in enumerate(process_inputs_json)
    )
    return f"""\
#!/bin/bash
#SBATCH --partition={resources.partition}
#SBATCH --nodes=1
#SBATCH --ntasks={task_count}
#SBATCH --gres={_scaled_gres(resources.gres, task_count)}
#SBATCH --cpus-per-task={resources.cpus_per_task}
#SBATCH --mem={resources.memory_gb * task_count}G
#SBATCH --time={resources.time_limit}
#SBATCH --output={remote.log_root}/%j.out
#SBATCH --chdir={shlex.quote(remote.storage_root)}
export STORAGE_ROOT={shlex.quote(remote.storage_root)}
pids=()
{steps}
status=0
for pid in "${{pids[@]}}"; do
    if ! wait "$pid"; then status=1; fi
done
exit "$status"
"""


def _scaled_gres(gres: str, count: int) -> str:
    resource, separator, configured_count = gres.rpartition(":")
    if not separator or configured_count != "1":
        raise ValueError("packed execution requires a one-GPU GRES")
    return f"{resource}:{count}"


def _parse_job_id(output: str) -> int:
    job_id = int(output.partition(";")[0])
    if job_id <= 0:
        raise ValueError(f"invalid sbatch --parsable output: {output!r}")
    return job_id
