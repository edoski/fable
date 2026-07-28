"""Submit one typed workflow through SSH and Slurm."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import Field, ValidationInfo, field_validator, model_validator

from .config import TuneRequest, WorkflowRequest
from .records import StrictFrozenRecord

_NonEmptyString = Annotated[str, Field(min_length=1)]
_NonNegativeInt = Annotated[int, Field(ge=0)]
_PositiveInt = Annotated[int, Field(gt=0)]
_JOB_ID_PATTERN = re.compile(r"([0-9]+)(?:;[^;\r\n]+)?\n?")


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
        _render_script(remote, request.model_dump_json(), "workflow"),
    )


def submit_candidate(request: TuneRequest, method_index: int) -> int:
    remote = _load_remote()
    candidate_json = CandidateProcessInput(
        request=request,
        method_index=method_index,
    ).model_dump_json()
    return _invoke_sbatch(remote, _render_script(remote, candidate_json, "candidate"))


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


def _render_script(
    remote: _Remote,
    process_input_json: str,
    leaf: Literal["workflow", "candidate"],
) -> str:
    resources = remote.resources
    return "\n".join(
        (
            "#!/bin/bash",
            f"#SBATCH --partition={resources.partition}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            f"#SBATCH --gres={resources.gres}",
            f"#SBATCH --cpus-per-task={resources.cpus_per_task}",
            f"#SBATCH --mem={resources.memory_gb}G",
            f"#SBATCH --time={resources.time_limit}",
            f"#SBATCH --output={remote.log_root}/%j.out",
            f"#SBATCH --chdir={shlex.quote(remote.storage_root)}",
            f"export STORAGE_ROOT={shlex.quote(remote.storage_root)}",
            (
                f"exec apptainer run --nv --bind {shlex.quote(remote.storage_root)} "
                f"{shlex.quote(remote.image)} remote {leaf} <<'FABLE_REQUEST'"
            ),
            process_input_json,
            "FABLE_REQUEST",
            "",
        )
    )


def _parse_job_id(output: str) -> int:
    match = _JOB_ID_PATTERN.fullmatch(output)
    if match is None or (job_id := int(match.group(1))) <= 0:
        raise ValueError(f"invalid sbatch --parsable output: {output!r}")
    return job_id
