from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace

from spice.config.models import ExecutionWorkflowSpec, WorkflowTask
from spice.remote.shell import build_remote_shell_argv, run_remote_command
from spice.remote.workflows import (
    RemoteJobSubmission,
    _render_sbatch_script,
    follow_remote_job,
    submit_remote_workflow,
)


def _target() -> SimpleNamespace:
    return SimpleNamespace(
        ssh_destination="edoardo.galli3@giano.cs.unibo.it",
        spec=SimpleNamespace(follow_by_default=True),
    )


def test_build_remote_shell_argv_quotes_full_command() -> None:
    argv = build_remote_shell_argv(_target(), "mkdir -p /scratch/test && cat | sbatch")

    assert argv == [
        "ssh",
        "edoardo.galli3@giano.cs.unibo.it",
        "bash",
        "-lc",
        "'mkdir -p /scratch/test && cat | sbatch'",
    ]


def test_run_remote_command_passes_quoted_command(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args[0]
        captured["kwargs"] = kwargs
        return CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")

    monkeypatch.setattr("spice.remote.shell.subprocess.run", fake_run)

    result = run_remote_command(_target(), "mkdir -p /scratch/test && cat | sbatch")

    assert result.returncode == 0
    assert captured["args"] == [
        "ssh",
        "edoardo.galli3@giano.cs.unibo.it",
        "bash",
        "-lc",
        "'mkdir -p /scratch/test && cat | sbatch'",
    ]


def test_follow_remote_job_uses_quoted_tail_command(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakePopen:
        def __init__(self, args, *, text):
            captured["args"] = args
            captured["text"] = text

        def poll(self):
            return 0

        def terminate(self) -> None:
            return None

        def wait(self, timeout=None) -> int:
            return 0

    monkeypatch.setattr("spice.remote.workflows.subprocess.Popen", FakePopen)
    monkeypatch.setattr("spice.remote.workflows.read_remote_job_state", lambda _submission: None)
    monkeypatch.setattr(
        "spice.remote.workflows.read_remote_job_final_state",
        lambda _submission: "COMPLETED",
    )

    submission = RemoteJobSubmission(
        task=WorkflowTask.TRAIN,
        execution_name="disi_l40",
        target=_target(),
        job_id="12345",
        log_path=tmp_path / "spice-train-12345.out",
    )

    state = follow_remote_job(submission)

    assert state == "COMPLETED"
    assert captured["args"] == [
        "ssh",
        "edoardo.galli3@giano.cs.unibo.it",
        "bash",
        "-lc",
        (
            "'while [ ! -f "
            + str(submission.log_path)
            + " ]; do sleep 2; done; tail -n +1 -F "
            + str(submission.log_path)
            + "'"
        ),
    ]
    assert captured["text"] is True


def test_render_sbatch_script_execs_spice_command(tmp_path: Path) -> None:
    target = SimpleNamespace(
        spec=SimpleNamespace(
            paths=SimpleNamespace(
                repo_root=Path("/repo"),
                venv_activate_path=Path("/venv/bin/activate"),
                storage_root=Path("/storage"),
                log_root=tmp_path,
                spice_path=Path("/venv/bin/spice"),
            )
        )
    )
    script = _render_sbatch_script(
        target=target,
        task=WorkflowTask.TRAIN,
        workflow_spec=ExecutionWorkflowSpec(
            partition="l40",
            gpus=1,
            cpus_per_task=4,
            memory_gb=24,
            time_limit="00:10:00",
        ),
        cli_args=["--preset", "icdcs_2026"],
        log_path_template=tmp_path / "spice-train-%j.out",
    )

    assert "\nexec /venv/bin/spice train --preset icdcs_2026 --storage-root /storage\n" in script


def test_submit_remote_workflow_forwards_sbatch_dependency(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    target = SimpleNamespace(
        name="disi_l40",
        spec=SimpleNamespace(
            paths=SimpleNamespace(
                repo_root=Path("/repo"),
                venv_activate_path=Path("/venv/bin/activate"),
                storage_root=Path("/storage"),
                log_root=tmp_path,
                spice_path=Path("/venv/bin/spice"),
            ),
            workflows=SimpleNamespace(
                train=ExecutionWorkflowSpec(
                    partition="l40",
                    gpus=1,
                    cpus_per_task=4,
                    memory_gb=24,
                    time_limit="00:10:00",
                ),
            ),
        ),
    )

    monkeypatch.setattr("spice.remote.workflows.resolve_remote_target", lambda _name: target)

    def fake_run_remote_command(_target, command: str, *, input_text: str | None = None):
        captured["command"] = command
        captured["input_text"] = input_text
        return CompletedProcess(
            args=["ssh", "giano"],
            returncode=0,
            stdout="Submitted batch job 12345\n",
            stderr="",
        )

    monkeypatch.setattr("spice.remote.workflows.run_remote_command", fake_run_remote_command)

    submission = submit_remote_workflow(
        WorkflowTask.TRAIN,
        cli_args=["--preset", "icdcs_2026"],
        execution_name="disi_l40",
        dependency="afterok:99999",
    )

    assert submission.job_id == "12345"
    assert captured["command"] == (
        f"mkdir -p {tmp_path} && mkdir -p /storage && "
        "cat | sbatch --dependency=afterok:99999"
    )
    assert isinstance(captured["input_text"], str)
