from __future__ import annotations

import os
import subprocess
from pathlib import Path

from tests.support import REPO_ROOT


def _write_script(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def test_spice_awake_uses_caffeinate_on_macos_when_available(tmp_path) -> None:
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    caffeinate_log = tmp_path / "caffeinate.log"
    target_log = tmp_path / "target.log"

    _write_script(
        shim_dir / "uname",
        "#!/bin/sh\n"
        "printf 'Darwin\\n'\n",
    )
    _write_script(
        shim_dir / "caffeinate",
        "#!/bin/sh\n"
        "printf '%s\\n' \"$@\" > \"$SPICE_AWAKE_CAFFEINATE_LOG\"\n"
        "if [ \"$1\" = \"-i\" ]; then\n"
        "    shift\n"
        "fi\n"
        "exec \"$@\"\n",
    )
    _write_script(
        shim_dir / "target",
        "#!/bin/sh\n"
        "printf '%s\\n' \"$@\" > \"$SPICE_AWAKE_TARGET_LOG\"\n",
    )

    env = os.environ.copy()
    env["PATH"] = f"{shim_dir}:{env['PATH']}"
    env["SPICE_AWAKE_CAFFEINATE_LOG"] = str(caffeinate_log)
    env["SPICE_AWAKE_TARGET_LOG"] = str(target_log)

    result = subprocess.run(
        [str(REPO_ROOT / "bin" / "spice-awake"), str(shim_dir / "target"), "alpha", "beta"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert caffeinate_log.read_text(encoding="utf-8").splitlines() == [
        "-i",
        str(shim_dir / "target"),
        "alpha",
        "beta",
    ]
    assert target_log.read_text(encoding="utf-8").splitlines() == ["alpha", "beta"]


def test_spice_awake_executes_command_directly_and_preserves_exit_code_off_macos(
    tmp_path,
) -> None:
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    target_log = tmp_path / "target.log"
    caffeinate_log = tmp_path / "unexpected.log"

    _write_script(
        shim_dir / "uname",
        "#!/bin/sh\n"
        "printf 'Linux\\n'\n",
    )
    _write_script(
        shim_dir / "caffeinate",
        "#!/bin/sh\n"
        "printf 'unexpected\\n' > \"$SPICE_AWAKE_UNEXPECTED_LOG\"\n"
        "exit 99\n",
    )
    _write_script(
        shim_dir / "target",
        "#!/bin/sh\n"
        "printf '%s\\n' \"$@\" > \"$SPICE_AWAKE_TARGET_LOG\"\n"
        "exit 7\n",
    )

    env = os.environ.copy()
    env["PATH"] = f"{shim_dir}:{env['PATH']}"
    env["SPICE_AWAKE_TARGET_LOG"] = str(target_log)
    env["SPICE_AWAKE_UNEXPECTED_LOG"] = str(caffeinate_log)

    result = subprocess.run(
        [str(REPO_ROOT / "bin" / "spice-awake"), str(shim_dir / "target"), "direct"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 7
    assert target_log.read_text(encoding="utf-8").splitlines() == ["direct"]
    assert not caffeinate_log.exists()
