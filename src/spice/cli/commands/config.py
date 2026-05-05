"""Config command routing."""

from __future__ import annotations

import os
import shlex
import subprocess
from typing import Annotated

import typer

from ...config.groups import public_group_help
from ...core.errors import SpiceOperatorError
from ..errors import OperatorTyper

app = OperatorTyper(
    help="Query and edit saved YAML config specs.",
    no_args_is_help=True,
)
_CONFIG_GROUP_HELP = public_group_help()


def _print_config_names(names: list[str]) -> None:
    for name in names:
        typer.echo(name)


def _resolve_editor() -> str:
    for env_name in ("VISUAL", "EDITOR"):
        value = os.getenv(env_name)
        if value:
            return value
    return "nvim"


@app.command(
    "list",
    short_help="List saved config specs.",
    help="List saved config names for one config group.",
)
def config_list_command(
    group: Annotated[
        str,
        typer.Argument(metavar="GROUP", help=_CONFIG_GROUP_HELP),
    ],
) -> None:
    from ...config.groups import list_group_names, normalize_public_group_name

    _print_config_names(list_group_names(normalize_public_group_name(group)))


@app.command(
    "show",
    short_help="Show one saved config spec.",
    help="Print one saved config spec as canonical YAML.",
)
def config_show_command(
    group: Annotated[
        str,
        typer.Argument(metavar="GROUP", help=_CONFIG_GROUP_HELP),
    ],
    name: Annotated[
        str,
        typer.Argument(metavar="NAME", help="Saved spec name."),
    ],
) -> None:
    from ...config.groups import normalize_public_group_name, show_named_group

    typer.echo(show_named_group(normalize_public_group_name(group), name), nl=False)


@app.command(
    "edit",
    short_help="Edit one saved config spec.",
    help="Open the real YAML file in $VISUAL, else $EDITOR, else nvim. Seeds missing files.",
)
def config_edit_command(
    group: Annotated[
        str,
        typer.Argument(metavar="GROUP", help=_CONFIG_GROUP_HELP),
    ],
    name: Annotated[
        str,
        typer.Argument(metavar="NAME", help="Saved spec name."),
    ],
) -> None:
    from ...config.groups import ensure_named_group_file, normalize_public_group_name

    editor = _resolve_editor()
    path = ensure_named_group_file(normalize_public_group_name(group), name)
    try:
        subprocess.run([*shlex.split(editor), str(path)], check=True)
    except FileNotFoundError as exc:
        raise SpiceOperatorError(f"Editor not found: {editor}") from exc
    except subprocess.CalledProcessError as exc:
        raise SpiceOperatorError(f"Editor exited with status {exc.returncode}: {editor}") from exc
