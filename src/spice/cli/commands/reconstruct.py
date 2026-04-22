"""Reference reconstruction analysis commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ...reconstruction import (
    DEFAULT_CHAINS,
    DEFAULT_DELAYS,
    new_run_name,
    run_current_parity_audit,
    run_reference_search,
    write_audit_artifacts,
    write_search_artifacts,
)

app = typer.Typer(
    help="Audit and search temporal reference-parity assumptions.",
    no_args_is_help=True,
)

_DEFAULT_REFERENCE_ROOT = Path("/Users/edo/dev/python/ICDCS-Model-Training")


def _resolved_choices(
    values: list[str] | None,
    *,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    return default if values is None or len(values) == 0 else tuple(values)


@app.command("audit", short_help="Audit the current SPICE parity path.")
def audit_command(
    preset: Annotated[
        str,
        typer.Option("--preset", metavar="PRESET", help="Named preset to audit."),
    ] = "icdcs_2026_professor",
    reference_root: Annotated[
        Path,
        typer.Option(
            "--reference-root",
            metavar="PATH",
            help="Path to the ICDCS-Model-Training reference repository.",
        ),
    ] = _DEFAULT_REFERENCE_ROOT,
    storage_root: Annotated[
        Path,
        typer.Option("--storage-root", metavar="PATH", help="Output root."),
    ] = Path("outputs"),
    run_name: Annotated[
        str | None,
        typer.Option("--run-name", metavar="NAME", help="Optional stable run directory name."),
    ] = None,
) -> None:
    resolved_run_name = run_name or new_run_name("audit")
    audit = run_current_parity_audit(
        preset=preset,
        reference_root=reference_root,
        storage_root=storage_root,
    )
    output_dir = write_audit_artifacts(
        audit=audit,
        storage_root=storage_root,
        run_name=resolved_run_name,
    )
    typer.echo(f"audit preset={preset} findings={len(audit.findings)} output={output_dir}")


@app.command("run", short_help="Run the full local reconstruction audit and search.")
def run_command(
    preset: Annotated[
        str,
        typer.Option("--preset", metavar="PRESET", help="Named preset to audit."),
    ] = "icdcs_2026_professor",
    reference_root: Annotated[
        Path,
        typer.Option(
            "--reference-root",
            metavar="PATH",
            help="Path to the ICDCS-Model-Training reference repository.",
        ),
    ] = _DEFAULT_REFERENCE_ROOT,
    chain: Annotated[
        list[str] | None,
        typer.Option(
            "--chain",
            metavar="CHAIN",
            help="Repeat to restrict the search to specific chains.",
        ),
    ] = None,
    delay_seconds: Annotated[
        list[int] | None,
        typer.Option(
            "--delay-seconds",
            metavar="SECONDS",
            help="Repeat to restrict the search to specific delay windows.",
        ),
    ] = None,
    storage_root: Annotated[
        Path,
        typer.Option("--storage-root", metavar="PATH", help="Output root."),
    ] = Path("outputs"),
    run_name: Annotated[
        str | None,
        typer.Option("--run-name", metavar="NAME", help="Optional stable run directory name."),
    ] = None,
) -> None:
    resolved_run_name = run_name or new_run_name("run")
    resolved_chains = _resolved_choices(chain, default=DEFAULT_CHAINS)
    resolved_delays = tuple(delay_seconds) if delay_seconds else DEFAULT_DELAYS
    audit = run_current_parity_audit(
        preset=preset,
        reference_root=reference_root,
        storage_root=storage_root,
    )
    search = run_reference_search(
        reference_root=reference_root,
        chains=resolved_chains,
        delays=resolved_delays,
    )
    audit_dir = write_audit_artifacts(
        audit=audit,
        storage_root=storage_root,
        run_name=resolved_run_name,
    )
    search_dir = write_search_artifacts(
        result=search,
        storage_root=storage_root,
        run_name=resolved_run_name,
    )
    typer.echo(
        " ".join(
            [
                "reconstruct",
                f"preset={preset}",
                f"chains={','.join(resolved_chains)}",
                f"delays={','.join(str(value) for value in resolved_delays)}",
                f"audit={audit_dir}",
                f"search={search_dir}",
            ]
        )
    )
