"""Artifact writers for reconstruction audit and search outputs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .search import ReconstructionSearchResult, best_candidates


def new_run_name(prefix: str) -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def analysis_root(storage_root: Path) -> Path:
    return storage_root / "analysis" / "reference_reconstruction"


def write_audit_artifacts(
    *,
    audit,
    storage_root: Path,
    run_name: str,
) -> Path:
    output_dir = analysis_root(storage_root) / "audit" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "audit.json", audit.payload())
    (output_dir / "audit.md").write_text(render_audit_markdown(audit), encoding="utf-8")
    return output_dir


def write_search_artifacts(
    *,
    result: ReconstructionSearchResult,
    storage_root: Path,
    run_name: str,
) -> Path:
    output_dir = analysis_root(storage_root) / "search" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "label_results": {
            key: [item.payload() for item in values]
            for key, values in result.label_results.items()
        },
        "feature_results": {
            key: [item.payload() for item in values]
            for key, values in result.feature_results.items()
        },
        "best_candidates": best_candidates(result),
    }
    _write_json(output_dir / "search.json", payload)
    (output_dir / "search.md").write_text(render_search_markdown(result), encoding="utf-8")
    return output_dir


def render_audit_markdown(audit) -> str:
    lines = [
        "# Current Parity Audit",
        "",
        f"- preset: `{audit.preset}`",
        f"- chain: `{audit.chain['name']}`",
        f"- problem: `{audit.problem['id']}`",
        f"- compiler: `{audit.compiler_runtime['id']}`",
        f"- dataset_builder: `{audit.dataset_builder['id']}`",
        f"- prediction: `{audit.prediction['family_id']}`",
        "",
        "## Findings",
        "",
    ]
    for finding in audit.findings:
        lines.append(f"- `{finding['area']}` `{finding['status']}`: {finding['detail']}")
    return "\n".join(lines) + "\n"


def render_search_markdown(result: ReconstructionSearchResult) -> str:
    lines = [
        "# Reference Reconstruction Search",
        "",
    ]
    for key in sorted(result.label_results):
        lines.append(f"## {key}")
        lines.append("")
        label_candidates = result.label_results[key][:3]
        feature_candidates = result.feature_results.get(key, [])[:3]
        lines.append("### Label Candidates")
        lines.append("")
        for item in label_candidates:
            lines.append(
                "- "
                f"`{item.label_candidate.candidate_id}` + `{item.split_candidate.candidate_id}` "
                f"score={item.score:.4f} train_classes={item.train_unique_classes} "
                f"expected={item.expected_unique_classes}"
            )
        lines.append("")
        lines.append("### Feature Candidates")
        lines.append("")
        for item in feature_candidates:
            note = item.notes[0] if item.notes else ""
            lines.append(
                f"- `{item.feature_candidate.candidate_id}` score={item.score:.4f} {note}"
            )
        lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
