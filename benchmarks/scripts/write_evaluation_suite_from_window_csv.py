from __future__ import annotations

import argparse
import csv
from pathlib import Path


def tag_values(row: dict[str, str]) -> list[str]:
    tags: list[str] = []
    for value in row.get("class_tags", "").split(";"):
        if not value:
            continue
        tags.append(value.replace(":", "_"))
    shortlist_class = row.get("shortlist_class", "")
    if shortlist_class:
        tags.append(f"shortlist_{shortlist_class}")
    return tags


def quote(value: str) -> str:
    return '"' + value.replace('"', '\\"') + '"'


def render_suite(suite_id: str, rows: list[dict[str, str]]) -> str:
    lines = [f"id: {suite_id}", "items:"]
    for row in rows:
        lines.extend(render_item(row, tag_values(row)))
    return "\n".join(lines) + "\n"


def unique_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    tags_by_id: dict[str, list[str]] = {}
    for row in rows:
        window_id = row["window_id"]
        if window_id not in merged:
            merged[window_id] = dict(row)
            tags_by_id[window_id] = []
        for tag in tag_values(row):
            if tag not in tags_by_id[window_id]:
                tags_by_id[window_id].append(tag)
    for window_id, row in merged.items():
        row["class_tags"] = ";".join(
            tag for tag in tags_by_id[window_id] if not tag.startswith("shortlist_")
        )
        row["shortlist_class"] = ""
        row["_merged_tags"] = ",".join(tags_by_id[window_id])
    return list(merged.values())


def tag_values_for_render(row: dict[str, str]) -> list[str]:
    if row.get("_merged_tags"):
        return row["_merged_tags"].split(",")
    return tag_values(row)


def render_unique_suite(suite_id: str, rows: list[dict[str, str]]) -> str:
    lines = [f"id: {suite_id}", "items:"]
    for row in unique_rows(rows):
        lines.extend(render_item(row, tag_values_for_render(row)))
    return "\n".join(lines) + "\n"


def render_item(row: dict[str, str], tags: list[str]) -> list[str]:
    lines = [f"  - id: {row['window_id']}"]
    if row.get("start_block") and row.get("block_count"):
        lines.extend(
            [
                f"    start_block: {int(float(row['start_block']))}",
                f"    block_count: {int(float(row['block_count']))}",
            ]
        )
    else:
        lines.extend(
            [
                f"    start: {quote(row['start_iso'])}",
                f"    duration_seconds: {int(float(row['duration_seconds']))}",
            ]
        )
    if tags:
        lines.append(f"    tags: [{', '.join(tags)}]")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open() as handle:
        rows = list(csv.DictReader(handle))
    unique = unique_rows(rows)
    args.output.write_text(render_unique_suite(args.suite_id, rows))
    print(f"wrote {args.output} rows={len(unique)} source_rows={len(rows)}")


if __name__ == "__main__":
    main()
