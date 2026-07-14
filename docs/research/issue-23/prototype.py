"""Interactive shell for the disposable Issue 23 synthetic prototype."""

from __future__ import annotations

import argparse
import json
import os

from prototype_task import run_observations


def _render(section: str) -> None:
    observations = run_observations()
    os.system("clear")
    print("\033[1mIssue 23 — shared task + three concrete models\033[0m")
    print("\033[2mSynthetic tensors only; disposable planning evidence.\033[0m\n")
    if section == "all":
        print(json.dumps(observations, indent=2))
    else:
        print(json.dumps({section: observations[section]}, indent=2))
    print(
        "\n\033[1m[m]\033[0m models  \033[1m[t]\033[0m task seam  "
        "\033[1m[l]\033[0m loss  \033[1m[d]\033[0m decode  \033[1m[s]\033[0m scorer  "
        "\033[1m[e]\033[0m evaluator  \033[1m[v]\033[0m serving  "
        "\033[1m[a]\033[0m all  \033[1m[q]\033[0m quit"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="print every observation as JSON")
    args = parser.parse_args()
    if args.all:
        print(json.dumps(run_observations(), indent=2))
        return

    sections = {
        "m": "architectures",
        "t": "task_seam",
        "l": "loss",
        "d": "decode",
        "s": "scorer",
        "e": "evaluator_input",
        "v": "serving",
        "a": "all",
    }
    selected = "all"
    while True:
        _render(selected)
        key = input("\n> ").strip().lower()
        if key == "q":
            return
        selected = sections.get(key, selected)


if __name__ == "__main__":
    main()
