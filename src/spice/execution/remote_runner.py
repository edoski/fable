"""Run one workflow request inside the remote execution environment."""

from __future__ import annotations

import argparse

from ..config.models import WorkflowTask
from ..config.resolution import WorkflowRequest, resolve_workflow_config


def run_remote_workflow(task: WorkflowTask, request: WorkflowRequest) -> None:
    if task is WorkflowTask.TRAIN:
        from ..workflows import train

        train.run(resolve_workflow_config(task, request))
        return
    if task is WorkflowTask.TUNE:
        from ..workflows import tune

        tune.run(resolve_workflow_config(task, request))
        return
    if task is WorkflowTask.EVALUATE:
        from ..workflows import evaluate

        evaluate.run(resolve_workflow_config(task, request))
        return
    raise ValueError(f"Unsupported remote workflow: {task.value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=[task.value for task in WorkflowTask if task is not WorkflowTask.ACQUIRE])
    parser.add_argument("request_json")
    args = parser.parse_args()
    run_remote_workflow(
        WorkflowTask(args.task),
        WorkflowRequest.model_validate_json(args.request_json),
    )


if __name__ == "__main__":
    main()
