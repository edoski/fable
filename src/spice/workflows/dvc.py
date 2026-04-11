"""DVC stage runner that consumes Hydra-composed params.yaml."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from ..core.config import ExperimentConfig, coerce_config, revalidate_config
from . import acquire, simulate, train, tune

StageRunner = Callable[[ExperimentConfig], None]

STAGE_RUNNERS: dict[str, StageRunner] = {
    "acquire": acquire.run,
    "tune": tune.run,
    "train": train.run,
    "simulate": simulate.run,
}


def load_stage_config(stage: str, params_path: Path) -> ExperimentConfig:
    raw_config = OmegaConf.load(params_path)
    if not isinstance(raw_config, DictConfig):
        raise TypeError(f"DVC params must be a mapping: {params_path}")
    config = coerce_config(raw_config, task=stage)
    if stage == "train":
        config.tuning.apply_best_params = True
        config = revalidate_config(config)
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="spice-dvc")
    parser.add_argument("stage", choices=sorted(STAGE_RUNNERS))
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args(argv)
    STAGE_RUNNERS[args.stage](load_stage_config(args.stage, args.params))


if __name__ == "__main__":
    main()
