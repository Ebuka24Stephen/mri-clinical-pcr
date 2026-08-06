"""Train the MRI-only (unimodal) pCR baseline.

Usage::

    python train_unimodal.py --config configs/unimodal.yaml
    python train_unimodal.py --config configs/unimodal.yaml --experiment my_run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config, load_config, save_config  # noqa: E402
from training.pipeline import run_experiment  # noqa: E402
from utils.logging_setup import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    """Parse arguments, load the config and run the unimodal experiment."""
    parser = argparse.ArgumentParser(description="Train the MRI-only pCR baseline.")
    parser.add_argument(
        "--config",
        default="configs/unimodal.yaml",
        help="Path to the experiment YAML configuration.",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Override the experiment name (default: from config).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.experiment:
        cfg.experiment_name = args.experiment
    setup_logging(Path(cfg.paths.log_dir) / f"{cfg.experiment_name}_train.log")
    logger.info("Training unimodal model '%s'.", cfg.experiment_name)

    summary = run_experiment(cfg)
    save_config(cfg, Path(cfg.paths.log_dir) / f"{cfg.experiment_name}_config.yaml")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()