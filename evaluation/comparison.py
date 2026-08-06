"""Comparison table: MRI-only vs MRI + clinical."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.io_utils import ensure_dir, load_json
from utils.logging_setup import get_logger

logger = get_logger(__name__)

METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def load_experiment_metrics(output_dir: str | Path, experiment_name: str) -> dict[str, object] | None:
    """Load the test metrics JSON for an experiment.

    Args:
        output_dir: Directory containing ``<name>_metrics.json``.
        experiment_name: Experiment name.

    Returns:
        Metrics dict, or ``None`` if the file is missing.
    """
    path = Path(output_dir) / f"{experiment_name}_metrics.json"
    if not path.exists():
        return None
    return load_json(path)


def build_comparison_table(
    output_dir: str | Path,
    experiments: list[str],
    comparison_dir: str | Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Compare multiple experiments on the shared test-split metrics.

    Args:
        output_dir: Directory holding per-experiment metrics JSON files.
        experiments: Experiment names to compare (e.g. unimodal, multimodal).
        comparison_dir: Directory for the comparison CSV/markdown.

    Returns:
        Tuple of ``(table, written_files)``.
    """
    ensure_dir(comparison_dir)
    rows: list[dict[str, object]] = []
    for name in experiments:
        metrics = load_experiment_metrics(output_dir, name)
        if metrics is None:
            logger.warning("No metrics found for experiment '%s'; skipping.", name)
            continue
        row: dict[str, object] = {"model": name}
        for key in METRIC_KEYS:
            row[key] = metrics.get(key)
        row["best_epoch"] = metrics.get("best_epoch")
        rows.append(row)

    table = pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()
    if table.empty:
        return table, []

    csv_path = Path(comparison_dir) / "comparison_table.csv"
    table.to_csv(csv_path)
    written = [str(csv_path)]

    md_path = Path(comparison_dir) / "comparison_table.md"
    md_lines = ["| model |" + " | ".join(table.columns) + " |",
                "|" + "---|" * (len(table.columns) + 1)]
    for name, row in table.iterrows():
        cells = " | ".join(
            f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
            for c in table.columns
        )
        md_lines.append(f"| {name} | {cells} |")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    written.append(str(md_path))
    logger.info("Comparison table written to %s", comparison_dir)
    return table, written