"""Bundle a Marketplace Package (.mq5 + .set + .md) per strategy."""
from __future__ import annotations

import re
from pathlib import Path

from config import settings
from factory.assets.description_writer import write_description
from factory.assets.set_writer import write_set_file
from factory.models import StrategyDefinition, ValidationReport
from factory.mql5.renderer import write_ea


def export_marketplace_package(strategy: StrategyDefinition,
                               report: ValidationReport,
                               out_root: Path = None) -> Path:
    """Write output/<strategy-name>/ with the three marketplace assets.

    The .set file carries the validated best parameters (report.best_params)
    with their optimization ranges.
    """
    out_root = out_root or settings.OUTPUT_DIR
    folder_name = re.sub(r"[^A-Za-z0-9_\- ]", "", strategy.name).replace(" ", "_")
    out_dir = Path(out_root) / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tuned = strategy.apply_flat_params(report.best_params) if report.best_params \
        else strategy
    write_ea(tuned, out_dir)
    write_set_file(tuned, out_dir)
    write_description(tuned, report, out_dir)
    return out_dir
