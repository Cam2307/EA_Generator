"""Empirical review of discovery results vs validation gates.

Loads ``results/`` (preferring ``candidates/*.json``), filters out incomplete
MT5/infra failures, then reports:

- Population mix (complete vs aborted)
- Metric distributions for tradeable candidates
- Current L1–L16 thresholds vs empirical percentiles
- Top failure reasons
- Calculation anomaly flags (WFE=0 with profitable OOS, PF=999, etc.)
- Suggested gate tweaks grounded in the data

Usage:
    python scripts/review_results.py
    python scripts/review_results.py --job-id auto_... --out data/review_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from factory import validation_levels  # noqa: E402
from factory.results_review import (  # noqa: E402
    METRIC_DEFINITIONS,
    analyze_results,
    format_report_text,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Review actual discovery results vs L1–L16 gates")
    parser.add_argument(
        "--results-root", default=None,
        help="Path to results/ (default: project results/)")
    parser.add_argument("--job-id", action="append", default=None,
                        help="Limit to one or more job ids (repeatable)")
    parser.add_argument("--out", default=None,
                        help="Write full JSON report to this path")
    parser.add_argument("--limit-candidates", type=int, default=0,
                        help="Max candidates to load (0 = all; useful for smoke)")
    args = parser.parse_args(argv)

    root = Path(args.results_root) if args.results_root else Path(settings.RESULTS_DIR)
    if not root.is_dir():
        print(f"No results directory at {root}", file=sys.stderr)
        return 1

    report = analyze_results(
        root,
        job_ids=args.job_id,
        limit_candidates=int(args.limit_candidates or 0) or None,
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"Wrote JSON report -> {out}")

    text = format_report_text(report)
    text += "\n\n--- Metric calculation notes ---\n"
    for name, note in METRIC_DEFINITIONS.items():
        text += f"  {name}: {note}\n"
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
