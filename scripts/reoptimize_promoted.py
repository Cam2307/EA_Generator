"""Re-optimize promoted strategies on their trailing window.

Schedule this weekly/monthly (Task Scheduler / cron). For every promoted or
edge-positive strategy it re-runs the IS optimizer on the last N days,
reports whether the incumbent parameters are still on the fitness plateau,
and writes an updated .set into output/reoptimized/ when they are not.

Usage:
    python scripts/reoptimize_promoted.py [--window-days 180] [--limit 20]
        [--email you@example.com]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings                                    # noqa: E402
from factory.reoptimize import (                               # noqa: E402
    format_reopt_report, reoptimize_promoted,
)
from factory.storage import Storage                            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window-days", type=int, default=180)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--email", default=None,
                    help="send the report via EA_SMTP_* configured email")
    args = ap.parse_args()

    storage = Storage()
    out_dir = settings.OUTPUT_DIR / "reoptimized"
    results = reoptimize_promoted(storage, window_days=args.window_days,
                                  limit=args.limit, out_dir=out_dir)
    if not results:
        print("No promoted / edge-positive strategies to re-optimize.")
        return 0

    report = format_reopt_report(results)
    print(report)

    if args.email:
        try:
            from factory.alerts import send_email
            send_email(args.email, "EA factory: re-optimization report", report)
            print(f"\nReport emailed to {args.email}")
        except Exception as exc:                   # noqa: BLE001
            print(f"\nEmail failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
