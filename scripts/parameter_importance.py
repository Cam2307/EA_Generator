"""Offline parameter importance over validated strategies.

Fits a RandomForestRegressor on parameter snapshots vs OOS profit factor
(fallback: WFE) and prints permutation importance, optionally writing JSON.

Usage:
    python scripts/parameter_importance.py [--symbol EURUSD] [--timeframe H1]
        [--mechanic standard_sltp] [--out data/param_importance.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factory.storage import Storage  # noqa: E402

_MIN_SAMPLES = 5


def _numeric_params(snapshot: dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(snapshot, dict):
        return out
    for key, val in snapshot.items():
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            out[str(key)] = float(val)
    return out


def _load_rows(
    storage: Storage,
    *,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    mechanic: Optional[str] = None,
) -> List[dict]:
    """Join strategy_metadata parameter snapshots with validation metrics."""
    with storage.connection() as con:
        meta_rows = con.execute(
            "SELECT strategy_id, sweep_symbol, sweep_timeframe, "
            "parameter_snapshot FROM strategy_metadata"
        ).fetchall()

    rows: List[dict] = []
    for mr in meta_rows:
        sid = mr["strategy_id"]
        strat = storage.get_strategy(sid)
        if strat is None:
            continue
        sym = (mr["sweep_symbol"] or strat.symbol or "").upper()
        tf = (mr["sweep_timeframe"] or strat.timeframe or "").upper()
        mech = (strat.mechanic.type.value if strat.mechanic else "")
        if symbol and sym != symbol.upper():
            continue
        if timeframe and tf != timeframe.upper():
            continue
        if mechanic and mech != mechanic:
            continue

        raw = mr["parameter_snapshot"]
        try:
            snapshot = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except (TypeError, json.JSONDecodeError):
            snapshot = {}
        if not snapshot and strat:
            snapshot = strat.all_params()
        params = _numeric_params(snapshot)
        if not params:
            continue

        report = storage.get_validation(sid)
        if report is None:
            continue
        target = float(report.oos_metrics.profit_factor or 0.0)
        if target <= 0.0:
            target = float(report.wfe or 0.0)

        rows.append({
            "strategy_id": sid,
            "symbol": sym,
            "timeframe": tf,
            "mechanic": mech,
            "params": params,
            "target": target,
        })
    return rows


def _group_key(row: dict) -> Tuple[str, str, str]:
    return (row["symbol"], row["timeframe"], row["mechanic"])


def _importance_for_group(
    group_rows: Sequence[dict],
) -> List[Dict[str, float]]:
    """Fit RF + permutation importance for one symbol/TF/mechanic group."""
    if len(group_rows) < _MIN_SAMPLES:
        return []

    # Features present in a majority of rows (avoid sparse one-offs).
    counts: Dict[str, int] = defaultdict(int)
    for row in group_rows:
        for k in row["params"]:
            counts[k] += 1
    features = sorted(
        k for k, c in counts.items() if c >= max(_MIN_SAMPLES, len(group_rows) // 2)
    )
    if not features:
        return []

    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance

    X = np.zeros((len(group_rows), len(features)), dtype=float)
    y = np.zeros(len(group_rows), dtype=float)
    for i, row in enumerate(group_rows):
        y[i] = row["target"]
        for j, feat in enumerate(features):
            X[i, j] = row["params"].get(feat, np.nan)

    # Impute missing with column median.
    for j in range(X.shape[1]):
        col = X[:, j]
        med = float(np.nanmedian(col))
        col[np.isnan(col)] = med
        X[:, j] = col

    if np.allclose(y, y[0]):
        return [{"feature": f, "importance": 0.0} for f in features]

    model = RandomForestRegressor(
        n_estimators=100, random_state=42, max_depth=6, n_jobs=-1)
    model.fit(X, y)
    result = permutation_importance(
        model, X, y, n_repeats=10, random_state=42, n_jobs=-1)

    ranked = sorted(
        zip(features, result.importances_mean.tolist()),
        key=lambda t: t[1],
        reverse=True,
    )
    return [{"feature": f, "importance": float(imp)} for f, imp in ranked]


def compute_parameter_importance(
    storage: Storage,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    mechanic: Optional[str] = None,
) -> List[Dict[str, float]]:
    """Return ``[{feature, importance}, ...]`` ranked high → low.

    When ``symbol`` / ``timeframe`` / ``mechanic`` are omitted, rows are still
    grouped internally and the largest eligible group is ranked (so the gallery
    Insights button can call this with only a symbol filter).
    """
    rows = _load_rows(
        storage, symbol=symbol, timeframe=timeframe, mechanic=mechanic)
    if not rows:
        return []

    groups: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)

    best: List[Dict[str, float]] = []
    best_n = 0
    for key, group_rows in sorted(groups.items()):
        ranked = _importance_for_group(group_rows)
        if ranked and len(group_rows) > best_n:
            best = ranked
            best_n = len(group_rows)
    return best


def compute_parameter_importance_by_group(
    storage: Storage,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    mechanic: Optional[str] = None,
) -> Dict[str, List[Dict[str, float]]]:
    """Same as :func:`compute_parameter_importance` but keyed by group label."""
    rows = _load_rows(
        storage, symbol=symbol, timeframe=timeframe, mechanic=mechanic)
    out: Dict[str, List[Dict[str, float]]] = {}
    groups: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)
    for (sym, tf, mech), group_rows in sorted(groups.items()):
        ranked = _importance_for_group(group_rows)
        if ranked:
            out[f"{sym}/{tf}/{mech or 'any'}"] = ranked
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--timeframe", default=None)
    ap.add_argument("--mechanic", default=None)
    ap.add_argument(
        "--out", default=None,
        help="Write JSON summary under data/ (or given path); default stdout")
    args = ap.parse_args()

    storage = Storage()
    by_group = compute_parameter_importance_by_group(
        storage,
        symbol=args.symbol,
        timeframe=args.timeframe,
        mechanic=args.mechanic,
    )
    payload = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "mechanic": args.mechanic,
        "groups": by_group,
        "top": compute_parameter_importance(
            storage,
            symbol=args.symbol,
            timeframe=args.timeframe,
            mechanic=args.mechanic,
        ),
    }

    text = json.dumps(payload, indent=2)
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute() and out_path.parts[0] != "data":
            out_path = Path("data") / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(text)

    if not payload["top"]:
        print("No group had enough overlapping parameter snapshots "
              f"(need ≥{_MIN_SAMPLES}).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
