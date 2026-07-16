"""Evidence-based review of archived discovery results vs validation gates.

Separates incomplete/infra failures from tradeable backtests, summarizes
metric distributions, compares them to L1–L16 thresholds, flags calculation
anomalies, and suggests gate adjustments from empirical percentiles.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from factory import validation_levels

# How each gated metric is computed (for audit / operator review).
METRIC_DEFINITIONS: Dict[str, str] = {
    "wfe": (
        "OOS annualized profit rate / IS annualized profit rate; "
        "forced to 0 when IS rate <= 0 (see compute_wfe)."
    ),
    "max_dd_pct": (
        "Peak-to-trough drawdown % using intrabar worst floating equity "
        "(conservative vs close-only)."
    ),
    "trade_count": "Closed trades in the OOS (or full-range) window.",
    "profit_factor": (
        "Gross profit / gross loss; capped at 999.0 when there are no losses."
    ),
    "sharpe": (
        "Bar-to-bar equity-return Sharpe, annualized by median bar interval "
        "(not trade-based or daily Sharpe)."
    ),
    "r_squared": (
        "Squared correlation of equity vs bar index (linearity of the curve)."
    ),
    "annualized_profit_rate": (
        "(net_profit / deposit) / years — linear rate, not CAGR."
    ),
}

_INFRA_PATTERNS = (
    re.compile(r"MT5RunnerError", re.I),
    re.compile(r"terminal is already running", re.I),
    re.compile(r"Validation did not complete", re.I),
    re.compile(r"MetaTrader", re.I),
    re.compile(r"failed to (?:initialize|connect)", re.I),
    re.compile(r"^INFRA:", re.I),
    re.compile(r"\bINFRA:", re.I),
)


def _percentile(sorted_vals: Sequence[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    p = max(0.0, min(100.0, float(p)))
    idx = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_vals[lo])
    w = idx - lo
    return float(sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w)


def _stats(vals: Sequence[float]) -> Dict[str, Optional[float]]:
    clean = sorted(float(v) for v in vals if v is not None and math.isfinite(float(v)))
    if not clean:
        return {"n": 0, "p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    return {
        "n": len(clean),
        "p10": round(_percentile(clean, 10), 4),
        "p25": round(_percentile(clean, 25), 4),
        "p50": round(_percentile(clean, 50), 4),
        "p75": round(_percentile(clean, 75), 4),
        "p90": round(_percentile(clean, 90), 4),
    }


def _is_infra_failure(
    reasons: Sequence[str],
    trade_count: int,
    *,
    infra_flag: bool = False,
) -> bool:
    if infra_flag:
        return True
    if trade_count > 0:
        return False
    text = " | ".join(str(r) for r in reasons)
    if not text:
        return trade_count <= 0
    return any(p.search(text) for p in _INFRA_PATTERNS) or trade_count <= 0


def _reason_bucket(reason: str) -> str:
    r = reason.lower()
    if r.startswith("infra:") or "infra:" in r:
        return "infra"
    if "drawdown" in r or "max dd" in r:
        return "drawdown"
    if "wfe" in r or "walk-forward" in r:
        return "wfe"
    if "profit factor" in r:
        return "profit_factor"
    if "sharpe" in r:
        return "sharpe"
    if "r-squared" in r or "r²" in r or "r_squared" in r or "smoothness" in r:
        return "r_squared"
    if "trade count" in r or "trades" in r:
        return "trades"
    if "monte carlo" in r:
        return "monte_carlo"
    if "mt5" in r or "validation did not complete" in r:
        return "infra"
    return "other"


def _load_candidate(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    report = data.get("report") or {}
    oos = report.get("oos_metrics") or {}
    reasons = list(report.get("reasons") or [])
    trade_count = int(oos.get("trade_count") or 0)
    infra_flag = bool(report.get("infra_failure"))
    return {
        "job_id": data.get("job_id") or path.parent.parent.name,
        "strategy_id": data.get("strategy_id") or path.stem,
        "passed": bool(report.get("passed")),
        "highest_level_passed": int(report.get("highest_level_passed") or 0),
        "levels_cleared": dict(report.get("levels_cleared") or {}),
        "wfe": float(report.get("wfe") or 0.0),
        "oos_net_profit": float(oos.get("net_profit") or 0.0),
        "oos_max_dd_pct": float(oos.get("max_dd_pct") or 0.0),
        "oos_trade_count": trade_count,
        "oos_profit_factor": float(oos.get("profit_factor") or 0.0),
        "oos_sharpe": float(oos.get("sharpe") or 0.0),
        "oos_r_squared": float(oos.get("r_squared") or 0.0),
        "reasons": reasons,
        "engine": report.get("engine"),
        "infra": _is_infra_failure(reasons, trade_count, infra_flag=infra_flag),
        "tradeable": trade_count > 0,
    }


def iter_candidates(
    results_root: Path,
    *,
    job_ids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> Iterable[dict]:
    """Yield normalized candidate dicts from ``results/*/candidates/*.json``."""
    root = Path(results_root)
    if job_ids:
        dirs = [root / jid for jid in job_ids if (root / jid).is_dir()]
    else:
        dirs = sorted(p for p in root.iterdir() if p.is_dir())
    n = 0
    for d in dirs:
        cand_dir = d / "candidates"
        if not cand_dir.is_dir():
            continue
        for path in cand_dir.glob("*.json"):
            row = _load_candidate(path)
            if row is None:
                continue
            yield row
            n += 1
            if limit is not None and n >= limit:
                return


def flag_anomalies(rows: Sequence[dict]) -> List[dict]:
    """Flag metric combinations that look wrong or misleading for gating."""
    flags: List[dict] = []
    for row in rows:
        if not row.get("tradeable"):
            continue
        sid = row["strategy_id"]
        job = row["job_id"]
        if row["wfe"] <= 0 and row["oos_net_profit"] > 0:
            flags.append({
                "kind": "wfe_zero_with_profitable_oos",
                "job_id": job, "strategy_id": sid,
                "detail": (
                    f"WFE={row['wfe']:.3f} but OOS net={row['oos_net_profit']:.2f} "
                    "(IS edge ≤ 0 → WFE forced to 0)"
                ),
            })
        if row["oos_profit_factor"] >= 900:
            flags.append({
                "kind": "profit_factor_cap",
                "job_id": job, "strategy_id": sid,
                "detail": f"PF={row['oos_profit_factor']:.1f} (no-loss cap at 999)",
            })
        if row["oos_sharpe"] > 8:
            flags.append({
                "kind": "extreme_bar_sharpe",
                "job_id": job, "strategy_id": sid,
                "detail": (
                    f"Sharpe={row['oos_sharpe']:.2f} on bar returns — "
                    "compare carefully to daily/trade Sharpe norms"
                ),
            })
        if row["oos_trade_count"] > 500 and row["highest_level_passed"] == 0:
            flags.append({
                "kind": "high_trades_level0",
                "job_id": job, "strategy_id": sid,
                "detail": (
                    f"{row['oos_trade_count']} OOS trades but level 0 — "
                    f"DD={row['oos_max_dd_pct']:.1f}% WFE={row['wfe']:.3f}"
                ),
            })
    return flags


def _gate_vs_distribution(tradeable: Sequence[dict]) -> List[dict]:
    """Compare each level's metric gates to empirical clear-rates / percentiles."""
    from factory.models import BacktestMetrics

    if not tradeable:
        return []
    metrics = {
        "wfe": [r["wfe"] for r in tradeable],
        "max_dd_pct": [r["oos_max_dd_pct"] for r in tradeable],
        "trade_count": [float(r["oos_trade_count"]) for r in tradeable],
        "profit_factor": [r["oos_profit_factor"] for r in tradeable],
        "sharpe": [r["oos_sharpe"] for r in tradeable],
        "r_squared": [r["oos_r_squared"] for r in tradeable],
    }
    dist = {k: _stats(v) for k, v in metrics.items()}
    rows = []
    for lvl in validation_levels.VALIDATION_LEVELS:
        c = lvl.criteria
        clears = 0
        for r in tradeable:
            m = BacktestMetrics(
                net_profit=r["oos_net_profit"],
                max_dd_pct=r["oos_max_dd_pct"],
                trade_count=r["oos_trade_count"],
                profit_factor=r["oos_profit_factor"],
                sharpe=r["oos_sharpe"],
                r_squared=r["oos_r_squared"],
            )
            # Nested: L_n only counts when L1..L_n all clear.
            hi = validation_levels.highest_level_cleared(
                m, r["wfe"], None, ceiling=lvl.level, floor=1)
            if hi >= lvl.level:
                clears += 1
        rows.append({
            "level": lvl.level,
            "name": lvl.name,
            "min_wfe": c.min_wfe,
            "max_dd_pct": c.max_dd_pct,
            "min_trades": c.min_trades,
            "min_profit_factor": c.min_profit_factor,
            "min_sharpe": c.min_sharpe,
            "min_r_squared": c.min_r_squared,
            "empirical": dist,
            "metric_clear_rate": round(clears / len(tradeable), 4),
            "vs_distribution": {
                "wfe_gate_vs_p50": (
                    None if dist["wfe"]["p50"] is None
                    else round(c.min_wfe - float(dist["wfe"]["p50"]), 4)
                ),
                "dd_gate_vs_p50": (
                    None if dist["max_dd_pct"]["p50"] is None
                    else round(c.max_dd_pct - float(dist["max_dd_pct"]["p50"]), 4)
                ),
                "trades_gate_vs_p50": (
                    None if dist["trade_count"]["p50"] is None
                    else round(c.min_trades - float(dist["trade_count"]["p50"]), 1)
                ),
            },
        })
    return rows


def suggest_adjustments(
    tradeable: Sequence[dict],
    level_rows: Sequence[dict],
) -> List[str]:
    """Human-readable suggestions grounded in percentiles + clear rates."""
    tips: List[str] = []
    if not tradeable:
        tips.append("No tradeable candidates — fix engine/infra failures before tuning gates.")
        return tips
    dist_wfe = _stats([r["wfe"] for r in tradeable])
    dist_dd = _stats([r["oos_max_dd_pct"] for r in tradeable])
    dist_tr = _stats([float(r["oos_trade_count"]) for r in tradeable])
    l1 = validation_levels.get_level(1).criteria
    # Clear rates from level_rows
    rates = {row["level"]: row.get("metric_clear_rate") for row in level_rows}

    if rates.get(1) is not None and rates[1] < 0.15:
        tips.append(
            f"L1 metric clear-rate is only {rates[1]:.0%} of tradeable candidates. "
            f"Tradeable median DD={dist_dd['p50']}% vs L1 max_dd={l1.max_dd_pct}%; "
            f"median WFE={dist_wfe['p50']} vs L1 min_wfe={l1.min_wfe}. "
            "Loosen L1 DD / WFE further, or separate infra failures from strategy quality."
        )
    elif rates.get(1) is not None and rates[1] > 0.6:
        tips.append(
            f"L1 clears {rates[1]:.0%} of tradeable — screener is doing its job as a wide net."
        )

    if dist_dd["p50"] is not None and dist_dd["p50"] > l1.max_dd_pct:
        tips.append(
            f"Median OOS DD ({dist_dd['p50']}%) exceeds even L1 max DD ({l1.max_dd_pct}%). "
            "Drawdown is the binding constraint — either accept higher DD at low tiers "
            "or change search/mechanics to target lower risk."
        )

    zero_wfe_profitable = sum(
        1 for r in tradeable
        if r["wfe"] <= 0 and r["oos_net_profit"] > 0
    )
    if zero_wfe_profitable > max(5, len(tradeable) // 10):
        tips.append(
            f"{zero_wfe_profitable} tradeable candidates have WFE=0 with profitable OOS "
            "(IS edge ≤ 0). Consider a screener that keys off OOS net/PF first, "
            "and only applies WFE once IS is positive — or treat WFE=0 as a soft flag."
        )

    if dist_tr["p50"] is not None and dist_tr["p50"] > 200:
        tips.append(
            f"Median OOS trade count is {dist_tr['p50']:.0f}. Min-trades gates "
            f"(L1={l1.min_trades} … L16={validation_levels.get_level(16).criteria.min_trades}) "
            "are rarely the bottleneck; focus calibration on DD, WFE, and Sharpe definition."
        )

    l7_rate = rates.get(7)
    if l7_rate is not None and l7_rate < 0.02:
        tips.append(
            f"L7 (MC unlock) metric clear-rate is {l7_rate:.1%}. Almost nothing reaches "
            "Monte Carlo tiers — verify Sharpe/R² semantics and Standard band thresholds "
            "before interpreting Elite as meaningful."
        )
    return tips


def analyze_results(
    results_root: Path,
    *,
    job_ids: Optional[Sequence[str]] = None,
    limit_candidates: Optional[int] = None,
) -> Dict[str, Any]:
    rows = list(iter_candidates(
        results_root, job_ids=job_ids, limit=limit_candidates))
    infra = [r for r in rows if r["infra"] and not r["tradeable"]]
    tradeable = [r for r in rows if r["tradeable"]]
    passed = [r for r in tradeable if r["passed"]]

    level_hist = Counter(r["highest_level_passed"] for r in tradeable)
    reason_buckets: Counter = Counter()
    reason_examples: Dict[str, Counter] = defaultdict(Counter)
    for r in tradeable:
        if r["passed"] and r["highest_level_passed"] > 0:
            continue
        if not r["reasons"]:
            reason_buckets["no_reason"] += 1
            continue
        # Count primary (first) reason bucket + all buckets lightly
        buckets = {_reason_bucket(x) for x in r["reasons"]}
        for b in buckets:
            reason_buckets[b] += 1
        reason_examples[_reason_bucket(r["reasons"][0])][r["reasons"][0][:120]] += 1

    distributions = {
        "wfe": _stats([r["wfe"] for r in tradeable]),
        "max_dd_pct": _stats([r["oos_max_dd_pct"] for r in tradeable]),
        "trade_count": _stats([float(r["oos_trade_count"]) for r in tradeable]),
        "profit_factor": _stats([r["oos_profit_factor"] for r in tradeable]),
        "sharpe": _stats([r["oos_sharpe"] for r in tradeable]),
        "r_squared": _stats([r["oos_r_squared"] for r in tradeable]),
    }

    level_rows = _gate_vs_distribution(tradeable)
    anomalies = flag_anomalies(tradeable)
    # Cap anomaly list in report body
    anomaly_summary = Counter(a["kind"] for a in anomalies)

    return {
        "results_root": str(results_root),
        "population": {
            "candidates_loaded": len(rows),
            "infra_or_empty": len(infra),
            "tradeable": len(tradeable),
            "passed_floor": len(passed),
            "infra_fraction": round(len(infra) / len(rows), 4) if rows else 0.0,
        },
        "highest_level_histogram_tradeable": {
            str(k): level_hist[k] for k in sorted(level_hist)
        },
        "distributions_tradeable": distributions,
        "failure_reason_buckets_tradeable": dict(reason_buckets.most_common()),
        "top_failure_reasons": {
            bucket: examples.most_common(5)
            for bucket, examples in reason_examples.items()
        },
        "levels_vs_data": level_rows,
        "anomaly_counts": dict(anomaly_summary),
        "anomaly_examples": anomalies[:40],
        "suggestions": suggest_adjustments(tradeable, level_rows),
        "metric_definitions": METRIC_DEFINITIONS,
    }


def format_report_text(report: Dict[str, Any]) -> str:
    pop = report["population"]
    lines = [
        "=== Discovery results review ===",
        f"Root: {report['results_root']}",
        f"Loaded {pop['candidates_loaded']} candidates — "
        f"{pop['tradeable']} tradeable, {pop['infra_or_empty']} infra/empty "
        f"({pop['infra_fraction']:.0%} incomplete), "
        f"{pop['passed_floor']} passed floor.",
        "",
        "Tradeable highest_level_passed:",
    ]
    for lvl, n in report["highest_level_histogram_tradeable"].items():
        lines.append(f"  L{lvl}: {n}")
    lines.append("")
    lines.append("Tradeable metric distributions (p10 / p50 / p90):")
    for name, st in report["distributions_tradeable"].items():
        if not st.get("n"):
            continue
        lines.append(
            f"  {name:16s}  n={st['n']:<6}  "
            f"{st['p10']} / {st['p50']} / {st['p90']}"
        )
    lines.append("")
    lines.append("Failure reason buckets (tradeable, non-clear):")
    for bucket, n in report["failure_reason_buckets_tradeable"].items():
        lines.append(f"  {bucket}: {n}")
    lines.append("")
    lines.append("Level metric clear-rates (MC ignored — criteria only):")
    for row in report["levels_vs_data"]:
        rate = row.get("metric_clear_rate")
        rate_s = f"{rate:.1%}" if rate is not None else "?"
        lines.append(
            f"  L{row['level']:>2} {row['name']:<12}  clear={rate_s:<7}  "
            f"WFE>={row['min_wfe']:.2f}  DD<{row['max_dd_pct']:.0f}%  "
            f"trades>={row['min_trades']}  PF>={row['min_profit_factor']:.2f}"
        )
    if report["anomaly_counts"]:
        lines.append("")
        lines.append("Calculation / interpretation anomaly counts:")
        for kind, n in report["anomaly_counts"].items():
            lines.append(f"  {kind}: {n}")
    lines.append("")
    lines.append("Suggested adjustments:")
    for tip in report["suggestions"]:
        lines.append(f"  - {tip}")
    return "\n".join(lines)
