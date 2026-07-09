"""Marketplace .md description generator, rendered from the validation report."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from factory.metrics_display import (
    data_source_label, gate_drawdown_pct, wfo_summary, zone_drawdown_label,
)
from factory.models import StrategyDefinition, ValidationReport


def _dt(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def build_description(strategy: StrategyDefinition,
                      report: ValidationReport) -> str:
    oos, is_m = report.oos_metrics, report.is_metrics
    filters = ", ".join(f.type.value.replace("_", " ").title()
                        for f in strategy.entry_filters)
    mechanic = strategy.mechanic.type.value.replace("_", " ").title()

    lines = [
        f"# {strategy.name}",
        "",
        f"A fully automated {strategy.symbol} {strategy.timeframe} Expert Advisor "
        f"built and validated by the EA Factory discovery pipeline.",
        "",
        "## Verified Edge",
        "",
        f"- **Walk-Forward Efficiency (WFE): {report.wfe:.2f}** "
        f"(out-of-sample performance retained vs in-sample; gate > 0.55)",
        f"- Out-of-sample net profit: **{oos.net_profit:,.2f}** on a "
        f"{oos.initial_deposit:,.0f} deposit "
        f"({_dt(report.oos_range[0])} to {_dt(report.oos_range[1])})",
        f"- **{zone_drawdown_label('OOS')}: {gate_drawdown_pct(oos):.1f}%** "
        f"(intrabar simulator metric; gate < 15%)",
        f"- In-sample {zone_drawdown_label('IS')}: "
        f"{gate_drawdown_pct(is_m):.1f}%",
        f"- Out-of-sample profit factor: {oos.profit_factor:.2f} over "
        f"{oos.trade_count} trades",
        f"- In-sample reference: net {is_m.net_profit:,.2f}, PF "
        f"{is_m.profit_factor:.2f}, {is_m.trade_count} trades",
        f"- Validation engine: {report.engine}; data: "
        f"{data_source_label(report.data_source)}; "
        f"{wfo_summary(report.wfo_windows, 'rolling')} rolling, "
        f"{wfo_summary(report.wfo_windows, 'anchored')} anchored"
        + (f" ({report.wfo_train_months}m train / "
           f"{report.wfo_test_months}m test)"
           if report.wfo_train_months and report.wfo_test_months else ""),
        "",
        "## Strategy Logic",
        "",
        f"- Entry filters: {filters}",
        f"- Execution mechanic: {mechanic}",
        "",
        "```",
        strategy.rule_description,
        "```",
        "",
        "## Features",
        "",
        "- Standalone .mq5 — no DLLs, no external indicators, only the standard "
        "library (`Trade.mqh`)",
        "- History-synchronization and array-bounds guards on every buffer access",
        "- Spread gate, market-open check, and zero-divide protection built in",
        "- Bounded retry with backoff on every trade operation (requote-safe)",
        "- On-chart dashboard: equity, margin, live drawdown, spread, basket state",
        "- Every parameter (including grid step / hedge distance) exposed as an "
        "optimizable input with a curated range in the bundled .set file",
        "",
        "## Recommended Setup",
        "",
        f"- Symbol: **{strategy.symbol}** (majors with tight spread preferred)",
        f"- Timeframe: **{strategy.timeframe}**",
        f"- Minimum deposit: {max(1000, int(oos.initial_deposit / 10)):,} "
        f"(account currency)",
        "- Leverage: 1:100 or higher",
        "- Broker profile: low-spread ECN/Raw account, 5-digit quotes",
        f"- Magic number: {strategy.magic_number}",
        "",
        "> Backtest results do not guarantee future performance. Always forward-"
        "test on a demo account first.",
        "",
    ]
    return "\n".join(lines)


def write_description(strategy: StrategyDefinition, report: ValidationReport,
                      out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{strategy.name.replace(' ', '_')}.md"
    path.write_text(build_description(strategy, report), encoding="utf-8")
    return path
