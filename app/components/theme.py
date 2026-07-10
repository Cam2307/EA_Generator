"""Shared design system for the dashboard: global CSS + HTML helpers.

One place for the visual language — hero header, KPI strip, chips/badges,
section headers — so Discovery, Gallery, and Export stay coherent. Palette
matches .streamlit/config.toml (charcoal base, amber primary, teal accent).
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import streamlit as st

# tone -> (foreground, background, border)
_CHIP_TONES = {
    "amber": ("#D4A853", "rgba(212,168,83,0.12)", "rgba(212,168,83,0.35)"),
    "teal": ("#2DD4BF", "rgba(45,212,191,0.10)", "rgba(45,212,191,0.35)"),
    "red": ("#F87171", "rgba(248,113,113,0.10)", "rgba(248,113,113,0.35)"),
    "blue": ("#38BDF8", "rgba(56,189,248,0.10)", "rgba(56,189,248,0.35)"),
    "violet": ("#A78BFA", "rgba(167,139,250,0.10)", "rgba(167,139,250,0.35)"),
    "gray": ("#8B95A5", "rgba(139,149,165,0.10)", "rgba(139,149,165,0.30)"),
}

_GLOBAL_CSS = """
<style>
/* ---- chrome ------------------------------------------------------- */
header[data-testid="stHeader"] {
    height: 0 !important; min-height: 0 !important;
    background: transparent !important; border: none !important;
}
header[data-testid="stHeader"] [data-testid="stToolbar"] { display: none !important; }
.block-container { padding-top: 1.4rem !important; max-width: 1400px; }

/* ---- hero ---------------------------------------------------------- */
.ea-hero {
    position: relative;
    padding: 1.35rem 1.6rem 1.2rem;
    margin-bottom: 1.1rem;
    border: 1px solid #2A3441;
    border-radius: 16px;
    background:
        radial-gradient(1200px 300px at 15% -40%, rgba(212,168,83,0.14), transparent 60%),
        radial-gradient(900px 260px at 85% -30%, rgba(45,212,191,0.10), transparent 55%),
        linear-gradient(180deg, #141A22 0%, #10151C 100%);
    overflow: hidden;
}
.ea-hero::after {
    content: ""; position: absolute; inset: 0 0 auto 0; height: 2px;
    background: linear-gradient(90deg, transparent, #D4A853 30%, #2DD4BF 70%, transparent);
    opacity: 0.7;
}
.ea-hero-title {
    margin: 0; font-size: 1.55rem; font-weight: 700; letter-spacing: -0.02em;
    color: #E8EAED; display: flex; align-items: center; gap: 0.55rem;
}
.ea-hero-mark {
    display: inline-flex; align-items: center; justify-content: center;
    width: 2.1rem; height: 2.1rem; border-radius: 10px; font-size: 1.1rem;
    background: linear-gradient(135deg, rgba(212,168,83,0.25), rgba(45,212,191,0.18));
    border: 1px solid rgba(212,168,83,0.4);
}
.ea-hero-sub {
    margin: 0.35rem 0 0; font-size: 0.88rem; color: #8B95A5;
    line-height: 1.55; max-width: 52rem;
}
.ea-hero-chips { margin-top: 0.7rem; display: flex; gap: 0.4rem; flex-wrap: wrap; }

/* ---- KPI strip ------------------------------------------------------ */
.ea-kpi-row { display: flex; gap: 0.7rem; flex-wrap: wrap; margin: 0 0 1.1rem; }
.ea-kpi {
    flex: 1 1 130px; min-width: 130px;
    padding: 0.7rem 0.95rem 0.65rem;
    border: 1px solid #2A3441; border-radius: 12px;
    background: linear-gradient(180deg, #161D26 0%, #121820 100%);
    transition: border-color 0.15s ease, transform 0.15s ease;
}
.ea-kpi:hover { border-color: #3D4F63; transform: translateY(-1px); }
.ea-kpi-label {
    font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: #8B95A5; margin-bottom: 0.25rem;
}
.ea-kpi-value {
    font-size: 1.35rem; font-weight: 700; color: #E8EAED;
    font-variant-numeric: tabular-nums; line-height: 1.15;
}
.ea-kpi-value.accent { color: #D4A853; }
.ea-kpi-value.good { color: #2DD4BF; }
.ea-kpi-value.bad { color: #F87171; }
.ea-kpi-delta { font-size: 0.72rem; color: #8B95A5; margin-top: 0.15rem; }

/* ---- chips ---------------------------------------------------------- */
.ea-chip {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.14rem 0.6rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em;
    white-space: nowrap;
}
.ea-chip .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }

/* ---- section headers ------------------------------------------------ */
.ea-section {
    display: flex; align-items: baseline; gap: 0.6rem;
    margin: 0.4rem 0 0.5rem; padding-bottom: 0.45rem;
    border-bottom: 1px solid #1E2733;
}
.ea-section-title { font-size: 1.05rem; font-weight: 650; color: #E8EAED; margin: 0; }
.ea-section-caption { font-size: 0.8rem; color: #8B95A5; margin: 0; }

/* ---- component polish ----------------------------------------------- */
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, #161D26 0%, #121820 100%);
    border: 1px solid #2A3441; border-radius: 12px;
    padding: 0.55rem 0.8rem;
}
div[data-testid="stMetric"] label { color: #8B95A5 !important; }
button[kind="primary"] {
    box-shadow: 0 2px 14px rgba(212,168,83,0.22) !important;
}
div[data-testid="stSegmentedControl"] button { font-weight: 600; }
[data-testid="stExpander"] details {
    border-radius: 12px !important;
}
</style>
"""


def inject_global_css() -> None:
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def chip(label: str, tone: str = "gray", dot: bool = True) -> str:
    """Inline HTML for a status chip (compose inside hero/cards)."""
    fg, bg, border = _CHIP_TONES.get(tone, _CHIP_TONES["gray"])
    dot_html = '<span class="dot"></span>' if dot else ""
    return (f'<span class="ea-chip" style="color:{fg};background:{bg};'
            f'border:1px solid {border}">{dot_html}{label}</span>')


def hero(title: str, subtitle: str,
         chips: Optional[Iterable[str]] = None) -> None:
    """Branded page header with optional live status chips (chip() HTML)."""
    chips_html = ""
    chip_list = list(chips or [])
    if chip_list:
        chips_html = f'<div class="ea-hero-chips">{"".join(chip_list)}</div>'
    st.markdown(
        f"""
        <div class="ea-hero">
          <p class="ea-hero-title"><span class="ea-hero-mark">⚙</span>{title}</p>
          <p class="ea-hero-sub">{subtitle}</p>
          {chips_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(items: Sequence[Tuple[str, str, str]],
            deltas: Optional[Sequence[str]] = None) -> None:
    """Render a row of KPI cards. ``items`` = (label, value, tone) where tone
    is one of ""|"accent"|"good"|"bad"; ``deltas`` adds a small footnote."""
    cells = []
    for i, (label, value, tone) in enumerate(items):
        delta = (f'<div class="ea-kpi-delta">{deltas[i]}</div>'
                 if deltas and i < len(deltas) and deltas[i] else "")
        cells.append(
            f'<div class="ea-kpi"><div class="ea-kpi-label">{label}</div>'
            f'<div class="ea-kpi-value {tone}">{value}</div>{delta}</div>')
    st.markdown(f'<div class="ea-kpi-row">{"".join(cells)}</div>',
                unsafe_allow_html=True)


def section(title: str, caption: str = "") -> None:
    """Consistent section header with underline + muted caption."""
    cap = f'<p class="ea-section-caption">{caption}</p>' if caption else ""
    st.markdown(
        f'<div class="ea-section"><p class="ea-section-title">{title}</p>{cap}</div>',
        unsafe_allow_html=True,
    )
