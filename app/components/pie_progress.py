"""Circular pie-slice progress indicator with animated percentage."""
from __future__ import annotations

import html as html_lib

import streamlit as st

# Always re-emit CSS. Gating on session_state breaks inside @st.fragment:
# the style tag is cleared on fragment rerun while the flag stays True.


def _ensure_pie_css() -> None:
    st.markdown(
        """
        <style>
        .ea-pie-wrap {
            display: flex;
            justify-content: center;
            width: 100%;
            margin: 0 0 0.15rem;
        }
        .ea-pie-wrap svg {
            display: block;
            overflow: visible;
        }
        .ea-pie-track {
            fill: none;
            stroke: #2A3040;
            stroke-width: 10;
        }
        .ea-pie-arc {
            fill: none;
            stroke: #D4A853;
            stroke-width: 10;
            stroke-linecap: round;
            transition: stroke-dashoffset 0.85s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .ea-pie-pct {
            fill: #E8EAED;
            font-size: 22px;
            font-weight: 700;
            font-family: "Source Sans Pro", sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_pie_progress(
    percent: float,
    *,
    label: str = "",
    size: int = 148,
) -> None:
    """Render a pie-slice circular progress ring with an animated percentage.

    Percentage is SVG text (no absolute HTML overlays). The optional label is a
    normal Streamlit caption so fragment reruns cannot collapse it onto the %.
    """
    _ensure_pie_css()
    pct = max(0.0, min(100.0, float(percent)))
    radius = 54.0
    circumference = 2 * 3.141592653589793 * radius
    offset = circumference * (1.0 - pct / 100.0)
    pct_txt = html_lib.escape(f"{pct:.0f}%")

    st.markdown(
        (
            f'<div class="ea-pie-wrap">'
            f'<svg viewBox="0 0 120 120" width="{size}" height="{size}" '
            f'aria-label="Progress {pct:.0f} percent">'
            f'<circle class="ea-pie-track" cx="60" cy="60" r="{radius}"/>'
            f'<circle class="ea-pie-arc" cx="60" cy="60" r="{radius}" '
            f'stroke-dasharray="{circumference:.3f}" '
            f'stroke-dashoffset="{offset:.3f}" '
            f'transform="rotate(-90 60 60)"/>'
            f'<text class="ea-pie-pct" x="60" y="60" text-anchor="middle" '
            f'dominant-baseline="central">{pct_txt}</text>'
            f"</svg></div>"
        ),
        unsafe_allow_html=True,
    )
    if label:
        st.caption(label)
