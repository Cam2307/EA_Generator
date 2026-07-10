"""Trading-terminal CSS for the unified discovery panel."""
from __future__ import annotations

import streamlit as st

_CSS_KEY = "_ea_discovery_terminal_css"


def inject_discovery_styles() -> None:
    if st.session_state.get(_CSS_KEY):
        return
    st.session_state[_CSS_KEY] = True
    st.markdown(
        """
        <style>
        :root {
            --ea-bg: #0b0f14;
            --ea-surface: #141a22;
            --ea-surface-2: #1a222d;
            --ea-border: #2a3441;
            --ea-amber: #d4a853;
            --ea-amber-dim: rgba(212, 168, 83, 0.14);
            --ea-teal: #2dd4bf;
            --ea-teal-dim: rgba(45, 212, 191, 0.12);
            --ea-muted: #8b95a5;
            --ea-text: #e8eaed;
        }

        .ea-terminal-shell {
            border: 1px solid var(--ea-border);
            border-radius: 14px;
            background: linear-gradient(
                165deg,
                rgba(26, 34, 45, 0.92) 0%,
                rgba(11, 15, 20, 0.98) 55%
            );
            padding: 1.35rem 1.45rem 1.1rem;
            margin-bottom: 0.35rem;
            box-shadow:
                0 0 0 1px rgba(212, 168, 83, 0.06) inset,
                0 18px 42px rgba(0, 0, 0, 0.28);
        }

        .ea-terminal-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.85rem;
        }

        .ea-terminal-title {
            margin: 0;
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            color: var(--ea-text);
        }

        .ea-terminal-sub {
            margin: 0.28rem 0 0;
            font-size: 0.86rem;
            line-height: 1.45;
            color: var(--ea-muted);
            max-width: 58rem;
        }

        .ea-terminal-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            border: 1px solid rgba(212, 168, 83, 0.35);
            background: var(--ea-amber-dim);
            color: var(--ea-amber);
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            white-space: nowrap;
        }

        .ea-section-label {
            margin: 0.15rem 0 0.55rem;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--ea-muted);
        }

        .ea-val-card {
            border: 1px solid var(--ea-border);
            border-left: 3px solid var(--ea-amber);
            border-radius: 10px;
            background: rgba(20, 26, 34, 0.72);
            padding: 0.75rem 0.95rem;
            margin: 0.35rem 0 0.85rem;
        }

        .ea-val-card strong {
            color: var(--ea-amber);
        }

        .ea-val-card p {
            margin: 0.35rem 0 0;
            color: var(--ea-muted);
            font-size: 0.84rem;
            line-height: 1.45;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(
                180deg,
                rgba(26, 34, 45, 0.95) 0%,
                rgba(15, 20, 27, 0.98) 100%
            ) !important;
            border: 1px solid var(--ea-border) !important;
            border-radius: 10px !important;
            padding: 0.55rem 0.75rem !important;
            box-shadow: 0 0 18px rgba(45, 212, 191, 0.04);
        }

        div[data-testid="stMetric"] label {
            color: var(--ea-muted) !important;
            font-size: 0.74rem !important;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--ea-text) !important;
        }

        div[data-testid="stMetric"]:hover {
            border-color: rgba(212, 168, 83, 0.35) !important;
            box-shadow: 0 0 22px rgba(212, 168, 83, 0.08);
        }

        .ea-run-card {
            border: 1px solid var(--ea-border);
            border-radius: 10px;
            background: rgba(20, 26, 34, 0.55);
            padding: 0.65rem 0.85rem;
            margin-bottom: 0.45rem;
        }

        .ea-run-card:hover {
            border-color: rgba(45, 212, 191, 0.28);
        }

        [data-testid="stSegmentedControl"] button[aria-checked="true"] {
            border-color: rgba(212, 168, 83, 0.55) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
