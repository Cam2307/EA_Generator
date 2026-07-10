"""Headless render checks for the dashboard (streamlit AppTest)."""
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

APP = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")

_NAV = (
    ":material/radar: Discovery",
    ":material/grid_view: Strategy gallery",
    ":material/download: Export",
)


def _apptest(tmp_path, monkeypatch, view: str):
    from config import settings
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "ui.db")
    at = st_testing.AppTest.from_file(APP, default_timeout=60)
    at.session_state["main_nav_section"] = view
    return at


@pytest.mark.parametrize("view", _NAV)
def test_all_three_pages_render_without_exception(tmp_path, monkeypatch, view):
    at = _apptest(tmp_path, monkeypatch, view)
    at.run()
    assert not at.exception, at.exception


def test_theme_helpers_emit_html():
    from app.components import theme
    html = theme.chip("Agent idle", "teal")
    assert "ea-chip" in html and "Agent idle" in html
    # unknown tone falls back to gray instead of raising
    assert "ea-chip" in theme.chip("x", "no-such-tone")
