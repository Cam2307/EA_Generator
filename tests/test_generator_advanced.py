import random

from factory.generator import GenerationSettings, random_strategy
from factory.models import ExecutionMechanicType


def test_advanced_generation_respects_complexity_cap() -> None:
    settings = GenerationSettings(
        advanced_mode=True,
        complexity_cap=4,
        enable_regime_switching=True,
        enable_mtf_context=True,
        feature_toggles=["momentum", "volatility"],
    )
    strat = random_strategy(
        "EURUSD",
        "M15",
        random.Random(7),
        allowed_mechanics=[ExecutionMechanicType.STANDARD_SLTP],
        generation_settings=settings,
    )
    assert strat.profile.advanced_mode is True
    assert strat.profile.complexity_score <= settings.complexity_cap
    assert strat.profile.regime_switching is True
    assert strat.profile.mtf_context is True
    assert strat.profile.portfolio_signature


def test_basic_generation_keeps_legacy_profile_defaults() -> None:
    strat = random_strategy("EURUSD", "M15", random.Random(11))
    assert strat.profile.advanced_mode is False
    assert strat.profile.complexity_cap == 2
    assert strat.profile.complexity_score >= 1
