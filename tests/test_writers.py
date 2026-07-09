"""Tester .ini writer and .set writer, including optimization ranges."""
from datetime import datetime
from pathlib import Path

from factory.assets.set_writer import build_set_content, write_set_file
from factory.backtest.mt5_runner import write_tester_ini
from factory.models import (
    EntryFilter, EntryFilterType, ExecutionMechanic, ExecutionMechanicType,
    ParamRange, RiskBlock, StrategyDefinition,
)


def _strategy():
    s = StrategyDefinition(
        name="Test Falcon ABC123", symbol="EURUSD", timeframe="M15",
        magic_number=771234,
        entry_filters=[EntryFilter(
            type=EntryFilterType.MA_CROSS,
            params={"fast_period": 10, "slow_period": 50},
            ranges={"fast_period": ParamRange(min=5, max=20, step=5),
                    "slow_period": ParamRange(min=30, max=100, step=10)})],
        mechanic=ExecutionMechanic(
            type=ExecutionMechanicType.HEDGE_LAYER,
            params={"sl_points": 400.0, "tp_points": 600.0,
                    "hedge_trigger_points": 250.0, "hedge_ratio": 1.0},
            ranges={"hedge_trigger_points": ParamRange(min=100, max=400, step=50),
                    "hedge_ratio": ParamRange(min=0.5, max=1.5, step=0.25)}),
        risk=RiskBlock(fixed_lots=0.1),
    )
    return s


def test_ini_writer_static_report_and_sections(tmp_path: Path):
    ini = tmp_path / "tester.ini"
    report = tmp_path / "reports" / "report_abc.xml"
    write_tester_ini(
        ini, expert="EAFactory\\test.ex5", symbol="EURUSD", timeframe="M15",
        from_date=datetime(2023, 1, 1), to_date=datetime(2024, 1, 1),
        deposit=10_000, report_path=report,
        inputs={"Inp_M_hedge_trigger_points": 250.0, "InpLots": 0.1},
        input_ranges={"Inp_M_hedge_trigger_points":
                      ParamRange(min=100, max=400, step=50)},
    )
    text = ini.read_text(encoding="utf-8")
    assert "[Tester]" in text
    assert "Expert=EAFactory\\test.ex5" in text
    assert "Symbol=EURUSD" in text
    assert "Period=M15" in text
    assert "FromDate=2023.01.01" in text
    assert "ToDate=2024.01.01" in text
    assert "ShutdownTerminal=1" in text
    # explicit static report path (extension stripped; MT5 appends it)
    assert f"Report={report.with_suffix('')}" in text
    # mechanic parameter emitted with an optimization range (Y flag)
    assert "Inp_M_hedge_trigger_points=250||100||50||400||Y" in text
    # plain input emitted fixed (N flag)
    assert "InpLots=0.1||0.1||0||0.1||N" in text


def test_ini_writer_optimization_mode(tmp_path: Path):
    ini = tmp_path / "opt.ini"
    write_tester_ini(
        ini, expert="e.ex5", symbol="EURUSD", timeframe="H1",
        from_date=datetime(2023, 1, 1), to_date=datetime(2024, 1, 1),
        deposit=5_000, report_path=tmp_path / "r.xml", optimization=2,
    )
    text = ini.read_text(encoding="utf-8")
    assert "Optimization=2" in text
    assert "Deposit=5000" in text


def test_set_writer_includes_mechanic_ranges():
    content = build_set_content(_strategy())
    # filter params optimizable
    assert "Inp_F0_fast_period=10||5||5||20||Y" in content
    assert "Inp_F0_slow_period=50||30||10||100||Y" in content
    # mechanic params optimizable (hedge distance!)
    assert "Inp_M_hedge_trigger_points=250||100||50||400||Y" in content
    assert "Inp_M_hedge_ratio=1||0.5||0.25||1.5||Y" in content
    # mechanic param without a declared range stays fixed
    assert "Inp_M_sl_points=400||400||0||400||N" in content
    # general inputs present
    assert "InpMagic=771234" in content
    assert "InpLots=0.1" in content


def test_set_file_roundtrip_utf16(tmp_path: Path):
    path = write_set_file(_strategy(), tmp_path)
    assert path.suffix == ".set"
    text = path.read_text(encoding="utf-16")
    assert "Inp_M_hedge_trigger_points=250||100||50||400||Y" in text
