import numpy as np
import pandas as pd
import pytest

from aq import config, ingest, live
from aq.features import FEATURES, WEATHER_COLS

WEATHER = pd.Series({c: 1.0 for c in WEATHER_COLS})


def make_no2(days=30, end="2026-10-16"):
    idx = pd.date_range(end=end, periods=days, freq="D")
    return pd.Series(np.linspace(20, 60, days), index=idx, name="no2")


def test_live_weather_aggregation_matches_training(monkeypatch):
    idx = pd.date_range("2024-03-01", periods=24 * 5, freq="h")
    rng = np.random.default_rng(1)
    wx = pd.DataFrame({"time": idx, **{v: rng.uniform(0, 10, len(idx)) for v in ingest.HOURLY_VARS}})
    no2 = pd.DataFrame({"time": idx, "no2": rng.uniform(10, 50, len(idx))})
    monkeypatch.setattr(config, "START_DATE", "2024-03-01")
    monkeypatch.setattr(config, "END_DATE", "2024-03-05")
    train_side = ingest.to_daily(no2, wx)
    live_side = live.aggregate_weather(wx)
    np.testing.assert_allclose(train_side[WEATHER_COLS].to_numpy(), live_side[WEATHER_COLS].to_numpy())


def test_feature_row_is_complete():
    row = live.build_feature_row(make_no2(), WEATHER)
    assert list(row.columns) == FEATURES
    assert not row.isna().to_numpy().any()
    assert row["no2_lag0"].iloc[0] == 60.0


def test_short_gap_is_filled():
    s = make_no2()
    row = live.build_feature_row(s.drop(s.index[-6:-3]), WEATHER)     # 3 missing days
    assert not row.isna().to_numpy().any()


def test_long_gap_is_rejected():
    s = make_no2()
    with pytest.raises(live.InsufficientData):
        live.build_feature_row(s.drop(s.index[-8:-3]), WEATHER)       # 5 missing days