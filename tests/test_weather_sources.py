import numpy as np
import pandas as pd
import pytest
import requests

from aq import live
from aq.features import WEATHER_COLS

DAY = pd.Timestamp("2026-10-17")


def fake_metno(n_hours=24):
    ts = []
    for h in range(n_hours):
        t = DAY + pd.Timedelta(hours=h)
        ts.append({
            "time": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "data": {
                "instant": {"details": {
                    "air_temperature": 10.0 + h, "relative_humidity": 80.0,
                    "air_pressure_at_sea_level": 1010.0, "wind_speed": 5.0,
                    "wind_from_direction": 270.0}},
                "next_1_hours": {"details": {"precipitation_amount": 0.5}},
            },
        })
    return {"properties": {"timeseries": ts}}


def test_metno_units_and_columns(monkeypatch):
    monkeypatch.setattr(live, "_get_json", lambda *a, **k: fake_metno())
    df = live.fetch_weather_metno(DAY)
    assert df["wind_speed_10m"].iloc[0] == pytest.approx(18.0)         # 5 m/s -> km/h
    daily = live.aggregate_weather(df).iloc[0]
    assert daily["temperature_2m"] == pytest.approx(21.5)
    assert daily["precipitation"] == pytest.approx(12.0)
    assert set(WEATHER_COLS) <= set(live.aggregate_weather(df).columns)


def test_metno_partial_day_is_rejected(monkeypatch):
    monkeypatch.setattr(live, "_get_json", lambda *a, **k: fake_metno(n_hours=5))
    with pytest.raises(requests.RequestException):
        live.fetch_weather_metno(DAY)


def test_chain_falls_through_to_open_meteo(monkeypatch):
    def boom(day):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(live, "fetch_weather_metno", boom)
    monkeypatch.setattr(live, "fetch_weather_forecast", lambda day: "open-meteo-data")
    assert live.fetch_weather_hourly(DAY) == ("open_meteo", "open-meteo-data")


def test_all_providers_down_uses_flagged_fallback(monkeypatch):
    def boom(day):
        raise requests.ConnectionError("down")
    idx = pd.date_range(end="2026-10-16", periods=30, freq="D")
    no2 = pd.Series(np.linspace(20, 60, 30), index=idx, name="no2")
    monkeypatch.setattr(live, "fetch_recent_no2", lambda site: no2)
    monkeypatch.setattr(live, "MAX_AGE_DAYS", 10**6)
    monkeypatch.setattr(live, "fetch_weather_metno", boom)
    monkeypatch.setattr(live, "fetch_weather_forecast", boom)
    monkeypatch.setattr(live, "fetch_weather_fallback",
                        lambda: pd.Series({c: 1.0 for c in WEATHER_COLS}))
    _, row = live._fetch_feature_row("MY1")
    assert row.attrs["weather_source"] == "fallback_recent_observed"
