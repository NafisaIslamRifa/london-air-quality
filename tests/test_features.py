import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from aq.features import FEATURES, TARGET, WEATHER_COLS, build_features


def make_daily(n=60, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n, freq="D"),
                       "no2": rng.uniform(10, 80, n)})
    for c in WEATHER_COLS:
        df[c] = rng.normal(size=n)
    return df


def test_target_is_next_day_no2():
    daily = make_daily()
    out = build_features(daily)
    d = pd.Timestamp("2024-02-01")
    expected = daily.loc[daily["date"] == d + pd.Timedelta(days=1), "no2"].iloc[0]
    assert out.loc[d, TARGET] == expected


def test_lags_and_rolling_are_correct():
    daily = make_daily()
    out = build_features(daily)
    s = daily.set_index("date")["no2"]
    d = pd.Timestamp("2024-02-10")
    assert out.loc[d, "no2_lag0"] == s[d]
    assert out.loc[d, "no2_lag1"] == s[d - pd.Timedelta(days=1)]
    assert out.loc[d, "no2_lag6"] == s[d - pd.Timedelta(days=6)]
    assert np.isclose(out.loc[d, "no2_roll3"], s[d - pd.Timedelta(days=2): d].mean())


def test_lags_use_calendar_days_not_row_position():
    daily = make_daily()
    gap_day = pd.Timestamp("2024-02-05")
    daily = daily[daily["date"] != gap_day]            # simulate a dropped day
    out = build_features(daily)
    after_gap = gap_day + pd.Timedelta(days=1)
    assert np.isnan(out.loc[after_gap, "no2_lag1"])    # must NOT silently use Feb 4
    assert not np.isnan(out.loc[after_gap, "no2_lag0"])


def test_no_future_leakage():
    daily = make_daily()
    d = pd.Timestamp("2024-02-10")
    base = build_features(daily).loc[d, FEATURES]

    tampered = daily.copy()
    tampered.loc[tampered["date"] > d, "no2"] = 9999.0                 # any NO2 after d
    later = tampered["date"] > d + pd.Timedelta(days=1)
    tampered.loc[later, WEATHER_COLS] = 9999.0                          # weather after d+1
    after = build_features(tampered).loc[d, FEATURES]

    assert_series_equal(base, after)


def test_calendar_features_describe_tomorrow():
    out = build_features(make_daily())
    d = pd.Timestamp("2024-02-10")                     # a Saturday -> tomorrow is Sunday
    assert out.loc[d, "fc_dayofweek"] == 6
    assert out.loc[d, "fc_is_weekend"] == 1