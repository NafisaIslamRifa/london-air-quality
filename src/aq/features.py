"""Feature engineering. Row at date d predicts NO2 on d+1.

NO2 features use only data up to and including d.
fc_* features describe day d+1 (tomorrow's weather forecast + calendar).
"""
import numpy as np
import pandas as pd

WEATHER_COLS = ["temperature_2m", "relative_humidity_2m", "precipitation",
                "surface_pressure", "wind_speed_10m", "wind_u", "wind_v"]
LAGS = [0, 1, 2, 6]
ROLL_WINDOWS = [3, 7, 14]
TARGET = "target_no2"

FEATURES = (
    [f"no2_lag{k}" for k in LAGS]
    + [f"no2_roll{w}" for w in ROLL_WINDOWS]
    + ["no2_roll7_std"]
    + [f"fc_{c}" for c in WEATHER_COLS]
    + ["fc_dayofweek", "fc_is_weekend", "fc_month", "fc_doy_sin", "fc_doy_cos"]
)


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    # full calendar index: missing days become NaN, so shifts are by calendar day
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))
    df.index.name = "date"

    out = pd.DataFrame(index=df.index)
    no2 = df["no2"]
    for k in LAGS:
        out[f"no2_lag{k}"] = no2.shift(k)
    for w in ROLL_WINDOWS:
        out[f"no2_roll{w}"] = no2.rolling(w, min_periods=w).mean()   # window ends today
    out["no2_roll7_std"] = no2.rolling(7, min_periods=7).std()

    for c in WEATHER_COLS:
        out[f"fc_{c}"] = df[c].shift(-1)                              # tomorrow's weather

    tomorrow = out.index + pd.Timedelta(days=1)
    out["fc_dayofweek"] = tomorrow.dayofweek
    out["fc_is_weekend"] = (tomorrow.dayofweek >= 5).astype(int)
    out["fc_month"] = tomorrow.month
    out["fc_doy_sin"] = np.sin(2 * np.pi * tomorrow.dayofyear / 365.25)
    out["fc_doy_cos"] = np.cos(2 * np.pi * tomorrow.dayofyear / 365.25)

    out[TARGET] = no2.shift(-1)
    return out[FEATURES + [TARGET]]


def make_dataset(daily: pd.DataFrame) -> pd.DataFrame:
    """Training-ready table: complete rows only, all float."""
    return build_features(daily).dropna().astype(float)