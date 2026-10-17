"""Live data for the API: recent NO2 (LAQN) + tomorrow's weather -> one feature row."""
import os
import time

import numpy as np
import pandas as pd
import requests

from aq import config
from aq.features import FEATURES, WEATHER_COLS, build_features
from aq.ingest import HOURLY_VARS, LAQN_URL

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
METNO_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
# MET Norway requires an identifying User-Agent with a contact point
USER_AGENT = os.getenv(
    "AQ_USER_AGENT",
    "london-air-quality/1.0 https://github.com/NafisaIslamRifa/london-air-quality",
)

MAX_AGE_DAYS = 3          # refuse to forecast from NO2 data older than this
MIN_HOURS_LIVE = 12       # serving tolerates partial days; training requires 18
MAX_FILL_DAYS = int(os.getenv("AQ_MAX_FILL_DAYS", "3"))   # keep 3 in production
CACHE_TTL_SECONDS = 600
SOURCE_ERRORS = (requests.RequestException, KeyError, ValueError, TypeError)


class InsufficientData(Exception):
    pass


def _utc_today() -> pd.Timestamp:
    return pd.Timestamp.now("UTC").tz_localize(None).normalize()


def _get_json(url, params=None, headers=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=20)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def fetch_recent_no2(site: str, history_days: int = 35) -> pd.Series:
    """Daily mean NO2 for days with enough hourly readings."""
    today = _utc_today()
    url = LAQN_URL.format(
        site=site, species=config.SPECIES,
        start=(today - pd.Timedelta(days=history_days)).strftime("%Y-%m-%d"),
        end=today.strftime("%Y-%m-%d"),
    )
    data = _get_json(url).get("RawAQData", {}).get("Data", [])
    if isinstance(data, dict):
        data = [data]
    if not data:
        raise InsufficientData("LAQN returned no recent NO2 data")
    df = pd.DataFrame(data).rename(columns={"@MeasurementDateGMT": "time", "@Value": "no2"})
    s = pd.to_numeric(df["no2"], errors="coerce").set_axis(pd.to_datetime(df["time"])).sort_index()
    daily = s.resample("D").mean().where(s.resample("D").count() >= MIN_HOURS_LIVE)
    daily = daily.dropna().rename("no2")
    if daily.empty:
        raise InsufficientData("No recent day has enough hourly NO2 readings")
    return daily


# ---------------------------------------------------------------- weather providers
# Each returns hourly rows with the same columns as training (ingest.HOURLY_VARS).

def fetch_weather_metno(day: pd.Timestamp) -> pd.DataFrame:
    payload = _get_json(
        METNO_URL,
        params={"lat": round(config.LAT, 4), "lon": round(config.LON, 4)},
        headers={"User-Agent": USER_AGENT},
    )
    rows = []
    for ts in payload["properties"]["timeseries"]:
        t = pd.to_datetime(ts["time"], utc=True).tz_localize(None)
        if t.normalize() != day:
            continue
        inst = ts["data"]["instant"]["details"]
        nxt = ts["data"].get("next_1_hours")
        if nxt is None:                      # steps beyond ~2.5 days are not hourly
            continue
        rows.append({
            "time": t,
            "temperature_2m": inst["air_temperature"],
            "relative_humidity_2m": inst["relative_humidity"],
            "precipitation": nxt["details"]["precipitation_amount"],
            "surface_pressure": inst["air_pressure_at_sea_level"],      # sea level, not station
            "wind_speed_10m": inst["wind_speed"] * 3.6,                  # m/s -> km/h
            "wind_direction_10m": inst["wind_from_direction"],
        })
    if len(rows) < 18:
        raise requests.RequestException(
            f"MET Norway has only {len(rows)} hourly points for {day:%Y-%m-%d}")
    return pd.DataFrame(rows)


def fetch_weather_forecast(day: pd.Timestamp) -> pd.DataFrame:
    """Open-Meteo forecast API."""
    params = {
        "latitude": config.LAT, "longitude": config.LON,
        "hourly": ",".join(HOURLY_VARS), "timezone": "UTC",
        "start_date": day.strftime("%Y-%m-%d"), "end_date": day.strftime("%Y-%m-%d"),
    }
    wx = pd.DataFrame(_get_json(OPEN_METEO_FORECAST_URL, params)["hourly"])
    wx["time"] = pd.to_datetime(wx["time"])
    return wx


def fetch_weather_hourly(day: pd.Timestamp):
    """Try providers in order. Returns (source_name, hourly_dataframe)."""
    errors = []
    for name, fn in (("met_norway", fetch_weather_metno), ("open_meteo", fetch_weather_forecast)):
        try:
            return name, fn(day)
        except SOURCE_ERRORS as exc:
            errors.append(f"{name}: {exc}")
    raise requests.RequestException("; ".join(errors))


def aggregate_weather(wx: pd.DataFrame) -> pd.DataFrame:
    """Hourly -> daily. Must match ingest.to_daily exactly (a test enforces this)."""
    wx = wx.set_index("time")
    rad = np.deg2rad(wx["wind_direction_10m"])
    wx["wind_u"] = -wx["wind_speed_10m"] * np.sin(rad)
    wx["wind_v"] = -wx["wind_speed_10m"] * np.cos(rad)
    return wx.drop(columns="wind_direction_10m").resample("D").agg({
        "temperature_2m": "mean", "relative_humidity_2m": "mean",
        "precipitation": "sum", "surface_pressure": "mean",
        "wind_speed_10m": "mean", "wind_u": "mean", "wind_v": "mean",
    })


def fetch_weather_fallback() -> pd.Series:
    """Most recent COMPLETE day of observed weather (a few days old): degraded stand-in."""
    today = _utc_today()
    params = {
        "latitude": config.LAT, "longitude": config.LON,
        "hourly": ",".join(HOURLY_VARS), "timezone": "UTC",
        "start_date": (today - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        "end_date": (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    }
    wx = pd.DataFrame(_get_json(OPEN_METEO_ARCHIVE_URL, params)["hourly"])
    wx["time"] = pd.to_datetime(wx["time"])
    wx = wx.dropna()
    full = wx.groupby(wx["time"].dt.normalize()).filter(lambda g: len(g) == 24)
    if full.empty:
        raise InsufficientData("No complete day of fallback weather available")
    return aggregate_weather(full).dropna().iloc[-1]


# ---------------------------------------------------------------- feature row
def build_feature_row(no2_daily: pd.Series, tomorrow_weather: pd.Series) -> pd.DataFrame:
    d = no2_daily.index.max()
    idx = pd.date_range(d - pd.Timedelta(days=29), d + pd.Timedelta(days=1), freq="D")
    frame = pd.DataFrame({"no2": no2_daily.reindex(idx)})
    frame["no2"] = frame["no2"].interpolate(limit=MAX_FILL_DAYS, limit_area="inside")
    for c in WEATHER_COLS:
        frame[c] = np.nan
    frame.loc[d + pd.Timedelta(days=1), WEATHER_COLS] = tomorrow_weather[WEATHER_COLS].to_numpy()

    feats = build_features(frame.rename_axis("date").reset_index())
    row = feats.loc[[d], FEATURES].astype(float)
    if row.isna().to_numpy().any():
        missing = list(row.columns[row.isna().iloc[0]])
        raise InsufficientData(f"Not enough recent NO2 history for features: {missing}")
    return row


def _fetch_feature_row(site: str):
    no2 = fetch_recent_no2(site)
    based_on = no2.index.max()
    age = (_utc_today() - based_on).days
    if age > MAX_AGE_DAYS:
        raise InsufficientData(f"Latest complete NO2 day is {based_on:%Y-%m-%d}, too old")
    day = based_on + pd.Timedelta(days=1)
    try:
        source, hourly = fetch_weather_hourly(day)
        weather = aggregate_weather(hourly).iloc[0]
    except requests.RequestException:
        source, weather = "fallback_recent_observed", fetch_weather_fallback()
    row = build_feature_row(no2, weather)
    row.attrs["weather_source"] = source
    return based_on, row


_CACHE: dict = {}


def get_feature_row(site: str):
    """Cached for 10 minutes so repeated calls don't hammer LAQN / weather APIs."""
    hit = _CACHE.get(site)
    if hit and time.time() - hit[0] < CACHE_TTL_SECONDS:
        return hit[1], hit[2]
    based_on, row = _fetch_feature_row(site)
    _CACHE[site] = (time.time(), based_on, row)
    return based_on, row
