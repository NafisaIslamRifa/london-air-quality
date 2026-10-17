"""Download hourly NO2 (LAQN) and weather (Open-Meteo), aggregate to daily, join."""
import time

import numpy as np
import pandas as pd
import requests

from aq import config

LAQN_URL = (
    "https://api.erg.ic.ac.uk/AirQuality/Data/SiteSpecies/"
    "SiteCode={site}/SpeciesCode={species}/StartDate={start}/EndDate={end}/Json"
)
METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "surface_pressure", "wind_speed_10m", "wind_direction_10m",
]


def _get(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def month_ranges(start, end):
    for p in pd.period_range(start, end, freq="M"):
        s = max(p.start_time.normalize(), pd.Timestamp(start))
        e = min(p.end_time.normalize(), pd.Timestamp(end))
        yield s, e


def fetch_no2() -> pd.DataFrame:
    frames = []
    for s, e in month_ranges(config.START_DATE, config.END_DATE):
        url = LAQN_URL.format(
            site=config.SITE, species=config.SPECIES,
            start=s.strftime("%Y-%m-%d"), end=e.strftime("%Y-%m-%d"),
        )
        data = _get(url).get("RawAQData", {}).get("Data", [])
        if isinstance(data, dict):          # single record comes back as a dict
            data = [data]
        print(f"LAQN {s:%Y-%m}: {len(data)} rows")
        if data:
            df = pd.DataFrame(data).rename(
                columns={"@MeasurementDateGMT": "time", "@Value": "no2"})
            frames.append(df[["time", "no2"]])
    if not frames:
        raise RuntimeError("LAQN returned no data - check SITE/SPECIES/dates in config.py")
    raw = pd.concat(frames, ignore_index=True)
    raw["time"] = pd.to_datetime(raw["time"])
    raw["no2"] = pd.to_numeric(raw["no2"], errors="coerce")
    return raw.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def fetch_weather() -> pd.DataFrame:
    params = {
        "latitude": config.LAT, "longitude": config.LON,
        "start_date": config.START_DATE, "end_date": config.END_DATE,
        "hourly": ",".join(HOURLY_VARS), "timezone": "UTC",
    }
    wx = pd.DataFrame(_get(METEO_URL, params)["hourly"])
    wx["time"] = pd.to_datetime(wx["time"])
    print(f"Open-Meteo: {len(wx)} hourly rows")
    return wx


def to_daily(no2: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    s = no2.set_index("time")["no2"]
    daily_no2 = s.resample("D").mean().where(s.resample("D").count() >= config.MIN_HOURS_PER_DAY)

    wx = wx.set_index("time")
    rad = np.deg2rad(wx["wind_direction_10m"])      # circular variable -> u/v components
    wx["wind_u"] = -wx["wind_speed_10m"] * np.sin(rad)
    wx["wind_v"] = -wx["wind_speed_10m"] * np.cos(rad)
    daily_wx = wx.drop(columns="wind_direction_10m").resample("D").agg({
        "temperature_2m": "mean", "relative_humidity_2m": "mean",
        "precipitation": "sum", "surface_pressure": "mean",
        "wind_speed_10m": "mean", "wind_u": "mean", "wind_v": "mean",
    })

    full = pd.date_range(config.START_DATE, config.END_DATE, freq="D")
    daily = daily_wx.reindex(full).join(daily_no2.rename("no2"))
    daily.index.name = "date"
    return daily.reset_index()


def main():
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    no2, wx = fetch_no2(), fetch_weather()
    no2.to_parquet(config.DATA_RAW / "laqn_no2_hourly.parquet", index=False)
    wx.to_parquet(config.DATA_RAW / "weather_hourly.parquet", index=False)
    daily = to_daily(no2, wx)
    daily.to_parquet(config.DATA_PROCESSED / "daily_raw.parquet", index=False)
    print(f"Saved {len(daily)} days, {daily['no2'].isna().sum()} without a valid NO2 mean")


if __name__ == "__main__":
    main()