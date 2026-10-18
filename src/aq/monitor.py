"""Daily monitoring job.

1. Ask the deployed API for a forecast (logs it, times it, measures cold-start wake time).
2. Score logged forecasts against actual LAQN NO2 once the actual day is available.
3. Compare recent inputs with same-month training statistics (simple drift check).
4. Push everything to Grafana Cloud via the Influx line-protocol endpoint.

  python -m aq.monitor --build-reference   # one-off, needs data/processed/daily.parquet
  python -m aq.monitor --dry-run           # logs the forecast, prints metrics, pushes nothing
  python -m aq.monitor                     # real run
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import requests

from aq import config, live

LOG_PATH = config.ROOT / "monitoring" / "predictions.csv"
REF_PATH = config.ROOT / "monitoring" / "reference_stats.json"
DRIFT_FEATURES = ["no2", "temperature_2m", "wind_speed_10m"]
DRIFT_THRESHOLD = 1.0   # heuristic: recent 14-day mean is > 1 daily-std from the same-month training mean
LOG_COLUMNS = ["made_at", "forecast_date", "based_on_date", "lead_days",
               "predicted_no2_ugm3", "model_version", "model_source", "weather_source"]


def load_dotenv():
    path = config.ROOT / ".env"
    if path.exists():
        for raw in path.read_text().splitlines():
            raw = raw.strip()
            if raw and not raw.startswith("#") and "=" in raw:
                key, value = raw.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _today() -> pd.Timestamp:
    return pd.Timestamp.now("UTC").tz_localize(None).normalize()


# ------------------------------------------------------------------ reference stats
def build_reference():
    df = pd.read_parquet(config.DATA_PROCESSED / "daily.parquet")
    df["month"] = pd.to_datetime(df["date"]).dt.month
    ref = {str(m): {f: {"mean": float(g[f].mean()), "std": float(g[f].std())}
                    for f in DRIFT_FEATURES}
           for m, g in df.groupby("month")}
    REF_PATH.parent.mkdir(parents=True, exist_ok=True)
    REF_PATH.write_text(json.dumps(ref, indent=2))
    print(f"Wrote {REF_PATH} ({len(ref)} months)")


# ------------------------------------------------------------------ API check
def check_api(api_url):
    base = api_url.rstrip("/")
    out = {"api_up": 0.0, "forecast_ok": 0.0}
    resp = None
    for attempt in range(3):                 # a sleeping free instance needs time to wake
        t0 = time.perf_counter()
        try:
            r = requests.get(f"{base}/health", timeout=180)
            if r.status_code == 200:
                out["api_up"] = 1.0
                out["api_wake_seconds"] = time.perf_counter() - t0
                break
        except requests.RequestException as exc:
            print(f"health attempt {attempt + 1} failed: {exc}")
        time.sleep(10)
    if not out["api_up"]:
        return out, None
    t1 = time.perf_counter()
    try:
        r = requests.get(f"{base}/forecast", params={"site": config.SITE}, timeout=120)
        out["api_latency_seconds"] = time.perf_counter() - t1
        if r.status_code == 200:
            out["forecast_ok"] = 1.0
            resp = r.json()
        else:
            print(f"forecast returned {r.status_code}: {r.text[:300]}")
    except requests.RequestException as exc:
        print(f"forecast call failed: {exc}")
    return out, resp


# ------------------------------------------------------------------ prediction log
def read_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH, parse_dates=["forecast_date", "based_on_date"])
    return pd.DataFrame(columns=LOG_COLUMNS)


def append_log(resp) -> pd.DataFrame:
    log = read_log()
    now = pd.Timestamp.now("UTC").tz_localize(None)
    fdate = pd.Timestamp(resp["forecast_date"])
    row = {
        "made_at": now.strftime("%Y-%m-%dT%H:%M:%S"), "forecast_date": fdate,
        "based_on_date": pd.Timestamp(resp["based_on_date"]),
        "lead_days": (fdate - now.normalize()).days,
        "predicted_no2_ugm3": resp["predicted_no2_ugm3"],
        "model_version": resp["model_version"], "model_source": resp["model_source"],
        "weather_source": resp.get("weather_source", "unknown"),
    }
    new = pd.DataFrame([row])
    log = new if log.empty else pd.concat([log, new], ignore_index=True)
    log = log.drop_duplicates("forecast_date", keep="first").sort_values("forecast_date")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(LOG_PATH, index=False, date_format="%Y-%m-%d")
    return log


def score(log: pd.DataFrame, actuals: pd.Series) -> pd.DataFrame:
    df = log.copy()
    df["actual"] = df["forecast_date"].map(actuals)
    df = df.dropna(subset=["actual"])
    df["abs_error"] = (df["predicted_no2_ugm3"] - df["actual"]).abs()
    return df.sort_values("forecast_date")


# ------------------------------------------------------------------ drift
def recent_weather_daily(days=16) -> pd.DataFrame:
    today = _today()
    params = {
        "latitude": config.LAT, "longitude": config.LON,
        "hourly": ",".join(live.HOURLY_VARS), "timezone": "UTC",
        "start_date": (today - pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
        "end_date": (today - pd.Timedelta(days=2)).strftime("%Y-%m-%d"),   # archive lags ~2 days
    }
    wx = pd.DataFrame(live._get_json(live.OPEN_METEO_ARCHIVE_URL, params)["hourly"])
    wx["time"] = pd.to_datetime(wx["time"])
    wx = wx.dropna()
    full = wx.groupby(wx["time"].dt.normalize()).filter(lambda g: len(g) == 24)
    return live.aggregate_weather(full).dropna()


def drift_shifts(actuals: pd.Series) -> dict:
    if not REF_PATH.exists():
        print("No reference_stats.json - run: python -m aq.monitor --build-reference")
        return {}
    ref = json.loads(REF_PATH.read_text())[str(_today().month)]
    cutoff = _today() - pd.Timedelta(days=14)
    recent = {"no2": actuals[actuals.index >= cutoff]}
    try:
        wx = recent_weather_daily()
        wx = wx[wx.index >= cutoff]
        recent["temperature_2m"] = wx["temperature_2m"]
        recent["wind_speed_10m"] = wx["wind_speed_10m"]
    except live.SOURCE_ERRORS as exc:
        print(f"weather drift inputs unavailable: {exc}")
    shifts = {}
    for feat, values in recent.items():
        if len(values) >= 5:
            shifts[feat] = float((values.mean() - ref[feat]["mean"]) / ref[feat]["std"])
    return shifts


# ------------------------------------------------------------------ Grafana Cloud push
def to_line(fields: dict, tags: dict | None = None):
    fields = {k: float(v) for k, v in fields.items() if v is not None and np.isfinite(v)}
    if not fields:
        return None
    tag_s = "".join(f",{k}={v}" for k, v in (tags or {}).items())
    return f"aq{tag_s} " + ",".join(f"{k}={v}" for k, v in fields.items())


def push(lines):
    r = requests.post(
        os.environ["GRAFANA_PUSH_URL"], data="\n".join(lines).encode(),
        auth=(os.environ["GRAFANA_USER"], os.environ["GRAFANA_API_KEY"]),
        headers={"Content-Type": "text/plain"}, timeout=30)
    print("push status:", r.status_code, r.text[:200])
    r.raise_for_status()


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-reference", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.build_reference:
        return build_reference()

    api_url = os.environ.get("AQ_API_URL")
    if not api_url:
        raise SystemExit("Set AQ_API_URL in .env (your Render URL)")

    api_metrics, resp = check_api(api_url)
    log = append_log(resp) if resp else read_log()
    if resp:
        print(f"forecast {resp['predicted_no2_ugm3']} for {resp['forecast_date']} "
              f"(weather: {resp.get('weather_source')})")

    actuals = live.fetch_recent_no2(config.SITE)
    scored = score(log, actuals) if not log.empty else log
    shifts = drift_shifts(actuals)

    fields = dict(api_metrics)
    if resp:
        fields["predicted_no2_ugm3"] = resp["predicted_no2_ugm3"]
    if len(scored):
        last = scored.iloc[-1]
        fields.update(
            eval_predicted_no2_ugm3=last["predicted_no2_ugm3"], eval_actual_no2_ugm3=last["actual"],
            eval_abs_error_ugm3=last["abs_error"], eval_lead_days=last["lead_days"],
            rolling_mae_7d_ugm3=scored.tail(7)["abs_error"].mean(), eval_count=len(scored.tail(7)))
    else:
        print("No forecast has an actual yet - predicted-vs-actual starts after the next run(s).")
    if shifts:
        fields["drift_score"] = max(abs(v) for v in shifts.values())
        fields["drift_flag"] = float(fields["drift_score"] > DRIFT_THRESHOLD)

    lines = [to_line(fields)] + [to_line({"drift_shift": v}, {"feature": f}) for f, v in shifts.items()]
    lines = [ln for ln in lines if ln]
    print("\n".join(lines))
    if args.dry_run:
        print("(dry run: nothing pushed)")
    else:
        push(lines)


if __name__ == "__main__":
    main()