import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

SITE = "MY1"          # Westminster - Marylebone Road
SPECIES = "NO2"
LAT, LON = 51.5225, -0.1546

# Fixed dates keep the dataset reproducible (important for DVC)
START_DATE = "2023-10-01"
END_DATE = os.getenv("AQ_END_DATE", "2026-09-30")

MIN_HOURS_PER_DAY = 18   # fewer valid hourly readings -> day counts as missing

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")


EXPERIMENT_NAME = "london-no2-forecast"
REGISTERED_MODEL_NAME = "aq-no2-forecaster"

_user = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") or (
    f"https://dagshub.com/{_user}/london-air-quality.mlflow" if _user else ""
)