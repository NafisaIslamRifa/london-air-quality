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
END_DATE = "2026-09-30"

MIN_HOURS_PER_DAY = 18   # fewer valid hourly readings -> day counts as missing

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")