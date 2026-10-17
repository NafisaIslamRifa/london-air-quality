"""FastAPI service: /forecast, /health, /metrics."""
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Query, Request, Response
from mlflow import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from aq import config, live

log = logging.getLogger("aq.api")
BAKED_DIR = Path(os.getenv("AQ_BAKED_MODEL_DIR", "model_baked"))

REQUESTS = Counter("aq_requests_total", "HTTP requests", ["endpoint", "method", "status"])
LATENCY = Histogram("aq_request_latency_seconds", "Request latency", ["endpoint"],
                    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30))
LAST_PREDICTION = Gauge("aq_last_prediction_ugm3", "Most recent NO2 forecast (ug/m3)")
MODEL_INFO = Gauge("aq_model_info", "Loaded model", ["version", "source"])


def load_model():
    """Production model from the MLflow registry; fall back to the copy baked into the image."""
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "15")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    name = config.REGISTERED_MODEL_NAME
    try:
        if not config.MLFLOW_TRACKING_URI:
            raise RuntimeError("MLflow tracking URI not configured")
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        version = MlflowClient().get_model_version_by_alias(name, "production").version
        model = mlflow.pyfunc.load_model(f"models:/{name}@production")
        return model, str(version), "mlflow"
    except Exception as exc:  # noqa: BLE001
        log.warning("MLflow unavailable (%s) - falling back to baked model", exc)
    model = mlflow.pyfunc.load_model(str(BAKED_DIR))
    vfile = BAKED_DIR / "model_version.txt"
    return model, (vfile.read_text().strip() if vfile.exists() else "baked"), "baked"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model, app.state.model_version, app.state.model_source = load_model()
        MODEL_INFO.labels(version=app.state.model_version, source=app.state.model_source).set(1)
        log.info("Model v%s loaded from %s", app.state.model_version, app.state.model_source)
    except Exception:  # noqa: BLE001
        log.exception("Could not load any model")
        app.state.model = None
    yield


app = FastAPI(title="London NO2 forecaster", lifespan=lifespan)


@app.middleware("http")
async def track(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    path = route.path if route else "unmatched"
    if path != "/metrics":
        REQUESTS.labels(path, request.method, str(response.status_code)).inc()
        LATENCY.labels(path).observe(time.perf_counter() - start)
    return response


@app.get("/health")
def health(response: Response):
    if getattr(app.state, "model", None) is None:
        response.status_code = 503
        return {"status": "model_not_loaded"}
    return {"status": "ok", "model_version": app.state.model_version,
            "model_source": app.state.model_source}


@app.get("/forecast")
def forecast(site: str = Query(config.SITE)):
    if site.upper() != config.SITE:
        raise HTTPException(400, f"Only site {config.SITE} is supported")
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(503, "Model not loaded")
    try:
        based_on, row = live.get_feature_row(config.SITE)
    except live.InsufficientData as exc:
        raise HTTPException(503, str(exc))
    except requests.RequestException as exc:
        raise HTTPException(502, f"Upstream data source failed: {exc}")
    pred = max(float(np.asarray(model.predict(row)).ravel()[0]), 0.0)
    LAST_PREDICTION.set(pred)
    return {
        "site": config.SITE,
        "based_on_date": based_on.strftime("%Y-%m-%d"),
        "forecast_date": (based_on + __import__("pandas").Timedelta(days=1)).strftime("%Y-%m-%d"),
        "predicted_no2_ugm3": round(pred, 2),
        "model_version": app.state.model_version,
        "model_source": app.state.model_source,
        "weather_source": row.attrs.get("weather_source", "unknown"),
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)