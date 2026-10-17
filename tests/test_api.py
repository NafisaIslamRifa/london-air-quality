import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as main
from aq.features import FEATURES


class FakeModel:
    def predict(self, df):
        return np.array([42.0])


@pytest.fixture
def client(monkeypatch):
    main.app.state.model = FakeModel()          # no `with`: the real model load is skipped
    main.app.state.model_version = "7"
    main.app.state.model_source = "test"
    row = pd.DataFrame([[1.0] * len(FEATURES)], columns=FEATURES)
    monkeypatch.setattr(main.live, "get_feature_row",
                        lambda site: (pd.Timestamp("2026-10-16"), row))
    return TestClient(main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_version"] == "7"


def test_forecast(client):
    body = client.get("/forecast?site=MY1").json()
    assert body["predicted_no2_ugm3"] == 42.0
    assert body["forecast_date"] == "2026-10-17" and body["model_version"] == "7"


def test_unsupported_site(client):
    assert client.get("/forecast?site=XX9").status_code == 400


def test_insufficient_data(client, monkeypatch):
    def boom(site):
        raise main.live.InsufficientData("no data")
    monkeypatch.setattr(main.live, "get_feature_row", boom)
    assert client.get("/forecast").status_code == 503


def test_metrics_exposed(client):
    client.get("/forecast")
    text = client.get("/metrics").text
    assert "aq_requests_total" in text and "aq_last_prediction_ugm3" in text
