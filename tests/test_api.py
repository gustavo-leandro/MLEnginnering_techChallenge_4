"""Testes de integracao dos endpoints da API."""

import numpy as np
import pickle
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parent.parent
client = TestClient(app)


def make_prices(n: int = 90) -> list[float]:
    """Gera uma lista de precos reais a partir do conjunto de teste."""
    X = np.load(ROOT / "data/processed/X_test.npy")[0]
    with open(ROOT / "models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    prices = scaler.inverse_transform(X).flatten().tolist()
    return prices[:n]


# --- /health ---

def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- /predict ---

def test_predict_returns_valid_price():
    payload = {"prices": make_prices(90)}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_close" in body
    assert body["predicted_close"] > 0
    assert body["ticker"] == "VALE3.SA"
    assert body["look_back_used"] == 90


def test_predict_uses_last_90_prices():
    """Envia mais de 90 precos: a API deve usar apenas os ultimos 90."""
    payload = {"prices": make_prices(90) + [99.0, 98.0, 97.0]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["look_back_used"] == 90


def test_predict_rejects_insufficient_prices():
    payload = {"prices": [60.0] * 50}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_negative_prices():
    payload = {"prices": [-1.0] + [60.0] * 89}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_zero_price():
    payload = {"prices": [0.0] + [60.0] * 89}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
