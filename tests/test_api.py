"""Tests for the prediction API.

These are the checks that would catch a broken deployment: the service starts,
it returns a sane number for a sane car, and it rejects nonsense input instead
of guessing.
"""

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import app  # noqa: E402

VALID_CAR = {
    "model": "peugeot 206",
    "year": 2018,
    "mileage_km": 120_000,
    "engine_size_l": 1.4,
    "fuel_type": "petrol",
    "transmission": "manual",
    "body_condition": "intact",
    "color": "white",
    "previous_owners": 2,
}


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    """Make sure a model artifact exists before any test runs."""
    if not (PROJECT_ROOT / "models" / "model.pkl").exists():
        if not (PROJECT_ROOT / "data" / "cars.csv").exists():
            subprocess.run(
                [sys.executable, "data/generate_data.py"], cwd=PROJECT_ROOT, check=True
            )
        subprocess.run(
            [sys.executable, "src/train.py"], cwd=PROJECT_ROOT, check=True
        )


@pytest.fixture(scope="session")
def client(trained_model):
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_model_info_exposes_metrics(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "r2" in response.json()["metrics"]


def test_predict_returns_positive_price(client):
    response = client.post("/predict", json=VALID_CAR)
    assert response.status_code == 200
    assert response.json()["predicted_price_million_toman"] > 0


def test_older_car_is_cheaper_than_newer_one(client):
    """A sanity check on the model's logic, not just its plumbing."""
    old = {**VALID_CAR, "year": 2008, "mileage_km": 300_000}
    new = {**VALID_CAR, "year": 2023, "mileage_km": 40_000}

    old_price = client.post("/predict", json=old).json()
    new_price = client.post("/predict", json=new).json()

    assert (
        old_price["predicted_price_million_toman"]
        < new_price["predicted_price_million_toman"]
    )


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("year", 1899),
        ("mileage_km", -50),
        ("transmission", "rocket_powered"),
        ("previous_owners", 0),
    ],
)
def test_invalid_input_is_rejected(client, field, bad_value):
    response = client.post("/predict", json={**VALID_CAR, field: bad_value})
    assert response.status_code == 422


def test_missing_field_is_rejected(client):
    payload = {key: value for key, value in VALID_CAR.items() if key != "year"}
    assert client.post("/predict", json=payload).status_code == 422
