import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to prompt-optimizer-API"}


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_optimize_prompt(client):
    test_data = {
        "prompt": "Please help me write a story about a brave knight.",
        "model": "gpt",
        "level": "light"
    }
    response = client.post("/optimize", json=test_data)
    assert response.status_code == 200

    data = response.json()
    assert "optimized_prompt" in data
    assert "original_tokens" in data
    assert "token_count" in data
    assert data["compression_ratio"] <= 1.0


def test_invalid_model(client):
    test_data = {
        "prompt": "Test prompt",
        "model": "invalid",
        "level": "light"
    }
    response = client.post("/optimize", json=test_data)
    assert response.status_code == 400


def test_empty_prompt(client):
    test_data = {
        "prompt": "",
        "model": "gpt",
        "level": "light"
    }
    response = client.post("/optimize", json=test_data)
    assert response.status_code == 400


def test_invalid_level(client):
    test_data = {
        "prompt": "Test prompt",
        "model": "gpt",
        "level": "invalid"
    }
    response = client.post("/optimize", json=test_data)
    assert response.status_code == 400


def test_max_tokens_validation(client):
    test_data = {
        "prompt": "Test prompt",
        "model": "gpt",
        "level": "light",
        "max_tokens": -1
    }
    response = client.post("/optimize", json=test_data)
    assert response.status_code == 400