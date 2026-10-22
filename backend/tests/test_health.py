from fastapi.testclient import TestClient

from app.main import create_app


def test_application_initializes() -> None:
    app = create_app()
    assert app.title == "DaMeme API"


def test_health_endpoint_returns_valid_response() -> None:
    client = TestClient(create_app())
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "dameme-api",
        "environment": "development",
    }
