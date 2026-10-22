import asyncio
import sys

from httpx import ASGITransport, AsyncClient

from app.main import create_app


def test_application_initializes() -> None:
    app = create_app()
    assert app.title == "DaMeme API"


def test_health_endpoint_returns_valid_response() -> None:
    async def request_health() -> tuple[int, dict[str, str]]:
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://testserver"
        ) as client:
            response = await client.get("/api/v1/health")
            return response.status_code, response.json()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    status_code, payload = asyncio.run(request_health())

    assert status_code == 200
    assert payload == {
        "status": "ok",
        "service": "dameme-api",
        "environment": "development",
    }
