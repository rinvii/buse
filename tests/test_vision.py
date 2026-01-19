import base64

import httpx
import pytest

from buse.models import ViewportInfo
from buse.vision import VisionClient


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text="OK", json_exc=None):
        request = httpx.Request("POST", "http://example/parse/")
        self._response = httpx.Response(
            status_code=status_code, request=request, text=text
        )
        self._json_data = json_data
        self._json_exc = json_exc

    @property
    def status_code(self):
        return self._response.status_code

    @property
    def text(self):
        return self._response.text

    @property
    def request(self):
        return self._response.request

    def raise_for_status(self):
        self._response.raise_for_status()

    def json(self):
        if self._json_exc is not None:
            raise self._json_exc
        return self._json_data


class FakeAsyncClient:
    def __init__(self, response=None, exc=None, **_kwargs):
        self._response = response
        self._exc = exc
        self.last_url = None
        self.last_payload = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url, json=None):
        self.last_url = url
        self.last_payload = json
        if self._exc is not None:
            raise self._exc
        return self._response


class ClientFactory:
    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.instance: FakeAsyncClient | None = None

    def __call__(self, *args, **kwargs):
        self.instance = FakeAsyncClient(response=self.response, exc=self.exc, **kwargs)
        return self.instance


def _viewport():
    return ViewportInfo(width=1000, height=800, device_pixel_ratio=1.0)


@pytest.mark.asyncio
async def test_analyze_requires_server_url(monkeypatch):
    monkeypatch.delenv("BUSE_OMNIPARSER_URL", raising=False)
    client = VisionClient(server_url="")
    with pytest.raises(RuntimeError, match="OmniParser server URL is not set"):
        await client.analyze("data", _viewport())


@pytest.mark.asyncio
async def test_analyze_http_status_error(monkeypatch):
    response = FakeResponse(status_code=500, text="boom", json_data={})
    factory = ClientFactory(response=response)
    monkeypatch.setattr("buse.vision.httpx.AsyncClient", factory)
    client = VisionClient(server_url="http://example")
    with pytest.raises(RuntimeError, match="returned 500: boom"):
        await client.analyze("data", _viewport())


@pytest.mark.asyncio
async def test_analyze_request_error(monkeypatch):
    factory = ClientFactory(exc=RuntimeError("boom"))
    monkeypatch.setattr("buse.vision.httpx.AsyncClient", factory)
    client = VisionClient(server_url="http://example")
    with pytest.raises(RuntimeError, match="Failed to connect to OmniParser: boom"):
        await client.analyze("data", _viewport())


@pytest.mark.asyncio
async def test_analyze_invalid_json(monkeypatch):
    response = FakeResponse(json_exc=ValueError("bad json"), text="not json")
    factory = ClientFactory(response=response)
    monkeypatch.setattr("buse.vision.httpx.AsyncClient", factory)
    client = VisionClient(server_url="http://example")
    with pytest.raises(RuntimeError, match="invalid JSON"):
        await client.analyze("data", _viewport())


@pytest.mark.asyncio
async def test_analyze_non_object_payload(monkeypatch):
    response = FakeResponse(json_data=["nope"])
    factory = ClientFactory(response=response)
    monkeypatch.setattr("buse.vision.httpx.AsyncClient", factory)
    client = VisionClient(server_url="http://example")
    with pytest.raises(RuntimeError, match="JSON object"):
        await client.analyze("data", _viewport())


@pytest.mark.asyncio
async def test_analyze_missing_parsed_content(monkeypatch):
    response = FakeResponse(json_data={"parsed_content_list": "nope"})
    factory = ClientFactory(response=response)
    monkeypatch.setattr("buse.vision.httpx.AsyncClient", factory)
    client = VisionClient(server_url="http://example")
    with pytest.raises(RuntimeError, match="parsed_content_list"):
        await client.analyze("data", _viewport())


@pytest.mark.asyncio
async def test_analyze_parses_elements(monkeypatch):
    response = FakeResponse(
        json_data={
            "parsed_content_list": [
                {"bbox": [0.123456, 0.234567, 0.567891, 0.678912], "type": "button"},
                {"bbox": [0.1, 0.2], "type": "bad"},
                "nope",
            ],
            "som_image_base64": 123,
        }
    )
    factory = ClientFactory(response=response)
    monkeypatch.setattr("buse.vision.httpx.AsyncClient", factory)
    client = VisionClient(server_url="http://example")
    analysis, som = await client.analyze("data", _viewport())
    assert factory.instance is not None
    assert factory.instance.last_url == "http://example/parse/"
    assert factory.instance.last_payload == {"base64_image": "data"}
    assert len(analysis.elements) == 1
    element = analysis.elements[0]
    assert element.type == "button"
    assert element.center_x == 345.67
    assert element.center_y == 365.39
    assert element.bbox == [123.45, 187.65, 567.89, 543.12]
    assert som == ""


def test_save_som_image(tmp_path):
    payload = base64.b64encode(b"hello").decode("ascii")
    out_path = tmp_path / "som.jpg"
    VisionClient.save_som_image(payload, str(out_path))
    assert out_path.read_bytes() == b"hello"
