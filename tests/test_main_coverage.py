import base64
import json
from types import SimpleNamespace

import httpx
import pytest
import typer

import buse.main as main
from buse.models import VisualAnalysis, VisualElement


class DummyCtx:
    def __init__(self, instance_id="b1"):
        self.obj = {"instance_id": instance_id}


class FakeDomState:
    def __init__(self, value="dom"):
        self._value = value

    def llm_representation(self):
        return self._value


class FakeTab:
    def __init__(self, target_id="t1", title="T", url="U"):
        self.target_id = target_id
        self.title = title
        self.url = url


class FakeEvent:
    def __init__(self, session, include_screenshot: bool):
        self._session = session
        self._include_screenshot = include_screenshot

    async def event_result(self, **_kwargs):
        return await self._session.get_browser_state_summary(
            include_screenshot=self._include_screenshot
        )


class FakeEventBus:
    def __init__(self, session):
        self._session = session

    def dispatch(self, event):
        include_screenshot = bool(getattr(event, "include_screenshot", False))
        return FakeEvent(self._session, include_screenshot)


def _make_summary(screenshot=None, dom_state=None):
    if dom_state is None:
        dom_state = FakeDomState()
    return SimpleNamespace(
        url="http://example",
        title="Title",
        screenshot=screenshot,
        dom_state=dom_state,
    )


def _make_cdp_session(
    viewport_value=None, capture_result=None, capture_exc=None, eval_exc=None
):
    async def evaluate(params, session_id=None):
        if eval_exc:
            raise eval_exc
        return {"result": {"value": viewport_value}}

    async def capture_screenshot(params, session_id=None):
        if capture_exc:
            raise capture_exc
        return capture_result

    send = SimpleNamespace(
        Runtime=SimpleNamespace(evaluate=evaluate),
        Page=SimpleNamespace(captureScreenshot=capture_screenshot),
    )
    return SimpleNamespace(session_id="s1", cdp_client=SimpleNamespace(send=send))


class FakeBrowserSession:
    def __init__(self, summaries, cdp_session):
        self._summaries = list(summaries)
        self._cdp_session = cdp_session
        self.agent_focus_target_id = "focus"
        self.event_bus = FakeEventBus(self)

    async def start(self):
        return None

    async def stop(self):
        return None

    async def get_browser_state_summary(self, include_screenshot=False, cached=False):
        if not self._summaries:
            raise RuntimeError("no summaries")
        item = self._summaries.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def get_or_create_cdp_session(self):
        if isinstance(self._cdp_session, Exception):
            raise self._cdp_session
        return self._cdp_session

    async def get_tabs(self):
        return [FakeTab()]


class FakeResponse:
    def __init__(self, status_code=200):
        request = httpx.Request("GET", "http://example/probe/")
        self._response = httpx.Response(
            status_code=status_code, request=request, text="boom"
        )

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


class FakeAsyncClient:
    def __init__(self, response=None, exc=None, **_kwargs):
        self._response = response
        self._exc = exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def get(self, url):
        if self._exc:
            raise self._exc
        return self._response


class FakeVisionClient:
    analysis: VisualAnalysis | None
    som_image: str

    def __init__(self, server_url=None):
        self.server_url = server_url
        self.analysis = None
        self.som_image = ""

    async def analyze(self, base64_image, viewport):
        return self.analysis, self.som_image

    def save_som_image(self, base64_data, output_path):
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(base64_data))


@pytest.fixture(autouse=True)
def reset_globals(monkeypatch):
    main._browser_sessions.clear()
    main._file_systems.clear()
    main._controllers.clear()
    main._selector_cache.clear()
    main._omniparser_probe_cache.clear()
    monkeypatch.delenv("BUSE_IMAGE_QUALITY", raising=False)
    monkeypatch.delenv("BUSE_OMNIPARSER_URL", raising=False)

    monkeypatch.setattr(main, "BrowserSession", FakeBrowserSession)

    yield
    main.state.profile = False


def test_parse_image_quality_invalid_env(monkeypatch):
    monkeypatch.setenv("BUSE_IMAGE_QUALITY", "nope")
    with pytest.raises(typer.BadParameter):
        main._parse_image_quality(None)


def test_parse_image_quality_out_of_range():
    with pytest.raises(typer.BadParameter):
        main._parse_image_quality(0)


def test_parse_image_quality_valid_env_and_normalize(monkeypatch):
    monkeypatch.setenv("BUSE_IMAGE_QUALITY", "80")
    assert main._parse_image_quality(None) == 80
    assert main._normalize_omniparser_endpoint(" http://x/ ") == "http://x"


def test_settings_load_save_and_probe_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(main.session_manager, "config_dir", tmp_path)
    main._omniparser_probe_cache.clear()

    assert main._load_settings() == {}

    main._save_settings({"a": 1})
    settings_path = tmp_path / "settings.json"
    assert json.loads(settings_path.read_text())["a"] == 1

    settings_path.write_text("{bad")
    assert main._load_settings() == {}

    settings_path.write_text("[]")
    assert main._load_settings() == {}

    settings_path.write_text(json.dumps({"omniparser_probe": "nope"}))
    main._omniparser_probe_cache.clear()
    main._load_omniparser_probe_cache()
    assert main._omniparser_probe_cache == {}

    settings_path.write_text(
        json.dumps(
            {
                "omniparser_probe": {
                    "http://x": 1,
                    "http://y": {"timestamp": 2},
                    "bad": "no",
                }
            }
        )
    )
    main._omniparser_probe_cache.clear()
    main._load_omniparser_probe_cache()
    assert main._omniparser_probe_cache["http://x"] == 1.0
    assert main._omniparser_probe_cache["http://y"] == 2.0

    main._omniparser_probe_cache["cached"] = 3.0
    main._load_omniparser_probe_cache()


def test_save_settings_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(main.session_manager, "config_dir", tmp_path)

    def boom(*_args, **_kwargs):
        raise OSError("fail")

    monkeypatch.setattr(main.Path, "write_text", boom)
    main._save_settings({"a": 1})


def test_should_probe_and_mark(monkeypatch, tmp_path):
    monkeypatch.setattr(main.session_manager, "config_dir", tmp_path)
    main._omniparser_probe_cache.clear()
    monkeypatch.setattr(main.time, "time", lambda: 1000.0)
    assert main._should_probe_omniparser("http://x") is True
    main._mark_omniparser_probe("http://x")
    monkeypatch.setattr(main.time, "time", lambda: 1100.0)
    assert main._should_probe_omniparser("http://x") is False
    monkeypatch.setattr(main.time, "time", lambda: 2000.0)
    assert main._should_probe_omniparser("http://x") is True


def test_build_error_details():
    details = main._build_error_details("stage", retryable=True, foo="bar")
    assert details["stage"] == "stage"
    assert details["retryable"] is True
    assert details["context"]["foo"] == "bar"


def test_output_error_includes_details(capsys):
    main._output_error("observe", {}, "boom", error_details={"stage": "test"})
    out = json.loads(capsys.readouterr().out)
    assert out["error_details"]["stage"] == "test"


def test_observe_requires_omniparser_env():
    with pytest.raises(typer.BadParameter):
        main.observe(
            DummyCtx(),
            screenshot=False,
            path=None,
            omniparser=True,
            no_dom=False,
        )


def test_observe_omniparser_probe_http_error(monkeypatch):
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://example")
    monkeypatch.setattr(main, "_should_probe_omniparser", lambda *_: True)
    monkeypatch.setattr(
        main.httpx,
        "AsyncClient",
        lambda **kwargs: FakeAsyncClient(response=FakeResponse(status_code=500)),
    )
    with pytest.raises(SystemExit):
        main.observe(
            DummyCtx(),
            screenshot=False,
            path=None,
            omniparser=True,
            no_dom=False,
        )


def test_observe_omniparser_probe_connection_error(monkeypatch):
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://example")
    monkeypatch.setattr(main, "_should_probe_omniparser", lambda *_: True)
    monkeypatch.setattr(
        main.httpx,
        "AsyncClient",
        lambda **kwargs: FakeAsyncClient(exc=RuntimeError("boom")),
    )
    with pytest.raises(SystemExit):
        main.observe(
            DummyCtx(),
            screenshot=False,
            path=None,
            omniparser=True,
            no_dom=False,
        )


def test_observe_omniparser_success(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://example")
    monkeypatch.setattr(main, "_should_probe_omniparser", lambda *_: True)
    monkeypatch.setattr(
        main.httpx,
        "AsyncClient",
        lambda **kwargs: FakeAsyncClient(response=FakeResponse(status_code=200)),
    )
    session_info = SimpleNamespace(cdp_url="x", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)
    screenshot_data = base64.b64encode(b"img").decode("ascii")
    summary = _make_summary(screenshot=screenshot_data)
    cdp_session = _make_cdp_session(
        viewport_value={"width": 100, "height": 200, "device_pixel_ratio": 1.0}
    )
    monkeypatch.setattr(
        main,
        "BrowserSession",
        lambda cdp_url=None: FakeBrowserSession([summary], cdp_session),
    )

    analysis = VisualAnalysis(
        elements=[
            VisualElement(
                index=0,
                type="button",
                content="ok",
                interactivity=True,
                center_x=1.0,
                center_y=2.0,
                bbox=[0.1, 0.2, 0.3, 0.4],
            )
        ]
    )
    fake_client = FakeVisionClient()
    fake_client.analysis = analysis
    fake_client.som_image = base64.b64encode(b"som").decode("ascii")

    monkeypatch.setattr("buse.vision.VisionClient", lambda server_url=None: fake_client)
    monkeypatch.setattr("buse.utils.downscale_image", lambda *a, **k: a[0])
    main.state.profile = True

    out_path = tmp_path / "shots" / "out.png"
    main.observe(
        DummyCtx(),
        screenshot=False,
        path=str(out_path),
        omniparser=True,
        no_dom=False,
    )

    out = json.loads(capsys.readouterr().out)
    assert out["screenshot_path"].endswith("image_som.jpg")
    assert (tmp_path / "shots" / "image.jpg").exists()
    assert (tmp_path / "shots" / "image_som.jpg").exists()


def test_observe_timeout_twice(monkeypatch, tmp_path, capsys):
    session_info = SimpleNamespace(cdp_url="x", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)
    timeout_exc = RuntimeError("timeout")
    cdp_session = _make_cdp_session(
        viewport_value={"width": 1, "height": 1, "device_pixel_ratio": 1.0}
    )
    monkeypatch.setattr(
        main,
        "BrowserSession",
        lambda cdp_url=None: FakeBrowserSession(
            [timeout_exc, timeout_exc], cdp_session
        ),
    )
    with pytest.raises(SystemExit):
        main.observe(
            DummyCtx(),
            screenshot=True,
            path=None,
            omniparser=False,
            no_dom=False,
        )
    assert "timed out twice" in capsys.readouterr().out


def test_observe_non_timeout_error(monkeypatch, tmp_path, capsys):
    session_info = SimpleNamespace(cdp_url="x", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)
    cdp_session = _make_cdp_session(
        viewport_value={"width": 1, "height": 1, "device_pixel_ratio": 1.0}
    )
    monkeypatch.setattr(
        main,
        "BrowserSession",
        lambda cdp_url=None: FakeBrowserSession([RuntimeError("boom")], cdp_session),
    )
    with pytest.raises(SystemExit):
        main.observe(
            DummyCtx(),
            screenshot=False,
            path=None,
            omniparser=False,
            no_dom=False,
        )
    assert "Failed to capture browser state" in capsys.readouterr().out


def test_observe_cdp_access_error(monkeypatch, tmp_path, capsys):
    session_info = SimpleNamespace(cdp_url="x", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)
    summary = _make_summary(screenshot=None)
    monkeypatch.setattr(
        main,
        "BrowserSession",
        lambda cdp_url=None: FakeBrowserSession([summary], RuntimeError("no cdp")),
    )
    with pytest.raises(SystemExit):
        main.observe(
            DummyCtx(),
            screenshot=False,
            path=None,
            omniparser=False,
            no_dom=False,
        )
    assert "Failed to access browser via CDP" in capsys.readouterr().out


def test_observe_cdp_capture_no_data_profile(monkeypatch, tmp_path):
    session_info = SimpleNamespace(cdp_url="x", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)
    summary = _make_summary(screenshot=None)
    cdp_session = _make_cdp_session(
        viewport_value={"width": 1, "height": 1, "device_pixel_ratio": 1.0},
        capture_result={"data": ""},
    )
    monkeypatch.setattr(
        main,
        "BrowserSession",
        lambda cdp_url=None: FakeBrowserSession([summary], cdp_session),
    )
    main.state.profile = True
    main.observe(
        DummyCtx(),
        screenshot=True,
        path=None,
        omniparser=False,
        no_dom=False,
    )


def test_observe_omniparser_missing_screenshot_with_cdp_error(monkeypatch, tmp_path):
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://example")
    monkeypatch.setattr(main, "_should_probe_omniparser", lambda *_: False)
    session_info = SimpleNamespace(cdp_url="x", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)
    summary = _make_summary(screenshot=None)
    cdp_session = _make_cdp_session(
        viewport_value={"width": 1, "height": 1, "device_pixel_ratio": 1.0},
        capture_exc=RuntimeError("snap failed"),
    )
    monkeypatch.setattr(
        main,
        "BrowserSession",
        lambda cdp_url=None: FakeBrowserSession([summary], cdp_session),
    )
    with pytest.raises(SystemExit):
        main.observe(
            DummyCtx(),
            screenshot=False,
            path=None,
            omniparser=True,
            no_dom=False,
        )


def test_observe_no_dom_timeout_retry(monkeypatch, tmp_path):
    session_info = SimpleNamespace(cdp_url="x", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)
    timeout_exc = RuntimeError("timeout")
    cdp_session = _make_cdp_session(
        viewport_value={"width": 1, "height": 1, "device_pixel_ratio": 1.0}
    )
    monkeypatch.setattr(
        main,
        "BrowserSession",
        lambda cdp_url=None: FakeBrowserSession(
            [timeout_exc, timeout_exc], cdp_session
        ),
    )
    with pytest.raises(SystemExit):
        main.observe(
            DummyCtx(),
            screenshot=True,
            path=None,
            omniparser=False,
            no_dom=True,
        )


def test_observe_omniparser_missing_viewport(monkeypatch, tmp_path):
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://example")
    monkeypatch.setattr(main, "_should_probe_omniparser", lambda *_: False)
    session_info = SimpleNamespace(cdp_url="x", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)
    summary = _make_summary(screenshot=base64.b64encode(b"img").decode("ascii"))
    cdp_session = _make_cdp_session(viewport_value=None)
    monkeypatch.setattr(
        main,
        "BrowserSession",
        lambda cdp_url=None: FakeBrowserSession([summary], cdp_session),
    )
    with pytest.raises(SystemExit):
        main.observe(
            DummyCtx(),
            screenshot=False,
            path=None,
            omniparser=True,
            no_dom=False,
        )


def test_observe_cdp_capture_sets_screenshot(monkeypatch, tmp_path):
    session_info = SimpleNamespace(cdp_url="x", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)
    summary = _make_summary(screenshot=None)
    capture_data = base64.b64encode(b"shot").decode("ascii")
    cdp_session = _make_cdp_session(
        viewport_value={"width": 1, "height": 1, "device_pixel_ratio": 1.0},
        capture_result={"data": capture_data},
    )
    monkeypatch.setattr(
        main,
        "BrowserSession",
        lambda cdp_url=None: FakeBrowserSession([summary], cdp_session),
    )
    main.observe(
        DummyCtx(),
        screenshot=True,
        path=None,
        omniparser=False,
        no_dom=False,
    )
    shot_path = tmp_path / "screenshots" / "last_state.png"
    assert shot_path.exists()


def test_observe_omniparser_analyze_empty_elements(monkeypatch, tmp_path):
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://example")
    monkeypatch.setattr(main, "_should_probe_omniparser", lambda *_: False)
    session_info = SimpleNamespace(cdp_url="x", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)
    summary = _make_summary(screenshot=base64.b64encode(b"img").decode("ascii"))
    cdp_session = _make_cdp_session(
        viewport_value={"width": 10, "height": 10, "device_pixel_ratio": 1.0}
    )
    monkeypatch.setattr(
        main,
        "BrowserSession",
        lambda cdp_url=None: FakeBrowserSession([summary], cdp_session),
    )
    fake_client = FakeVisionClient()
    fake_client.analysis = VisualAnalysis(elements=[])
    fake_client.som_image = ""
    monkeypatch.setattr("buse.vision.VisionClient", lambda server_url=None: fake_client)
    monkeypatch.setattr("buse.utils.downscale_image", lambda *a, **k: a[0])
    main.state.profile = True
    with pytest.raises(SystemExit):
        main.observe(
            DummyCtx(),
            screenshot=False,
            path=None,
            omniparser=True,
            no_dom=False,
        )


def test_observe_omniparser_analyze_error(monkeypatch, tmp_path):
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://example")
    monkeypatch.setattr(main, "_should_probe_omniparser", lambda *_: False)
    session_info = SimpleNamespace(cdp_url="x", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)
    summary = _make_summary(screenshot=base64.b64encode(b"img").decode("ascii"))
    cdp_session = _make_cdp_session(
        viewport_value={"width": 10, "height": 10, "device_pixel_ratio": 1.0}
    )
    monkeypatch.setattr(
        main,
        "BrowserSession",
        lambda cdp_url=None: FakeBrowserSession([summary], cdp_session),
    )

    class BoomClient(FakeVisionClient):
        async def analyze(self, base64_image, viewport):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "buse.vision.VisionClient", lambda server_url=None: BoomClient()
    )
    monkeypatch.setattr("buse.utils.downscale_image", lambda *a, **k: a[0])
    with pytest.raises(SystemExit):
        main.observe(
            DummyCtx(),
            screenshot=False,
            path=None,
            omniparser=True,
            no_dom=False,
        )


def test_observe_omniparser_path_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://example")
    monkeypatch.setattr(main, "_should_probe_omniparser", lambda *_: False)
    session_info = SimpleNamespace(cdp_url="x", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)
    summary = _make_summary(screenshot=base64.b64encode(b"img").decode("ascii"))
    cdp_session = _make_cdp_session(
        viewport_value={"width": 10, "height": 10, "device_pixel_ratio": 1.0}
    )
    monkeypatch.setattr(
        main,
        "BrowserSession",
        lambda cdp_url=None: FakeBrowserSession([summary], cdp_session),
    )
    fake_client = FakeVisionClient()
    fake_client.analysis = VisualAnalysis(
        elements=[
            VisualElement(
                index=0,
                type="button",
                content="ok",
                interactivity=True,
                center_x=1.0,
                center_y=2.0,
                bbox=[0.1, 0.2, 0.3, 0.4],
            )
        ]
    )
    fake_client.som_image = ""
    monkeypatch.setattr("buse.vision.VisionClient", lambda server_url=None: fake_client)
    monkeypatch.setattr("buse.utils.downscale_image", lambda *a, **k: a[0])
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    main.observe(
        DummyCtx(),
        screenshot=False,
        path=str(shots_dir),
        omniparser=True,
        no_dom=False,
    )
    assert (shots_dir / "image.jpg").exists()


def test_observe_omniparser_missing_image_data(monkeypatch, tmp_path):
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://example")
    monkeypatch.setattr(main, "_should_probe_omniparser", lambda *_: False)
    session_info = SimpleNamespace(cdp_url="x", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)
    summary = _make_summary(screenshot=base64.b64encode(b"img").decode("ascii"))
    cdp_session = _make_cdp_session(
        viewport_value={"width": 10, "height": 10, "device_pixel_ratio": 1.0}
    )
    monkeypatch.setattr(
        main,
        "BrowserSession",
        lambda cdp_url=None: FakeBrowserSession([summary], cdp_session),
    )
    fake_client = FakeVisionClient()
    fake_client.analysis = VisualAnalysis(
        elements=[
            VisualElement(
                index=0,
                type="button",
                content="ok",
                interactivity=True,
                center_x=1.0,
                center_y=2.0,
                bbox=[0.1, 0.2, 0.3, 0.4],
            )
        ]
    )
    fake_client.som_image = ""
    monkeypatch.setattr("buse.vision.VisionClient", lambda server_url=None: fake_client)
    monkeypatch.setattr("buse.utils.downscale_image", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        main.observe(
            DummyCtx(),
            screenshot=False,
            path=None,
            omniparser=True,
            no_dom=False,
        )


def test_observe_screenshot_path_file(monkeypatch, tmp_path):
    session_info = SimpleNamespace(cdp_url="x", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)
    summary = _make_summary(screenshot=base64.b64encode(b"img").decode("ascii"))
    cdp_session = _make_cdp_session(
        viewport_value={"width": 1, "height": 1, "device_pixel_ratio": 1.0}
    )
    monkeypatch.setattr(
        main,
        "BrowserSession",
        lambda cdp_url=None: FakeBrowserSession([summary], cdp_session),
    )
    out_path = tmp_path / "shots" / "state.png"
    main.observe(
        DummyCtx(),
        screenshot=True,
        path=str(out_path),
        omniparser=False,
        no_dom=False,
    )
    assert out_path.exists()


def test_observe_screenshot_path_dir(monkeypatch, tmp_path):
    session_info = SimpleNamespace(cdp_url="x", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)
    summary = _make_summary(screenshot=base64.b64encode(b"img").decode("ascii"))
    cdp_session = _make_cdp_session(
        viewport_value={"width": 1, "height": 1, "device_pixel_ratio": 1.0}
    )
    monkeypatch.setattr(
        main,
        "BrowserSession",
        lambda cdp_url=None: FakeBrowserSession([summary], cdp_session),
    )
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    main.observe(
        DummyCtx(),
        screenshot=True,
        path=str(shots_dir),
        omniparser=False,
        no_dom=False,
    )
    assert (shots_dir / "last_state.png").exists()
