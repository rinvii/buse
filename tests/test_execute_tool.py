import json
from types import SimpleNamespace
import time

import pytest

import buse.main as main


class FakeActionResult:
    def __init__(self, error=None, extracted_content="ok"):
        self.error = error
        self.extracted_content = extracted_content


class FakeRegistry:
    async def execute_action(self, action_name, params, **kwargs):
        FakeController.last_action = action_name
        FakeController.last_params = params
        return FakeActionResult()


class FakeController:
    last_action = None
    last_params = None
    coordinate_enabled = False

    def __init__(self):
        self.registry = FakeRegistry()

    def set_coordinate_clicking(self, enabled: bool) -> None:
        FakeController.coordinate_enabled = enabled


class FakeRuntime:
    async def evaluate(self, params, session_id=None):
        return FakeCDPSession.evaluate_result


class FakeSend:
    Runtime = FakeRuntime()


class FakeCDPClient:
    send = FakeSend()


class FakeCDPSession:
    evaluate_result = {"result": {"value": {}}}
    cdp_client = FakeCDPClient()
    session_id = "fake"


class FakeBrowserSession:
    index_by_id = {}
    index_by_class = {}
    refreshed_index_by_id = {}
    refreshed_index_by_class = {}
    refreshed = False
    start_calls = 0
    state_calls = 0

    def __init__(self, cdp_url=None):
        self.cdp_url = cdp_url

    async def start(self):
        FakeBrowserSession.start_calls += 1
        return None

    async def stop(self):
        return None

    async def get_browser_state_summary(self, include_screenshot=False, cached=False):
        FakeBrowserSession.state_calls += 1
        FakeBrowserSession.refreshed = True
        return None

    async def get_index_by_id(self, element_id):
        if FakeBrowserSession.refreshed and element_id in FakeBrowserSession.refreshed_index_by_id:
            return FakeBrowserSession.refreshed_index_by_id[element_id]
        return FakeBrowserSession.index_by_id.get(element_id)

    async def get_index_by_class(self, class_name):
        if FakeBrowserSession.refreshed and class_name in FakeBrowserSession.refreshed_index_by_class:
            return FakeBrowserSession.refreshed_index_by_class[class_name]
        return FakeBrowserSession.index_by_class.get(class_name)

    async def get_or_create_cdp_session(self):
        return FakeCDPSession()


@pytest.fixture(autouse=True)
def reset_fakes(monkeypatch):
    FakeController.last_action = None
    FakeController.last_params = None
    FakeController.coordinate_enabled = False
    FakeBrowserSession.index_by_id = {}
    FakeBrowserSession.index_by_class = {}
    FakeBrowserSession.refreshed_index_by_id = {}
    FakeBrowserSession.refreshed_index_by_class = {}
    FakeBrowserSession.refreshed = False
    FakeBrowserSession.start_calls = 0
    FakeBrowserSession.state_calls = 0
    FakeCDPSession.evaluate_result = {"result": {"value": {}}}
    main._browser_sessions.clear()
    main._file_systems.clear()
    main._selector_cache.clear()

    import browser_use
    import browser_use.browser

    monkeypatch.setattr(browser_use, "Controller", FakeController)
    monkeypatch.setattr(browser_use.browser, "BrowserSession", FakeBrowserSession)

    class DummySession:
        cdp_url = "http://localhost:0"
        user_data_dir = "/tmp"

    monkeypatch.setattr(main.session_manager, "get_session", lambda _: DummySession())

    yield


@pytest.mark.asyncio
async def test_execute_tool_reuses_session(capsys):
    import os
    os.environ["BUSE_KEEP_SESSION"] = "1"
    await main.execute_tool(
        "b1",
        "click",
        {"index": 1},
        needs_selector_map=False,
    )
    first = main._browser_sessions["b1"]
    await main.execute_tool(
        "b1",
        "click",
        {"index": 2},
        needs_selector_map=False,
    )
    second = main._browser_sessions["b1"]
    assert first is second
    assert FakeBrowserSession.start_calls == 1
    output = capsys.readouterr().out
    decoder = json.JSONDecoder()
    first_obj, idx = decoder.raw_decode(output)
    second_obj, _ = decoder.raw_decode(output, idx=output.find("{", idx))
    assert second_obj["success"] is True
    os.environ.pop("BUSE_KEEP_SESSION", None)


@pytest.mark.asyncio
async def test_execute_tool_profile_output(capsys):
    main.state.profile = True
    await main.execute_tool(
        "b1",
        "click",
        {"index": 1},
        needs_selector_map=False,
    )
    captured = json.loads(capsys.readouterr().out)
    assert "profile" in captured
    assert "total_ms" in captured["profile"]
    main.state.profile = False


@pytest.mark.asyncio
async def test_execute_tool_exception_outputs_error(capsys, monkeypatch):
    class ErrorRegistry(FakeRegistry):
        async def execute_action(self, action_name, params, **kwargs):
            raise RuntimeError("boom")

    class ErrorController(FakeController):
        def __init__(self):
            self.registry = ErrorRegistry()

    import browser_use

    original = browser_use.Controller
    try:
        browser_use.Controller = ErrorController  # type: ignore[assignment]
        await main.execute_tool(
            "b1",
            "click",
            {"index": 1},
            needs_selector_map=False,
        )
    finally:
        browser_use.Controller = original

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is False
@pytest.mark.asyncio
async def test_selector_cache_ttl_skips_refresh():
    session_info = SimpleNamespace(cdp_url="http://localhost:0", user_data_dir="/tmp")
    browser_session, _ = await main._get_browser_session("b1", session_info)
    await main._ensure_selector_map(browser_session, "b1")
    await main._ensure_selector_map(browser_session, "b1")
    assert FakeBrowserSession.state_calls == 1
    await main._ensure_selector_map(browser_session, "b1", force=True)
    assert FakeBrowserSession.state_calls == 2


@pytest.mark.asyncio
async def test_stop_cached_session_clears_cache():
    session_info = SimpleNamespace(cdp_url="http://localhost:0", user_data_dir="/tmp")
    await main._get_browser_session("b1", session_info)
    assert "b1" in main._browser_sessions
    await main._stop_cached_browser_session("b1")
    assert "b1" not in main._browser_sessions


@pytest.mark.asyncio
async def test_navigation_invalidates_selector_cache():
    main._selector_cache["b1"] = time.time()
    await main.execute_tool(
        "b1",
        "navigate",
        {"url": "http://x", "new_tab": False},
        needs_selector_map=False,
    )
    assert "b1" not in main._selector_cache


@pytest.mark.asyncio
async def test_resolve_index_by_id_after_refresh(capsys):
    FakeBrowserSession.refreshed_index_by_id = {"foo": 7}

    await main.execute_tool(
        "b1",
        "input",
        {"element_id": "foo", "text": "hi"},
        needs_selector_map=True,
    )

    assert FakeController.last_params is not None
    assert FakeController.last_params["index"] == 7
    captured = capsys.readouterr().out
    assert json.loads(captured)["success"] is True


@pytest.mark.asyncio
async def test_resolve_index_by_class(capsys):
    FakeBrowserSession.index_by_class = {"cls": 3}
    await main.execute_tool(
        "b1",
        "click",
        {"element_class": "cls"},
        needs_selector_map=True,
    )
    assert FakeController.last_params is not None
    assert FakeController.last_params["index"] == 3
    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is True


@pytest.mark.asyncio
async def test_resolve_index_by_class_after_refresh(capsys):
    FakeBrowserSession.refreshed_index_by_class = {"cls": 9}
    await main.execute_tool(
        "b1",
        "click",
        {"element_class": "cls"},
        needs_selector_map=True,
    )
    assert FakeController.last_params is not None
    assert FakeController.last_params["index"] == 9
    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is True


@pytest.mark.asyncio
async def test_coordinate_click_enables_coordinate_mode(capsys):
    await main.execute_tool(
        "b1",
        "click",
        {"coordinate_x": 10, "coordinate_y": 20},
        needs_selector_map=False,
    )

    assert FakeController.coordinate_enabled is True
    captured = capsys.readouterr().out
    assert json.loads(captured)["success"] is True


@pytest.mark.asyncio
async def test_dropdown_options_fallback_by_id(capsys):
    FakeCDPSession.evaluate_result = {
        "result": {
            "value": {
                "id": "sel",
                "name": "sel",
                "options": [
                    {"i": 0, "text": "A", "value": "A", "selected": True},
                    {"i": 1, "text": "B", "value": "B", "selected": False},
                ],
            }
        }
    }

    await main.execute_tool(
        "b1",
        "dropdown_options",
        {"element_id": "sel"},
        needs_selector_map=True,
    )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is True
    assert "Found select dropdown" in captured["message"]
    assert "A" in captured["message"]
    assert "B" in captured["message"]


@pytest.mark.asyncio
async def test_dropdown_options_fallback_by_class(capsys):
    FakeCDPSession.evaluate_result = {
        "result": {"value": {"id": "sel", "name": "sel", "options": []}}
    }

    await main.execute_tool(
        "b1",
        "dropdown_options",
        {"element_class": "sel"},
        needs_selector_map=True,
    )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is True


@pytest.mark.asyncio
async def test_select_dropdown_fallback_by_id(capsys):
    FakeCDPSession.evaluate_result = {
        "result": {"value": {"text": "B", "value": "B"}}
    }

    await main.execute_tool(
        "b1",
        "select_dropdown",
        {"element_id": "sel", "text": "B"},
        needs_selector_map=True,
    )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is True
    assert captured["message"] == "Selected option: B (value: B)"


@pytest.mark.asyncio
async def test_dropdown_options_fallback_error(capsys):
    FakeCDPSession.evaluate_result = {
        "result": {"value": {"error": "Select element not found"}}
    }

    await main.execute_tool(
        "b1",
        "dropdown_options",
        {"element_id": "sel"},
        needs_selector_map=True,
    )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is False


@pytest.mark.asyncio
async def test_select_dropdown_fallback_option_missing(capsys):
    FakeCDPSession.evaluate_result = {"result": {"value": {"error": "Option not found"}}}

    await main.execute_tool(
        "b1",
        "select_dropdown",
        {"element_id": "sel", "text": "Z"},
        needs_selector_map=True,
    )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is False


@pytest.mark.asyncio
async def test_resolve_index_failure_outputs_error(capsys):
    await main.execute_tool(
        "b1",
        "input",
        {"element_id": "missing", "text": "hi"},
        needs_selector_map=True,
    )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is False


@pytest.mark.asyncio
async def test_execute_tool_no_session(monkeypatch):
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: None)
    with pytest.raises(SystemExit):
        await main.execute_tool("b1", "click", {"index": 1}, needs_selector_map=False)


def test_augment_error_hints():
    msg = main._augment_error("click", {}, "bad")
    assert "Hint" in msg

    msg = main._augment_error("click", {"coordinate_x": 1}, "bad")
    assert "both --x and --y" in msg

    msg = main._augment_error("dropdown_options", {"element_id": "x"}, "bad")
    assert "select" in msg

    msg = main._augment_error("click", {}, "Instance b1 not found.")
    assert "Run `buse <id>`" in msg

    msg = main._augment_error("navigate", {"url": "x"}, "bad")
    assert msg == "bad"

    msg = main._augment_error("select_dropdown", {"element_id": "x"}, "Option not found")
    assert "dropdown-options" in msg

    msg = main._augment_error("dropdown_options", {"element_id": "x"}, "Select element not found")
    assert "select" in msg

    msg = main._augment_error("switch-tab", {"tab_id": "A"}, "bad")
    assert "tab ID" in msg

    msg = main._augment_error("scroll", {"pages": 0}, "bad")
    assert "positive" in msg

    msg = main._augment_error(
        "input",
        {"coordinate_x": 1, "coordinate_y": 2},
        "bad",
    )
    assert "run observe" in msg

    msg = main._augment_error("wait", {}, "seconds integer")
    assert "whole seconds" in msg

    msg = main._augment_error("search", {}, "Unsupported search engine: nope")
    assert "engine" in msg

    msg = main._augment_error(
        "navigate",
        {"url": "example.com"},
        "Navigation failed - site unavailable: example.com",
    )
    assert "scheme" in msg

    msg = main._augment_error(
        "navigate",
        {"url": "https://example.com"},
        "Navigation failed - site unavailable: https://example.com",
    )
    assert "URL" in msg

    msg = main._augment_error("evaluate", {}, "Failed to execute JavaScript: boom")
    assert "Wrap code" in msg

    msg = main._augment_error("extract", {}, "API key missing")
    assert "OPENAI_API_KEY" in msg

    msg = main._augment_error("click", {}, "Element index 2 not available")
    assert "Run observe" in msg

    msg = main._augment_error("input", {}, "Could not resolve element index")
    assert "observe" in msg


@pytest.mark.asyncio
async def test_execute_tool_augments_error(capsys):
    class ErrorRegistry(FakeRegistry):
        async def execute_action(self, action_name, params, **kwargs):
            FakeController.last_action = action_name
            FakeController.last_params = params
            return FakeActionResult(error="bad")

    class ErrorController(FakeController):
        def __init__(self):
            self.registry = ErrorRegistry()

    import browser_use

    original = browser_use.Controller
    try:
        browser_use.Controller = ErrorController  # type: ignore[assignment]
        await main.execute_tool(
            "b1",
            "click",
            {"index": 1},
            needs_selector_map=False,
        )
    finally:
        browser_use.Controller = original

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is False
