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
        FakeController.last_kwargs = kwargs
        if action_name == "close" and params.get("tab_id"):
            tab_id = params["tab_id"]
            for tid in list(FakeBrowserSession.open_tabs):
                if tid.endswith(tab_id):
                    FakeBrowserSession.open_tabs.remove(tid)
                    break
        return FakeActionResult()


class FakeController:
    last_action = None
    last_params = None
    last_kwargs = None
    coordinate_enabled = False

    def __init__(self):
        self.registry = FakeRegistry()

    def set_coordinate_clicking(self, enabled: bool) -> None:
        FakeController.coordinate_enabled = enabled


class FakeRuntime:
    async def evaluate(self, params, session_id=None):
        return FakeCDPSession.evaluate_result


class FakeInput:
    async def dispatchMouseEvent(self, params, session_id=None):
        return {}


class FakeDOM:
    should_fail = False

    async def focus(self, params, session_id=None):
        if FakeDOM.should_fail:
            raise RuntimeError("focus failed")
        return {}


class FakeSend:
    Runtime = FakeRuntime()
    Input = FakeInput()
    DOM = FakeDOM()


class FakeCDPClient:
    send = FakeSend()


class FakeCDPSession:
    evaluate_result = {"result": {"value": {}}}
    cdp_client = FakeCDPClient()
    session_id = "fake"


class FakeBounds:
    def __init__(self, x=0.0, y=0.0, width=10.0, height=10.0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


_UNSET = object()


class FakeElement:
    def __init__(self, bounds=_UNSET):
        if bounds is _UNSET:
            bounds = FakeBounds()
        self.absolute_position = bounds
        self.backend_node_id = 1


class FakeBrowserSession:
    index_by_id = {}
    index_by_class = {}
    refreshed_index_by_id = {}
    refreshed_index_by_class = {}
    refreshed = False
    start_calls = 0
    state_calls = 0
    selector_map = {}
    open_tabs = []

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
        if (
            FakeBrowserSession.refreshed
            and element_id in FakeBrowserSession.refreshed_index_by_id
        ):
            return FakeBrowserSession.refreshed_index_by_id[element_id]
        return FakeBrowserSession.index_by_id.get(element_id)

    async def get_index_by_class(self, class_name):
        if (
            FakeBrowserSession.refreshed
            and class_name in FakeBrowserSession.refreshed_index_by_class
        ):
            return FakeBrowserSession.refreshed_index_by_class[class_name]
        return FakeBrowserSession.index_by_class.get(class_name)

    async def get_or_create_cdp_session(self):
        return FakeCDPSession()

    async def get_selector_map(self):
        return FakeBrowserSession.selector_map

    async def get_tabs(self):
        return [SimpleNamespace(target_id=tid) for tid in FakeBrowserSession.open_tabs]

    async def get_target_id_from_tab_id(self, tab_id: str):
        for tid in FakeBrowserSession.open_tabs:
            if tid.endswith(tab_id):
                return tid
        raise ValueError("not found")

    async def _cdp_close_page(self, target_id: str) -> None:
        if target_id in FakeBrowserSession.open_tabs:
            FakeBrowserSession.open_tabs.remove(target_id)


@pytest.fixture(autouse=True)
def reset_fakes(monkeypatch):
    FakeController.last_action = None
    FakeController.last_params = None
    FakeController.last_kwargs = None
    FakeController.coordinate_enabled = False
    FakeBrowserSession.index_by_id = {}
    FakeBrowserSession.index_by_class = {}
    FakeBrowserSession.refreshed_index_by_id = {}
    FakeBrowserSession.refreshed_index_by_class = {}
    FakeBrowserSession.refreshed = False
    FakeBrowserSession.start_calls = 0
    FakeBrowserSession.state_calls = 0
    FakeBrowserSession.selector_map = {}
    FakeBrowserSession.open_tabs = []
    FakeCDPSession.evaluate_result = {"result": {"value": {}}}
    FakeDOM.should_fail = False
    monkeypatch.delenv("BUSE_SELECTOR_CACHE_TTL", raising=False)
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
async def test_execute_tool_upload_file_passes_available_paths(capsys):
    await main.execute_tool(
        "b1",
        "upload_file",
        {"index": 1, "path": "/tmp/file.txt"},
        needs_selector_map=False,
    )
    assert FakeController.last_kwargs is not None
    assert FakeController.last_kwargs["available_file_paths"] == ["/tmp/file.txt"]
    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is True


@pytest.mark.asyncio
async def test_send_keys_focus_by_index(capsys):
    FakeBrowserSession.selector_map = {2: FakeElement()}
    await main.execute_tool(
        "b1",
        "send_keys",
        {"index": 2, "keys": "Enter"},
        needs_selector_map=True,
    )
    assert FakeController.last_action == "send_keys"
    assert FakeController.last_params == {"keys": "Enter"}
    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is True


@pytest.mark.asyncio
async def test_send_keys_focus_fallback_click(capsys):
    FakeDOM.should_fail = True
    FakeBrowserSession.selector_map = {3: FakeElement()}
    await main.execute_tool(
        "b1",
        "send_keys",
        {"index": 3, "keys": "Hello"},
        needs_selector_map=True,
    )
    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is True


@pytest.mark.asyncio
async def test_send_keys_focus_missing_index(capsys):
    FakeBrowserSession.selector_map = {}
    with pytest.raises(SystemExit):
        await main.execute_tool(
            "b1",
            "send_keys",
            {"index": 9, "keys": "A"},
            needs_selector_map=True,
        )
    captured = json.loads(capsys.readouterr().out)
    assert "Element index 9 not available" in captured["error"]


@pytest.mark.asyncio
async def test_send_keys_focus_no_position(capsys):
    FakeDOM.should_fail = True
    FakeBrowserSession.selector_map = {4: FakeElement(bounds=None)}
    with pytest.raises(SystemExit):
        await main.execute_tool(
            "b1",
            "send_keys",
            {"index": 4, "keys": "A"},
            needs_selector_map=True,
        )
    captured = json.loads(capsys.readouterr().out)
    assert "Could not resolve element position for focus" in captured["error"]


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
async def test_execute_tool_coerces_index_message_to_error(capsys):
    class IndexMessageRegistry(FakeRegistry):
        async def execute_action(self, action_name, params, **kwargs):
            return FakeActionResult(
                extracted_content=(
                    "Element index 1 not available - page may have changed. "
                    "Try refreshing browser state."
                )
            )

    class IndexMessageController(FakeController):
        def __init__(self):
            self.registry = IndexMessageRegistry()

    import browser_use

    original = browser_use.Controller
    try:
        browser_use.Controller = IndexMessageController  # type: ignore[assignment]
        with pytest.raises(SystemExit):
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
    assert captured["error_details"]["stage"] == "execute_tool"
    assert captured["error_details"]["context"]["action"] == "click"
    assert "buse <id> observe" in captured["error"]


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
        with pytest.raises(SystemExit):
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
async def test_selector_cache_ttl_skips_refresh(monkeypatch):
    monkeypatch.setenv("BUSE_SELECTOR_CACHE_TTL", "2")
    session_info = SimpleNamespace(cdp_url="http://localhost:0", user_data_dir="/tmp")
    browser_session, _ = await main._get_browser_session("b1", session_info)
    await main._ensure_selector_map(browser_session, "b1")
    await main._ensure_selector_map(browser_session, "b1")
    assert FakeBrowserSession.state_calls == 1
    await main._ensure_selector_map(browser_session, "b1", force=True)
    assert FakeBrowserSession.state_calls == 2


def test_selector_cache_ttl_invalid_value(monkeypatch):
    monkeypatch.setenv("BUSE_SELECTOR_CACHE_TTL", "nope")
    assert main._get_selector_cache_ttl_seconds() == 0.0


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
    FakeCDPSession.evaluate_result = {"result": {"value": {"text": "B", "value": "B"}}}

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

    with pytest.raises(SystemExit):
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
    FakeCDPSession.evaluate_result = {
        "result": {"value": {"error": "Option not found"}}
    }

    with pytest.raises(SystemExit):
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
    with pytest.raises(SystemExit):
        await main.execute_tool(
            "b1",
            "input",
            {"element_id": "missing", "text": "hi"},
            needs_selector_map=True,
        )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is False


@pytest.mark.asyncio
async def test_resolve_index_failure_profiled(capsys):
    main.state.profile = True
    with pytest.raises(SystemExit):
        await main.execute_tool(
            "b1",
            "input",
            {"element_id": "missing", "text": "hi"},
            needs_selector_map=True,
        )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is False
    assert "profile" in captured
    main.state.profile = False


@pytest.mark.asyncio
async def test_js_fallback_click(capsys):
    FakeCDPSession.evaluate_result = {"result": {"value": {"ok": True}}}

    await main.execute_tool(
        "b1",
        "click",
        {"element_id": "x"},
        needs_selector_map=True,
    )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is True
    assert captured["message"] == "Clicked element"


@pytest.mark.asyncio
async def test_js_fallback_input(capsys):
    FakeCDPSession.evaluate_result = {"result": {"value": {"ok": True}}}

    await main.execute_tool(
        "b1",
        "input",
        {"element_class": "cls", "text": "hi"},
        needs_selector_map=True,
    )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is True
    assert captured["message"] == "Typed 'hi'"


@pytest.mark.asyncio
async def test_js_fallback_hover(capsys):
    FakeCDPSession.evaluate_result = {"result": {"value": {"ok": True}}}

    await main.execute_tool(
        "b1",
        "hover",
        {"element_id": "hover"},
        needs_selector_map=True,
    )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is True
    assert captured["message"] == "Hovered element"


@pytest.mark.asyncio
async def test_js_fallback_profiled(capsys):
    FakeCDPSession.evaluate_result = {"result": {"value": {"ok": True}}}
    main.state.profile = True

    await main.execute_tool(
        "b1",
        "click",
        {"element_id": "x"},
        needs_selector_map=True,
    )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is True
    assert "profile" in captured
    main.state.profile = False


@pytest.mark.asyncio
async def test_execute_tool_hover_index(capsys):
    FakeBrowserSession.selector_map = {5: FakeElement()}

    await main.execute_tool(
        "b1",
        "hover",
        {"index": 5},
        needs_selector_map=True,
    )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is True
    assert captured["message"] == "Hovered element"


@pytest.mark.asyncio
async def test_execute_tool_hover_index_profiled(capsys):
    FakeBrowserSession.selector_map = {5: FakeElement()}
    main.state.profile = True

    await main.execute_tool(
        "b1",
        "hover",
        {"index": 5},
        needs_selector_map=True,
    )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is True
    assert "profile" in captured
    main.state.profile = False


@pytest.mark.asyncio
async def test_execute_tool_hover_missing_index(capsys):
    with pytest.raises(SystemExit):
        await main.execute_tool(
            "b1",
            "hover",
            {},
            needs_selector_map=False,
        )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is False


@pytest.mark.asyncio
async def test_execute_tool_hover_missing_index_profiled(capsys):
    main.state.profile = True
    with pytest.raises(SystemExit):
        await main.execute_tool(
            "b1",
            "hover",
            {},
            needs_selector_map=False,
        )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is False
    assert "profile" in captured
    main.state.profile = False


@pytest.mark.asyncio
async def test_execute_tool_hover_missing_position(capsys):
    FakeBrowserSession.selector_map = {5: FakeElement(bounds=None)}

    with pytest.raises(SystemExit):
        await main.execute_tool(
            "b1",
            "hover",
            {"index": 5},
            needs_selector_map=True,
        )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is False


@pytest.mark.asyncio
async def test_execute_tool_hover_missing_position_profiled(capsys):
    FakeBrowserSession.selector_map = {5: FakeElement(bounds=None)}
    main.state.profile = True

    with pytest.raises(SystemExit):
        await main.execute_tool(
            "b1",
            "hover",
            {"index": 5},
            needs_selector_map=True,
        )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is False
    assert "profile" in captured
    main.state.profile = False


@pytest.mark.asyncio
async def test_close_tab_prefix_normalizes(capsys):
    FakeBrowserSession.open_tabs = ["92320F7260631C592324DEBC601EA405"]

    await main.execute_tool(
        "b1",
        "close",
        {"tab_id": "9232"},
        needs_selector_map=False,
    )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is True
    assert captured["message"] == "Closed tab #A405"
    assert FakeController.last_params is not None
    assert FakeController.last_params["tab_id"] == "A405"
    assert FakeBrowserSession.open_tabs == []


@pytest.mark.asyncio
async def test_close_tab_not_found(capsys):
    FakeBrowserSession.open_tabs = []

    with pytest.raises(SystemExit):
        await main.execute_tool(
            "b1",
            "close",
            {"tab_id": "ABCD"},
            needs_selector_map=False,
        )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is False
    assert captured["error"] == "Tab #ABCD not found"


@pytest.mark.asyncio
async def test_close_tab_still_open(capsys, monkeypatch):
    class StickyRegistry(FakeRegistry):
        async def execute_action(self, action_name, params, **kwargs):
            FakeController.last_action = action_name
            FakeController.last_params = params
            return FakeActionResult()

    class StickyController(FakeController):
        def __init__(self):
            self.registry = StickyRegistry()

    import browser_use

    original = browser_use.Controller
    try:
        browser_use.Controller = StickyController  # type: ignore[assignment]
        FakeBrowserSession.open_tabs = ["1111222233334444ABCD"]
        with pytest.raises(SystemExit):
            await main.execute_tool(
                "b1",
                "close",
                {"tab_id": "ABCD"},
                needs_selector_map=False,
            )
    finally:
        browser_use.Controller = original

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is False
    assert captured["error"] == "Tab #ABCD still open"


@pytest.mark.asyncio
async def test_dropdown_options_fallback_error_profiled(capsys):
    main.state.profile = True
    FakeCDPSession.evaluate_result = {
        "result": {"value": {"error": "Select element not found"}}
    }

    with pytest.raises(SystemExit):
        await main.execute_tool(
            "b1",
            "dropdown_options",
            {"element_id": "sel"},
            needs_selector_map=True,
        )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is False
    assert "profile" in captured
    main.state.profile = False


@pytest.mark.asyncio
async def test_select_dropdown_fallback_error_exits(capsys):
    FakeCDPSession.evaluate_result = {
        "result": {"value": {"error": "Option not found"}}
    }

    with pytest.raises(SystemExit):
        await main.execute_tool(
            "b1",
            "select_dropdown",
            {"element_id": "sel", "text": "Z"},
            needs_selector_map=True,
        )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is False


@pytest.mark.asyncio
async def test_select_dropdown_fallback_error_profiled(capsys):
    main.state.profile = True
    FakeCDPSession.evaluate_result = {
        "result": {"value": {"error": "Option not found"}}
    }

    with pytest.raises(SystemExit):
        await main.execute_tool(
            "b1",
            "select_dropdown",
            {"element_id": "sel", "text": "Z"},
            needs_selector_map=True,
        )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is False
    assert "profile" in captured
    main.state.profile = False


@pytest.mark.asyncio
async def test_dropdown_options_fallback_profiled_success(capsys):
    main.state.profile = True
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
    assert "profile" in captured
    main.state.profile = False


@pytest.mark.asyncio
async def test_dropdown_options_fallback_error_exits(capsys):
    FakeCDPSession.evaluate_result = {
        "result": {"value": {"error": "Select element not found"}}
    }

    with pytest.raises(SystemExit):
        await main.execute_tool(
            "b1",
            "dropdown_options",
            {"element_id": "sel"},
            needs_selector_map=True,
        )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is False


@pytest.mark.asyncio
async def test_select_dropdown_fallback_profiled_success(capsys):
    main.state.profile = True
    FakeCDPSession.evaluate_result = {"result": {"value": {"text": "B", "value": "B"}}}

    await main.execute_tool(
        "b1",
        "select_dropdown",
        {"element_id": "sel", "text": "B"},
        needs_selector_map=True,
    )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is True
    assert "profile" in captured
    main.state.profile = False


@pytest.mark.asyncio
async def test_execute_tool_profiled_error_exits(capsys, monkeypatch):
    class ErrorRegistry(FakeRegistry):
        async def execute_action(self, action_name, params, **kwargs):
            return FakeActionResult(error="bad")

    class ErrorController(FakeController):
        def __init__(self):
            self.registry = ErrorRegistry()

    import browser_use

    original = browser_use.Controller
    try:
        browser_use.Controller = ErrorController  # type: ignore[assignment]
        main.state.profile = True
        with pytest.raises(SystemExit):
            await main.execute_tool(
                "b1",
                "click",
                {"index": 1},
                needs_selector_map=False,
            )
    finally:
        browser_use.Controller = original
        main.state.profile = False

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is False
    assert "profile" in captured


@pytest.mark.asyncio
async def test_execute_tool_profiled_navigate_timings(capsys):
    FakeCDPSession.evaluate_result = {
        "result": {
            "value": {
                "dom_content_loaded_ms": 10,
                "load_event_ms": 20,
                "response_end_ms": 30,
                "ttfb_ms": 5,
            }
        }
    }
    main.state.profile = True
    await main.execute_tool(
        "b1",
        "navigate",
        {"url": "http://example.com", "new_tab": False},
        needs_selector_map=False,
    )
    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is True
    assert "nav_dom_content_loaded_ms" in captured["profile"]
    main.state.profile = False


@pytest.mark.asyncio
async def test_execute_tool_profiled_navigate_timings_error(capsys, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("fail")

    main.state.profile = True
    monkeypatch.setattr(main, "_get_navigation_timings", boom)
    await main.execute_tool(
        "b1",
        "navigate",
        {"url": "http://example.com", "new_tab": False},
        needs_selector_map=False,
    )
    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is True
    assert "nav_dom_content_loaded_ms" not in captured["profile"]
    main.state.profile = False


@pytest.mark.asyncio
async def test_get_navigation_timings_non_dict():
    FakeCDPSession.evaluate_result = {"result": {"value": "nope"}}
    timings = await main._get_navigation_timings(FakeBrowserSession())
    assert timings == {}


@pytest.mark.asyncio
async def test_execute_tool_error_exits(capsys, monkeypatch):
    class ErrorRegistry(FakeRegistry):
        async def execute_action(self, action_name, params, **kwargs):
            return FakeActionResult(error="bad")

    class ErrorController(FakeController):
        def __init__(self):
            self.registry = ErrorRegistry()

    import browser_use

    original = browser_use.Controller
    try:
        browser_use.Controller = ErrorController  # type: ignore[assignment]
        with pytest.raises(SystemExit):
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
async def test_try_dropdown_fallback_no_selector():
    handled, error_msg, success_msg = await main._try_dropdown_fallback(
        "dropdown_options",
        None,
        "",
        FakeBrowserSession(),
        main.Profiler(),
    )
    assert handled is False
    assert error_msg is None
    assert success_msg is None


@pytest.mark.asyncio
async def test_try_dropdown_fallback_unsupported_action():
    handled, error_msg, success_msg = await main._try_dropdown_fallback(
        "click",
        "#sel",
        "",
        FakeBrowserSession(),
        main.Profiler(),
    )
    assert handled is False
    assert error_msg is None
    assert success_msg is None


@pytest.mark.asyncio
async def test_execute_tool_no_session(monkeypatch):
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: None)
    with pytest.raises(SystemExit):
        await main.execute_tool("b1", "click", {"index": 1}, needs_selector_map=False)


def test_augment_error_hints():
    msg = main._augment_error("click", {}, "bad")
    assert "Provide an index" in msg

    msg = main._augment_error("click", {"coordinate_x": 1}, "bad")
    assert "both --x and --y" in msg

    msg = main._augment_error("dropdown_options", {"element_id": "x"}, "bad")
    assert "select" in msg

    msg = main._augment_error("click", {}, "Instance b1 not found.")
    assert "Run `buse <id>`" in msg

    msg = main._augment_error("navigate", {"url": "x"}, "bad")
    assert msg == "bad"

    msg = main._augment_error(
        "select_dropdown", {"element_id": "x"}, "Option not found"
    )
    assert "dropdown-options" in msg

    msg = main._augment_error(
        "dropdown_options", {"element_id": "x"}, "Select element not found"
    )
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
    assert "buse <id> observe" in msg

    msg = main._augment_error("click", {}, "Element with index 3 does not exist")
    assert "buse <id> observe" in msg

    msg = main._augment_error(
        "click",
        {},
        "Element index 4 not available - page may have changed. Try refreshing browser state.",
    )
    assert "Try refreshing browser state" not in msg

    msg = main._augment_error("input", {}, "Could not resolve element index")
    assert "observe" in msg

    msg = main._augment_error("send_keys", {"keys": "Hello"}, "send failed")
    assert "--index/--id/--class" in msg

    msg = main._augment_error("send_keys", {"keys": "Enter"}, "send failed")
    assert "--index/--id/--class" not in msg

    msg = main._augment_error("send_keys", {"keys": "Control+L"}, "send failed")
    assert "--index/--id/--class" not in msg

    msg = main._augment_error("send_keys", {"keys": "Hello", "index": 1}, "send failed")
    assert "--index/--id/--class" not in msg


def test_coerce_index_error_variants():
    assert main._coerce_index_error(None) is None
    msg = "Element with index 9 does not exist"
    assert main._coerce_index_error(msg) == msg


def test_is_reserved_key_sequence():
    assert main._is_reserved_key_sequence(None) is False
    assert main._is_reserved_key_sequence("Enter") is True
    assert main._is_reserved_key_sequence("Control+L") is True
    assert main._is_reserved_key_sequence("space") is True
    assert main._is_reserved_key_sequence("F13") is True
    assert main._is_reserved_key_sequence("Hello") is False
    assert main._is_reserved_key_sequence(" ") is False


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
        with pytest.raises(SystemExit):
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
