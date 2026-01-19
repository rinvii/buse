import json
from types import SimpleNamespace
import time
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
import browser_use
import buse.main as main


class FakeActionResult:
    def __init__(self, error=None, extracted_content="ok"):
        self.error = error
        self.extracted_content = extracted_content


class FakeRegistry:
    def __init__(self):
        self.actions = {}

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

    def __init__(self, *args, **kwargs):
        self.registry = FakeRegistry()

    def set_coordinate_clicking(self, enabled: bool) -> None:
        FakeController.coordinate_enabled = enabled


class FakeElement:
    def __init__(self, x=0, y=0, width=10, height=10):
        self.absolute_position = SimpleNamespace(x=x, y=y, width=width, height=height)
        self.backend_node_id = 1


class FakeCDPSession:
    evaluate_result = {"result": {"value": {}}}

    def __init__(self):
        self.session_id = "fake"
        self.cdp_client = MagicMock()
        self.cdp_client.send.Runtime.evaluate = AsyncMock(
            side_effect=lambda *args, **kwargs: FakeCDPSession.evaluate_result
        )
        self.cdp_client.send.Page.captureScreenshot = AsyncMock(return_value={})
        self.cdp_client.send.Input.dispatchMouseEvent = AsyncMock(return_value={})
        self.cdp_client.send.DOM.focus = AsyncMock(return_value={})
        self.cdp_client.send.DOM.scrollIntoViewIfNeeded = AsyncMock(return_value={})


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

    def __init__(self, *args, **kwargs):
        self.cdp_url = kwargs.get("cdp_url")
        self.agent_focus_target_id = None
        self.cdp_client = MagicMock()

    async def start(self):
        FakeBrowserSession.start_calls += 1
        return None

    async def stop(self):
        return None

    async def get_browser_state_summary(self, include_screenshot=False, cached=False):
        FakeBrowserSession.state_calls += 1
        FakeBrowserSession.refreshed = True
        return MagicMock(url="http://u", title="t", screenshot="data", dom_state=None)

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
        return [
            SimpleNamespace(target_id=tid, title="t", url="u")
            for tid in FakeBrowserSession.open_tabs
        ]

    async def get_target_id_from_tab_id(self, tab_id: str):
        for tid in FakeBrowserSession.open_tabs:
            if tid.endswith(tab_id):
                return tid
        raise ValueError("not found")

    async def _cdp_close_page(self, target_id: str) -> None:
        if target_id in FakeBrowserSession.open_tabs:
            FakeBrowserSession.open_tabs.remove(target_id)

    async def get_current_page_url(self):
        return "http://u"


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

    monkeypatch.delenv("BUSE_SELECTOR_CACHE_TTL", raising=False)
    main._browser_sessions.clear()
    main._file_systems.clear()
    main._selector_cache.clear()
    main._controllers.clear()

    class DummySession:
        cdp_url = "http://localhost:0"
        user_data_dir = "/tmp"

    monkeypatch.setattr(main.session_manager, "get_session", lambda _: DummySession())

    monkeypatch.setattr(browser_use, "Controller", FakeController)
    monkeypatch.setattr(browser_use.browser, "BrowserSession", FakeBrowserSession)

    monkeypatch.setattr(main, "BrowserSession", FakeBrowserSession)

    class FakeFileSystem:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(main, "FileSystem", FakeFileSystem)

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
    FakeBrowserSession.selector_map = {3: FakeElement()}

    mock_cdp = FakeCDPSession()
    mock_cdp.cdp_client.send.DOM.focus = AsyncMock(side_effect=Exception("focus fail"))

    with patch.object(
        FakeBrowserSession, "get_or_create_cdp_session", return_value=mock_cdp
    ):
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
    class NoPosElement:
        def __init__(self):
            self.absolute_position = None
            self.backend_node_id = 1

    FakeBrowserSession.selector_map = {4: NoPosElement()}

    mock_cdp = FakeCDPSession()
    mock_cdp.cdp_client.send.DOM.focus = AsyncMock(side_effect=Exception("focus fail"))

    with patch.object(
        FakeBrowserSession, "get_or_create_cdp_session", return_value=mock_cdp
    ):
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
        def __init__(self, *args, **kwargs):
            self.registry = IndexMessageRegistry()

    with patch("browser_use.Controller", IndexMessageController):
        with pytest.raises(SystemExit):
            await main.execute_tool(
                "b1",
                "click",
                {"index": 1},
                needs_selector_map=False,
            )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is False
    assert captured["error_details"]["stage"] == "execute_tool"
    assert captured["error_details"]["context"]["action"] == "click"
    assert "buse <id> observe" in captured["error"]


@pytest.mark.asyncio
async def test_execute_tool_exception_outputs_error(capsys):
    class ErrorRegistry(FakeRegistry):
        async def execute_action(self, action_name, params, **kwargs):
            raise RuntimeError("boom")

    class ErrorController(FakeController):
        def __init__(self, *args, **kwargs):
            self.registry = ErrorRegistry()

    with patch("browser_use.Controller", ErrorController):
        with pytest.raises(SystemExit):
            await main.execute_tool(
                "b1",
                "click",
                {"index": 1},
                needs_selector_map=False,
            )

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
    with patch(
        "buse.main._try_dropdown_fallback",
        return_value=(
            True,
            None,
            'Found select dropdown\n0: text="A", value="A" (selected)\n1: text="B", value="B"',
        ),
    ):
        await main.execute_tool(
            "b1",
            "dropdown_options",
            {"element_id": "sel"},
            needs_selector_map=True,
        )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is True
    assert "Found select dropdown" in captured["message"]


@pytest.mark.asyncio
async def test_dropdown_options_fallback_by_class(capsys):
    FakeCDPSession.evaluate_result = {
        "result": {"value": {"id": "sel", "name": "sel", "options": []}}
    }
    with patch(
        "buse.main._try_dropdown_fallback",
        return_value=(True, None, "Found select dropdown"),
    ):
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
    with patch(
        "buse.main._try_dropdown_fallback",
        return_value=(True, None, "Selected option: B (value: B)"),
    ):
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

    with patch(
        "buse.main._try_dropdown_fallback",
        return_value=(True, "Select element not found", None),
    ):
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

    with patch(
        "buse.main._try_dropdown_fallback",
        return_value=(True, "Option not found", None),
    ):
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
    with patch("buse.main._try_js_fallback", return_value=(True, "Clicked element")):
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
    with patch("buse.main._try_js_fallback", return_value=(True, "Typed 'hi'")):
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
    with patch("buse.main._try_js_fallback", return_value=(True, "Hovered element")):
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
    main.state.profile = True
    with patch("buse.main._try_js_fallback", return_value=(True, "Clicked element")):
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
    class NoPosElement:
        def __init__(self):
            self.absolute_position = None
            self.backend_node_id = 1

    FakeBrowserSession.selector_map = {5: NoPosElement()}

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
    class NoPosElement:
        def __init__(self):
            self.absolute_position = None
            self.backend_node_id = 1

    FakeBrowserSession.selector_map = {5: NoPosElement()}
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
async def test_close_tab_still_open(capsys):
    class StickyRegistry(FakeRegistry):
        async def execute_action(self, action_name, params, **kwargs):
            FakeController.last_action = action_name
            FakeController.last_params = params
            return FakeActionResult()

    class StickyController(FakeController):
        def __init__(self, *args, **kwargs):
            self.registry = StickyRegistry()

    with patch("browser_use.Controller", StickyController):
        FakeBrowserSession.open_tabs = ["1111222233334444ABCD"]
        with pytest.raises(SystemExit):
            await main.execute_tool(
                "b1",
                "close",
                {"tab_id": "ABCD"},
                needs_selector_map=False,
            )

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is False
    assert captured["error"] == "Tab #ABCD still open"


@pytest.mark.asyncio
async def test_dropdown_options_fallback_error_profiled(capsys):
    main.state.profile = True
    with patch(
        "buse.main._try_dropdown_fallback",
        return_value=(True, "Select element not found", None),
    ):
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
    with patch(
        "buse.main._try_dropdown_fallback",
        return_value=(True, "Option not found", None),
    ):
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
    with patch(
        "buse.main._try_dropdown_fallback",
        return_value=(True, "Option not found", None),
    ):
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
    with patch(
        "buse.main._try_dropdown_fallback",
        return_value=(True, None, "Found select dropdown"),
    ):
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
    with patch(
        "buse.main._try_dropdown_fallback",
        return_value=(True, "Select element not found", None),
    ):
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
    with patch(
        "buse.main._try_dropdown_fallback",
        return_value=(True, None, "Selected option: B (value: B)"),
    ):
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
async def test_execute_tool_profiled_error_exits(capsys):
    class ErrorRegistry(FakeRegistry):
        async def execute_action(self, action_name, params, **kwargs):
            return FakeActionResult(error="bad")

    class ErrorController(FakeController):
        def __init__(self, *args, **kwargs):
            self.registry = ErrorRegistry()

    with patch("browser_use.Controller", ErrorController):
        main.state.profile = True
        with pytest.raises(SystemExit):
            await main.execute_tool(
                "b1",
                "click",
                {"index": 1},
                needs_selector_map=False,
            )
    main.state.profile = False

    captured = json.loads(capsys.readouterr().out)
    assert captured["success"] is False
    assert "profile" in captured


@pytest.mark.asyncio
async def test_execute_tool_profiled_navigate_timings(capsys):
    main.state.profile = True
    with patch(
        "buse.main._get_navigation_timings", return_value={"dom_content_loaded_ms": 10}
    ):
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
    mock_session = MagicMock()
    mock_session.get_or_create_cdp_session = AsyncMock()
    mock_session.get_or_create_cdp_session.return_value.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={"result": {"value": "nope"}}
    )
    timings = await main._get_navigation_timings(mock_session)
    assert timings == {}


@pytest.mark.asyncio
async def test_execute_tool_error_exits(capsys):
    class ErrorRegistry(FakeRegistry):
        async def execute_action(self, action_name, params, **kwargs):
            return FakeActionResult(error="bad")

    class ErrorController(FakeController):
        def __init__(self, *args, **kwargs):
            self.registry = ErrorRegistry()

    with patch("browser_use.Controller", ErrorController):
        with pytest.raises(SystemExit):
            await main.execute_tool(
                "b1",
                "click",
                {"index": 1},
                needs_selector_map=False,
            )

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
