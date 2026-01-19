import pytest
import json
import base64
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch
from types import SimpleNamespace
import typer
import buse.main as main
from buse.models import ActionResult, VisualAnalysis, VisualElement, ViewportInfo
from buse.session import SessionInfo, SessionManager


@pytest.fixture(autouse=True)
def cleanup_globals(monkeypatch):
    main._browser_sessions.clear()
    main._file_systems.clear()
    main._selector_cache.clear()
    main.state.format = main.OutputFormat.json
    main.state.profile = False

    class GlobalFakeBrowser:
        def __init__(self, *args, **kwargs):
            self.event_bus = MagicMock()
            self.agent_focus_target_id = None
            self.cdp_url = kwargs.get("cdp_url")

        async def start(self):
            pass

        async def stop(self):
            pass

        async def get_browser_state_summary(self, **kwargs):
            return SimpleNamespace(url="u", title="t", screenshot=None, dom_state=None)

        async def get_or_create_cdp_session(self):
            cdp = MagicMock()
            cdp.session_id = "s1"
            cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
                return_value={"result": {"value": {}}}
            )
            cdp.cdp_client.send.Page.captureScreenshot = AsyncMock(return_value={})
            return cdp

        async def get_tabs(self):
            return []

        async def export_storage_state(self, output_path=None):
            return {"cookies": []}

    monkeypatch.setattr(main, "BrowserSession", GlobalFakeBrowser)

    class FakeFS:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(main, "FileSystem", FakeFS)
    yield


def test_is_reserved_key_sequence_plus():
    assert main._is_reserved_key_sequence("Control+L") is True
    assert main._is_reserved_key_sequence("A") is False
    assert main._is_reserved_key_sequence("") is False
    assert main._is_reserved_key_sequence(None) is False


@pytest.mark.asyncio
async def test_get_navigation_timings_not_dict():
    mock_session = MagicMock()
    mock_session.get_or_create_cdp_session = AsyncMock()
    mock_session.get_or_create_cdp_session.return_value.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={"result": {"value": "not_a_dict"}}
    )
    res = await main._get_navigation_timings(mock_session)
    assert res == {}


def test_augment_error_hints_extended():
    res = main._augment_error(
        "click", {"coordinate_x": 10, "coordinate_y": None}, "fail"
    )
    assert "Provide both --x and --y" in res
    res = main._augment_error("search", {}, "Unsupported search engine")
    assert "Use --engine duckduckgo|google|bing" in res
    res = main._augment_error("navigate", {"url": "example.com"}, "site unavailable")
    assert "Include a scheme" in res
    res = main._augment_error(
        "navigate", {"url": "https://example.com"}, "site unavailable"
    )
    assert "Check the URL" in res
    res = main._augment_error("evaluate", {}, "JavaScript execution error")
    assert "Wrap code in (function(){...})()" in res
    res = main._augment_error("extract", {}, "API key")
    assert "Set OPENAI_API_KEY" in res
    res = main._augment_error("wait", {}, "seconds should be integer")
    assert "Use whole seconds" in res
    res = main._augment_error("any", {}, "Instance b1 not found")
    assert "Run `buse <id>` first" in res
    res = main._augment_error("send_keys", {"keys": "hello"}, "fail")
    assert "If you intended to type into a field" in res
    res = main._augment_error("select_dropdown", {}, "Option not found")
    assert "Run dropdown-options" in res
    res = main._augment_error("select_dropdown", {}, "Select element not found")
    assert "Pass the actual <select>" in res
    res = main._augment_error("select_dropdown", {"element_id": "x"}, "some error")
    assert "If a select is wrapped" in res
    res = main._augment_error("scroll", {"pages": 0}, "fail")
    assert "Use a positive --pages value" in res
    res = main._augment_error(
        "click",
        {},
        "Element index 1 not available - page may have changed. Try refreshing browser state.",
    )
    assert "in the current DOM" in res
    res = main._augment_error("click", {}, "element with index 1 does not exist")
    assert "refresh indices" in res


def sync_run_in_new_loop(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_observe_omniparser_no_screenshot_with_error(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://omni")
    monkeypatch.setattr(main, "_should_probe_omniparser", lambda *_: False)
    session_info = SimpleNamespace(cdp_url="http://u", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)

    cdp = MagicMock()
    cdp.cdp_client.send.Page.captureScreenshot = AsyncMock(
        side_effect=Exception("CDP FAIL")
    )
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={
            "result": {
                "value": {"width": 100, "height": 100, "device_pixel_ratio": 1.0}
            }
        }
    )
    cdp.session_id = "s1"

    class LocalFakeBrowser:
        def __init__(self, *args, **kwargs):
            self.event_bus = MagicMock()
            self.agent_focus_target_id = None
            self.cdp_url = kwargs.get("cdp_url")

        async def start(self):
            pass

        async def stop(self):
            pass

        async def get_browser_state_summary(self, **kwargs):
            return SimpleNamespace(url="u", title="t", screenshot=None, dom_state=None)

        async def get_or_create_cdp_session(self):
            return cdp

        async def get_tabs(self):
            return []

    monkeypatch.setattr(main, "BrowserSession", LocalFakeBrowser)
    ctx = MagicMock()
    ctx.obj = {"instance_id": "b1"}

    with patch("buse.main.asyncio.run", side_effect=sync_run_in_new_loop):
        with pytest.raises((SystemExit, typer.Exit)):
            main.observe(
                ctx, screenshot=False, path=None, omniparser=True, no_dom=False
            )

    out = json.loads(capsys.readouterr().out)
    assert out["success"] is False


def test_observe_omniparser_no_viewport(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://omni")
    monkeypatch.setattr(main, "_should_probe_omniparser", lambda *_: False)
    session_info = SimpleNamespace(cdp_url="http://u", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)

    cdp = MagicMock()
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={"result": {"value": None}}
    )
    cdp.session_id = "s1"

    summary = SimpleNamespace(url="u", title="t", screenshot="data", dom_state=None)

    class LocalFakeBrowser:
        def __init__(self, *args, **kwargs):
            self.event_bus = MagicMock()
            self.agent_focus_target_id = None
            self.cdp_url = kwargs.get("cdp_url")

        async def start(self):
            pass

        async def stop(self):
            pass

        async def get_browser_state_summary(self, **kwargs):
            return summary

        async def get_or_create_cdp_session(self):
            return cdp

        async def get_tabs(self):
            return []

    monkeypatch.setattr(main, "BrowserSession", LocalFakeBrowser)
    ctx = MagicMock()
    ctx.obj = {"instance_id": "b1"}

    with patch("buse.main.asyncio.run", side_effect=sync_run_in_new_loop):
        with pytest.raises((SystemExit, typer.Exit)):
            main.observe(
                ctx, screenshot=False, path=None, omniparser=True, no_dom=False
            )

    out = json.loads(capsys.readouterr().out)
    assert out["success"] is False


@pytest.mark.asyncio
async def test_observe_omniparser_som_saving_direct(monkeypatch, tmp_path):
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://omni")
    monkeypatch.setattr(main, "_should_probe_omniparser", lambda *_: False)
    session_info = SimpleNamespace(cdp_url="http://u", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)

    cdp = MagicMock()
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={
            "result": {
                "value": {"width": 100, "height": 100, "device_pixel_ratio": 1.0}
            }
        }
    )
    cdp.session_id = "s1"

    img_data = base64.b64encode(b"img").decode()
    summary = SimpleNamespace(url="u", title="t", screenshot=img_data, dom_state=None)

    class LocalFakeBrowser:
        def __init__(self, *args, **kwargs):
            self.event_bus = MagicMock()
            self.agent_focus_target_id = None
            self.cdp_url = kwargs.get("cdp_url")

        async def start(self):
            pass

        async def stop(self):
            pass

        async def get_browser_state_summary(self, **kwargs):
            return summary

        async def get_or_create_cdp_session(self):
            return cdp

        async def get_tabs(self):
            return []

    monkeypatch.setattr(main, "BrowserSession", LocalFakeBrowser)

    mock_analysis = VisualAnalysis(
        elements=[
            VisualElement(
                index=0,
                type="b",
                content="c",
                interactivity=True,
                center_x=0,
                center_y=0,
                bbox=[0, 0, 1, 1],
            )
        ]
    )

    from buse.vision import VisionClient

    def fake_save(self, data, path):
        with open(path, "wb") as f:
            f.write(base64.b64decode(data))

    monkeypatch.setattr(VisionClient, "save_som_image", fake_save)

    mock_vision = MagicMock(spec=VisionClient)
    mock_vision.analyze = AsyncMock(
        return_value=(mock_analysis, base64.b64encode(b"som").decode())
    )

    mock_vision.save_som_image = lambda data, path: fake_save(None, data, path)

    monkeypatch.setattr("buse.vision.VisionClient", lambda **kwargs: mock_vision)
    monkeypatch.setattr("buse.utils.downscale_image", lambda x, **kwargs: x)

    main.state.profile = True
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    res = await main.get_observation("b1", omniparser=True, path=str(shots_dir))
    assert res["screenshot_path"] is not None
    assert (shots_dir / "image_som.jpg").exists()
    main.state.profile = False


def test_save_state_no_session_outputs_error(monkeypatch, capsys):
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: None)
    ctx = MagicMock()
    ctx.obj = {"instance_id": "missing"}

    with patch("buse.main.asyncio.run", side_effect=sync_run_in_new_loop):
        with pytest.raises((SystemExit, typer.Exit)):
            main.save_state(ctx, "path.json")

    out = json.loads(capsys.readouterr().out)
    assert "Instance missing not found" in out["error"]


def test_run_list_sessions(monkeypatch, capsys):
    monkeypatch.setattr(main.session_manager, "list_sessions", lambda: {"a": 1})
    main._run(["list"])
    out = json.loads(capsys.readouterr().out)
    assert out == {"a": 1}


def test_run_format_arg(monkeypatch):
    monkeypatch.setattr(main.session_manager, "list_sessions", lambda: {})
    main.state.format = main.OutputFormat.json
    args = ["list", "-f", "toon"]
    while "-f" in args:
        idx = args.index("-f")
        main.state.format = main.OutputFormat(args[idx + 1])
        args.pop(idx)
        args.pop(idx)
    assert main.state.format == main.OutputFormat.toon


@pytest.mark.asyncio
async def test_get_observation_omniparser_returned_no_elements_direct(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://omni")
    monkeypatch.setattr(main, "_should_probe_omniparser", lambda *_: False)
    session_info = SimpleNamespace(cdp_url="http://u", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)

    cdp = MagicMock()
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={
            "result": {
                "value": {"width": 100, "height": 100, "device_pixel_ratio": 1.0}
            }
        }
    )
    cdp.session_id = "s1"

    summary = SimpleNamespace(url="u", title="t", screenshot="data", dom_state=None)

    class LocalFakeBrowser:
        def __init__(self, *args, **kwargs):
            self.event_bus = MagicMock()
            self.agent_focus_target_id = None
            self.cdp_url = kwargs.get("cdp_url")

        async def start(self):
            pass

        async def stop(self):
            pass

        async def get_browser_state_summary(self, **kwargs):
            return summary

        async def get_or_create_cdp_session(self):
            return cdp

        async def get_tabs(self):
            return []

    monkeypatch.setattr(main, "BrowserSession", LocalFakeBrowser)

    mock_analysis = VisualAnalysis(elements=[])
    mock_vision = MagicMock()
    mock_vision.analyze = AsyncMock(return_value=(mock_analysis, None))
    monkeypatch.setattr("buse.vision.VisionClient", lambda **kwargs: mock_vision)
    monkeypatch.setattr("buse.utils.downscale_image", lambda x, **kwargs: x)

    with pytest.raises(RuntimeError, match="OmniParser returned no elements"):
        await main.get_observation("b1", omniparser=True)


@pytest.mark.asyncio
async def test_execute_tool_no_dom_deferred(monkeypatch, tmp_path):
    session_info = SimpleNamespace(cdp_url="u", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)
    main.state.profile = True
    with patch("buse.main._try_js_fallback", return_value=(True, "Clicked")):
        try:
            await main.execute_tool(
                "b1", "click", {"element_id": "x"}, needs_selector_map=False
            )
        except main.ResultEmitter.EarlyExit:
            pass
        except Exception as e:
            if e.__class__.__name__ == "EarlyExit":
                pass
            else:
                raise
    main.state.profile = False


def test_run_wait_negative_rewrite(monkeypatch):
    mock_instance_app = MagicMock()
    monkeypatch.setattr(main, "instance_app", mock_instance_app)
    main._run(["b1", "wait", "-1"])
    args, kwargs = mock_instance_app.call_args
    assert kwargs["args"] == ["wait", "--", "-1"]


def test_utils_serialize_flatten_bbox():
    from buse.utils import _serialize

    data = {"bbox": [1, 2, 3, 4]}
    res = _serialize(data)
    assert res["bbox"] == "1,2,3,4"


@pytest.mark.asyncio
async def test_session_manager_stop_missing():
    sm = SessionManager()
    sm.stop_session("not-existent")


@pytest.mark.asyncio
async def test_vision_client_unsupported_image_type():
    from buse.vision import VisionClient

    client = VisionClient(server_url="http://x")
    with patch("httpx.AsyncClient.post", side_effect=Exception("Connection failed")):
        with pytest.raises(RuntimeError, match="Failed to connect to OmniParser"):
            await client.analyze(
                "ZGF0YQ==", ViewportInfo(width=1, height=1, device_pixel_ratio=1.0)
            )


@pytest.mark.asyncio
async def test_run_start_fail(monkeypatch):
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: None)
    monkeypatch.setattr(
        main.session_manager,
        "start_session",
        MagicMock(side_effect=Exception("start fail")),
    )
    with pytest.raises(Exception, match="start fail"):
        await main.run_start("b1")


@pytest.mark.asyncio
async def test_run_stop_fail(monkeypatch):
    monkeypatch.setattr(
        main.session_manager,
        "get_session",
        lambda _: SessionInfo(
            instance_id="b1", cdp_url="http://u", pid=1, user_data_dir="d"
        ),
    )
    monkeypatch.setattr(main, "_stop_cached_browser_session", AsyncMock())
    monkeypatch.setattr(
        main.session_manager,
        "stop_session",
        MagicMock(side_effect=Exception("stop fail")),
    )
    with pytest.raises(Exception, match="stop fail"):
        await main.run_stop("b1")


def test_main_run_help(monkeypatch, capsys):
    main._run(["--help"])
    out = capsys.readouterr().out
    assert "Usage" in out


def test_main_run_no_args(monkeypatch, capsys):
    main._run([])
    out = capsys.readouterr().out
    assert "Usage" in out


@pytest.mark.asyncio
async def test_get_observation_original_image_saving(monkeypatch, tmp_path):
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://omni")
    monkeypatch.setattr(main, "_should_probe_omniparser", lambda *_: False)
    session_info = SimpleNamespace(cdp_url="http://u", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)
    img_data = base64.b64encode(b"img").decode()
    summary = SimpleNamespace(url="u", title="t", screenshot=img_data, dom_state=None)
    cdp = MagicMock()
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={
            "result": {
                "value": {"width": 100, "height": 100, "device_pixel_ratio": 1.0}
            }
        }
    )
    cdp.session_id = "s1"

    class LocalFakeBrowser:
        def __init__(self, *args, **kwargs):
            self.event_bus = MagicMock()
            self.agent_focus_target_id = None
            self.cdp_url = kwargs.get("cdp_url")

        async def start(self):
            pass

        async def stop(self):
            pass

        async def get_browser_state_summary(self, **kwargs):
            return summary

        async def get_or_create_cdp_session(self):
            return cdp

        async def get_tabs(self):
            return []

    monkeypatch.setattr(main, "BrowserSession", LocalFakeBrowser)
    mock_analysis = VisualAnalysis(
        elements=[
            VisualElement(
                index=0,
                type="b",
                content="c",
                interactivity=True,
                center_x=0,
                center_y=0,
                bbox=[0, 0, 1, 1],
            )
        ]
    )
    mock_vision = MagicMock()
    mock_vision.analyze = AsyncMock(return_value=(mock_analysis, None))
    monkeypatch.setattr("buse.vision.VisionClient", lambda **kwargs: mock_vision)
    monkeypatch.setattr("buse.utils.downscale_image", lambda x, **kwargs: x)
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    await main.get_observation("b1", omniparser=True, path=str(shots_dir))
    assert (shots_dir / "image.jpg").exists()


def test_is_reserved_key_sequence_empty():
    assert main._is_reserved_key_sequence("   ") is False


@pytest.mark.asyncio
async def test_get_browser_session_start_fail(monkeypatch):
    class FailBrowser:
        def __init__(self, **kwargs):
            pass

        async def start(self):
            raise Exception("fail")

    monkeypatch.setattr(main, "BrowserSession", FailBrowser)
    with pytest.raises(RuntimeError, match="Failed to start browser session"):
        await main._get_browser_session(
            "b1", SimpleNamespace(cdp_url="u", user_data_dir="d")
        )


@pytest.mark.asyncio
async def test_ensure_selector_map_cached(monkeypatch):
    monkeypatch.setattr(main, "_get_selector_cache_ttl_seconds", lambda: 100.0)
    main._selector_cache["b1"] = time.time()
    mock_session = MagicMock()
    await main._ensure_selector_map(mock_session, "b1")
    mock_session.get_browser_state_summary.assert_not_called()


def test_normalize_tab_id_prefix(monkeypatch):
    tabs = [SimpleNamespace(target_id="12345678")]
    tid, matched = main._normalize_tab_id("1234", tabs)
    assert tid == "5678"
    assert matched is True


@pytest.mark.asyncio
async def test_dispatch_focus_click_no_pos():
    with pytest.raises(ValueError, match="Could not resolve element position"):
        await main._dispatch_focus_click(None, SimpleNamespace(absolute_position=None))


@pytest.mark.asyncio
async def test_focus_element_not_found(monkeypatch):
    mock_session = MagicMock()
    mock_session.get_selector_map = AsyncMock(return_value={})
    monkeypatch.setattr(main, "_ensure_selector_map", AsyncMock())
    res = await main._focus_element(mock_session, "b1", 1, main.Profiler())
    assert res is not None
    assert "not available" in res


@pytest.mark.asyncio
async def test_focus_element_cdp_fail_fallback_fail(monkeypatch):
    mock_session = MagicMock()
    mock_session.get_selector_map = AsyncMock(
        return_value={
            1: SimpleNamespace(
                backend_node_id=1,
                absolute_position=SimpleNamespace(x=0, y=0, width=1, height=1),
            )
        }
    )
    cdp = MagicMock()
    cdp.cdp_client.send.DOM.focus = AsyncMock(side_effect=Exception("focus fail"))

    cdp.cdp_client.send.Input.dispatchMouseEvent = AsyncMock(
        side_effect=Exception("click fail")
    )
    mock_session.get_or_create_cdp_session = AsyncMock(return_value=cdp)
    monkeypatch.setattr(main, "_ensure_selector_map", AsyncMock())
    res = await main._focus_element(mock_session, "b1", 1, main.Profiler())
    assert res is not None
    assert "click fail" in res


@pytest.mark.asyncio
async def test_try_js_fallback_no_result(monkeypatch):
    mock_session = MagicMock()
    cdp = MagicMock()
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(return_value=None)
    mock_session.get_or_create_cdp_session = AsyncMock(return_value=cdp)
    handled, msg = await main._try_js_fallback(
        "click", "#x", "", mock_session, main.Profiler()
    )
    assert handled is False


@pytest.mark.asyncio
async def test_try_dropdown_fallback_no_result(monkeypatch):
    mock_session = MagicMock()
    cdp = MagicMock()
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(return_value=None)
    mock_session.get_or_create_cdp_session = AsyncMock(return_value=cdp)
    handled, err, msg = await main._try_dropdown_fallback(
        "dropdown_options", "#x", "", mock_session, main.Profiler()
    )
    assert handled is True
    assert err is None
    assert msg is not None


@pytest.mark.asyncio
async def test_verify_close_tab_not_found():
    mock_session = MagicMock()
    mock_session.get_tabs = AsyncMock(return_value=[])
    res = await main._verify_close_tab(mock_session, "1234", False, {})
    assert res is not None
    assert "not found" in res


@pytest.mark.asyncio
async def test_verify_close_tab_still_open():
    mock_session = MagicMock()
    mock_session.get_tabs = AsyncMock(return_value=[SimpleNamespace(target_id="1234")])
    res = await main._verify_close_tab(mock_session, "1234", True, {})
    assert res is not None
    assert "still open" in res


@pytest.mark.asyncio
async def test_execute_tool_session_missing(monkeypatch, capsys):
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: None)
    monkeypatch.setattr(main, "_stop_cached_browser_session", AsyncMock())
    with pytest.raises((SystemExit, main.ResultEmitter.EarlyExit, Exception)):
        await main.execute_tool("missing", "click", {})


@pytest.mark.asyncio
async def test_execute_tool_resolve_index_fail(monkeypatch, capsys):
    mock_session = MagicMock()
    cdp = MagicMock()
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={"result": {"value": {"ok": False}}}
    )
    mock_session.get_or_create_cdp_session = AsyncMock(return_value=cdp)
    monkeypatch.setattr(
        main.session_manager,
        "get_session",
        lambda _: SimpleNamespace(cdp_url="http://u", user_data_dir="d"),
    )
    monkeypatch.setattr(
        main,
        "_get_browser_session",
        AsyncMock(return_value=(mock_session, MagicMock())),
    )
    monkeypatch.setattr(main, "_resolve_index", AsyncMock(return_value=None))
    with pytest.raises((SystemExit, main.ResultEmitter.EarlyExit, Exception)):
        await main.execute_tool("b1", "click", {"element_id": "x"})


@pytest.mark.asyncio
async def test_execute_tool_hover_no_index(monkeypatch, capsys):
    mock_session = MagicMock()
    monkeypatch.setattr(
        main.session_manager,
        "get_session",
        lambda _: SimpleNamespace(cdp_url="http://u", user_data_dir="d"),
    )
    monkeypatch.setattr(
        main,
        "_get_browser_session",
        AsyncMock(return_value=(mock_session, MagicMock())),
    )
    with pytest.raises((SystemExit, main.ResultEmitter.EarlyExit, Exception)):
        await main.execute_tool("b1", "hover", {})


@pytest.mark.asyncio
async def test_execute_tool_error_augmented(monkeypatch, capsys):
    mock_session = MagicMock()
    monkeypatch.setattr(
        main.session_manager,
        "get_session",
        lambda _: SimpleNamespace(cdp_url="http://u", user_data_dir="d"),
    )
    monkeypatch.setattr(
        main,
        "_get_browser_session",
        AsyncMock(return_value=(mock_session, MagicMock())),
    )
    controller = MagicMock()
    controller.registry.execute_action = AsyncMock(
        return_value=ActionResult(success=False, action="click", error="some error")
    )
    monkeypatch.setattr(main, "_controllers", {"b1": controller})
    with pytest.raises((SystemExit, main.ResultEmitter.EarlyExit, Exception)):
        await main.execute_tool("b1", "click", {"index": 1})


@pytest.mark.asyncio
async def test_execute_tool_exception_unhandled(monkeypatch, capsys):
    monkeypatch.setattr(
        main.session_manager,
        "get_session",
        lambda _: SimpleNamespace(cdp_url="http://u", user_data_dir="d"),
    )
    monkeypatch.setattr(
        main, "_get_browser_session", MagicMock(side_effect=Exception("boom"))
    )
    with pytest.raises((SystemExit, main.ResultEmitter.EarlyExit, Exception)):
        await main.execute_tool("b1", "click", {})


def test_main_run_start_already_running(monkeypatch, capsys):
    monkeypatch.setattr(
        main.session_manager, "get_session", lambda _: SimpleNamespace()
    )
    main._run(["b1"])
    out = json.loads(capsys.readouterr().out)
    assert out["already_running"] is True


def test_main_run_start_new(monkeypatch, capsys):
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: None)
    monkeypatch.setattr(main.session_manager, "start_session", MagicMock())
    main._run(["b1"])
    out = json.loads(capsys.readouterr().out)
    assert out["already_running"] is False


def test_main_run_cli_exception(monkeypatch, capsys):
    monkeypatch.setattr(main, "instance_app", MagicMock(side_effect=Exception("boom")))
    with pytest.raises(SystemExit):
        main._run(["b1", "observe"])
    out = json.loads(capsys.readouterr().out)
    assert "boom" in out["error"]


@pytest.mark.asyncio
async def test_get_observation_cdp_capture_no_data(monkeypatch):
    mock_session = MagicMock()
    cdp = MagicMock()
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={
            "result": {"value": {"width": 1, "height": 1, "device_pixel_ratio": 1.0}}
        }
    )
    cdp.cdp_client.send.Page.captureScreenshot = AsyncMock(return_value={})
    mock_session.get_or_create_cdp_session = AsyncMock(return_value=cdp)
    monkeypatch.setattr(
        main.session_manager,
        "get_session",
        lambda _: SimpleNamespace(cdp_url="http://u", user_data_dir="d"),
    )
    monkeypatch.setattr(
        main,
        "_get_browser_session",
        AsyncMock(return_value=(mock_session, MagicMock())),
    )
    summary = SimpleNamespace(url="u", title="t", screenshot=None, dom_state=None)
    mock_session.get_browser_state_summary = AsyncMock(return_value=summary)
    mock_session.get_tabs = AsyncMock(return_value=[])
    mock_session.agent_focus_target_id = None
    res = await main.get_observation("b1", screenshot=True)
    assert res["screenshot_path"] is None


@pytest.mark.asyncio
async def test_get_observation_cdp_capture_exception(monkeypatch):
    mock_session = MagicMock()
    cdp = MagicMock()
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={
            "result": {"value": {"width": 1, "height": 1, "device_pixel_ratio": 1.0}}
        }
    )
    cdp.cdp_client.send.Page.captureScreenshot = AsyncMock(
        side_effect=Exception("snap fail")
    )
    mock_session.get_or_create_cdp_session = AsyncMock(return_value=cdp)
    monkeypatch.setattr(
        main.session_manager,
        "get_session",
        lambda _: SimpleNamespace(cdp_url="http://u", user_data_dir="d"),
    )
    monkeypatch.setattr(
        main,
        "_get_browser_session",
        AsyncMock(return_value=(mock_session, MagicMock())),
    )
    summary = SimpleNamespace(url="u", title="t", screenshot=None, dom_state=None)
    mock_session.get_browser_state_summary = AsyncMock(return_value=summary)
    mock_session.get_tabs = AsyncMock(return_value=[])
    mock_session.agent_focus_target_id = None
    with pytest.raises(RuntimeError, match="CDP access failed: snap fail"):
        await main.get_observation("b1", screenshot=True)


@pytest.mark.asyncio
async def test_resolve_index_refresh_retry(monkeypatch):
    mock_session = MagicMock()
    mock_session.get_index_by_id = AsyncMock(side_effect=[None, 7])
    monkeypatch.setattr(main, "_ensure_selector_map", AsyncMock())
    res = await main._resolve_index(mock_session, "b1", "id", None, main.Profiler())
    assert res == 7


@pytest.mark.asyncio
async def test_try_dropdown_fallback_options(monkeypatch):
    mock_session = MagicMock()
    cdp = MagicMock()
    opts = [{"i": 0, "text": "A", "value": "vA", "selected": True}]
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={"result": {"value": {"id": "s", "name": "n", "options": opts}}}
    )
    mock_session.get_or_create_cdp_session = AsyncMock(return_value=cdp)
    handled, err, msg = await main._try_dropdown_fallback(
        "dropdown_options", "#x", "", mock_session, main.Profiler()
    )
    assert msg is not None
    assert "Found select dropdown" in msg
    assert "vA" in msg


@pytest.mark.asyncio
async def test_try_dropdown_fallback_select(monkeypatch):
    mock_session = MagicMock()
    cdp = MagicMock()
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={"result": {"value": {"text": "A", "value": "vA"}}}
    )
    mock_session.get_or_create_cdp_session = AsyncMock(return_value=cdp)
    handled, err, msg = await main._try_dropdown_fallback(
        "select_dropdown", "#x", "A", mock_session, main.Profiler()
    )
    assert msg is not None
    assert "Selected option: A" in msg


def test_coerce_index_error_none():
    assert main._coerce_index_error("other error") is None


def test_observe_omniparser_som_ms_profile(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://omni")
    monkeypatch.setattr(main, "_should_probe_omniparser", lambda *_: False)
    session_info = SimpleNamespace(cdp_url="http://u", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)
    img_data = base64.b64encode(b"img").decode()
    summary = SimpleNamespace(url="u", title="t", screenshot=img_data, dom_state=None)
    cdp = MagicMock()
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={
            "result": {
                "value": {"width": 100, "height": 100, "device_pixel_ratio": 1.0}
            }
        }
    )
    cdp.session_id = "s1"

    class LocalFakeBrowser:
        def __init__(self, *args, **kwargs):
            self.event_bus = MagicMock()
            self.agent_focus_target_id = None
            self.cdp_url = kwargs.get("cdp_url")

        async def start(self):
            pass

        async def stop(self):
            pass

        async def get_browser_state_summary(self, **kwargs):
            return summary

        async def get_or_create_cdp_session(self):
            return cdp

        async def get_tabs(self):
            return []

    monkeypatch.setattr(main, "BrowserSession", LocalFakeBrowser)
    mock_analysis = VisualAnalysis(
        elements=[
            VisualElement(
                index=0,
                type="b",
                content="c",
                interactivity=True,
                center_x=0,
                center_y=0,
                bbox=[0, 0, 1, 1],
            )
        ]
    )
    mock_vision = MagicMock()
    mock_vision.analyze = AsyncMock(
        return_value=(mock_analysis, base64.b64encode(b"som").decode())
    )
    monkeypatch.setattr("buse.vision.VisionClient", lambda **kwargs: mock_vision)
    monkeypatch.setattr("buse.utils.downscale_image", lambda x, **kwargs: x)
    main.state.profile = True
    ctx = MagicMock()
    ctx.obj = {"instance_id": "b1"}
    with patch("buse.main.asyncio.run", side_effect=sync_run_in_new_loop):
        main.observe(
            ctx, screenshot=False, path=str(tmp_path), omniparser=True, no_dom=False
        )
    out = json.loads(capsys.readouterr().out)
    assert "omniparser_som_ms" in out["profile"]
    main.state.profile = False


def test_save_state_success_output(monkeypatch, capsys, tmp_path):
    session_info = SimpleNamespace(cdp_url="http://u", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)

    class LocalFakeBrowser:
        def __init__(self, *args, **kwargs):
            self.cdp_url = kwargs.get("cdp_url")

        async def start(self):
            pass

        async def stop(self):
            pass

        async def export_storage_state(self, output_path):
            return {"cookies": [1]}

    monkeypatch.setattr(main, "BrowserSession", LocalFakeBrowser)
    ctx = MagicMock()
    ctx.obj = {"instance_id": "b1"}
    with patch("buse.main.asyncio.run", side_effect=sync_run_in_new_loop):
        main.save_state(ctx, "path.json")
    out = json.loads(capsys.readouterr().out)
    assert out["success"] is True
    assert out["cookies_count"] == 1


def test_extract_calls_execute_tool(monkeypatch):
    ctx = MagicMock()
    ctx.obj = {"instance_id": "b1"}
    mock_execute = AsyncMock()
    monkeypatch.setattr(main, "execute_tool", mock_execute)
    monkeypatch.setattr("browser_use.llm.openai.chat.ChatOpenAI", MagicMock())
    with patch("buse.main.asyncio.run", side_effect=sync_run_in_new_loop):
        main.extract(ctx, "query")
    mock_execute.assert_called_once()


@pytest.mark.asyncio
async def test_get_observation_omniparser_missing_screenshot(monkeypatch, tmp_path):
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://omni")
    monkeypatch.setattr(main, "_should_probe_omniparser", lambda *_: False)
    session_info = SimpleNamespace(cdp_url="http://u", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)

    browser_session = MagicMock()
    browser_session.agent_focus_target_id = None
    browser_session.get_tabs = AsyncMock(return_value=[])
    browser_session.get_browser_state_summary = AsyncMock(
        return_value=SimpleNamespace(
            url="u", title="t", screenshot=None, dom_state=None
        )
    )
    monkeypatch.setattr(
        main,
        "_get_browser_session",
        AsyncMock(return_value=(browser_session, MagicMock())),
    )

    cdp = MagicMock()
    cdp.session_id = "s1"
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={
            "result": {
                "value": {"width": 100, "height": 100, "device_pixel_ratio": 1.0}
            }
        }
    )
    cdp.cdp_client.send.Page.captureScreenshot = AsyncMock(return_value={})
    browser_session.get_or_create_cdp_session = AsyncMock(return_value=cdp)

    with pytest.raises(RuntimeError, match="Missing screenshot for OmniParser"):
        await main.get_observation("b1", omniparser=True)


@pytest.mark.asyncio
async def test_get_observation_omniparser_missing_viewport(monkeypatch, tmp_path):
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://omni")
    monkeypatch.setattr(main, "_should_probe_omniparser", lambda *_: False)
    session_info = SimpleNamespace(cdp_url="http://u", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)

    img_data = base64.b64encode(b"img").decode()
    browser_session = MagicMock()
    browser_session.agent_focus_target_id = None
    browser_session.get_tabs = AsyncMock(return_value=[])
    browser_session.get_browser_state_summary = AsyncMock(
        return_value=SimpleNamespace(
            url="u", title="t", screenshot=img_data, dom_state=None
        )
    )
    monkeypatch.setattr(
        main,
        "_get_browser_session",
        AsyncMock(return_value=(browser_session, MagicMock())),
    )

    cdp = MagicMock()
    cdp.session_id = "s1"
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={"result": {"value": None}}
    )
    browser_session.get_or_create_cdp_session = AsyncMock(return_value=cdp)

    with pytest.raises(RuntimeError, match="Missing viewport for OmniParser"):
        await main.get_observation("b1", omniparser=True)


@pytest.mark.asyncio
async def test_get_observation_path_file_saved(monkeypatch, tmp_path):
    session_info = SimpleNamespace(cdp_url="http://u", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)

    img_data = base64.b64encode(b"img").decode()
    browser_session = MagicMock()
    browser_session.agent_focus_target_id = None
    browser_session.get_tabs = AsyncMock(return_value=[])
    browser_session.get_browser_state_summary = AsyncMock(
        return_value=SimpleNamespace(
            url="u", title="t", screenshot=img_data, dom_state=None
        )
    )
    monkeypatch.setattr(
        main,
        "_get_browser_session",
        AsyncMock(return_value=(browser_session, MagicMock())),
    )

    cdp = MagicMock()
    cdp.session_id = "s1"
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={"result": {"value": {}}}
    )
    browser_session.get_or_create_cdp_session = AsyncMock(return_value=cdp)

    out_path = tmp_path / "shots" / "state.png"
    res = await main.get_observation("b1", screenshot=True, path=str(out_path))
    assert res["screenshot_path"] == str(out_path)
    assert out_path.exists()


@pytest.mark.asyncio
async def test_get_observation_path_dir_saved(monkeypatch, tmp_path):
    session_info = SimpleNamespace(cdp_url="http://u", user_data_dir=str(tmp_path))
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: session_info)

    img_data = base64.b64encode(b"img").decode()
    browser_session = MagicMock()
    browser_session.agent_focus_target_id = None
    browser_session.get_tabs = AsyncMock(return_value=[])
    browser_session.get_browser_state_summary = AsyncMock(
        return_value=SimpleNamespace(
            url="u", title="t", screenshot=img_data, dom_state=None
        )
    )
    monkeypatch.setattr(
        main,
        "_get_browser_session",
        AsyncMock(return_value=(browser_session, MagicMock())),
    )

    cdp = MagicMock()
    cdp.session_id = "s1"
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={"result": {"value": {}}}
    )
    browser_session.get_or_create_cdp_session = AsyncMock(return_value=cdp)

    out_dir = tmp_path / "shots"
    out_dir.mkdir()
    res = await main.get_observation("b1", screenshot=True, path=str(out_dir))
    expected_path = out_dir / "last_state.png"
    assert res["screenshot_path"] == str(expected_path)
    assert expected_path.exists()


@pytest.mark.asyncio
async def test_get_navigation_timings_success():
    mock_session = MagicMock()
    cdp = MagicMock()
    cdp.session_id = "s1"
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={
            "result": {"value": {"load_event_ms": 10, "ttfb_ms": 5.5, "x": -1}}
        }
    )
    mock_session.get_or_create_cdp_session = AsyncMock(return_value=cdp)
    res = await main._get_navigation_timings(mock_session)
    assert res == {"load_event_ms": 10.0, "ttfb_ms": 5.5}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action_name,text,expected",
    [
        ("input", "hello", "Typed 'hello'"),
        ("click", "", "Clicked element"),
        ("hover", "", "Hovered element"),
    ],
)
async def test_try_js_fallback_success(action_name, text, expected):
    mock_session = MagicMock()
    cdp = MagicMock()
    cdp.session_id = "s1"
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={"result": {"value": {"ok": True}}}
    )
    mock_session.get_or_create_cdp_session = AsyncMock(return_value=cdp)
    handled, msg = await main._try_js_fallback(
        action_name, "#x", text, mock_session, main.Profiler()
    )
    assert handled is True
    assert msg == expected


@pytest.mark.asyncio
async def test_try_dropdown_fallback_options_error():
    mock_session = MagicMock()
    cdp = MagicMock()
    cdp.session_id = "s1"
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={"result": {"value": {"error": "Select element not found"}}}
    )
    mock_session.get_or_create_cdp_session = AsyncMock(return_value=cdp)
    handled, err, msg = await main._try_dropdown_fallback(
        "dropdown_options", "#x", "", mock_session, main.Profiler()
    )
    assert handled is True
    assert err == "Select element not found"
    assert msg is None


@pytest.mark.asyncio
async def test_try_dropdown_fallback_select_error():
    mock_session = MagicMock()
    cdp = MagicMock()
    cdp.session_id = "s1"
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={"result": {"value": {"error": "Option not found"}}}
    )
    mock_session.get_or_create_cdp_session = AsyncMock(return_value=cdp)
    handled, err, msg = await main._try_dropdown_fallback(
        "select_dropdown", "#x", "A", mock_session, main.Profiler()
    )
    assert handled is True
    assert err == "Option not found"
    assert msg is None


def test_augment_error_coords_missing_resolver():
    res = main._augment_error("click", {"coordinate_x": 1, "coordinate_y": 2}, "fail")
    assert "Provide an index or use --id/--class" in res


def test_augment_error_switch_tab_hint():
    res = main._augment_error("switch-tab", {"tab_id": "abc"}, "fail")
    assert "Use the 4-char tab ID" in res


def test_coerce_index_error_missing_index():
    res = main._coerce_index_error("element with index 1 does not exist")
    assert res == "element with index 1 does not exist"
