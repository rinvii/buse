import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from typer.testing import CliRunner
import buse.main as main
from buse.session import SessionInfo
from buse.models import ActionResult, VisualAnalysis, VisualElement
import buse.mcp_server as mcp_server

runner = CliRunner()


class FakeSessionManager:
    def __init__(self, sessions):
        self._sessions = sessions

    def list_sessions(self):
        return self._sessions

    def get_session(self, instance_id):
        return self._sessions.get(instance_id)


class FakeFastMCP:
    def __init__(self, *args, **kwargs):
        self.resources = {}
        self.tools = {}

    def resource(self, uri):
        def decorator(func):
            self.resources[uri] = func
            return func

        return decorator

    def tool(self):
        def decorator(func):
            self.tools[func.__name__] = func
            return func

        return decorator

    def run(self, **kwargs):
        pass

    def streamable_http_app(self):
        return "app"


@pytest.mark.asyncio
async def test_mcp_resources_coverage(monkeypatch):
    monkeypatch.setattr(mcp_server, "FastMCP", FakeFastMCP)
    session = SessionInfo(instance_id="b1", cdp_url="u", pid=1, user_data_dir="d")
    manager = FakeSessionManager({"b1": session})
    server = mcp_server.BuseMCPServer(manager)  # type: ignore
    res = getattr(server.mcp, "resources")["buse://sessions"]()
    assert len(res["instances"]) == 1
    res = getattr(server.mcp, "resources")["buse://session/{instance_id}"]("b1")
    assert res["instance_id"] == "b1"


@pytest.mark.asyncio
async def test_get_observation_coverage_ultimate(monkeypatch):
    session = SessionInfo(instance_id="b1", cdp_url="u", pid=1, user_data_dir="d")
    monkeypatch.setattr(
        main.session_manager, "get_session", MagicMock(return_value=session)
    )
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://omni")
    monkeypatch.setattr(main, "_should_probe_omniparser", MagicMock(return_value=False))
    monkeypatch.setattr(main.state, "profile", True)

    browser_session = MagicMock()
    monkeypatch.setattr(
        main,
        "_get_browser_session",
        AsyncMock(return_value=(browser_session, MagicMock())),
    )
    browser_session.get_tabs = AsyncMock(return_value=[])
    browser_session.get_or_create_cdp_session = AsyncMock(return_value=AsyncMock())

    state_summary = MagicMock()
    state_summary.url = "u"
    state_summary.title = "t"
    state_summary.screenshot = "ZGF0YQ=="
    state_summary.dom_state = None
    browser_session.get_browser_state_summary = AsyncMock(return_value=state_summary)

    cdp = MagicMock()
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={
            "result": {
                "value": {"width": 100, "height": 100, "device_pixel_ratio": 1.0}
            }
        }
    )
    cdp.cdp_client.send.Page.captureScreenshot = AsyncMock(
        return_value={"data": "ZGF0YQ=="}
    )
    browser_session.get_or_create_cdp_session = AsyncMock(return_value=cdp)

    monkeypatch.setattr("builtins.open", MagicMock())
    monkeypatch.setattr(
        "buse.utils.downscale_image", MagicMock(return_value="ZGF0YQ==")
    )
    mock_client = MagicMock()
    mock_client.analyze = AsyncMock(
        return_value=(
            VisualAnalysis(
                elements=[
                    VisualElement(
                        index=1,
                        type="text",
                        content="a",
                        interactivity=False,
                        center_x=0,
                        center_y=0,
                        bbox=[0, 0, 0, 0],
                    )
                ]
            ),
            "ZGF0YQ==",
        )
    )
    monkeypatch.setattr("buse.vision.VisionClient", MagicMock(return_value=mock_client))

    with patch("pathlib.Path.mkdir"), patch("pathlib.Path.is_dir", return_value=False):
        with patch("pathlib.Path.suffix", ".png"):
            await main.get_observation(
                "b1", screenshot=True, path="subdir/file.png", omniparser=True
            )

    state_summary.screenshot = None
    cdp.cdp_client.send.Page.captureScreenshot = AsyncMock(return_value={})
    await main.get_observation("b1", screenshot=True)


@pytest.mark.asyncio
async def test_get_observation_timeout_retry_no_dom(monkeypatch):
    session = SessionInfo(instance_id="b1", cdp_url="u", pid=1, user_data_dir="d")
    monkeypatch.setattr(
        main.session_manager, "get_session", MagicMock(return_value=session)
    )
    browser_session = MagicMock()
    monkeypatch.setattr(
        main,
        "_get_browser_session",
        AsyncMock(return_value=(browser_session, MagicMock())),
    )
    browser_session.get_tabs = AsyncMock(return_value=[])

    mock_cdp = MagicMock()
    mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={"result": {"value": {}}}
    )
    mock_cdp.cdp_client.send.Page.captureScreenshot = AsyncMock(return_value={})
    browser_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp)

    browser_session.get_browser_state_summary = AsyncMock(
        side_effect=Exception("timeout error")
    )
    browser_session.event_bus = MagicMock()
    event = MagicMock()
    event.event_result = AsyncMock(
        side_effect=[
            Exception("timeout error"),
            MagicMock(url="u", title="t", screenshot="data", dom_state=None),
        ]
    )
    browser_session.event_bus.dispatch.return_value = event
    await main.get_observation("b1", no_dom=True, screenshot=True)

    browser_session.get_browser_state_summary = AsyncMock(side_effect=RuntimeError)
    with pytest.raises(RuntimeError):
        await main.get_observation("b1")


@pytest.mark.asyncio
async def test_execute_tool_return_result_hover_none(monkeypatch):
    monkeypatch.setattr(
        main.session_manager,
        "get_session",
        MagicMock(
            return_value=SessionInfo(
                instance_id="b1", cdp_url="u", pid=1, user_data_dir="d"
            )
        ),
    )
    monkeypatch.setattr(
        main,
        "_get_browser_session",
        AsyncMock(return_value=(browser_session := MagicMock(), MagicMock())),
    )
    browser_session.get_tabs = AsyncMock(return_value=[])

    mock_controller = MagicMock()
    mock_controller.registry.execute_action = AsyncMock(
        return_value=ActionResult(success=True, action="a")
    )
    monkeypatch.setattr(main, "_controllers", {"b1": mock_controller})
    res = await main.execute_tool("b1", "click", {"index": 1}, return_result=True)
    assert res.success is True

    mock_emitter = MagicMock()
    monkeypatch.setattr(main, "ResultEmitter", MagicMock(return_value=mock_emitter))
    await main.execute_tool("b1", "hover", {"index": None})
    mock_emitter.fail.assert_called_with("hover", "Hover requires an element index")


@pytest.mark.asyncio
async def test_save_state_stop_exception(monkeypatch):
    monkeypatch.setattr(
        main.session_manager,
        "get_session",
        MagicMock(
            return_value=SessionInfo(
                instance_id="b1", cdp_url="u", pid=1, user_data_dir="d"
            )
        ),
    )
    mock_bs = MagicMock()
    mock_bs.start = AsyncMock()
    mock_bs.stop = AsyncMock(side_effect=Exception)
    mock_bs.export_storage_state = AsyncMock(return_value={"cookies": []})

    with patch("buse.main.BrowserSession", return_value=mock_bs):
        await main.run_save_state("b1", "p")


@pytest.mark.asyncio
async def test_get_browser_session_start_exception(monkeypatch):
    mock_bs = MagicMock()
    mock_bs.start = AsyncMock(side_effect=Exception)
    if "b_fresh_gap" in main._browser_sessions:
        del main._browser_sessions["b_fresh_gap"]
    with patch("buse.main.BrowserSession", return_value=mock_bs):
        with pytest.raises(RuntimeError):
            await main._get_browser_session(
                "b_fresh_gap",
                SessionInfo(
                    instance_id="b_fresh_gap", cdp_url="u", pid=1, user_data_dir="d"
                ),
            )


def test_toon_format_coverage():
    main.state.format = main.OutputFormat.toon
    mock_toon = MagicMock()
    mock_toon.encode.return_value = "toon"
    with patch.dict("sys.modules", {"toon_format": mock_toon}):
        assert main._format_mcp_output({}) == "toon"
    main.state.format = main.OutputFormat.json


@pytest.mark.asyncio
async def test_mcp_tool_handler_model_dump_coverage(monkeypatch):
    monkeypatch.setattr(
        main,
        "execute_tool",
        AsyncMock(return_value=ActionResult(success=True, action="a")),
    )
    handler = main._make_mcp_tool_handler()
    res = await handler("b1", "click")
    assert res["success"] is True


@pytest.mark.asyncio
async def test_stop_cached_browser_session_exception_coverage():
    mock_bs = MagicMock()
    mock_bs.stop = AsyncMock(side_effect=Exception)
    main._browser_sessions["b_cached_gap"] = mock_bs
    await main._stop_cached_browser_session("b_cached_gap")
