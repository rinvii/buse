import pytest

import buse.mcp_server as mcp_server
from buse.session import SessionInfo
from unittest.mock import MagicMock
import buse.main as main


class FakeSessionManager:
    def __init__(self, sessions: dict[str, SessionInfo]):
        self._sessions = sessions

    def list_sessions(self) -> dict[str, SessionInfo]:
        return self._sessions

    def get_session(self, instance_id: str) -> SessionInfo | None:
        return self._sessions.get(instance_id)


class FakeFastMCP:
    def __init__(self, *args, **kwargs):
        self.resources = {}
        self.tools = {}

    def resource(self, uri: str):
        def decorator(func):
            self.resources[uri] = func
            return func

        return decorator

    def tool(self):
        def decorator(func):
            self.tools[func.__name__] = func
            return func

        return decorator

    def run(self, transport: str = "stdio"):
        self.transport = transport

    def streamable_http_app(self):
        return "streamable"

    def sse_app(self):
        return "sse"


@pytest.fixture(autouse=True)
def _patch_fastmcp(monkeypatch):
    monkeypatch.setattr(mcp_server, "FastMCP", FakeFastMCP)


def test_session_summary_to_dict():
    summary = mcp_server.SessionSummary(
        instance_id="b1",
        cdp_url="http://localhost:9222",
        user_data_dir="/tmp/buse/b1",
    )
    assert summary.to_dict() == {
        "instance_id": "b1",
        "cdp_url": "http://localhost:9222",
        "user_data_dir": "/tmp/buse/b1",
    }


def test_serialize_and_load_session():
    session = SessionInfo(
        instance_id="b1",
        cdp_url="http://localhost:9222",
        pid=123,
        user_data_dir="/tmp/buse/b1",
    )
    manager = FakeSessionManager({"b1": session})
    server = mcp_server.BuseMCPServer(manager)  # type: ignore

    serialized = server._serialize_session(session)
    assert serialized["instance_id"] == "b1"
    assert serialized["cdp_url"] == "http://localhost:9222"
    assert serialized["user_data_dir"] == "/tmp/buse/b1"

    loaded = server._load_session("b1")
    assert loaded is session


def test_load_missing_session_raises():
    manager = FakeSessionManager({})
    server = mcp_server.BuseMCPServer(manager)  # type: ignore
    with pytest.raises(ValueError, match="Instance b2 not found"):
        server._load_session("b2")


@pytest.mark.asyncio
async def test_mcp_tool_wiring_navigate():
    manager = FakeSessionManager({})
    calls = []

    async def handler(instance_id: str, action_name: str, **kwargs):
        calls.append((instance_id, action_name, kwargs))
        return {"ok": True}

    server = mcp_server.BuseMCPServer(manager, tool_handler=handler)  # type: ignore
    await getattr(server.mcp, "tools")["navigate"](
        "b1", "https://example.com", new_tab=True
    )
    assert calls == [
        ("b1", "navigate", {"url": "https://example.com", "new_tab": True})
    ]


@pytest.mark.asyncio
async def test_mcp_tool_wiring_refresh():
    manager = FakeSessionManager({})
    calls = []

    async def handler(instance_id: str, action_name: str, **kwargs):
        calls.append((instance_id, action_name, kwargs))
        return {"ok": True}

    server = mcp_server.BuseMCPServer(manager, tool_handler=handler)  # type: ignore
    await getattr(server.mcp, "tools")["refresh"]("b1")
    assert calls == [
        (
            "b1",
            "evaluate",
            {"code": "window.location.reload()", "action_label": "refresh"},
        )
    ]


@pytest.mark.asyncio
async def test_mcp_click_coerces_empty_element_fields():
    manager = FakeSessionManager({})
    calls = []

    async def handler(instance_id: str, action_name: str, **kwargs):
        calls.append((instance_id, action_name, kwargs))
        return {"ok": True}

    server = mcp_server.BuseMCPServer(manager, tool_handler=handler)  # type: ignore
    await getattr(server.mcp, "tools")["click"](
        "b1",
        x=15,
        y=15,
        element_id={},
        element_class={},
    )
    assert calls == [
        (
            "b1",
            "click",
            {
                "index": None,
                "element_id": None,
                "element_class": None,
                "coordinate_x": 15,
                "coordinate_y": 15,
            },
        )
    ]


@pytest.mark.asyncio
async def test_mcp_click_requires_coordinate_pair():
    manager = FakeSessionManager({})

    async def handler(instance_id: str, action_name: str, **kwargs):
        return {"ok": True}

    server = mcp_server.BuseMCPServer(manager, tool_handler=handler)  # type: ignore
    with pytest.raises(ValueError, match="Provide both x and y"):
        await getattr(server.mcp, "tools")["click"]("b1", x=15)


@pytest.mark.asyncio
async def test_mcp_observe_wiring_omniparser():
    manager = FakeSessionManager({})
    calls = []

    async def observer(instance_id: str, **kwargs):
        calls.append((instance_id, kwargs))
        return {"ok": True}

    server = mcp_server.BuseMCPServer(manager, observation_handler=observer)  # type: ignore
    await getattr(server.mcp, "tools")["observe"](
        "b1", omniparser=True, screenshot=True, no_dom=True
    )
    assert calls == [
        (
            "b1",
            {"screenshot": True, "no_dom": True, "omniparser": True},
        )
    ]


def test_access_guard_helpers():
    assert mcp_server._is_loopback_address("127.0.0.1")
    assert mcp_server._is_loopback_address("::1")
    assert not mcp_server._is_loopback_address("8.8.8.8")

    headers = [(b"authorization", b"Bearer token")]
    assert mcp_server._extract_auth_token(headers) == "token"
    headers = [(b"x-buse-token", b"token")]
    assert mcp_server._extract_auth_token(headers) == "token"


def test_mcp_server_cli_command(monkeypatch):
    mock_server_cls = MagicMock()
    mock_server_instance = MagicMock()
    mock_server_cls.return_value = mock_server_instance
    monkeypatch.setattr(main, "BuseMCPServer", mock_server_cls)

    mock_session_manager = MagicMock()
    monkeypatch.setattr(main, "session_manager", mock_session_manager)

    code, out = main.run_cli(["mcp-server", "--port", "9000", "--name", "test-server"])

    mock_server_cls.assert_called_once()
    _, kwargs = mock_server_cls.call_args
    assert kwargs["server_name"] == "test-server"

    mock_server_instance.run.assert_called_once()
    _, run_kwargs = mock_server_instance.run.call_args
    assert run_kwargs["port"] == 9000


@pytest.mark.asyncio
async def test_mcp_tool_handler_passes_action_label(monkeypatch):
    captured = {}

    async def fake_execute_tool(
        instance_id,
        action_name,
        params,
        return_result=False,
        action_label=None,
        **kwargs,
    ):
        captured["instance_id"] = instance_id
        captured["action_name"] = action_name
        captured["params"] = params
        captured["return_result"] = return_result
        captured["action_label"] = action_label
        return {"ok": True}

    monkeypatch.setattr(main, "execute_tool", fake_execute_tool)
    handler = main._make_mcp_tool_handler()
    await handler(
        "b1",
        "evaluate",
        action_label="refresh",
        code="window.location.reload()",
    )
    assert captured["instance_id"] == "b1"
    assert captured["action_name"] == "evaluate"
    assert captured["params"] == {"code": "window.location.reload()"}
    assert captured["return_result"] is True
    assert captured["action_label"] == "refresh"


@pytest.mark.asyncio
async def test_mcp_observation_handler_passes_omniparser(monkeypatch):
    captured = {}

    async def fake_get_observation(
        instance_id,
        screenshot=False,
        path=None,
        omniparser=False,
        no_dom=False,
    ):
        captured["instance_id"] = instance_id
        captured["screenshot"] = screenshot
        captured["no_dom"] = no_dom
        captured["omniparser"] = omniparser
        return {"ok": True}

    monkeypatch.setattr(main, "get_observation", fake_get_observation)
    handler = main._make_mcp_observation_handler()
    await handler("b1", screenshot=True, no_dom=True, omniparser=True)
    assert captured["instance_id"] == "b1"
    assert captured["screenshot"] is True
    assert captured["no_dom"] is True
    assert captured["omniparser"] is True
