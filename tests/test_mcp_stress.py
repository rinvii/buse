import pytest
from buse.mcp_server import BuseMCPServer
from buse.session import SessionInfo


class FakeSessionManager:
    def __init__(self, sessions: dict[str, SessionInfo]):
        self._sessions = sessions

    def list_sessions(self) -> dict[str, SessionInfo]:
        return self._sessions

    def get_session(self, instance_id: str) -> SessionInfo | None:
        return self._sessions.get(instance_id)


class CapturingHandler:
    def __init__(self):
        self.calls = []

    async def __call__(
        self, instance_id: str, action_name: str | None = None, **kwargs
    ):
        if action_name is None:
            self.calls.append({"instance_id": instance_id, "kwargs": kwargs})
        else:
            self.calls.append(
                {
                    "instance_id": instance_id,
                    "action_name": action_name,
                    "kwargs": kwargs,
                }
            )
        return {"status": "ok"}


@pytest.fixture
def session_manager():
    return FakeSessionManager({})


@pytest.fixture
def tool_handler():
    return CapturingHandler()


@pytest.fixture
def observation_handler():
    return CapturingHandler()


@pytest.fixture
def server(session_manager, tool_handler, observation_handler):
    import buse.mcp_server as mcp_server

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
            pass

        def streamable_http_app(self):
            return "app"

    monkeypatch_fastmcp = FakeFastMCP
    mcp_server.FastMCP = monkeypatch_fastmcp  # type: ignore

    return BuseMCPServer(
        session_manager,
        tool_handler=tool_handler,
        observation_handler=observation_handler,
    )


@pytest.mark.asyncio
async def test_navigate(server, tool_handler):
    await server.mcp.tools["navigate"]("id1", "https://test.com", new_tab=True)
    assert tool_handler.calls[-1] == {
        "instance_id": "id1",
        "action_name": "navigate",
        "kwargs": {"url": "https://test.com", "new_tab": True},
    }

    await server.mcp.tools["navigate"]("id1", "https://test.com")
    assert tool_handler.calls[-1]["kwargs"]["new_tab"] is False


@pytest.mark.asyncio
async def test_click_combinations(server, tool_handler):
    await server.mcp.tools["click"]("id1", index=5)
    assert tool_handler.calls[-1] == {
        "instance_id": "id1",
        "action_name": "click",
        "kwargs": {
            "index": 5,
            "element_id": None,
            "element_class": None,
            "coordinate_x": None,
            "coordinate_y": None,
        },
    }

    await server.mcp.tools["click"]("id1", element_id="submit-btn")
    assert tool_handler.calls[-1]["kwargs"]["element_id"] == "submit-btn"

    await server.mcp.tools["click"]("id1", element_class="btn-primary")
    assert tool_handler.calls[-1]["kwargs"]["element_class"] == "btn-primary"

    await server.mcp.tools["click"]("id1", x=100, y=200)
    assert tool_handler.calls[-1]["kwargs"]["coordinate_x"] == 100
    assert tool_handler.calls[-1]["kwargs"]["coordinate_y"] == 200

    await server.mcp.tools["click"]("id1", index=1, element_id="", element_class="  ")
    assert tool_handler.calls[-1]["kwargs"]["element_id"] is None
    assert tool_handler.calls[-1]["kwargs"]["element_class"] is None


@pytest.mark.asyncio
async def test_click_validation(server):
    with pytest.raises(
        ValueError, match="Provide an index, element_id/element_class, or x/y"
    ):
        await server.mcp.tools["click"]("id1")

    with pytest.raises(ValueError, match="Provide both x and y"):
        await server.mcp.tools["click"]("id1", x=10)

    with pytest.raises(ValueError, match="Provide both x and y"):
        await server.mcp.tools["click"]("id1", y=10)


@pytest.mark.asyncio
async def test_input_text(server, tool_handler):
    await server.mcp.tools["input_text"]("id1", "hello", index=1)
    assert tool_handler.calls[-1] == {
        "instance_id": "id1",
        "action_name": "input",
        "kwargs": {
            "text": "hello",
            "index": 1,
            "element_id": None,
            "element_class": None,
        },
    }

    with pytest.raises(
        ValueError, match="Provide an index or element_id/element_class"
    ):
        await server.mcp.tools["input_text"]("id1", "text")


@pytest.mark.asyncio
async def test_send_keys(server, tool_handler):
    await server.mcp.tools["send_keys"]("id1", "Enter", index=1)
    assert tool_handler.calls[-1] == {
        "instance_id": "id1",
        "action_name": "send_keys",
        "kwargs": {
            "keys": "Enter",
            "index": 1,
            "element_id": None,
            "element_class": None,
        },
    }
    await server.mcp.tools["send_keys"]("id1", "Enter")
    assert tool_handler.calls[-1]["kwargs"]["index"] is None


@pytest.mark.asyncio
async def test_scroll(server, tool_handler):
    await server.mcp.tools["scroll"]("id1", pages=0.5, down=False, index=2)
    assert tool_handler.calls[-1] == {
        "instance_id": "id1",
        "action_name": "scroll",
        "kwargs": {"pages": 0.5, "down": False, "index": 2},
    }

    await server.mcp.tools["scroll"]("id1")
    assert tool_handler.calls[-1]["kwargs"]["down"] is True
    assert tool_handler.calls[-1]["kwargs"]["pages"] == 1.0


@pytest.mark.asyncio
async def test_tabs(server, tool_handler):
    await server.mcp.tools["switch_tab"]("id1", "tab1")
    assert tool_handler.calls[-1] == {
        "instance_id": "id1",
        "action_name": "switch",
        "kwargs": {"tab_id": "tab1"},
    }

    await server.mcp.tools["close_tab"]("id1", "tab1")
    assert tool_handler.calls[-1] == {
        "instance_id": "id1",
        "action_name": "close",
        "kwargs": {"tab_id": "tab1"},
    }


@pytest.mark.asyncio
async def test_search(server, tool_handler):
    await server.mcp.tools["search"]("id1", "query", engine="bing")
    assert tool_handler.calls[-1] == {
        "instance_id": "id1",
        "action_name": "search",
        "kwargs": {"query": "query", "engine": "bing"},
    }


@pytest.mark.asyncio
async def test_upload_file(server, tool_handler):
    await server.mcp.tools["upload_file"]("id1", index=1, path="/tmp/file")
    assert tool_handler.calls[-1] == {
        "instance_id": "id1",
        "action_name": "upload_file",
        "kwargs": {"index": 1, "path": "/tmp/file"},
    }


@pytest.mark.asyncio
async def test_find_text(server, tool_handler):
    await server.mcp.tools["find_text"]("id1", "some text")
    assert tool_handler.calls[-1] == {
        "instance_id": "id1",
        "action_name": "find_text",
        "kwargs": {"text": "some text"},
    }


@pytest.mark.asyncio
async def test_dropdown(server, tool_handler):
    await server.mcp.tools["dropdown_options"]("id1", index=1)
    assert tool_handler.calls[-1]["action_name"] == "dropdown_options"

    with pytest.raises(ValueError):
        await server.mcp.tools["dropdown_options"]("id1")

    await server.mcp.tools["select_dropdown"]("id1", "Option A", index=1)
    assert tool_handler.calls[-1] == {
        "instance_id": "id1",
        "action_name": "select_dropdown",
        "kwargs": {
            "text": "Option A",
            "index": 1,
            "element_id": None,
            "element_class": None,
        },
    }

    with pytest.raises(ValueError):
        await server.mcp.tools["select_dropdown"]("id1", "Option A")


@pytest.mark.asyncio
async def test_go_back(server, tool_handler):
    await server.mcp.tools["go_back"]("id1")
    assert tool_handler.calls[-1]["action_name"] == "go_back"


@pytest.mark.asyncio
async def test_hover(server, tool_handler):
    await server.mcp.tools["hover"]("id1", index=1)
    assert tool_handler.calls[-1]["action_name"] == "hover"

    with pytest.raises(ValueError):
        await server.mcp.tools["hover"]("id1")


@pytest.mark.asyncio
async def test_refresh(server, tool_handler):
    await server.mcp.tools["refresh"]("id1")
    assert tool_handler.calls[-1] == {
        "instance_id": "id1",
        "action_name": "evaluate",
        "kwargs": {"code": "window.location.reload()", "action_label": "refresh"},
    }


@pytest.mark.asyncio
async def test_wait(server, tool_handler):
    await server.mcp.tools["wait"]("id1", seconds=2.5)
    assert tool_handler.calls[-1] == {
        "instance_id": "id1",
        "action_name": "wait",
        "kwargs": {"seconds": 2.5},
    }


@pytest.mark.asyncio
async def test_save_state(server, tool_handler):
    await server.mcp.tools["save_state"]("id1", path="state.json")
    assert tool_handler.calls[-1] == {
        "instance_id": "id1",
        "action_name": "save_state",
        "kwargs": {"path": "state.json"},
    }


@pytest.mark.asyncio
async def test_extract(server, tool_handler):
    await server.mcp.tools["extract"]("id1", query="data")
    assert tool_handler.calls[-1] == {
        "instance_id": "id1",
        "action_name": "extract",
        "kwargs": {"query": "data"},
    }


@pytest.mark.asyncio
async def test_evaluate(server, tool_handler):
    await server.mcp.tools["evaluate"]("id1", code="alert(1)")
    assert tool_handler.calls[-1] == {
        "instance_id": "id1",
        "action_name": "evaluate",
        "kwargs": {"code": "alert(1)"},
    }


@pytest.mark.asyncio
async def test_session_lifecycle(server, tool_handler):
    await server.mcp.tools["start_session"]("id1")
    assert tool_handler.calls[-1]["action_name"] == "start"

    await server.mcp.tools["stop_session"]("id1")
    assert tool_handler.calls[-1]["action_name"] == "stop"


@pytest.mark.asyncio
async def test_observe(server, observation_handler):
    await server.mcp.tools["observe"](
        "id1", screenshot=True, no_dom=True, omniparser=False
    )

    assert observation_handler.calls[-1] == {
        "instance_id": "id1",
        "kwargs": {"screenshot": True, "no_dom": True, "omniparser": False},
    }


@pytest.mark.asyncio
async def test_negative_values(server, tool_handler):
    await server.mcp.tools["click"]("id1", index=-1)
    assert tool_handler.calls[-1]["kwargs"]["index"] == -1

    await server.mcp.tools["click"]("id1", x=-100, y=-100)
    assert tool_handler.calls[-1]["kwargs"]["coordinate_x"] == -100


@pytest.mark.asyncio
async def test_large_inputs(server, tool_handler):
    large_text = "a" * 10000
    await server.mcp.tools["input_text"]("id1", text=large_text, index=1)
    assert tool_handler.calls[-1]["kwargs"]["text"] == large_text


@pytest.mark.asyncio
async def test_weird_inputs(server, tool_handler):
    weird_id = '"; DROP TABLE sessions; --'
    await server.mcp.tools["click"]("id1", element_id=weird_id)
    assert tool_handler.calls[-1]["kwargs"]["element_id"] == weird_id

    await server.mcp.tools["switch_tab"]("id1", "tab #1")
    assert tool_handler.calls[-1]["kwargs"]["tab_id"] == "tab #1"


@pytest.mark.asyncio
async def test_coerce_non_string(server, tool_handler):
    await server.mcp.tools["click"]("id1", element_id=123)
    assert tool_handler.calls[-1]["kwargs"]["element_id"] == "123"
