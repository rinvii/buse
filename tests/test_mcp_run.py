import pytest
from unittest.mock import MagicMock, patch
from buse.mcp_server import BuseMCPServer


class FakeSessionManager:
    def list_sessions(self):
        return {}


@pytest.fixture
def server():
    return BuseMCPServer(
        FakeSessionManager()  # type: ignore
    )


def test_run_stdio(server):
    with patch("sys.stderr") as mock_stderr:
        server.mcp.run = MagicMock()
        server.run(transport="stdio")
        server.mcp.run.assert_called_with(transport="stdio")
        mock_stderr.write.assert_called()


def test_run_streamable_http(server):
    mock_app = MagicMock()
    server.mcp.streamable_http_app = MagicMock(return_value=mock_app)

    with patch("buse.mcp_server.Server") as MockUvicornServer:
        mock_uvicorn_instance = MockUvicornServer.return_value
        server.run(transport="streamable-http", port=9000)

        server.mcp.streamable_http_app.assert_called_once()
        MockUvicornServer.assert_called_once()

        config = MockUvicornServer.call_args[0][0]
        assert config.port == 9000
        mock_uvicorn_instance.run.assert_called_once()


def test_run_sse(server):
    mock_app = MagicMock()
    server.mcp.sse_app = MagicMock(return_value=mock_app)

    with patch("buse.mcp_server.Server") as MockUvicornServer:
        mock_uvicorn_instance = MockUvicornServer.return_value
        server.run(transport="sse", port=9001)

        server.mcp.sse_app.assert_called_once()
        MockUvicornServer.assert_called_once()
        config = MockUvicornServer.call_args[0][0]
        assert config.port == 9001
        mock_uvicorn_instance.run.assert_called_once()


def test_run_invalid_transport(server):
    with pytest.raises(ValueError, match="Unsupported transport"):
        server.run(transport="carrier-pigeon")
