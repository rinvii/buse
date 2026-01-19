import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import buse.main as main
from buse.session import SessionInfo


@pytest.fixture
def mock_session_manager(monkeypatch):
    manager = MagicMock()
    monkeypatch.setattr(main, "session_manager", manager)
    return manager


@pytest.fixture
def mock_stop_cached(monkeypatch):
    mock = AsyncMock()
    monkeypatch.setattr(main, "_stop_cached_browser_session", mock)
    return mock


@pytest.fixture
def mock_browser_session(monkeypatch):
    mock_cls = MagicMock()
    monkeypatch.setattr("browser_use.browser.BrowserSession", mock_cls)
    monkeypatch.setattr(main, "BrowserSession", mock_cls)
    return mock_cls


@pytest.mark.asyncio
async def test_run_start_new(mock_session_manager):
    mock_session_manager.get_session.return_value = None

    result = await main.run_start("b1")

    mock_session_manager.start_session.assert_called_with("b1")
    assert result["success"] is True
    assert result["already_running"] is False


@pytest.mark.asyncio
async def test_run_start_existing(mock_session_manager):
    mock_session_manager.get_session.return_value = SessionInfo(
        instance_id="b1", cdp_url="url", pid=123, user_data_dir="dir"
    )

    result = await main.run_start("b1")

    mock_session_manager.start_session.assert_not_called()
    assert result["success"] is True
    assert result["already_running"] is True


@pytest.mark.asyncio
async def test_run_stop_existing(mock_session_manager, mock_stop_cached):
    mock_session_manager.get_session.return_value = SessionInfo(
        instance_id="b1", cdp_url="url", pid=123, user_data_dir="dir"
    )

    result = await main.run_stop("b1")

    mock_stop_cached.assert_awaited_with("b1")
    mock_session_manager.stop_session.assert_called_with("b1")
    assert result["success"] is True


@pytest.mark.asyncio
async def test_run_stop_missing(mock_session_manager, mock_stop_cached):
    mock_session_manager.get_session.return_value = None

    with pytest.raises(ValueError, match="Instance b1 not found"):
        await main.run_stop("b1")

    mock_stop_cached.assert_awaited_with("b1")


@pytest.mark.asyncio
async def test_run_save_state(mock_session_manager, mock_browser_session):
    mock_session_manager.get_session.return_value = SessionInfo(
        instance_id="b1", cdp_url="http://cdp", pid=123, user_data_dir="dir"
    )

    session_instance = mock_browser_session.return_value
    session_instance.start = AsyncMock()
    session_instance.stop = AsyncMock()
    session_instance.export_storage_state = AsyncMock(
        return_value={"cookies": [1, 2, 3]}
    )

    result = await main.run_save_state("b1", "state.json")

    mock_browser_session.assert_called_with(cdp_url="http://cdp")
    session_instance.start.assert_awaited_once()
    session_instance.export_storage_state.assert_awaited_with(output_path="state.json")
    session_instance.stop.assert_awaited_once()

    assert result["success"] is True
    assert result["cookies_count"] == 3


@pytest.mark.asyncio
async def test_run_save_state_missing(mock_session_manager):
    mock_session_manager.get_session.return_value = None
    with pytest.raises(ValueError, match="Instance b1 not found"):
        await main.run_save_state("b1", "state.json")


@pytest.mark.asyncio
async def test_mcp_tool_handler_routing(monkeypatch):
    handler = main._make_mcp_tool_handler()

    mock_start = AsyncMock(return_value={"status": "started"})
    mock_stop = AsyncMock(return_value={"status": "stopped"})
    mock_save = AsyncMock(return_value={"status": "saved"})
    mock_execute = AsyncMock(return_value={"status": "executed"})

    monkeypatch.setattr(main, "run_start", mock_start)
    monkeypatch.setattr(main, "run_stop", mock_stop)
    monkeypatch.setattr(main, "run_save_state", mock_save)
    monkeypatch.setattr(main, "execute_tool", mock_execute)

    await handler("b1", "start")
    mock_start.assert_awaited_with("b1")

    await handler("b1", "stop")
    mock_stop.assert_awaited_with("b1")

    await handler("b1", "save_state", path="p")
    mock_save.assert_awaited_with("b1", path="p")

    await handler("b1", "click", index=1)
    mock_execute.assert_awaited()


@pytest.mark.asyncio
async def test_new_tab_command(monkeypatch):
    mock_execute = AsyncMock()
    monkeypatch.setattr(main, "execute_tool", mock_execute)

    ctx = MagicMock()
    ctx.obj = {"instance_id": "b1"}

    with patch("asyncio.run") as mock_run:
        main.new_tab(ctx, "http://example.com")
        mock_run.assert_called_once()
