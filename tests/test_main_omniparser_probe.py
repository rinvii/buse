import pytest
from unittest.mock import MagicMock, AsyncMock
import buse.main as main
import time


@pytest.fixture
def mock_settings(monkeypatch):
    mock_settings = {}
    monkeypatch.setattr(main, "_load_settings", MagicMock(return_value=mock_settings))
    mock_save = MagicMock()
    monkeypatch.setattr(main, "_save_settings", mock_save)
    return mock_save


def test_should_probe_omniparser(mock_settings, monkeypatch):
    endpoint = "http://omni"

    main._omniparser_probe_cache.clear()
    assert main._should_probe_omniparser(endpoint) is True

    main._omniparser_probe_cache[endpoint] = time.time()
    assert main._should_probe_omniparser(endpoint) is False

    main._omniparser_probe_cache[endpoint] = time.time() - (
        main._OMNIPARSER_PROBE_TTL_SECONDS + 10
    )
    assert main._should_probe_omniparser(endpoint) is True

    main._omniparser_probe_cache[endpoint] = 0.0
    assert main._should_probe_omniparser(endpoint) is True


def test_mark_omniparser_probe(mock_settings):
    endpoint = "http://omni"
    main._mark_omniparser_probe(endpoint)
    assert endpoint in main._omniparser_probe_cache
    mock_settings.assert_called()


@pytest.mark.asyncio
async def test_get_observation_probe_success(monkeypatch):
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://omni")
    monkeypatch.setattr(main, "_should_probe_omniparser", MagicMock(return_value=True))
    mock_mark = MagicMock()
    monkeypatch.setattr(main, "_mark_omniparser_probe", mock_mark)

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.get.return_value.raise_for_status = MagicMock()
    monkeypatch.setattr("httpx.AsyncClient", MagicMock(return_value=mock_client))

    monkeypatch.setattr(
        main.session_manager, "get_session", MagicMock(return_value=None)
    )
    monkeypatch.setattr(main, "_stop_cached_browser_session", AsyncMock())

    try:
        await main.get_observation("b1", omniparser=True)
    except ValueError:
        pass

    mock_client.get.assert_called_with("http://omni/probe/")
    mock_mark.assert_called_with("http://omni")


@pytest.mark.asyncio
async def test_get_observation_probe_fail(monkeypatch):
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://omni")
    monkeypatch.setattr(main, "_should_probe_omniparser", MagicMock(return_value=True))

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.get.side_effect = Exception("Connection refused")
    monkeypatch.setattr("httpx.AsyncClient", MagicMock(return_value=mock_client))

    with pytest.raises(RuntimeError, match="OmniParser probe failed"):
        await main.get_observation("b1", omniparser=True)
