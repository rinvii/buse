import pytest
from unittest.mock import MagicMock, AsyncMock
import buse.main as main
from buse.session import SessionInfo


@pytest.mark.asyncio
async def test_get_observation_omniparser_prepare_fail(monkeypatch):
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://omni")
    monkeypatch.setattr(main, "_should_probe_omniparser", MagicMock(return_value=False))

    monkeypatch.setattr(
        main.session_manager,
        "get_session",
        MagicMock(
            return_value=SessionInfo(
                instance_id="b1", cdp_url="u", pid=1, user_data_dir="d"
            )
        ),
    )
    browser_session = MagicMock()
    file_system = MagicMock()

    async def get_session(*args):
        return browser_session, file_system

    monkeypatch.setattr(main, "_get_browser_session", get_session)

    state_summary = MagicMock()
    state_summary.screenshot = "ZGF0YQ=="
    state_summary.dom_state.llm_representation.return_value = "dom"
    browser_session.get_browser_state_summary = AsyncMock(return_value=state_summary)

    browser_session.get_tabs = AsyncMock(return_value=[])

    cdp = MagicMock()
    cdp.session_id = "s1"
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={
            "result": {
                "value": {"width": 100, "height": 100, "device_pixel_ratio": 1.0}
            }
        }
    )
    browser_session.get_or_create_cdp_session = AsyncMock(return_value=cdp)
    browser_session.agent_focus_target_id = None

    monkeypatch.setattr(
        "buse.utils.downscale_image",
        MagicMock(side_effect=Exception("Downscale failed")),
    )

    mock_client_cls = MagicMock()
    monkeypatch.setattr("buse.vision.VisionClient", mock_client_cls)

    with pytest.raises(Exception, match="Downscale failed"):
        await main.get_observation("b1", omniparser=True)


@pytest.mark.asyncio
async def test_get_observation_omniparser_analysis_fail(monkeypatch):
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://omni")
    monkeypatch.setattr(main, "_should_probe_omniparser", MagicMock(return_value=False))

    monkeypatch.setattr(
        main.session_manager,
        "get_session",
        MagicMock(
            return_value=SessionInfo(
                instance_id="b1", cdp_url="u", pid=1, user_data_dir="d"
            )
        ),
    )
    browser_session = MagicMock()

    async def get_session(*args):
        return browser_session, MagicMock()

    monkeypatch.setattr(main, "_get_browser_session", get_session)

    state_summary = MagicMock()
    state_summary.screenshot = "ZGF0YQ=="
    state_summary.dom_state.llm_representation.return_value = "dom"
    browser_session.get_browser_state_summary = AsyncMock(return_value=state_summary)
    browser_session.get_tabs = AsyncMock(return_value=[])

    cdp = MagicMock()
    cdp.session_id = "s1"
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={
            "result": {
                "value": {"width": 100, "height": 100, "device_pixel_ratio": 1.0}
            }
        }
    )
    browser_session.get_or_create_cdp_session = AsyncMock(return_value=cdp)
    browser_session.agent_focus_target_id = None

    monkeypatch.setattr(
        "buse.utils.downscale_image", MagicMock(return_value="ZGF0YQ==")
    )

    mock_client = MagicMock()
    mock_client.analyze = AsyncMock(side_effect=Exception("Analysis failed"))
    monkeypatch.setattr("buse.vision.VisionClient", MagicMock(return_value=mock_client))

    with pytest.raises(Exception, match="Analysis failed"):
        await main.get_observation("b1", omniparser=True)


@pytest.mark.asyncio
async def test_run_save_state_fail(monkeypatch):
    monkeypatch.setattr(
        main.session_manager,
        "get_session",
        MagicMock(
            return_value=SessionInfo(
                instance_id="b1", cdp_url="u", pid=1, user_data_dir="d"
            )
        ),
    )

    mock_browser_cls = MagicMock()
    session_instance = mock_browser_cls.return_value
    session_instance.start = AsyncMock()
    session_instance.stop = AsyncMock()
    session_instance.export_storage_state = AsyncMock(
        side_effect=Exception("Export failed")
    )
    monkeypatch.setattr("browser_use.browser.BrowserSession", mock_browser_cls)
    monkeypatch.setattr(main, "BrowserSession", mock_browser_cls)

    try:
        await main.run_save_state("b1", "path")
    except Exception:
        pass

    session_instance.stop.assert_awaited()


@pytest.mark.asyncio
async def test_get_observation_omniparser_no_screenshot_error(monkeypatch):
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://omni")
    monkeypatch.setattr(main, "_should_probe_omniparser", MagicMock(return_value=False))

    monkeypatch.setattr(
        main.session_manager,
        "get_session",
        MagicMock(
            return_value=SessionInfo(
                instance_id="b1", cdp_url="u", pid=1, user_data_dir="d"
            )
        ),
    )
    browser_session = MagicMock()

    async def get_session(*args):
        return browser_session, MagicMock()

    monkeypatch.setattr(main, "_get_browser_session", get_session)

    state_summary = MagicMock()
    state_summary.screenshot = None
    state_summary.dom_state.llm_representation.return_value = "dom"
    browser_session.get_browser_state_summary = AsyncMock(return_value=state_summary)

    cdp = MagicMock()
    cdp.session_id = "s1"
    cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={
            "result": {
                "value": {"width": 100, "height": 100, "device_pixel_ratio": 1.0}
            }
        }
    )
    with pytest.raises(RuntimeError, match="CDP access failed"):
        await main.get_observation("b1", omniparser=True)
