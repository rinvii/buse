import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import buse.main as main
from buse.session import SessionInfo
from buse.models import VisualAnalysis, VisualElement


@pytest.fixture(autouse=True)
def mock_file_ops(monkeypatch):
    mock_open = MagicMock()
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file
    monkeypatch.setattr("builtins.open", mock_open)

    mock_mkdir = MagicMock()
    monkeypatch.setattr("pathlib.Path.mkdir", mock_mkdir)
    return mock_open


@pytest.fixture
def mock_session_manager(monkeypatch):
    manager = MagicMock()
    monkeypatch.setattr(main, "session_manager", manager)
    return manager


@pytest.fixture
def mock_browser_session_setup(monkeypatch):
    browser_session = MagicMock()
    file_system = MagicMock()

    async def get_session(*args):
        return browser_session, file_system

    monkeypatch.setattr(main, "_get_browser_session", get_session)
    return browser_session


@pytest.fixture
def mock_cdp_session():
    cdp = MagicMock()
    cdp.session_id = "sess1"
    cdp.cdp_client = MagicMock()
    cdp.cdp_client.send = AsyncMock()

    runtime = AsyncMock()
    runtime.evaluate = AsyncMock(
        return_value={
            "result": {"value": {"width": 100, "height": 100, "device_pixel_ratio": 1}}
        }
    )

    page = AsyncMock()
    page.captureScreenshot = AsyncMock(return_value={"data": "ZGF0YQ=="})

    class SendMock:
        Runtime = runtime
        Page = page
        Input = AsyncMock()
        DOM = AsyncMock()

    cdp.cdp_client.send = SendMock()
    return cdp


@pytest.mark.asyncio
async def test_get_observation_basic(
    mock_session_manager, mock_browser_session_setup, mock_cdp_session
):
    mock_session_manager.get_session.return_value = SessionInfo(
        instance_id="b1", cdp_url="u", pid=1, user_data_dir="d"
    )
    browser_session = mock_browser_session_setup

    state_summary = MagicMock()
    state_summary.url = "http://example.com"
    state_summary.title = "Example"
    state_summary.dom_state.llm_representation.return_value = "<html></html>"
    state_summary.screenshot = None

    browser_session.get_browser_state_summary = AsyncMock(return_value=state_summary)
    browser_session.get_tabs = AsyncMock(return_value=[])
    browser_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp_session)
    browser_session.agent_focus_target_id = None

    data = await main.get_observation("b1", screenshot=True)

    assert data["url"] == "http://example.com"
    assert data["screenshot_path"] is not None

    mock_cdp_session.cdp_client.send.Page.captureScreenshot.assert_called()


@pytest.mark.asyncio
async def test_get_observation_no_dom(
    mock_session_manager, mock_browser_session_setup, mock_cdp_session
):
    mock_session_manager.get_session.return_value = SessionInfo(
        instance_id="b1", cdp_url="u", pid=1, user_data_dir="d"
    )
    browser_session = mock_browser_session_setup

    browser_session.event_bus = MagicMock()
    event = MagicMock()
    event.event_result = AsyncMock(
        return_value=MagicMock(url="u", title="t", dom_state=None, screenshot="data")
    )
    browser_session.event_bus.dispatch.return_value = event

    browser_session.get_tabs = AsyncMock(return_value=[])
    browser_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp_session)
    browser_session.agent_focus_target_id = None

    data = await main.get_observation("b1", no_dom=True, screenshot=False)

    assert data["dom_minified"] == ""
    browser_session.event_bus.dispatch.assert_called()


@pytest.mark.asyncio
async def test_get_observation_omniparser(
    mock_session_manager, mock_browser_session_setup, mock_cdp_session, monkeypatch
):
    mock_session_manager.get_session.return_value = SessionInfo(
        instance_id="b1", cdp_url="u", pid=1, user_data_dir="d"
    )
    browser_session = mock_browser_session_setup

    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://omni")

    mock_client_cls = MagicMock()
    mock_client = mock_client_cls.return_value
    analysis_res = VisualAnalysis(
        elements=[
            VisualElement(
                index=1,
                type="button",
                content="ok",
                interactivity=True,
                center_x=10,
                center_y=10,
                bbox=[0, 0, 20, 20],
            )
        ]
    )

    mock_client.analyze = AsyncMock(return_value=(analysis_res, "ZGF0YQ=="))
    monkeypatch.setattr("buse.vision.VisionClient", mock_client_cls)

    monkeypatch.setattr(
        "buse.utils.downscale_image", MagicMock(return_value="ZGF0YQ==")
    )

    monkeypatch.setattr(main, "_should_probe_omniparser", MagicMock(return_value=False))

    state_summary = MagicMock()
    state_summary.url = "u"
    state_summary.title = "t"
    state_summary.screenshot = "ZGF0YQ=="
    state_summary.dom_state.llm_representation.return_value = "dom"
    browser_session.get_browser_state_summary = AsyncMock(return_value=state_summary)
    browser_session.get_tabs = AsyncMock(return_value=[])
    browser_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp_session)
    browser_session.agent_focus_target_id = None

    data = await main.get_observation("b1", omniparser=True)

    assert data["visual_analysis"] is not None
    mock_client.analyze.assert_awaited()
    assert data["screenshot_path"].endswith("image_som.jpg")


@pytest.mark.asyncio
async def test_get_observation_timeout_retry(
    mock_session_manager, mock_browser_session_setup, mock_cdp_session
):
    mock_session_manager.get_session.return_value = SessionInfo(
        instance_id="b1", cdp_url="u", pid=1, user_data_dir="d"
    )
    browser_session = mock_browser_session_setup

    dom_mock = MagicMock()
    dom_mock.llm_representation.return_value = "<html>"
    browser_session.get_browser_state_summary = AsyncMock(
        side_effect=[
            Exception("Timeout"),
            MagicMock(url="u", title="t", screenshot="ZGF0YQ==", dom_state=dom_mock),
        ]
    )
    browser_session.get_tabs = AsyncMock(return_value=[])
    browser_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp_session)
    browser_session.agent_focus_target_id = None

    await main.get_observation("b1", screenshot=True)

    assert browser_session.get_browser_state_summary.call_count == 2


@pytest.mark.asyncio
async def test_get_observation_missing_env_omniparser(monkeypatch):
    monkeypatch.delenv("BUSE_OMNIPARSER_URL", raising=False)
    with pytest.raises(
        ValueError, match="BUSE_OMNIPARSER_URL environment variable is required"
    ):
        await main.get_observation("b1", omniparser=True)


@pytest.mark.asyncio
async def test_get_observation_profile(
    mock_session_manager, mock_browser_session_setup, mock_cdp_session, monkeypatch
):
    mock_session_manager.get_session.return_value = SessionInfo(
        instance_id="b1", cdp_url="u", pid=1, user_data_dir="d"
    )
    browser_session = mock_browser_session_setup

    monkeypatch.setattr(main.state, "profile", True)

    state_summary = MagicMock()
    state_summary.url = "u"
    state_summary.title = "t"
    state_summary.screenshot = None
    state_summary.dom_state.llm_representation.return_value = "dom"
    browser_session.get_browser_state_summary = AsyncMock(return_value=state_summary)
    browser_session.get_tabs = AsyncMock(return_value=[])
    browser_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp_session)
    browser_session.agent_focus_target_id = None

    data = await main.get_observation("b1")
    assert "profile" in data
    assert data["profile"]["get_session_ms"] > 0


@pytest.mark.asyncio
async def test_get_observation_omniparser_custom_path(
    mock_session_manager, mock_browser_session_setup, mock_cdp_session, monkeypatch
):
    mock_session_manager.get_session.return_value = SessionInfo(
        instance_id="b1", cdp_url="u", pid=1, user_data_dir="d"
    )
    browser_session = mock_browser_session_setup

    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://omni")
    monkeypatch.setattr(main, "_should_probe_omniparser", MagicMock(return_value=False))

    mock_client_cls = MagicMock()
    monkeypatch.setattr("buse.vision.VisionClient", mock_client_cls)
    mock_client_cls.return_value.analyze = AsyncMock(
        return_value=(VisualAnalysis(elements=[]), None)
    )

    mock_client_cls.return_value.analyze = AsyncMock(
        return_value=(
            VisualAnalysis(
                elements=[
                    VisualElement(
                        index=1,
                        type="t",
                        content="c",
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
    monkeypatch.setattr(
        "buse.utils.downscale_image", MagicMock(return_value="ZGF0YQ==")
    )

    state_summary = MagicMock()
    state_summary.url = "u"
    state_summary.title = "t"
    state_summary.screenshot = "ZGF0YQ=="
    state_summary.dom_state.llm_representation.return_value = "dom"
    browser_session.get_browser_state_summary = AsyncMock(return_value=state_summary)
    browser_session.get_tabs = AsyncMock(return_value=[])
    browser_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp_session)
    browser_session.agent_focus_target_id = None

    path = MagicMock()
    path.is_dir.return_value = True
    with patch("buse.main.Path", return_value=path):
        await main.get_observation("b1", omniparser=True, path="/custom/dir")


@pytest.mark.asyncio
async def test_get_observation_session_not_found(mock_session_manager, monkeypatch):
    mock_session_manager.get_session.return_value = None
    monkeypatch.setattr(main, "_stop_cached_browser_session", AsyncMock())

    with pytest.raises(ValueError, match="Instance b1 not found"):
        await main.get_observation("b1")


@pytest.mark.asyncio
async def test_get_observation_cdp_failure(
    mock_session_manager, mock_browser_session_setup, mock_cdp_session
):
    mock_session_manager.get_session.return_value = SessionInfo(
        instance_id="b1", cdp_url="u", pid=1, user_data_dir="d"
    )
    browser_session = mock_browser_session_setup

    state_summary = MagicMock()
    state_summary.url = "u"
    state_summary.title = "t"
    state_summary.screenshot = None
    state_summary.dom_state.llm_representation.return_value = "dom"
    browser_session.get_browser_state_summary = AsyncMock(return_value=state_summary)
    browser_session.get_tabs = AsyncMock(return_value=[])

    browser_session.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp_session)
    mock_cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(
        side_effect=Exception("CDP error")
    )

    data = await main.get_observation("b1")
    assert data["dom_minified"] == "dom"


@pytest.mark.asyncio
async def test_get_observation_retry_failure(
    mock_session_manager, mock_browser_session_setup
):
    mock_session_manager.get_session.return_value = SessionInfo(
        instance_id="b1", cdp_url="u", pid=1, user_data_dir="d"
    )
    browser_session = mock_browser_session_setup

    browser_session.get_browser_state_summary = AsyncMock(
        side_effect=[Exception("Timeout"), Exception("Timeout")]
    )

    with pytest.raises(Exception, match="Timeout"):
        await main.get_observation("b1", screenshot=True)

    assert browser_session.get_browser_state_summary.call_count == 2
