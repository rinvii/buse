import pytest
from unittest.mock import MagicMock
import buse.main as main
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


def test_navigate_cli(monkeypatch):
    mock_run = MagicMock()
    monkeypatch.setattr("asyncio.run", mock_run)

    ctx = MagicMock()
    ctx.obj = {"instance_id": "b1"}

    main.navigate(ctx, "http://example.com", new_tab=True)

    mock_run.assert_called_once()


def test_new_tab_cli(monkeypatch):
    mock_run = MagicMock()
    monkeypatch.setattr("asyncio.run", mock_run)

    ctx = MagicMock()
    ctx.obj = {"instance_id": "b1"}

    main.new_tab(ctx, "http://example.com")
    mock_run.assert_called_once()


def test_observe_cli_basic(monkeypatch):
    mock_run = MagicMock()
    monkeypatch.setattr("asyncio.run", mock_run)

    ctx = MagicMock()
    ctx.obj = {"instance_id": "b1"}

    main.observe(ctx, screenshot=True, path=None, omniparser=False, no_dom=False)
    mock_run.assert_called_once()


def test_observe_cli_omniparser_missing_env(monkeypatch):
    monkeypatch.delenv("BUSE_OMNIPARSER_URL", raising=False)
    ctx = MagicMock()
    ctx.obj = {"instance_id": "b1"}

    with pytest.raises(
        main.typer.BadParameter,
        match="BUSE_OMNIPARSER_URL environment variable is required",
    ):
        main.observe(ctx, omniparser=True)


def test_observe_cli_omniparser_valid_env(monkeypatch):
    monkeypatch.setenv("BUSE_OMNIPARSER_URL", "http://omni")
    mock_run = MagicMock()
    monkeypatch.setattr("asyncio.run", mock_run)

    ctx = MagicMock()
    ctx.obj = {"instance_id": "b1"}

    main.observe(ctx, omniparser=True)
    mock_run.assert_called_once()
