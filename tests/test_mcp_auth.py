import pytest
from unittest.mock import AsyncMock, MagicMock
from buse.mcp_server import (
    _normalize_auth_token,
    _is_loopback_address,
    _client_is_loopback,
    _get_header,
    _extract_auth_token,
    MCPAccessGuard,
    _wrap_with_access_guard,
    _coerce_optional_str,
)


def test_normalize_auth_token():
    assert _normalize_auth_token(None) is None
    assert _normalize_auth_token("") is None
    assert _normalize_auth_token("   ") is None
    assert _normalize_auth_token(" token ") == "token"


def test_coerce_optional_str():
    assert _coerce_optional_str(None) is None
    assert _coerce_optional_str("  ") is None
    assert _coerce_optional_str(" foo ") == "foo"
    assert _coerce_optional_str(123) == "123"
    assert _coerce_optional_str(12.34) == "12.34"
    assert _coerce_optional_str([]) is None


def test_is_loopback_address():
    assert _is_loopback_address("localhost") is True
    assert _is_loopback_address("127.0.0.1") is True
    assert _is_loopback_address("::1") is True
    assert _is_loopback_address("8.8.8.8") is False
    assert _is_loopback_address("invalid-ip") is False


def test_client_is_loopback():
    assert _client_is_loopback(None) is False
    assert _client_is_loopback(("127.0.0.1", 1234)) is True
    assert _client_is_loopback(("8.8.8.8", 1234)) is False


def test_get_header():
    headers = [
        (b"content-type", b"application/json"),
        (b"Authorization", b"Bearer 123"),
    ]
    assert _get_header(headers, b"Content-Type") == "application/json"
    assert _get_header(headers, b"authorization") == "Bearer 123"
    assert _get_header(headers, b"Missing") is None


def test_extract_auth_token():
    assert _extract_auth_token([(b"authorization", b"Bearer secret")]) == "secret"
    assert _extract_auth_token([(b"authorization", b"bearer secret")]) == "secret"

    assert _extract_auth_token([(b"authorization", b"secret")]) == "secret"

    assert _extract_auth_token([(b"x-buse-token", b"secret")]) == "secret"

    assert (
        _extract_auth_token(
            [(b"x-buse-token", b"token2"), (b"authorization", b"Bearer token1")]
        )
        == "token1"
    )

    assert _extract_auth_token([]) is None


@pytest.mark.asyncio
async def test_access_guard_passthrough_non_http():
    app = AsyncMock()
    guard = MCPAccessGuard(app, allow_remote=False, auth_token=None)
    scope = {"type": "websocket"}
    await guard(scope, {}, {})
    app.assert_called_once()


@pytest.mark.asyncio
async def test_access_guard_blocks_remote():
    app = AsyncMock()
    guard = MCPAccessGuard(app, allow_remote=False, auth_token=None)

    scope = {"type": "http", "client": ("8.8.8.8", 1234)}
    send = AsyncMock()

    await guard(scope, {}, send)

    app.assert_not_called()
    assert send.call_count == 2
    assert send.call_args_list[0][0][0]["status"] == 403


@pytest.mark.asyncio
async def test_access_guard_allows_remote_if_configured():
    app = AsyncMock()
    guard = MCPAccessGuard(app, allow_remote=True, auth_token=None)

    scope = {"type": "http", "client": ("8.8.8.8", 1234)}
    await guard(scope, {}, {})
    app.assert_called_once()


@pytest.mark.asyncio
async def test_access_guard_auth_failure():
    app = AsyncMock()
    guard = MCPAccessGuard(app, allow_remote=True, auth_token="secret")

    scope = {"type": "http", "headers": [], "client": ("127.0.0.1", 1234)}
    send = AsyncMock()
    await guard(scope, {}, send)

    app.assert_not_called()
    assert send.call_args_list[0][0][0]["status"] == 401

    send.reset_mock()
    scope["headers"] = [(b"authorization", b"Bearer wrong")]
    await guard(scope, {}, send)
    app.assert_not_called()
    assert send.call_args_list[0][0][0]["status"] == 401


@pytest.mark.asyncio
async def test_access_guard_auth_success():
    app = AsyncMock()
    guard = MCPAccessGuard(app, allow_remote=True, auth_token="secret")

    scope = {
        "type": "http",
        "headers": [(b"authorization", b"Bearer secret")],
        "client": ("127.0.0.1", 1234),
    }
    await guard(scope, {}, {})
    app.assert_called_once()


def test_wrap_with_access_guard():
    app = MagicMock()

    wrapper = _wrap_with_access_guard(app, allow_remote=False, auth_token=None)
    assert isinstance(wrapper, MCPAccessGuard)

    wrapper = _wrap_with_access_guard(app, allow_remote=True, auth_token=None)
    assert wrapper is app

    wrapper = _wrap_with_access_guard(app, allow_remote=True, auth_token="token")
    assert isinstance(wrapper, MCPAccessGuard)
