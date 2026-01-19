from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from typing import Dict, Iterable, Optional, Any, Union

from uvicorn import Config, Server

from mcp.server.fastmcp import FastMCP

from .session import SessionInfo, SessionManager


def _normalize_auth_token(token: Optional[str]) -> Optional[str]:
    if token is None:
        return None
    cleaned = token.strip()
    return cleaned or None


def _coerce_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _is_loopback_address(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _client_is_loopback(client: Optional[tuple[str, int]]) -> bool:
    if not client:
        return False
    return _is_loopback_address(client[0])


def _get_header(headers: Iterable[tuple[bytes, bytes]], name: bytes) -> Optional[str]:
    name_lower = name.lower()
    for key, value in headers:
        if key.lower() == name_lower:
            return value.decode("latin-1")
    return None


def _extract_auth_token(headers: Iterable[tuple[bytes, bytes]]) -> Optional[str]:
    auth = _get_header(headers, b"authorization")
    if auth:
        parts = auth.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
        return auth.strip()
    token = _get_header(headers, b"x-buse-token")
    if token:
        return token.strip()
    return None


async def _send_error(
    send, status: int, message: str, *, auth_required: bool = False
) -> None:
    headers = [(b"content-type", b"text/plain; charset=utf-8")]
    if auth_required:
        headers.append((b"www-authenticate", b"Bearer"))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": message.encode("utf-8")})


class MCPAccessGuard:
    def __init__(
        self,
        app,
        *,
        allow_remote: bool,
        auth_token: Optional[str],
    ) -> None:
        self.app = app
        self.allow_remote = allow_remote
        self.auth_token = _normalize_auth_token(auth_token)

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        if not self.allow_remote and not _client_is_loopback(scope.get("client")):
            await _send_error(send, 403, "Forbidden")
            return
        if self.auth_token:
            headers = scope.get("headers", [])
            token = _extract_auth_token(headers)
            if token != self.auth_token:
                await _send_error(send, 401, "Unauthorized", auth_required=True)
                return
        await self.app(scope, receive, send)


def _wrap_with_access_guard(app, *, allow_remote: bool, auth_token: Optional[str]):
    normalized_token = _normalize_auth_token(auth_token)
    if allow_remote and not normalized_token:
        return app
    return MCPAccessGuard(
        app,
        allow_remote=allow_remote,
        auth_token=normalized_token,
    )


@dataclass
class SessionSummary:
    instance_id: str
    cdp_url: str
    user_data_dir: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "instance_id": self.instance_id,
            "cdp_url": self.cdp_url,
            "user_data_dir": self.user_data_dir,
        }


class BuseMCPServer:
    """Simple MCP server exposing buse session metadata."""

    def __init__(
        self,
        session_manager: SessionManager,
        server_name: str = "buse",
        *,
        stateless_http: bool = True,
        json_response: bool = True,
        allow_remote: bool = False,
        auth_token: Optional[str] = None,
        tool_handler=None,
        observation_handler=None,
    ):
        self.session_manager = session_manager
        self.tool_handler = tool_handler
        self.observation_handler = observation_handler
        self.allow_remote = allow_remote
        self.auth_token = _normalize_auth_token(auth_token)
        self.mcp = FastMCP(
            server_name,
            stateless_http=stateless_http,
            json_response=json_response,
        )
        self._register_resources()
        if self.tool_handler or self.observation_handler:
            self._register_tools()

    def _register_resources(self) -> None:
        @self.mcp.resource("buse://sessions")
        def all_sessions() -> Dict[str, Iterable[Dict[str, str]]]:
            return {
                "instances": [
                    self._serialize_session(session)
                    for session in self.session_manager.list_sessions().values()
                ]
            }

        @self.mcp.resource("buse://session/{instance_id}")
        def session_detail(instance_id: str) -> Dict[str, str]:
            session = self._load_session(instance_id)
            return self._serialize_session(session)

    def _register_tools(self) -> None:
        if self.tool_handler:

            @self.mcp.tool()
            async def navigate(
                instance_id: str, url: str, new_tab: bool = False
            ) -> Union[str, Dict[str, Any]]:
                """Navigate to a URL."""
                assert self.tool_handler
                return await self.tool_handler(
                    instance_id, "navigate", url=url, new_tab=new_tab
                )

            @self.mcp.tool()
            async def click(
                instance_id: str,
                index: Optional[int] = None,
                element_id: Optional[Union[str, dict]] = None,
                element_class: Optional[Union[str, dict]] = None,
                x: Optional[int] = None,
                y: Optional[int] = None,
            ) -> Union[str, Dict[str, Any]]:
                """Click an element by index, ID, class, or coordinates."""
                assert self.tool_handler
                element_id = _coerce_optional_str(element_id)
                element_class = _coerce_optional_str(element_class)
                if (x is None) != (y is None):
                    raise ValueError("Provide both x and y for coordinate clicks.")
                if (
                    index is None
                    and x is None
                    and y is None
                    and element_id is None
                    and element_class is None
                ):
                    raise ValueError(
                        "Provide an index, element_id/element_class, or x/y for clicks."
                    )
                return await self.tool_handler(
                    instance_id,
                    "click",
                    index=index,
                    element_id=element_id,
                    element_class=element_class,
                    coordinate_x=x,
                    coordinate_y=y,
                )

            @self.mcp.tool()
            async def input_text(
                instance_id: str,
                text: str,
                index: Optional[int] = None,
                element_id: Optional[Union[str, dict]] = None,
                element_class: Optional[Union[str, dict]] = None,
            ) -> Union[str, Dict[str, Any]]:
                """Input text into a field."""
                assert self.tool_handler
                element_id = _coerce_optional_str(element_id)
                element_class = _coerce_optional_str(element_class)
                if index is None and element_id is None and element_class is None:
                    raise ValueError(
                        "Provide an index or element_id/element_class for input."
                    )
                return await self.tool_handler(
                    instance_id,
                    "input",
                    text=text,
                    index=index,
                    element_id=element_id,
                    element_class=element_class,
                )

            @self.mcp.tool()
            async def send_keys(
                instance_id: str,
                keys: str,
                index: Optional[int] = None,
                element_id: Optional[Union[str, dict]] = None,
                element_class: Optional[Union[str, dict]] = None,
            ) -> Union[str, Dict[str, Any]]:
                """Send keys (e.g. 'Enter', 'Tab') to the browser or a specific element."""
                assert self.tool_handler
                element_id = _coerce_optional_str(element_id)
                element_class = _coerce_optional_str(element_class)
                return await self.tool_handler(
                    instance_id,
                    "send_keys",
                    keys=keys,
                    index=index,
                    element_id=element_id,
                    element_class=element_class,
                )

            @self.mcp.tool()
            async def scroll(
                instance_id: str,
                pages: float = 1.0,
                down: bool = True,
                index: Optional[int] = None,
            ) -> Union[str, Dict[str, Any]]:
                """Scroll the page or an element."""
                assert self.tool_handler
                return await self.tool_handler(
                    instance_id, "scroll", pages=pages, down=down, index=index
                )

            @self.mcp.tool()
            async def switch_tab(
                instance_id: str, tab_id: str
            ) -> Union[str, Dict[str, Any]]:
                """Switch to a tab by its 4-char ID."""
                assert self.tool_handler
                return await self.tool_handler(instance_id, "switch", tab_id=tab_id)

            @self.mcp.tool()
            async def close_tab(
                instance_id: str, tab_id: str
            ) -> Union[str, Dict[str, Any]]:
                """Close a tab by its 4-char ID."""
                assert self.tool_handler
                return await self.tool_handler(instance_id, "close", tab_id=tab_id)

            @self.mcp.tool()
            async def search(
                instance_id: str, query: str, engine: str = "google"
            ) -> Union[str, Dict[str, Any]]:
                """Search the web (google, bing, duckduckgo)."""
                assert self.tool_handler
                return await self.tool_handler(
                    instance_id, "search", query=query, engine=engine
                )

            @self.mcp.tool()
            async def upload_file(
                instance_id: str, index: int, path: str
            ) -> Union[str, Dict[str, Any]]:
                """Upload a file to a specific file input element."""
                assert self.tool_handler
                return await self.tool_handler(
                    instance_id, "upload_file", index=index, path=path
                )

            @self.mcp.tool()
            async def find_text(
                instance_id: str, text: str
            ) -> Union[str, Dict[str, Any]]:
                """Find and scroll to specific text on the page."""
                assert self.tool_handler
                return await self.tool_handler(instance_id, "find_text", text=text)

            @self.mcp.tool()
            async def dropdown_options(
                instance_id: str,
                index: Optional[int] = None,
                element_id: Optional[Union[str, dict]] = None,
                element_class: Optional[Union[str, dict]] = None,
            ) -> Union[str, Dict[str, Any]]:
                """Get options from a dropdown element."""
                assert self.tool_handler
                element_id = _coerce_optional_str(element_id)
                element_class = _coerce_optional_str(element_class)
                if index is None and element_id is None and element_class is None:
                    raise ValueError(
                        "Provide an index or element_id/element_class for dropdown options."
                    )
                return await self.tool_handler(
                    instance_id,
                    "dropdown_options",
                    index=index,
                    element_id=element_id,
                    element_class=element_class,
                )

            @self.mcp.tool()
            async def select_dropdown(
                instance_id: str,
                text: str,
                index: Optional[int] = None,
                element_id: Optional[Union[str, dict]] = None,
                element_class: Optional[Union[str, dict]] = None,
            ) -> Union[str, Dict[str, Any]]:
                """Select an option in a dropdown by text."""
                assert self.tool_handler
                element_id = _coerce_optional_str(element_id)
                element_class = _coerce_optional_str(element_class)
                if index is None and element_id is None and element_class is None:
                    raise ValueError(
                        "Provide an index or element_id/element_class for dropdown selection."
                    )
                return await self.tool_handler(
                    instance_id,
                    "select_dropdown",
                    text=text,
                    index=index,
                    element_id=element_id,
                    element_class=element_class,
                )

            @self.mcp.tool()
            async def go_back(instance_id: str) -> Union[str, Dict[str, Any]]:
                """Go back in browser history."""
                assert self.tool_handler
                return await self.tool_handler(instance_id, "go_back")

            @self.mcp.tool()
            async def hover(
                instance_id: str,
                index: Optional[int] = None,
                element_id: Optional[Union[str, dict]] = None,
                element_class: Optional[Union[str, dict]] = None,
            ) -> Union[str, Dict[str, Any]]:
                """Hover over an element."""
                assert self.tool_handler
                element_id = _coerce_optional_str(element_id)
                element_class = _coerce_optional_str(element_class)
                if index is None and element_id is None and element_class is None:
                    raise ValueError(
                        "Provide an index or element_id/element_class for hover."
                    )
                return await self.tool_handler(
                    instance_id,
                    "hover",
                    index=index,
                    element_id=element_id,
                    element_class=element_class,
                )

            @self.mcp.tool()
            async def refresh(instance_id: str) -> Union[str, Dict[str, Any]]:
                """Refresh the current page."""
                assert self.tool_handler
                return await self.tool_handler(
                    instance_id,
                    "evaluate",
                    code="window.location.reload()",
                    action_label="refresh",
                )

            @self.mcp.tool()
            async def wait(
                instance_id: str, seconds: float
            ) -> Union[str, Dict[str, Any]]:
                """Wait for a specified number of seconds."""
                assert self.tool_handler
                return await self.tool_handler(instance_id, "wait", seconds=seconds)

            @self.mcp.tool()
            async def save_state(
                instance_id: str, path: str
            ) -> Union[str, Dict[str, Any]]:
                """Save current browser state (cookies/storage) to a JSON file."""
                assert self.tool_handler
                return await self.tool_handler(instance_id, "save_state", path=path)

            @self.mcp.tool()
            async def extract(
                instance_id: str, query: str
            ) -> Union[str, Dict[str, Any]]:
                """Extract structured data from the page using an LLM."""
                assert self.tool_handler
                return await self.tool_handler(instance_id, "extract", query=query)

            @self.mcp.tool()
            async def evaluate(
                instance_id: str, code: str
            ) -> Union[str, Dict[str, Any]]:
                """Execute custom JavaScript on the page."""
                assert self.tool_handler
                return await self.tool_handler(instance_id, "evaluate", code=code)

            @self.mcp.tool()
            async def stop_session(instance_id: str) -> Union[str, Dict[str, Any]]:
                """Stop and close a specific browser session."""
                assert self.tool_handler
                return await self.tool_handler(instance_id, "stop")

            @self.mcp.tool()
            async def start_session(instance_id: str) -> Union[str, Dict[str, Any]]:
                """Start and initialize a new browser session with the given ID."""
                assert self.tool_handler
                return await self.tool_handler(instance_id, "start")

        if self.observation_handler:

            @self.mcp.tool()
            async def observe(
                instance_id: str,
                screenshot: bool = False,
                no_dom: bool = False,
                omniparser: bool = False,
            ) -> Union[str, Dict[str, Any]]:
                """Observe the current state of the browser (DOM, tabs, screenshot, OmniParser)."""
                assert self.observation_handler
                return await self.observation_handler(
                    instance_id,
                    screenshot=screenshot,
                    no_dom=no_dom,
                    omniparser=omniparser,
                )

    def _serialize_session(self, session: SessionInfo) -> Dict[str, str]:
        return SessionSummary(
            instance_id=session.instance_id,
            cdp_url=session.cdp_url,
            user_data_dir=session.user_data_dir,
        ).to_dict()

    def _load_session(self, instance_id: str) -> SessionInfo:
        session = self.session_manager.get_session(instance_id)
        if session is None:
            raise ValueError(f"Instance {instance_id} not found")
        return session

    def run(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        transport: str = "streamable-http",
    ) -> None:
        """Start the FastMCP server. This call blocks until the process is interrupted."""

        if transport == "stdio":
            import sys

            sys.stderr.write("INFO: Buse MCP server starting in 'stdio' mode...\n")
            sys.stderr.flush()
            self.mcp.run(transport="stdio")
            return

        if transport == "streamable-http":
            app = self.mcp.streamable_http_app()
        elif transport == "sse":
            app = self.mcp.sse_app()
        else:
            raise ValueError(
                "Unsupported transport. Choose 'stdio', 'streamable-http' or 'sse'."
            )

        app = _wrap_with_access_guard(
            app,
            allow_remote=self.allow_remote,
            auth_token=self.auth_token,
        )
        config = Config(app=app, host=host, port=port)
        server = Server(config)
        server.run()
