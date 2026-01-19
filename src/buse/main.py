import asyncio
import base64
import json
import logging
import os
import sys
import time
import httpx
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Any

import typer

from .models import ActionResult, Observation, TabInfo
from .mcp_server import BuseMCPServer
from .session import SessionManager
from .utils import OutputFormat, handle_errors, output_data, state, _serialize

os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["BROWSER_USE_LOGGING_LEVEL"] = "critical"
os.environ["CDP_LOGGING_LEVEL"] = "critical"
os.environ["BROWSER_USE_SETUP_LOGGING"] = "true"

logging.getLogger("browser_use").setLevel(logging.CRITICAL)
logging.getLogger("bubus").setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)
logging.disable(logging.CRITICAL)

session_manager = SessionManager()
instance_app = typer.Typer(no_args_is_help=True)


async def get_observation(
    instance_id: str,
    screenshot: bool = False,
    path: Optional[str] = None,
    omniparser: bool = False,
    no_dom: bool = False,
) -> dict:
    if omniparser and not os.getenv("BUSE_OMNIPARSER_URL"):
        raise ValueError(
            "BUSE_OMNIPARSER_URL environment variable is required when using --omniparser."
        )
    quality_override = _parse_image_quality(None)
    endpoint = None
    if omniparser:
        endpoint = _normalize_omniparser_endpoint(os.getenv("BUSE_OMNIPARSER_URL", ""))
        if _should_probe_omniparser(endpoint):
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(f"{endpoint}/probe/")
                    resp.raise_for_status()
                _mark_omniparser_probe(endpoint)
            except Exception as e:
                raise RuntimeError(f"OmniParser probe failed: {e}")

    start_session = time.perf_counter()
    session_info = session_manager.get_session(instance_id)
    get_session_ms = (time.perf_counter() - start_session) * 1000.0
    if not session_info:
        await _stop_cached_browser_session(instance_id)
        raise ValueError(f"Instance {instance_id} not found.")

    start_browser = time.perf_counter()
    browser_session, _ = await _get_browser_session(instance_id, session_info)
    get_browser_session_ms = (time.perf_counter() - start_browser) * 1000.0

    try:
        profile_extra: dict[str, float] = {}
        start_state = time.perf_counter()
        include_screenshot = screenshot

        try:
            if no_dom:
                from browser_use.browser.events import BrowserStateRequestEvent

                event = browser_session.event_bus.dispatch(
                    BrowserStateRequestEvent(
                        include_dom=False, include_screenshot=include_screenshot
                    )
                )
                state_summary = await asyncio.wait_for(
                    event.event_result(raise_if_none=True, raise_if_any=True),
                    timeout=30.0,
                )
            else:
                state_summary = await asyncio.wait_for(
                    browser_session.get_browser_state_summary(
                        include_screenshot=include_screenshot
                    ),
                    timeout=30.0,
                )
        except Exception as e:
            if "timeout" in str(e).lower() and include_screenshot:
                if no_dom:
                    from browser_use.browser.events import BrowserStateRequestEvent

                    event = browser_session.event_bus.dispatch(
                        BrowserStateRequestEvent(
                            include_dom=False, include_screenshot=include_screenshot
                        )
                    )
                    state_summary = await asyncio.wait_for(
                        event.event_result(raise_if_none=True, raise_if_any=True),
                        timeout=30.0,
                    )
                else:
                    state_summary = await asyncio.wait_for(
                        browser_session.get_browser_state_summary(
                            include_screenshot=include_screenshot
                        ),
                        timeout=30.0,
                    )
            else:
                raise

        state_ms = (time.perf_counter() - start_state) * 1000.0
        if not no_dom:
            _selector_cache[instance_id] = time.time()
        focused_id = browser_session.agent_focus_target_id
        screenshot_data = state_summary.screenshot
        cdp_screenshot_error = None
        needs_screenshot = screenshot or omniparser
        omniparser_shot_dir = None
        omniparser_image_data = None
        som_path = None

        viewport_data = None
        try:
            cdp_session = await browser_session.get_or_create_cdp_session()
            viewport_res = await asyncio.wait_for(
                cdp_session.cdp_client.send.Runtime.evaluate(
                    params={
                        "expression": "({width: window.innerWidth, height: window.innerHeight, device_pixel_ratio: window.devicePixelRatio})",
                        "returnByValue": True,
                    },
                    session_id=cdp_session.session_id,
                ),
                timeout=5.0,
            )
            viewport_data = viewport_res.get("result", {}).get("value")
            if needs_screenshot and not screenshot_data:
                capture = await asyncio.wait_for(
                    cdp_session.cdp_client.send.Page.captureScreenshot(
                        params={"format": "png"},
                        session_id=cdp_session.session_id,
                    ),
                    timeout=15.0,
                )
                capture_data = (
                    capture.get("data") if isinstance(capture, dict) else None
                )
                if isinstance(capture_data, str) and capture_data:
                    screenshot_data = capture_data
                else:
                    cdp_screenshot_error = "CDP capture returned no data"
        except Exception as e:
            if not screenshot_data and needs_screenshot:
                raise RuntimeError(f"CDP access failed: {e}")

        viewport = None
        if viewport_data:
            from .models import ViewportInfo

            viewport = ViewportInfo(**viewport_data)

        omniparser_quality = quality_override if quality_override is not None else 95
        som_quality = quality_override if quality_override is not None else 75
        visual_analysis = None

        if omniparser:
            omniparser_start = time.perf_counter() if state.profile else None
            if not screenshot_data:
                raise RuntimeError(
                    f"Missing screenshot for OmniParser. CDP error: {cdp_screenshot_error}"
                )
            if not viewport:
                raise RuntimeError("Missing viewport for OmniParser")

            omniparser_shot_dir = Path(session_info.user_data_dir) / "screenshots"
            if path:
                p = Path(path)
                if p.is_dir() or not p.suffix:
                    omniparser_shot_dir = p
                else:
                    omniparser_shot_dir = p.parent
            if omniparser_shot_dir:
                omniparser_shot_dir.mkdir(exist_ok=True, parents=True)

            from .vision import VisionClient
            from .utils import downscale_image

            client = VisionClient(server_url=endpoint)
            omniparser_image_data = downscale_image(
                screenshot_data, quality=omniparser_quality
            )
            analysis, som_image_base64 = await asyncio.wait_for(
                client.analyze(omniparser_image_data, viewport), timeout=60.0
            )

            if not analysis.elements:
                raise RuntimeError("OmniParser returned no elements")

            if som_image_base64 and omniparser_shot_dir:
                som_image_base64 = downscale_image(
                    som_image_base64, quality=som_quality
                )
                som_path = str(omniparser_shot_dir / "image_som.jpg")
                client.save_som_image(som_image_base64, som_path)
                analysis.som_image_path = som_path

            visual_analysis = analysis
            if omniparser_start is not None:
                profile_extra["omniparser_total_ms"] = (
                    time.perf_counter() - omniparser_start
                ) * 1000.0

        start_tabs = time.perf_counter()
        tabs = [
            TabInfo(id=t.target_id, title=t.title, url=t.url)
            for t in await browser_session.get_tabs()
        ]
        tabs_ms = (time.perf_counter() - start_tabs) * 1000.0

        screenshot_path = None
        screenshot_ms = 0.0

        if (screenshot or omniparser) and screenshot_data:
            start_shot = time.perf_counter()
            write_path = None
            if omniparser and omniparser_shot_dir:
                shot_data = omniparser_image_data

                write_path = str(omniparser_shot_dir / "image.jpg")

                screenshot_path = som_path or write_path
            else:
                shot_data = screenshot_data
                ext = "png"
                if not path:
                    shot_dir = Path(session_info.user_data_dir) / "screenshots"
                    shot_dir.mkdir(exist_ok=True)
                    screenshot_path = str(shot_dir / f"last_state.{ext}")
                else:
                    p = Path(path)
                    if p.is_dir():
                        p.mkdir(exist_ok=True, parents=True)
                        screenshot_path = str(p / f"last_state.{ext}")
                    else:
                        if p.parent:
                            p.parent.mkdir(exist_ok=True, parents=True)
                        screenshot_path = str(p)
                write_path = screenshot_path

            if write_path and shot_data:
                with open(write_path, "wb") as f:
                    f.write(base64.b64decode(shot_data))
                screenshot_ms = (time.perf_counter() - start_shot) * 1000.0

        obs = Observation(
            session_id=instance_id,
            url=state_summary.url,
            title=state_summary.title,
            visual_analysis=visual_analysis,
            tabs=tabs,
            viewport=viewport,
            screenshot_path=screenshot_path,
            dom_minified=(
                ""
                if no_dom
                else state_summary.dom_state.llm_representation()
                if state_summary.dom_state
                else ""
            ),
        )

        data = obs.model_dump()
        data["focused_tab_id"] = focused_id
        if state.profile:
            data["profile"] = {
                "get_session_ms": get_session_ms,
                "get_browser_session_ms": get_browser_session_ms,
                "get_state_ms": state_ms,
                "get_tabs_ms": tabs_ms,
                "write_screenshot_ms": screenshot_ms,
                **profile_extra,
            }

        return data

    finally:
        if not _should_keep_session():
            await _stop_cached_browser_session(instance_id)


async def run_save_state(instance_id: str, path: str) -> dict:
    session_info = session_manager.get_session(instance_id)
    if session_info is None:
        await _stop_cached_browser_session(instance_id)
        raise ValueError(f"Instance {instance_id} not found.")

    browser_session = BrowserSession(cdp_url=session_info.cdp_url)
    await browser_session.start()
    try:
        state_data = await browser_session.export_storage_state(output_path=path)
        return {
            "success": True,
            "path": path,
            "cookies_count": len(state_data.get("cookies", [])),
        }
    finally:
        try:
            await asyncio.wait_for(browser_session.stop(), timeout=10.0)
        except Exception:
            pass


async def run_stop(instance_id: str) -> dict:
    if session_manager.get_session(instance_id) is None:
        await _stop_cached_browser_session(instance_id)
        raise ValueError(f"Instance {instance_id} not found.")
    await _stop_cached_browser_session(instance_id)
    session_manager.stop_session(instance_id)
    return {"message": f"Stopped {instance_id}", "success": True}


async def run_start(instance_id: str) -> dict:
    existing = session_manager.get_session(instance_id)
    if existing:
        return {
            "message": f"Instance {instance_id} already running",
            "already_running": True,
            "success": True,
        }
    session_manager.start_session(instance_id)
    return {
        "message": f"Initialized {instance_id}",
        "already_running": False,
        "success": True,
    }


def _parse_bool_env(name: str) -> Optional[bool]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    return raw.lower() in {"1", "true", "yes", "on"}


server_app = typer.Typer(no_args_is_help=False)


def _format_mcp_output(result: Any) -> Any:
    dumped = _serialize(result)
    if state.format == OutputFormat.toon:
        import toon_format as toon

        return toon.encode(dumped)
    return dumped


def _make_mcp_tool_handler():
    async def tool_handler(instance_id: str, action_name: str, **kwargs):
        action_label = kwargs.pop("action_label", None)
        if action_name == "save_state":
            result = await run_save_state(instance_id, **kwargs)
            return _format_mcp_output(result)
        if action_name == "stop":
            result = await run_stop(instance_id)
            return _format_mcp_output(result)
        if action_name == "start":
            result = await run_start(instance_id)
            return _format_mcp_output(result)
        result = await execute_tool(
            instance_id,
            action_name,
            kwargs,
            return_result=True,
            action_label=action_label,
        )
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        return _format_mcp_output(result)

    return tool_handler


def _make_mcp_observation_handler():
    async def observation_handler(
        instance_id: str,
        screenshot: bool = False,
        no_dom: bool = False,
        omniparser: bool = False,
    ):
        result = await get_observation(
            instance_id,
            screenshot=screenshot,
            no_dom=no_dom,
            omniparser=omniparser,
        )
        return _format_mcp_output(result)

    return observation_handler


@server_app.callback(invoke_without_command=True)
def run_mcp_server(
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Host/address for the MCP server."
    ),
    port: int = typer.Option(
        8000, "--port", help="Port for the MCP server (default: 8000)."
    ),
    transport: str = typer.Option(
        "streamable-http",
        "--transport",
        help="MCP transport (stdio, streamable-http, or sse).",
    ),
    name: str = typer.Option("buse", "--name", help="Name reported by the MCP server."),
    stateless: bool = typer.Option(
        True,
        "--stateless/--stateful",
        help="Run the MCP server as stateless HTTP (default: stateless).",
    ),
    json_response: bool = typer.Option(
        True,
        "--json-response/--no-json-response",
        help="Wrap responses as JSON (default: true).",
    ),
    allow_remote: bool = typer.Option(
        _parse_bool_env("BUSE_MCP_ALLOW_REMOTE") or False,
        "--allow-remote",
        help="Permit non-local clients (default: local-only).",
    ),
    auth_token: Optional[str] = typer.Option(
        os.getenv("BUSE_MCP_AUTH_TOKEN"),
        "--auth-token",
        help="Require a Bearer or X-Buse-Token header for MCP access.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.json,
        "--format",
        "-f",
        help="Output format for tool results (json or toon).",
    ),
):
    """Run a minimal MCP server that exposes active buse sessions."""
    state.format = output_format

    server = BuseMCPServer(
        session_manager,
        server_name=name,
        stateless_http=stateless,
        json_response=json_response,
        allow_remote=allow_remote,
        auth_token=auth_token,
        tool_handler=_make_mcp_tool_handler(),
        observation_handler=_make_mcp_observation_handler(),
    )
    server.run(host=host, port=port, transport=transport)


_browser_sessions = {}
_file_systems = {}
_controllers = {}
_selector_cache = {}


def _get_selector_cache_ttl_seconds() -> float:
    raw = os.getenv("BUSE_SELECTOR_CACHE_TTL", "").strip()
    if not raw:
        return 0.0
    try:
        ttl = float(raw)
    except ValueError:
        return 0.0
    return ttl if ttl > 0 else 0.0


def _parse_image_quality(quality: Optional[int]) -> Optional[int]:
    if quality is None:
        raw = os.getenv("BUSE_IMAGE_QUALITY", "").strip()
        if raw:
            try:
                quality = int(raw)
            except ValueError as exc:
                raise typer.BadParameter(
                    "BUSE_IMAGE_QUALITY must be an integer between 1 and 100."
                ) from exc
    if quality is None:
        return None
    if quality < 1 or quality > 100:
        raise typer.BadParameter("Image quality must be between 1 and 100.")
    return quality


_OMNIPARSER_PROBE_TTL_SECONDS = 600.0
_omniparser_probe_cache: dict[str, float] = {}


def _normalize_omniparser_endpoint(endpoint: str) -> str:
    return endpoint.strip().rstrip("/")


def _settings_path() -> Path:
    return session_manager.config_dir / "settings.json"


def _load_settings() -> dict:
    path = _settings_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_settings(data: dict) -> None:
    path = _settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _load_omniparser_probe_cache() -> None:
    if _omniparser_probe_cache:
        return
    data = _load_settings()
    probe_cache = data.get("omniparser_probe", {})
    if not isinstance(probe_cache, dict):
        return
    for key, value in probe_cache.items():
        timestamp = value
        if isinstance(value, dict):
            timestamp = value.get("timestamp")
        if isinstance(timestamp, (int, float)):
            _omniparser_probe_cache[str(key)] = float(timestamp)


def _save_omniparser_probe_cache() -> None:
    data = _load_settings()
    data["omniparser_probe"] = _omniparser_probe_cache
    _save_settings(data)


def _should_probe_omniparser(endpoint: str) -> bool:
    _load_omniparser_probe_cache()
    last_probe = _omniparser_probe_cache.get(endpoint)
    if not isinstance(last_probe, (int, float)):
        return True
    return (time.time() - last_probe) >= _OMNIPARSER_PROBE_TTL_SECONDS


def _mark_omniparser_probe(endpoint: str) -> None:
    _omniparser_probe_cache[endpoint] = time.time()
    _save_omniparser_probe_cache()


_SEND_KEYS_NAV_KEYS = [
    "Backspace",
    "Tab",
    "Enter",
    "Escape",
    "Space",
    "PageUp",
    "PageDown",
    "Home",
    "End",
    "ArrowLeft",
    "ArrowUp",
    "ArrowRight",
    "ArrowDown",
    "Insert",
    "Delete",
]
_SEND_KEYS_MODIFIER_KEYS = [
    "Shift",
    "ShiftLeft",
    "ShiftRight",
    "Control",
    "ControlLeft",
    "ControlRight",
    "Alt",
    "AltLeft",
    "AltRight",
    "Meta",
    "MetaLeft",
    "MetaRight",
]
_SEND_KEYS_FUNCTION_KEYS = [f"F{num}" for num in range(1, 25)]
_SEND_KEYS_NUMPAD_KEYS = (
    ["NumLock"]
    + [f"Numpad{num}" for num in range(10)]
    + [
        "NumpadMultiply",
        "NumpadAdd",
        "NumpadSubtract",
        "NumpadDecimal",
        "NumpadDivide",
    ]
)
_SEND_KEYS_LOCK_KEYS = ["CapsLock", "ScrollLock"]
_SEND_KEYS_PUNCTUATION_KEYS = [
    "Semicolon",
    "Equal",
    "Comma",
    "Minus",
    "Period",
    "Slash",
    "Backquote",
    "BracketLeft",
    "Backslash",
    "BracketRight",
    "Quote",
]
_SEND_KEYS_MEDIA_KEYS = [
    "AudioVolumeMute",
    "AudioVolumeDown",
    "AudioVolumeUp",
    "MediaTrackNext",
    "MediaTrackPrevious",
    "MediaStop",
    "MediaPlayPause",
    "BrowserBack",
    "BrowserForward",
    "BrowserRefresh",
    "BrowserStop",
    "BrowserSearch",
    "BrowserFavorites",
    "BrowserHome",
]
_SEND_KEYS_OTHER_KEYS = [
    "Clear",
    "Pause",
    "Select",
    "Print",
    "Execute",
    "PrintScreen",
    "Help",
    "ContextMenu",
]
_SEND_KEYS_SOLO_KEYS = [
    "Enter",
    "Tab",
    "Delete",
    "Backspace",
    "Escape",
    "ArrowUp",
    "ArrowDown",
    "ArrowLeft",
    "ArrowRight",
    "PageUp",
    "PageDown",
    "Home",
    "End",
    "Control",
    "Alt",
    "Meta",
    "Shift",
] + [f"F{num}" for num in range(1, 13)]
_SEND_KEYS_ALIAS_LINES = [
    "ctrl/control -> Control",
    "option -> Alt",
    "cmd/command/meta -> Meta",
    "esc/escape -> Escape",
    "return -> Enter",
    "tab -> Tab",
    "delete/backspace -> Delete/Backspace",
    'space -> " "',
    "up/down/left/right -> ArrowUp/ArrowDown/ArrowLeft/ArrowRight",
    "pageup/pagedown -> PageUp/PageDown",
    "home/end -> Home/End",
]
_SEND_KEYS_ALIAS_MAP = {
    "ctrl": "Control",
    "control": "Control",
    "option": "Alt",
    "alt": "Alt",
    "cmd": "Meta",
    "command": "Meta",
    "meta": "Meta",
    "shift": "Shift",
    "enter": "Enter",
    "return": "Enter",
    "tab": "Tab",
    "delete": "Delete",
    "backspace": "Backspace",
    "escape": "Escape",
    "esc": "Escape",
    "space": "Space",
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "pageup": "PageUp",
    "pagedown": "PageDown",
    "home": "Home",
    "end": "End",
}
_SEND_KEYS_RESERVED = set(
    _SEND_KEYS_NAV_KEYS
    + _SEND_KEYS_MODIFIER_KEYS
    + _SEND_KEYS_FUNCTION_KEYS
    + _SEND_KEYS_NUMPAD_KEYS
    + _SEND_KEYS_LOCK_KEYS
    + _SEND_KEYS_PUNCTUATION_KEYS
    + _SEND_KEYS_MEDIA_KEYS
    + _SEND_KEYS_OTHER_KEYS
)
_SEND_KEYS_RESERVED_LOWER = {key.lower() for key in _SEND_KEYS_RESERVED}


def _format_send_keys_help() -> str:
    sections = [
        ("Navigation", _SEND_KEYS_NAV_KEYS),
        ("Modifiers", _SEND_KEYS_MODIFIER_KEYS),
        ("Function", _SEND_KEYS_FUNCTION_KEYS),
        ("Numpad", _SEND_KEYS_NUMPAD_KEYS),
        ("Locks", _SEND_KEYS_LOCK_KEYS),
        ("Punctuation", _SEND_KEYS_PUNCTUATION_KEYS),
        ("Media/Browser", _SEND_KEYS_MEDIA_KEYS),
        ("Other", _SEND_KEYS_OTHER_KEYS),
    ]
    lines = ["Special keys (sent as key events when used alone):"]
    lines.append(f"  {', '.join(_SEND_KEYS_SOLO_KEYS)}")
    lines.append("  Everything else is typed as text unless used in a combo.")
    lines.append("")
    lines.append("Named keys (case-sensitive):")
    for label, keys in sections:
        lines.append(f"  {label}: {', '.join(keys)}")
    lines.append("  Letters/digits: A-Z, 0-9 (single characters).")
    lines.append("  Punctuation literals also work: ; = , - . / ` [ \\ ] ' and space.")
    lines.append("")
    lines.append("Aliases (case-insensitive):")
    for alias_line in _SEND_KEYS_ALIAS_LINES:
        lines.append(f"  {alias_line}")
    lines.append("")
    lines.append("Combos:")
    lines.append(
        "  Use '+' to combine modifiers with a key, e.g. Control+L, Shift+Tab, Alt+ArrowLeft."
    )
    lines.append("")
    lines.append("Text:")
    lines.append("  Any other string is typed as text (wrap spaces in quotes).")
    return "\n".join(lines)


def _is_reserved_key_sequence(keys: Optional[str]) -> bool:
    if not isinstance(keys, str):
        return False
    cleaned = keys.strip()
    if not cleaned:
        return False
    if "+" in cleaned:
        return True
    normalized = _SEND_KEYS_ALIAS_MAP.get(cleaned.lower(), cleaned)
    return normalized.lower() in _SEND_KEYS_RESERVED_LOWER


def _should_keep_session() -> bool:
    return os.getenv("BUSE_KEEP_SESSION", "").lower() in {"1", "true", "yes"}


async def _stop_cached_browser_session(instance_id: str) -> None:
    browser_session = _browser_sessions.pop(instance_id, None)
    if browser_session is not None:
        try:
            await asyncio.wait_for(browser_session.stop(), timeout=10.0)
        except Exception:
            pass
    _file_systems.pop(instance_id, None)
    _controllers.pop(instance_id, None)
    _selector_cache.pop(instance_id, None)


from browser_use.browser import BrowserSession  # noqa: E402
from browser_use.filesystem.file_system import FileSystem  # noqa: E402


async def _get_browser_session(instance_id: str, session_info):
    browser_session = _browser_sessions.get(instance_id)
    file_system = _file_systems.get(instance_id)

    if browser_session is None:
        browser_session = BrowserSession(cdp_url=session_info.cdp_url)
        try:
            await asyncio.wait_for(browser_session.start(), timeout=30.0)
        except Exception:
            raise RuntimeError("Failed to start browser session (timeout or error)")
        _browser_sessions[instance_id] = browser_session

    if file_system is None:
        file_system = FileSystem(base_dir=session_info.user_data_dir)
        _file_systems[instance_id] = file_system

    return browser_session, file_system


async def _ensure_selector_map(
    browser_session, instance_id: str, force: bool = False
) -> None:
    now = time.time()
    last = _selector_cache.get(instance_id, 0.0)
    ttl_seconds = _get_selector_cache_ttl_seconds()
    if not force and ttl_seconds > 0 and now - last < ttl_seconds:
        return
    await browser_session.get_browser_state_summary(
        include_screenshot=False, cached=False
    )
    _selector_cache[instance_id] = time.time()


def _tab_present(tabs, tab_id: str) -> bool:
    tab_id_lower = tab_id.lower()
    return any(t.target_id.lower().endswith(tab_id_lower) for t in tabs)


def _normalize_tab_id(tab_id: str, tabs) -> tuple[str, bool]:
    short_id = tab_id[-4:]
    if _tab_present(tabs, short_id):
        return short_id, False
    match = next(
        (t for t in tabs if t.target_id.lower().startswith(short_id.lower())),
        None,
    )
    if match:
        return match.target_id[-4:], True
    return short_id, False


async def _dispatch_hover(browser_session, hover_index: int) -> None:
    selector_map = await browser_session.get_selector_map()
    element = selector_map.get(hover_index)
    if element is None or element.absolute_position is None:
        raise ValueError("Could not resolve element position for hover")
    bounds = element.absolute_position
    x = bounds.x + (bounds.width or 0) / 2
    y = bounds.y + (bounds.height or 0) / 2
    cdp_session = await browser_session.get_or_create_cdp_session()
    await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
        params={"type": "mouseMoved", "x": x, "y": y, "buttons": 0},
        session_id=cdp_session.session_id,
    )


async def _dispatch_focus_click(browser_session, element) -> None:
    if element is None or element.absolute_position is None:
        raise ValueError("Could not resolve element position for focus")
    bounds = element.absolute_position
    x = bounds.x + (bounds.width or 0) / 2
    y = bounds.y + (bounds.height or 0) / 2
    cdp_session = await browser_session.get_or_create_cdp_session()
    await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
        params={
            "type": "mousePressed",
            "x": x,
            "y": y,
            "button": "left",
            "clickCount": 1,
        },
        session_id=cdp_session.session_id,
    )
    await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
        params={
            "type": "mouseReleased",
            "x": x,
            "y": y,
            "button": "left",
            "clickCount": 1,
        },
        session_id=cdp_session.session_id,
    )


async def _focus_element(
    browser_session,
    instance_id: str,
    index: int,
    profiler: "Profiler",
) -> Optional[str]:
    start_focus = time.perf_counter()
    try:
        await _ensure_selector_map(browser_session, instance_id)
        selector_map = await browser_session.get_selector_map()
        element = selector_map.get(index)
        if element is None:
            return f"Element index {index} not available"
        cdp_session = await browser_session.get_or_create_cdp_session()
        try:
            await cdp_session.cdp_client.send.DOM.focus(
                params={"backendNodeId": element.backend_node_id},
                session_id=cdp_session.session_id,
            )
            return None
        except Exception:
            try:
                await _dispatch_focus_click(browser_session, element)
                return None
            except Exception as exc:
                return str(exc) or f"Failed to focus element index {index}"
    finally:
        profiler.mark("focus_ms", start_focus)


class Profiler:
    def __init__(self) -> None:
        self.data: dict[str, float] = {}

    @contextmanager
    def span(self, key: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.data[key] = (time.perf_counter() - start) * 1000.0

    def mark(self, key: str, start: float) -> None:
        self.data[key] = (time.perf_counter() - start) * 1000.0


class ResultEmitter:
    class EarlyExit(Exception):
        pass

    def __init__(
        self,
        defer_output: bool,
        params_for_hint: dict,
        profile: dict,
        suppress_output: bool = False,
    ) -> None:
        self.defer_output = defer_output
        self.suppress_output = suppress_output
        self.params_for_hint = params_for_hint
        self.profile = profile
        self.result_payload = None
        self.exit_code = 0

    def _emit(self, payload, code: int = 0) -> None:
        self.result_payload = payload
        self.exit_code = code
        if self.suppress_output:
            return
        if self.defer_output:
            return
        output_data(payload)
        if code:
            sys.exit(code)

    def emit_error(
        self, action: str, message: str, error_details: Optional[dict[str, Any]] = None
    ) -> None:
        if error_details is None:
            error_details = _build_error_details("execute_tool", action=action)
        payload = ActionResult(
            success=False,
            action=action,
            message=None,
            error=_augment_error(action, self.params_for_hint, message),
            error_details=error_details,
            extracted_content=None,
            profile=self.profile if state.profile else None,
        )
        self._emit(payload, code=1)

    def emit_success(self, action: str, message: str) -> None:
        payload = ActionResult(
            success=True,
            action=action,
            message=message,
            error=None,
            extracted_content=message,
            profile=self.profile if state.profile else None,
        )
        self._emit(payload, code=0)

    def emit_result(
        self,
        action: str,
        message: Optional[str],
        error: Optional[str],
        extracted_content,
        error_details: Optional[dict[str, Any]] = None,
    ) -> None:
        if error and error_details is None:
            error_details = _build_error_details("execute_tool", action=action)
        payload = ActionResult(
            success=not error,
            action=action,
            message=message,
            error=error,
            error_details=error_details if error else None,
            extracted_content=extracted_content if not error else None,
            profile=self.profile if state.profile else None,
        )
        self._emit(payload, code=1 if error else 0)

    def fail(
        self, action: str, message: str, error_details: Optional[dict[str, Any]] = None
    ) -> None:
        self.emit_error(action, message, error_details=error_details)
        if self.defer_output or self.suppress_output:
            raise ResultEmitter.EarlyExit()

    def finalize(self) -> None:
        if self.suppress_output:
            return
        if self.defer_output and isinstance(self.result_payload, ActionResult):
            self.result_payload.profile = self.profile if state.profile else None
        if self.defer_output and self.result_payload is not None:
            output_data(self.result_payload)
            if self.exit_code:
                sys.exit(self.exit_code)


async def _get_navigation_timings(browser_session) -> dict[str, float]:
    cdp_session = await browser_session.get_or_create_cdp_session()
    code = (
        "(function(){ "
        "const entries=performance.getEntriesByType('navigation');"
        "const entry=entries && entries[0];"
        "if(!entry){return null;}"
        "return {"
        "dom_content_loaded_ms: entry.domContentLoadedEventEnd,"
        "load_event_ms: entry.loadEventEnd,"
        "response_end_ms: entry.responseEnd,"
        "ttfb_ms: entry.responseStart"
        "};"
        "})()"
    )
    result = await cdp_session.cdp_client.send.Runtime.evaluate(
        params={"expression": code, "returnByValue": True},
        session_id=cdp_session.session_id,
    )
    value = result.get("result", {}).get("value", {})
    if not isinstance(value, dict):
        return {}
    return {
        k: float(v) for k, v in value.items() if isinstance(v, (int, float)) and v >= 0
    }


async def _try_js_fallback(
    action_name: str,
    selector: str,
    text: str,
    browser_session,
    profiler: Profiler,
) -> tuple[bool, Optional[str]]:
    start_fallback = time.perf_counter()
    start_cdp = time.perf_counter()
    cdp_session = await browser_session.get_or_create_cdp_session()
    profiler.mark("js_cdp_session_ms", start_cdp)
    code = (
        "(function(){ "
        "const selector=" + json.dumps(selector) + ";"
        "const findDeep=(root)=>{ "
        "const el=root.querySelector(selector);"
        "if(el){return el;}"
        "const nodes=root.querySelectorAll('*');"
        "for(const node of nodes){ "
        "if(node.shadowRoot){ "
        "const found=findDeep(node.shadowRoot);"
        "if(found){return found;}"
        "}"
        "}"
        "return null;"
        "};"
        "const el=findDeep(document);"
        "if(!el){return {error:'Element not found'};}"
        "const tag=el.tagName;"
    )
    if action_name == "click":
        code += "el.click();return {ok:true,tag:tag};"
    elif action_name == "hover":
        code += (
            "el.dispatchEvent(new MouseEvent('mouseover',{bubbles:true}));"
            "return {ok:true,tag:tag};"
        )
    else:
        code += (
            "const value=" + json.dumps(text) + ";"
            "if('value' in el){ "
            "el.focus();"
            "el.value=value;"
            "el.dispatchEvent(new Event('input',{bubbles:true}));"
            "el.dispatchEvent(new Event('change',{bubbles:true}));"
            "return {ok:true,tag:tag};"
            "}"
            "return {error:'Element not writable'};"
        )
    code += "})()"
    start_eval = time.perf_counter()
    result = await cdp_session.cdp_client.send.Runtime.evaluate(
        params={"expression": code, "returnByValue": True},
        session_id=cdp_session.session_id,
    )
    profiler.mark("js_eval_ms", start_eval)
    profiler.mark("js_fallback_ms", start_fallback)
    value = result.get("result", {}).get("value", {}) if result else {}
    if not value.get("ok"):
        return False, None
    if action_name == "input":
        return True, f"Typed '{text}'"
    if action_name == "click":
        return True, "Clicked element"
    return True, "Hovered element"


async def _resolve_index(
    browser_session,
    instance_id: str,
    element_id: Optional[str],
    element_class: Optional[str],
    profiler: Profiler,
) -> Optional[int]:
    start_resolve = time.perf_counter()
    resolved_index = None
    if element_id:
        start_resolve_id = time.perf_counter()
        resolved_index = await browser_session.get_index_by_id(element_id)
        profiler.mark("resolve_id_ms", start_resolve_id)
    if resolved_index is None and element_class:
        start_resolve_class = time.perf_counter()
        resolved_index = await browser_session.get_index_by_class(element_class)
        profiler.mark("resolve_class_ms", start_resolve_class)
    if resolved_index is None:
        start_selector = time.perf_counter()
        await _ensure_selector_map(browser_session, instance_id, force=True)
        profiler.mark("selector_map_refresh_ms", start_selector)
        if element_id:
            start_resolve_id = time.perf_counter()
            resolved_index = await browser_session.get_index_by_id(element_id)
            profiler.mark("resolve_id_refresh_ms", start_resolve_id)
        if resolved_index is None and element_class:
            start_resolve_class = time.perf_counter()
            resolved_index = await browser_session.get_index_by_class(element_class)
            profiler.mark("resolve_class_refresh_ms", start_resolve_class)
    profiler.mark("resolve_index_ms", start_resolve)
    return resolved_index


async def _try_dropdown_fallback(
    action_name: str,
    selector: Optional[str],
    text: str,
    browser_session,
    profiler: Profiler,
) -> tuple[bool, Optional[str], Optional[str]]:
    if not selector:
        return False, None, None
    start_cdp = time.perf_counter()
    cdp_session = await browser_session.get_or_create_cdp_session()
    profiler.mark("dropdown_cdp_session_ms", start_cdp)
    if action_name == "dropdown_options":
        code = (
            "(function(){ "
            "const sel=document.querySelector(" + json.dumps(selector) + ");"
            "let el=sel;"
            "if(el && el.tagName!=='SELECT'){el=el.querySelector('select')||el.closest('select');}"
            "if(!el||el.tagName!=='SELECT'){return {error:'Select element not found'};}"
            "const opts=[...el.options].map((o,i)=>({i,text:o.text,value:o.value,selected:o.selected}));"
            "return {id:el.id||'',name:el.name||'',options:opts};"
            "})()"
        )
        start_eval = time.perf_counter()
        result = await cdp_session.cdp_client.send.Runtime.evaluate(
            params={"expression": code, "returnByValue": True},
            session_id=cdp_session.session_id,
        )
        profiler.mark("dropdown_eval_ms", start_eval)
        value = result.get("result", {}).get("value", {}) if result else {}
        if value.get("error"):
            return True, value.get("error"), None
        lines = []
        for opt in value.get("options", []):
            suffix = " (selected)" if opt.get("selected") else ""
            lines.append(
                f'{opt.get("i")}: text="{opt.get("text")}", value="{opt.get("value")}"{suffix}'
            )
        msg = (
            f"Found select dropdown (Index: n/a, ID: {value.get('id')}, Name: {value.get('name')}):\n"
            + "\n".join(lines)
        )
        return True, None, msg
    if action_name == "select_dropdown":
        code = (
            "(function(){ "
            "const sel=document.querySelector(" + json.dumps(selector) + ");"
            "let el=sel;"
            "if(el && el.tagName!=='SELECT'){el=el.querySelector('select')||el.closest('select');}"
            "if(!el||el.tagName!=='SELECT'){return {error:'Select element not found'};}"
            "const target=" + json.dumps(text) + ";"
            "const opt=[...el.options].find(o=>o.text===target||o.value===target);"
            "if(!opt){return {error:'Option not found'};}"
            "el.value=opt.value;"
            "el.dispatchEvent(new Event('change',{bubbles:true}));"
            "return {text:opt.text,value:opt.value};"
            "})()"
        )
        start_eval = time.perf_counter()
        result = await cdp_session.cdp_client.send.Runtime.evaluate(
            params={"expression": code, "returnByValue": True},
            session_id=cdp_session.session_id,
        )
        profiler.mark("dropdown_eval_ms", start_eval)
        value = result.get("result", {}).get("value", {}) if result else {}
        if value.get("error"):
            return True, value.get("error"), None
        msg = f"Selected option: {value.get('text')} (value: {value.get('value')})"
        return True, None, msg
    return False, None, None


async def _verify_close_tab(
    browser_session,
    tab_id: str,
    tab_present_before: Optional[bool],
    profile: dict[str, float],
) -> Optional[str]:
    start_tabs = time.perf_counter()
    tabs_after = await browser_session.get_tabs()
    profile["tabs_after_ms"] = (time.perf_counter() - start_tabs) * 1000.0
    tab_present_after = _tab_present(tabs_after, tab_id)
    profile["tab_present_after"] = 1.0 if tab_present_after else 0.0
    if tab_present_before is False:
        return f"Tab #{tab_id} not found"
    if not tab_present_after:
        return None
    return f"Tab #{tab_id} still open"


def _augment_error(action_name: str, params: dict, error: str) -> str:
    hint = None
    message = error
    if action_name in {
        "click",
        "input",
        "dropdown_options",
        "select_dropdown",
        "hover",
    }:
        missing_index = params.get("index") is None
        missing_resolver = not (params.get("element_id") or params.get("element_class"))
        missing_coords = not (
            params.get("coordinate_x") is not None
            and params.get("coordinate_y") is not None
        )
        if missing_index and missing_resolver and missing_coords:
            hint = (
                "Provide an index or use --id/--class, or use --x/--y for coordinates."
            )
        elif missing_index and missing_resolver:
            hint = "Provide an index or use --id/--class (run observe to get indices)."
    if action_name == "click" and (
        "coordinate_x" in params or "coordinate_y" in params
    ):
        if params.get("coordinate_x") is None or params.get("coordinate_y") is None:
            hint = "Provide both --x and --y for coordinate clicks."
    if "Could not resolve element index" in error:
        hint = "Run `buse <id> observe` and use an index or pass the actual <select> --id/--class."
    lowered_error = message.lower()
    if "element index" in lowered_error and (
        "not available" in lowered_error
        or "not found in browser state" in lowered_error
    ):
        message = (
            message.replace(" - page may have changed.", "")
            .replace(" Try refreshing browser state.", "")
            .replace("Try refreshing browser state.", "")
            .strip()
        )
        message = f"{message} in the current DOM"
        hint = 'Run "buse <id> observe" to refresh indices.'
    if "element with index" in lowered_error and "does not exist" in lowered_error:
        hint = 'Run "buse <id> observe" to refresh indices.'
    if action_name in {"dropdown_options", "select_dropdown"}:
        if "Option not found" in error:
            hint = "Run dropdown-options and use the exact option text or value."
        elif "Select element not found" in error:
            hint = "Pass the actual <select> id/class or provide an index from observe."
        elif params.get("element_id") or params.get("element_class"):
            hint = "If a select is wrapped, pass the actual <select> id/class or use observe to get an index."
    if action_name in {"switch-tab", "close-tab"}:
        if not params.get("tab_id") or len(str(params.get("tab_id"))) < 4:
            hint = "Use the 4-char tab ID shown in observe tabs list."
    if action_name == "scroll":
        pages = params.get("pages")
        if pages is not None and pages <= 0:
            hint = "Use a positive --pages value (e.g., 1 or 0.5)."
    if action_name == "search" and "Unsupported search engine" in error:
        hint = "Use --engine duckduckgo|google|bing."
    if action_name == "navigate" and "site unavailable" in error:
        url = str(params.get("url", ""))
        if not url.startswith(("http://", "https://")):
            hint = "Include a scheme, e.g. https://example.com."
        else:
            hint = "Check the URL or network connectivity."
    if action_name == "evaluate" and (
        "JavaScript execution error" in error or "Failed to execute JavaScript" in error
    ):
        hint = "Wrap code in (function(){...})() and return a value."
    if action_name == "extract" and "API key" in error:
        hint = "Set OPENAI_API_KEY or export BUSE_EXTRACT_MODEL."
    if "seconds" in error and "integer" in error:
        hint = "Use whole seconds (e.g., `buse b1 wait 2`)."
    if "Instance" in error and "not found" in error:
        hint = "Run `buse <id>` first to start an instance or use `buse list`."
    if action_name == "send_keys" and hint is None:
        has_focus_target = any(
            params.get(key) is not None
            for key in ("index", "element_id", "element_class")
        )
        keys = params.get("keys")
        if (
            not has_focus_target
            and isinstance(keys, str)
            and keys
            and not _is_reserved_key_sequence(keys)
        ):
            hint = (
                "If you intended to type into a field, focus it with --index/--id/--class "
                "(otherwise no focus is needed)."
            )
    if hint:
        return f"{message}. {hint}"
    return message


def _build_error_details(
    stage: str,
    *,
    retryable: Optional[bool] = None,
    **context: Any,
) -> dict[str, Any]:
    details: dict[str, Any] = {"stage": stage}
    if retryable is not None:
        details["retryable"] = retryable
    if context:
        details["context"] = context
    return details


def _coerce_index_error(message: Optional[str]) -> Optional[str]:
    if not isinstance(message, str):
        return None
    lowered = message.lower()
    if "element index" in lowered and (
        "not available" in lowered or "not found in browser state" in lowered
    ):
        return message
    if "element with index" in lowered and "does not exist" in lowered:
        return message
    return None


def _output_error(
    action: str,
    params: dict,
    message: str,
    profile: Optional[dict[str, float]] = None,
    error_details: Optional[dict[str, Any]] = None,
):
    output_data(
        ActionResult(
            success=False,
            action=action,
            message=None,
            error=_augment_error(action, params, message),
            error_details=error_details,
            extracted_content=None,
            profile=profile if state.profile else None,
        )
    )


async def execute_tool(
    instance_id: str,
    action_name: str,
    params: dict,
    needs_selector_map: bool = False,
    action_label: Optional[str] = None,
    return_result: bool = False,
    **extra_kwargs,
):
    from browser_use import Controller

    params_for_hint = dict(params)
    label = action_label or action_name
    defer_output = state.profile
    profiler = Profiler()
    profile = profiler.data
    emitter = ResultEmitter(
        defer_output, params_for_hint, profile, suppress_output=return_result
    )
    start_total = time.perf_counter()
    with profiler.span("get_session_ms"):
        session_info = session_manager.get_session(instance_id)
    if session_info is None:
        await _stop_cached_browser_session(instance_id)
        emitter.fail(
            label,
            f"Instance {instance_id} not found.",
            error_details=_build_error_details(
                "session", retryable=False, instance_id=instance_id
            ),
        )

    profile["browser_session_cached"] = 1.0 if instance_id in _browser_sessions else 0.0
    profile["file_system_cached"] = 1.0 if instance_id in _file_systems else 0.0
    with profiler.span("get_browser_session_ms"):
        browser_session, file_system = await _get_browser_session(
            instance_id, session_info
        )

    try:
        if needs_selector_map:
            with profiler.span("selector_map_ms"):
                await _ensure_selector_map(browser_session, instance_id)

        element_id = params.pop("element_id", None)
        element_class = params.pop("element_class", None)
        if params.get("index") is None and (element_id or element_class):
            selector = None
            if element_id:
                selector = f"#{element_id}"
            elif element_class:
                selector = f".{element_class}"
            if selector and action_name in {"click", "input", "hover"}:
                text = params.get("text", "")
                handled, msg = await _try_js_fallback(
                    action_name,
                    selector,
                    text,
                    browser_session,
                    profiler,
                )
                if handled and msg:
                    profile["total_ms"] = (time.perf_counter() - start_total) * 1000.0
                    emitter.emit_success(action_name, msg)
                    if defer_output:
                        raise ResultEmitter.EarlyExit()
                    return
            resolved_index = await _resolve_index(
                browser_session,
                instance_id,
                element_id,
                element_class,
                profiler,
            )
            if resolved_index is None and action_name in {
                "dropdown_options",
                "select_dropdown",
            }:
                selector = None
                if element_id:
                    selector = f"#{element_id}"
                elif element_class:
                    selector = f".{element_class}"

                if selector:
                    text = params.get("text", "")
                    handled, error_msg, success_msg = await _try_dropdown_fallback(
                        action_name,
                        selector,
                        text,
                        browser_session,
                        profiler,
                    )
                    if handled and error_msg:
                        emitter.fail(action_name, error_msg)
                    if handled and success_msg:
                        profile["total_ms"] = (
                            time.perf_counter() - start_total
                        ) * 1000.0
                        emitter.emit_success(action_name, success_msg)
                        if defer_output:
                            raise ResultEmitter.EarlyExit()
                        return

            if resolved_index is None:
                error_msg = "Could not resolve element index"
                emitter.fail(
                    action_name,
                    f"{error_msg} (id={element_id}, class={element_class})",
                )
            params["index"] = resolved_index

        if action_name == "hover":
            hover_index = params.get("index")
            if hover_index is None:
                emitter.fail(action_name, "Hover requires an element index")
                return
            start_hover = time.perf_counter()
            try:
                await _dispatch_hover(browser_session, hover_index)
            except ValueError as exc:
                emitter.fail(action_name, str(exc))
            profile["hover_ms"] = (time.perf_counter() - start_hover) * 1000.0
            profile["total_ms"] = (time.perf_counter() - start_total) * 1000.0
            emitter.emit_success("hover", "Hovered element")
            if defer_output:
                raise ResultEmitter.EarlyExit()
            return

        if action_name == "send_keys":
            focus_index = params.pop("index", None)
            if focus_index is not None:
                focus_error = await _focus_element(
                    browser_session,
                    instance_id,
                    focus_index,
                    profiler,
                )
                if focus_error:
                    emitter.fail(action_name, focus_error)

        controller = _controllers.get(instance_id)
        profile["controller_cached"] = 1.0 if controller is not None else 0.0
        if controller is None:
            controller = Controller()
            _controllers[instance_id] = controller
        if action_name == "click" and (
            params.get("coordinate_x") is not None
            or params.get("coordinate_y") is not None
        ):
            controller.set_coordinate_clicking(True)
            profile["coordinate_clicking_enabled"] = 1.0
        tab_id = params.get("tab_id")
        tab_present_before = None
        tabs_snapshot: list = []
        if action_name in {"close", "switch"} and tab_id:
            start_tabs = time.perf_counter()
            tabs_snapshot = await browser_session.get_tabs()
            profiler.mark("tabs_lookup_ms", start_tabs)
            tab_id, matched_prefix = _normalize_tab_id(tab_id, tabs_snapshot)
            params["tab_id"] = tab_id
            if matched_prefix:
                profile["tab_id_matched_prefix"] = 1.0
        if action_name == "close" and tab_id:
            profile["tabs_before_ms"] = profile.get("tabs_lookup_ms", 0.0)
            tab_present_before = _tab_present(tabs_snapshot, tab_id)
            profile["tab_present_before"] = 1.0 if tab_present_before else 0.0
        start_action = time.perf_counter()
        if action_name == "upload_file" and "path" in params:
            extra_kwargs["available_file_paths"] = [params["path"]]
        result = await controller.registry.execute_action(
            action_name,
            params,
            browser_session=browser_session,
            file_system=file_system,
            **extra_kwargs,
        )
        profile["action_ms"] = (time.perf_counter() - start_action) * 1000.0
        error = result.error
        if action_name == "close" and tab_id and not error:
            error = await _verify_close_tab(
                browser_session,
                tab_id,
                tab_present_before,
                profile,
            )
        if state.profile and action_name == "navigate" and not error:
            try:
                nav_timings = await _get_navigation_timings(browser_session)
            except Exception:
                nav_timings = {}
            profile.update({f"nav_{k}": v for k, v in nav_timings.items()})
        profile["total_ms"] = (time.perf_counter() - start_total) * 1000.0
        message = result.extracted_content if not error else None
        if not error:
            coerced_error = _coerce_index_error(message)
            if coerced_error:
                error = coerced_error
                message = None
        if error:
            error = _augment_error(label, params_for_hint, error)
        if action_name == "close" and tab_id and not error:
            message = f"Closed tab #{tab_id}"
        emitter.emit_result(
            label,
            message,
            error,
            result.extracted_content,
        )
        if error:
            if defer_output:
                raise ResultEmitter.EarlyExit()

        if action_name in {"navigate", "go_back", "search"} or label == "refresh":
            _selector_cache.pop(instance_id, None)
    except ResultEmitter.EarlyExit:
        pass
    except Exception as e:
        profile["total_ms"] = (time.perf_counter() - start_total) * 1000.0
        emitter.fail(
            label,
            f"{label} failed: {type(e).__name__}: {e}",
            error_details=_build_error_details(
                "execute_tool",
                exception_type=type(e).__name__,
                action=label,
            ),
        )
    finally:
        if not _should_keep_session():
            cleanup_start = time.perf_counter()
            await _stop_cached_browser_session(instance_id)
            profile["cleanup_ms"] = (time.perf_counter() - cleanup_start) * 1000.0
    emitter.finalize()
    if return_result:
        return emitter.result_payload


@instance_app.command(
    help="Observe current state.",
    epilog="Example: buse b1 observe --screenshot",
)
def observe(
    ctx: typer.Context,
    screenshot: bool = typer.Option(False, "--screenshot", help="Take a screenshot."),
    path: Optional[str] = typer.Option(
        None, "--path", help="Custom path for screenshot."
    ),
    omniparser: bool = typer.Option(
        False, "--omniparser", help="Analyze page with OmniParser."
    ),
    no_dom: bool = typer.Option(
        False, "--no-dom", help="Skip DOM processing and omit dom_minified."
    ),
):
    if omniparser and not os.getenv("BUSE_OMNIPARSER_URL"):
        raise typer.BadParameter(
            "BUSE_OMNIPARSER_URL environment variable is required when using --omniparser.\n"
        )
    quality_override = _parse_image_quality(None)
    instance_id = ctx.obj["instance_id"]

    async def run():
        endpoint = None
        if omniparser:
            endpoint = _normalize_omniparser_endpoint(
                os.getenv("BUSE_OMNIPARSER_URL", "")
            )

            if _should_probe_omniparser(endpoint):
                try:
                    async with httpx.AsyncClient(timeout=3.0) as client:
                        resp = await client.get(f"{endpoint}/probe/")
                        resp.raise_for_status()
                    _mark_omniparser_probe(endpoint)
                except httpx.HTTPStatusError as e:
                    _output_error(
                        "observe",
                        {"omniparser": True},
                        f"OmniParser server at {endpoint} returned {e.response.status_code}",
                        error_details=_build_error_details(
                            "omniparser_probe",
                            retryable=e.response.status_code >= 500,
                            endpoint=endpoint,
                            status_code=e.response.status_code,
                        ),
                    )
                    sys.exit(1)
                except Exception as e:
                    _output_error(
                        "observe",
                        {"omniparser": True},
                        f"Cannot reach OmniParser at {endpoint}: {e}",
                        error_details=_build_error_details(
                            "omniparser_probe",
                            retryable=True,
                            endpoint=endpoint,
                            exception_type=type(e).__name__,
                        ),
                    )
                    sys.exit(1)

        start_session = time.perf_counter()
        session_info = session_manager.get_session(instance_id)
        get_session_ms = (time.perf_counter() - start_session) * 1000.0
        if not session_info:
            await _stop_cached_browser_session(instance_id)
            _output_error(
                "observe",
                {},
                f"Instance {instance_id} not found.",
                error_details=_build_error_details(
                    "session", retryable=False, instance_id=instance_id
                ),
            )
            sys.exit(1)
        assert session_info is not None
        start_browser = time.perf_counter()
        browser_session, _ = await _get_browser_session(instance_id, session_info)
        get_browser_session_ms = (time.perf_counter() - start_browser) * 1000.0

        try:
            profile_extra: dict[str, float] = {}
            start_state = time.perf_counter()
            include_screenshot = screenshot

            try:
                if no_dom:
                    from browser_use.browser.events import BrowserStateRequestEvent

                    event = browser_session.event_bus.dispatch(
                        BrowserStateRequestEvent(
                            include_dom=False, include_screenshot=include_screenshot
                        )
                    )
                    state_summary = await event.event_result(
                        raise_if_none=True, raise_if_any=True
                    )
                else:
                    state_summary = await browser_session.get_browser_state_summary(
                        include_screenshot=include_screenshot
                    )
            except Exception as e:
                if "timeout" in str(e).lower() and include_screenshot:
                    try:
                        if no_dom:
                            from browser_use.browser.events import (
                                BrowserStateRequestEvent,
                            )

                            event = browser_session.event_bus.dispatch(
                                BrowserStateRequestEvent(
                                    include_dom=False,
                                    include_screenshot=include_screenshot,
                                )
                            )
                            state_summary = await event.event_result(
                                raise_if_none=True, raise_if_any=True
                            )
                        else:
                            state_summary = (
                                await browser_session.get_browser_state_summary(
                                    include_screenshot=include_screenshot
                                )
                            )
                    except Exception as e2:
                        _output_error(
                            "observe",
                            {"screenshot": include_screenshot},
                            f"Failed to capture browser state (timed out twice): {e2}",
                            error_details=_build_error_details(
                                "state_capture",
                                retryable=True,
                                include_screenshot=include_screenshot,
                                exception_type=type(e2).__name__,
                            ),
                        )
                        sys.exit(1)
                else:
                    _output_error(
                        "observe",
                        {"screenshot": include_screenshot},
                        f"Failed to capture browser state: {e}",
                        error_details=_build_error_details(
                            "state_capture",
                            retryable=False,
                            include_screenshot=include_screenshot,
                            exception_type=type(e).__name__,
                        ),
                    )
                    sys.exit(1)

            state_ms = (time.perf_counter() - start_state) * 1000.0
            if not no_dom:
                _selector_cache[instance_id] = time.time()
            focused_id = browser_session.agent_focus_target_id
            screenshot_data = state_summary.screenshot
            cdp_screenshot_error = None
            needs_screenshot = screenshot or omniparser
            omniparser_shot_dir = None
            omniparser_image_data = None
            som_path = None

            try:
                cdp_session = await browser_session.get_or_create_cdp_session()
                viewport_res = await asyncio.wait_for(
                    cdp_session.cdp_client.send.Runtime.evaluate(
                        params={
                            "expression": "({width: window.innerWidth, height: window.innerHeight, device_pixel_ratio: window.devicePixelRatio})",
                            "returnByValue": True,
                        },
                        session_id=cdp_session.session_id,
                    ),
                    timeout=5.0,
                )
                viewport_data = viewport_res.get("result", {}).get("value")
                if needs_screenshot and not screenshot_data:
                    try:
                        capture_start = time.perf_counter() if state.profile else None
                        capture = await asyncio.wait_for(
                            cdp_session.cdp_client.send.Page.captureScreenshot(
                                params={"format": "png"},
                                session_id=cdp_session.session_id,
                            ),
                            timeout=15.0,
                        )
                        if capture_start is not None:
                            profile_extra["cdp_screenshot_ms"] = (
                                time.perf_counter() - capture_start
                            ) * 1000.0
                        capture_data = (
                            capture.get("data") if isinstance(capture, dict) else None
                        )
                        if isinstance(capture_data, str) and capture_data:
                            screenshot_data = capture_data
                        else:
                            cdp_screenshot_error = "CDP capture returned no data"
                    except Exception as capture_exc:
                        cdp_screenshot_error = str(capture_exc) or "CDP capture failed"
            except Exception as e:
                _output_error(
                    "observe",
                    {},
                    f"Failed to access browser via CDP: {e}",
                    error_details=_build_error_details(
                        "cdp_access",
                        retryable=False,
                        exception_type=type(e).__name__,
                    ),
                )
                sys.exit(1)

            viewport = None
            if viewport_data:
                from .models import ViewportInfo

                viewport = ViewportInfo(**viewport_data)

            omniparser_quality = (
                quality_override if quality_override is not None else 95
            )
            som_quality = quality_override if quality_override is not None else 75

            visual_analysis = None
            if omniparser:
                omniparser_start = time.perf_counter() if state.profile else None
                if not screenshot_data:
                    message = "Failed to capture screenshot required for OmniParser."
                    if cdp_screenshot_error:
                        message = (
                            f"{message} CDP screenshot failed: {cdp_screenshot_error}"
                        )
                    _output_error(
                        "observe",
                        {"omniparser": True},
                        message,
                        error_details=_build_error_details(
                            "omniparser_input",
                            retryable=True,
                            reason="missing_screenshot",
                            cdp_screenshot_error=cdp_screenshot_error,
                        ),
                    )
                    sys.exit(1)
                if not viewport:
                    _output_error(
                        "observe",
                        {"omniparser": True},
                        "Failed to capture viewport info required for OmniParser.",
                        error_details=_build_error_details(
                            "omniparser_input",
                            retryable=True,
                            reason="missing_viewport",
                        ),
                    )
                    sys.exit(1)

                omniparser_shot_dir = Path(session_info.user_data_dir) / "screenshots"
                if path:
                    p = Path(path)
                    if p.is_dir() or not p.suffix:
                        omniparser_shot_dir = p
                    else:
                        omniparser_shot_dir = p.parent
                if omniparser_shot_dir:
                    omniparser_shot_dir.mkdir(exist_ok=True, parents=True)

                from .vision import VisionClient
                from .utils import downscale_image

                client = VisionClient(server_url=endpoint)

                try:
                    prepare_start = time.perf_counter() if state.profile else None
                    omniparser_image_data = downscale_image(
                        screenshot_data, quality=omniparser_quality
                    )
                    if prepare_start is not None:
                        profile_extra["omniparser_prepare_ms"] = (
                            time.perf_counter() - prepare_start
                        ) * 1000.0
                    request_start = time.perf_counter() if state.profile else None
                    assert viewport is not None
                    analysis, som_image_base64 = await client.analyze(
                        omniparser_image_data, viewport
                    )
                    if request_start is not None:
                        profile_extra["omniparser_request_ms"] = (
                            time.perf_counter() - request_start
                        ) * 1000.0

                    if not analysis.elements:
                        _output_error(
                            "observe",
                            {"omniparser": True},
                            "OmniParser returned no elements for the current page.",
                            error_details=_build_error_details(
                                "omniparser_analysis",
                                retryable=False,
                                elements=0,
                            ),
                        )
                        sys.exit(1)

                    if som_image_base64 and omniparser_shot_dir:
                        som_start = time.perf_counter() if state.profile else None
                        som_image_base64 = downscale_image(
                            som_image_base64, quality=som_quality
                        )

                        som_path = str(omniparser_shot_dir / "image_som.jpg")
                        client.save_som_image(som_image_base64, som_path)
                        analysis.som_image_path = som_path
                        if som_start is not None:
                            profile_extra["omniparser_som_ms"] = (
                                time.perf_counter() - som_start
                            ) * 1000.0

                    visual_analysis = analysis
                    if omniparser_start is not None:
                        profile_extra["omniparser_total_ms"] = (
                            time.perf_counter() - omniparser_start
                        ) * 1000.0
                except Exception as ve:
                    _output_error(
                        "observe",
                        {"omniparser": True},
                        f"OmniParser analysis failed: {ve}",
                        error_details=_build_error_details(
                            "omniparser_analysis",
                            exception_type=type(ve).__name__,
                        ),
                    )
                    sys.exit(1)

            start_tabs = time.perf_counter()
            tabs = [
                TabInfo(id=t.target_id, title=t.title, url=t.url)
                for t in await browser_session.get_tabs()
            ]
            tabs_ms = (time.perf_counter() - start_tabs) * 1000.0

            screenshot_path = None
            screenshot_ms = 0.0

            if (screenshot or omniparser) and screenshot_data:
                start_shot = time.perf_counter()

                if omniparser:
                    if omniparser_image_data is None:
                        _output_error(
                            "observe",
                            {"omniparser": True},
                            "Failed to prepare OmniParser screenshot.",
                            error_details=_build_error_details(
                                "omniparser_prepare",
                                retryable=False,
                            ),
                        )
                        sys.exit(1)
                    shot_dir = omniparser_shot_dir or (
                        Path(session_info.user_data_dir) / "screenshots"
                    )
                    shot_dir.mkdir(exist_ok=True, parents=True)
                    image_path = str(shot_dir / "image.jpg")
                    screenshot_path = som_path or image_path
                    shot_path = image_path
                    shot_data = omniparser_image_data
                else:
                    shot_data = screenshot_data
                    ext = "png"
                    if not path:
                        shot_dir = Path(session_info.user_data_dir) / "screenshots"
                        shot_dir.mkdir(exist_ok=True)
                        screenshot_path = str(shot_dir / f"last_state.{ext}")
                    else:
                        p = Path(path)
                        if p.is_dir():
                            p.mkdir(exist_ok=True, parents=True)
                            screenshot_path = str(p / f"last_state.{ext}")
                        else:
                            if p.parent:
                                p.parent.mkdir(exist_ok=True, parents=True)
                            screenshot_path = str(p)
                    shot_path = screenshot_path

                if shot_path and shot_data:
                    with open(shot_path, "wb") as f:
                        f.write(base64.b64decode(shot_data))
                    screenshot_ms = (time.perf_counter() - start_shot) * 1000.0

            obs = Observation(
                session_id=instance_id,
                url=state_summary.url,
                title=state_summary.title,
                visual_analysis=visual_analysis,
                tabs=tabs,
                viewport=viewport,
                screenshot_path=screenshot_path,
                dom_minified=(
                    ""
                    if no_dom
                    else state_summary.dom_state.llm_representation()
                    if state_summary.dom_state
                    else ""
                ),
            )

            data = obs.model_dump()
            data["focused_tab_id"] = focused_id
            if state.profile:
                data["profile"] = {
                    "get_session_ms": get_session_ms,
                    "get_browser_session_ms": get_browser_session_ms,
                    "get_state_ms": state_ms,
                    "get_tabs_ms": tabs_ms,
                    "write_screenshot_ms": screenshot_ms,
                    **profile_extra,
                }
            output_data(data)

        finally:
            if not _should_keep_session():
                await _stop_cached_browser_session(instance_id)

    @handle_errors(action="observe")
    async def wrapper():
        await run()

    asyncio.run(wrapper())


@instance_app.command(
    help="Navigate to URL.",
    epilog="Example: buse b1 navigate https://example.com --new-tab",
)
def navigate(
    ctx: typer.Context, url: str, new_tab: bool = typer.Option(False, "--new-tab")
):
    asyncio.run(
        execute_tool(
            ctx.obj["instance_id"], "navigate", {"url": url, "new_tab": new_tab}
        )
    )


@instance_app.command(
    help="Open a URL in a new tab (alias for navigate --new-tab).",
    epilog="Example: buse b1 new-tab https://example.com",
)
def new_tab(ctx: typer.Context, url: str):
    asyncio.run(
        execute_tool(ctx.obj["instance_id"], "navigate", {"url": url, "new_tab": True})
    )


@instance_app.command(
    help="Search the web (duckduckgo, google, bing).",
    epilog='Example: buse b1 search "site:example.com" --engine duckduckgo',
)
def search(ctx: typer.Context, query: str, engine: str = "google"):
    asyncio.run(
        execute_tool(
            ctx.obj["instance_id"], "search", {"query": query, "engine": engine}
        )
    )


@instance_app.command(
    help="Click by index OR coordinates.",
    epilog="Examples: buse b1 click 12 | buse b1 click --x 200 --y 300 | buse b1 click --id submit",
)
def click(
    ctx: typer.Context,
    index: Optional[int] = typer.Argument(None),
    x: Optional[int] = typer.Option(None, "--x"),
    y: Optional[int] = typer.Option(None, "--y"),
    element_id: Optional[str] = typer.Option(None, "--id"),
    element_class: Optional[str] = typer.Option(None, "--class"),
):
    if (
        index is None
        and x is None
        and y is None
        and element_id is None
        and element_class is None
    ):
        raise typer.BadParameter(
            "Provide an index, --id/--class, or --x/--y. Example: buse b1 click 12 or buse b1 click --x 200 --y 300."
        )
    if (x is None) != (y is None):
        raise typer.BadParameter("Provide both --x and --y for coordinate clicks.")
    params = {}
    if index is not None:
        params["index"] = index
    if x is not None:
        params["coordinate_x"] = x
    if y is not None:
        params["coordinate_y"] = y
    if element_id is not None:
        params["element_id"] = element_id
    if element_class is not None:
        params["element_class"] = element_class

    asyncio.run(
        execute_tool(
            ctx.obj["instance_id"],
            "click",
            params,
            needs_selector_map=(index is not None),
        )
    )


@instance_app.command(
    help="Input text into form fields.",
    epilog='Examples: buse b1 input 12 "hello" | buse b1 input --id email --text "a@b.com"',
)
def input(
    ctx: typer.Context,
    index: Optional[int] = typer.Argument(None),
    text: Optional[str] = typer.Argument(None),
    text_opt: Optional[str] = typer.Option(None, "--text"),
    element_id: Optional[str] = typer.Option(None, "--id"),
    element_class: Optional[str] = typer.Option(None, "--class"),
):
    resolved_text = text if text is not None else text_opt
    if resolved_text is None:
        raise typer.BadParameter(
            'Provide text as a positional arg or --text. Example: buse b1 input 12 "hello".'
        )
    if index is None and element_id is None and element_class is None:
        raise typer.BadParameter(
            'Provide an index or --id/--class. Example: buse b1 input 12 "hello".'
        )
    params = {"text": resolved_text}
    if index is not None:
        params["index"] = index
    if element_id is not None:
        params["element_id"] = element_id
    if element_class is not None:
        params["element_class"] = element_class
    asyncio.run(
        execute_tool(
            ctx.obj["instance_id"],
            "input",
            params,
            needs_selector_map=(index is not None),
        )
    )


@instance_app.command(
    help="Upload a file to an element.",
    epilog="Example: buse b1 upload-file 5 ./image.png",
)
def upload_file(
    ctx: typer.Context,
    index: int = typer.Argument(..., help="Index of the file input element"),
    path: str = typer.Argument(..., help="Path to the file"),
):
    import os

    full_path = os.path.abspath(path)
    if not os.path.exists(full_path):
        raise typer.BadParameter(f"File not found: {path}")
    if not os.path.isfile(full_path):
        raise typer.BadParameter(f"Path is not a file: {path}")

    asyncio.run(
        execute_tool(
            ctx.obj["instance_id"],
            "upload_file",
            {"index": index, "path": full_path},
            needs_selector_map=True,
        )
    )


@instance_app.command(
    help="Send keys to the browser.",
    epilog='Examples: buse b1 send-keys "Enter" | buse b1 send-keys "Control+L" | buse b1 send-keys --id search "Hello" | buse b1 send-keys --list-keys',
)
def send_keys(
    ctx: typer.Context,
    keys: Optional[str] = typer.Argument(
        None, help="Keys to send (e.g. Enter, Control+L, or text)."
    ),
    index: Optional[int] = typer.Option(
        None, "--index", help="Optional element index to focus before sending keys."
    ),
    element_id: Optional[str] = typer.Option(
        None, "--id", help="Optional element id to focus before sending keys."
    ),
    element_class: Optional[str] = typer.Option(
        None, "--class", help="Optional element class to focus before sending keys."
    ),
    list_keys: bool = typer.Option(
        False, "--list-keys", help="List supported key names and aliases, then exit."
    ),
):
    if list_keys is True:
        typer.echo(_format_send_keys_help())
        return
    if not keys:
        raise typer.BadParameter("Provide keys or use --list-keys.")
    resolved_index = index if isinstance(index, int) else None
    resolved_element_id = element_id if isinstance(element_id, str) else None
    resolved_element_class = element_class if isinstance(element_class, str) else None
    params = {"keys": keys}
    if resolved_index is not None:
        params["index"] = resolved_index
    if resolved_element_id is not None:
        params["element_id"] = resolved_element_id
    if resolved_element_class is not None:
        params["element_class"] = resolved_element_class
    asyncio.run(
        execute_tool(
            ctx.obj["instance_id"],
            "send_keys",
            params,
            needs_selector_map=(
                resolved_index is not None
                or resolved_element_id is not None
                or resolved_element_class is not None
            ),
        )
    )


@instance_app.command(
    help="Scroll to text on the page.",
    epilog='Example: buse b1 find-text "Contact Us"',
)
def find_text(
    ctx: typer.Context,
    text: str = typer.Argument(..., help="Text to find and scroll to"),
):
    asyncio.run(
        execute_tool(
            ctx.obj["instance_id"],
            "find_text",
            {"text": text},
        )
    )


@instance_app.command(
    help="Get dropdown options for a select element.",
    epilog="Examples: buse b1 dropdown-options 5 | buse b1 dropdown-options --id country",
)
def dropdown_options(
    ctx: typer.Context,
    index: Optional[int] = typer.Argument(None),
    element_id: Optional[str] = typer.Option(None, "--id"),
    element_class: Optional[str] = typer.Option(None, "--class"),
):
    if index is None and element_id is None and element_class is None:
        raise typer.BadParameter(
            "Provide an index or --id/--class. Example: buse b1 dropdown-options 5."
        )
    params = {}
    if index is not None:
        params["index"] = index
    if element_id is not None:
        params["element_id"] = element_id
    if element_class is not None:
        params["element_class"] = element_class
    asyncio.run(
        execute_tool(
            ctx.obj["instance_id"],
            "dropdown_options",
            params,
            needs_selector_map=True,
        )
    )


@instance_app.command(
    help="Select a dropdown option by visible text.",
    epilog='Examples: buse b1 select-dropdown 5 "Canada" | buse b1 select-dropdown --id country --text "Canada"',
)
def select_dropdown(
    ctx: typer.Context,
    index: Optional[int] = typer.Argument(None),
    text: Optional[str] = typer.Argument(None),
    text_opt: Optional[str] = typer.Option(None, "--text"),
    element_id: Optional[str] = typer.Option(None, "--id"),
    element_class: Optional[str] = typer.Option(None, "--class"),
):
    resolved_text = text if text is not None else text_opt
    if resolved_text is None:
        raise typer.BadParameter(
            'Provide text as a positional arg or --text. Example: buse b1 select-dropdown 5 "Option".'
        )
    if index is None and element_id is None and element_class is None:
        raise typer.BadParameter(
            'Provide an index or --id/--class. Example: buse b1 select-dropdown 5 "Option".'
        )
    params = {"text": resolved_text}
    if index is not None:
        params["index"] = index
    if element_id is not None:
        params["element_id"] = element_id
    if element_class is not None:
        params["element_class"] = element_class
    asyncio.run(
        execute_tool(
            ctx.obj["instance_id"],
            "select_dropdown",
            params,
            needs_selector_map=True,
        )
    )


@instance_app.command(
    help="Go back in browser history.",
    epilog="Example: buse b1 go-back",
)
def go_back(ctx: typer.Context):
    asyncio.run(execute_tool(ctx.obj["instance_id"], "go_back", {}))


@instance_app.command(
    help="Scroll the page or an element.",
    epilog="Examples: buse b1 scroll --pages 2 | buse b1 scroll --up --pages 1 | buse b1 scroll --index 12 --pages 0.5",
)
def scroll(
    ctx: typer.Context,
    down: bool = typer.Option(True, "--down/--up"),
    pages: float = 1.0,
    index: Optional[int] = typer.Option(None),
):
    if pages <= 0:
        raise typer.BadParameter("Use a positive --pages value (e.g., 1 or 0.5).")
    instance_id = ctx.obj["instance_id"]

    async def run():
        if index is not None:
            await execute_tool(
                instance_id,
                "scroll",
                {"down": down, "pages": pages, "index": index},
                needs_selector_map=True,
                action_label="scroll",
            )
        else:
            direction = 1 if down else -1
            code = f"window.scrollBy({{top: window.innerHeight * {pages} * {direction}, behavior: 'smooth'}})"
            await execute_tool(
                instance_id, "evaluate", {"code": code}, action_label="scroll"
            )

    asyncio.run(run())


@instance_app.command(
    help="Hover over an element (via JS).",
    epilog="Examples: buse b1 hover 12 | buse b1 hover --id menu",
)
def hover(
    ctx: typer.Context,
    index: Optional[int] = typer.Argument(None),
    element_id: Optional[str] = typer.Option(None, "--id"),
    element_class: Optional[str] = typer.Option(None, "--class"),
):
    if index is None and element_id is None and element_class is None:
        raise typer.BadParameter("Provide index, --id, or --class for hover.")
    params = {}
    if index is not None:
        params["index"] = index
    if element_id is not None:
        params["element_id"] = element_id
    if element_class is not None:
        params["element_class"] = element_class
    asyncio.run(
        execute_tool(
            ctx.obj["instance_id"],
            "hover",
            params,
            needs_selector_map=True,
            action_label="hover",
        )
    )


@instance_app.command(
    help="Refresh the current page.",
    epilog="Example: buse b1 refresh",
)
def refresh(ctx: typer.Context):
    asyncio.run(
        execute_tool(
            ctx.obj["instance_id"],
            "evaluate",
            {"code": "window.location.reload()"},
            action_label="refresh",
        )
    )


@instance_app.command(
    help="Wait for specified seconds.",
    epilog="Example: buse b1 wait 2",
)
def wait(ctx: typer.Context, seconds: float):
    if seconds < 0:
        raise typer.BadParameter("Use a non-negative number of seconds.")
    asyncio.run(
        execute_tool(
            ctx.obj["instance_id"],
            "wait",
            {"seconds": seconds},
            action_label="wait",
        )
    )


@instance_app.command(
    help="Switch to a tab by 4-char ID.",
    epilog="Example: buse b1 switch-tab a1b2",
)
def switch_tab(ctx: typer.Context, tab_id: str):
    if len(tab_id) < 4:
        raise typer.BadParameter(
            "Provide the 4-char tab ID shown in observe. Example: buse b1 switch-tab a1b2."
        )
    asyncio.run(
        execute_tool(
            ctx.obj["instance_id"],
            "switch",
            {"tab_id": tab_id},
            action_label="switch-tab",
        )
    )


@instance_app.command(
    help="Close a tab by 4-char ID.",
    epilog="Example: buse b1 close-tab a1b2",
)
def close_tab(ctx: typer.Context, tab_id: str):
    if len(tab_id) < 4:
        raise typer.BadParameter(
            "Provide the 4-char tab ID shown in observe. Example: buse b1 close-tab a1b2."
        )
    asyncio.run(
        execute_tool(
            ctx.obj["instance_id"],
            "close",
            {"tab_id": tab_id},
            action_label="close-tab",
        )
    )


@instance_app.command(
    help="Export cookies/storage to a JSON file.",
    epilog="Example: buse b1 save-state ./state.json",
)
def save_state(ctx: typer.Context, path: str):
    instance_id = ctx.obj["instance_id"]

    async def run():
        session_info = session_manager.get_session(instance_id)
        if session_info is None:
            _output_error(
                "save_state",
                {"path": path},
                f"Instance {instance_id} not found.",
                error_details=_build_error_details(
                    "session", retryable=False, instance_id=instance_id
                ),
            )
            sys.exit(1)
        assert session_info is not None
        browser_session = BrowserSession(cdp_url=session_info.cdp_url)
        await browser_session.start()
        try:
            state = await browser_session.export_storage_state(output_path=path)
            output_data(
                {
                    "success": True,
                    "path": path,
                    "cookies_count": len(state.get("cookies", [])),
                }
            )
        finally:
            await browser_session.stop()

    asyncio.run(run())


@instance_app.command(
    help="Extract structured data using LLM.",
    epilog='Example: buse b1 extract "List all form fields"',
)
def extract(ctx: typer.Context, query: str):
    from browser_use.llm.openai.chat import ChatOpenAI

    model = os.getenv("BUSE_EXTRACT_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model)
    asyncio.run(
        execute_tool(
            ctx.obj["instance_id"],
            "extract",
            {"query": query},
            page_extraction_llm=llm,
        )
    )


@instance_app.command(
    help="Execute custom JavaScript.",
    epilog='Example: buse b1 evaluate "(function(){return document.title})()"',
)
def evaluate(ctx: typer.Context, code: str):
    asyncio.run(
        execute_tool(
            ctx.obj["instance_id"],
            "evaluate",
            {"code": code},
            action_label="evaluate",
        )
    )


@instance_app.command(
    help="Stop instance.",
    epilog="Example: buse b1 stop",
)
def stop(ctx: typer.Context):
    instance_id = ctx.obj["instance_id"]
    if session_manager.get_session(instance_id) is None:
        asyncio.run(_stop_cached_browser_session(instance_id))
        _output_error(
            "stop",
            {},
            f"Instance {instance_id} not found.",
            error_details=_build_error_details(
                "session", retryable=False, instance_id=instance_id
            ),
        )
        sys.exit(1)
    asyncio.run(_stop_cached_browser_session(instance_id))
    session_manager.stop_session(instance_id)
    output_data({"message": f"Stopped {instance_id}"})


def _run(args: list[str]) -> None:
    state.format = OutputFormat.json
    state.profile = False

    if args and args[0] == "list":
        output_data(session_manager.list_sessions())
        return

    if args and args[0] == "mcp-server":
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s: %(message)s",
            stream=sys.stderr,
            force=True,
        )
        logging.disable(logging.NOTSET)
        server_app(args=args[1:], standalone_mode=False)
        return

    if "--profile" in args:
        state.profile = True
        args = [arg for arg in args if arg != "--profile"]
    if "-p" in args:
        state.profile = True
        args = [arg for arg in args if arg != "-p"]

    while "--format" in args:
        idx = args.index("--format")
        if idx + 1 < len(args):
            state.format = OutputFormat(args[idx + 1])
            args.pop(idx)
            args.pop(idx)
    while "-f" in args:
        idx = args.index("-f")
        if idx + 1 < len(args):
            state.format = OutputFormat(args[idx + 1])
            args.pop(idx)
            args.pop(idx)

    if len(args) >= 3 and args[1] == "wait":
        candidate = args[2]
        if candidate.startswith("-") and candidate not in {"-", "--"}:
            try:
                float(candidate)
            except ValueError:
                pass
            else:
                args.insert(2, "--")

    if not args or args[0] in ["--help", "-h"]:
        print(
            "buse: Stateless CLI for browser-use\n\n"
            "Usage:\n"
            "  buse [--format json|toon] [--profile] list              List active instances\n"
            "  buse [--format json|toon] [--profile] <id>              Start/initialize an instance\n"
            "  buse [--format json|toon] [--profile] <id> observe      Observe instance state\n"
            "  buse [--format json|toon] [--profile] <id> <command>    Execute an action\n\n"
            "Instance commands:\n"
            "  observe            Observe current state\n"
            "  navigate           Navigate to URL\n"
            "  new-tab            Open a URL in a new tab\n"
            "  search             Search the web (duckduckgo, google, bing)\n"
            "  click              Click by index OR coordinates\n"
            "  input              Input text into form fields\n"
            "  upload-file        Upload a file to an element\n"
            "  send-keys          Send keys to the browser\n"
            "  find-text          Scroll to text on the page\n"
            "  dropdown-options   Get dropdown options for a select element\n"
            "  select-dropdown    Select a dropdown option by visible text\n"
            "  go-back            Go back in browser history\n"
            "  scroll             Scroll the page or an element\n"
            "  hover              Hover over an element (via JS)\n"
            "  refresh            Refresh the current page\n"
            "  wait               Wait for specified seconds\n"
            "  switch-tab         Switch to a tab by 4-char ID\n"
            "  close-tab          Close a tab by 4-char ID\n"
            "  save-state         Export cookies/storage to a JSON file\n"
            "  extract            Extract structured data using LLM\n"
            "  evaluate           Execute custom JavaScript\n"
            "  stop               Stop instance\n\n"
            "Tip: run `buse <id> --help` to see per-command options."
        )
        return

    instance_id = args[0]
    if len(args) == 1:
        existing = session_manager.get_session(instance_id)
        if existing:
            output_data(
                {"message": f"Instance {instance_id} running", "already_running": True}
            )
        else:
            session_manager.start_session(instance_id)
            output_data(
                {"message": f"Initialized {instance_id}", "already_running": False}
            )
        return

    try:
        instance_app(
            args=args[1:], obj={"instance_id": instance_id}, standalone_mode=False
        )
    except Exception as e:
        if not isinstance(e, SystemExit):
            _output_error(
                "cli",
                {},
                str(e),
                error_details=_build_error_details(
                    "cli",
                    exception_type=type(e).__name__,
                    args=args,
                ),
            )
            sys.exit(1)


def run_cli(args: list[str]) -> tuple[int, str]:
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            _run(list(args))
            return 0, buf.getvalue()
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            return code, buf.getvalue()


def app():
    _run(sys.argv[1:])


if __name__ == "__main__":
    app()
