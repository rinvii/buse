import asyncio
import base64
import json
import logging
import os
import sys
import time
import httpx
import importlib
import inspect
from enum import Enum
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Any, Union
import typer
from browser_use.browser import BrowserSession
from .models import ActionResult, Observation, TabInfo
from .mcp_server import BuseMCPServer
from .session import SessionManager
from .utils import OutputFormat, handle_errors, output_data, state, _serialize
from .features import inspection, interaction

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


def _resolve_option_default(value, default):
    if hasattr(value, "default"):
        return value.default
    return default if value is None else value


async def get_observation(
    instance_id: str,
    screenshot: bool = False,
    path: Optional[str] = None,
    omniparser: bool = False,
    no_dom: bool = False,
    som: bool = False,
    diagnostics: bool = False,
    semantic: bool = False,
    mode: str = "efficient",
    max_chars: Optional[int] = None,
    max_labels: Optional[int] = None,
    selector: Optional[str] = None,
    frame: Optional[str] = None,
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
    event = None
    try:
        profile_extra: dict[str, float] = {}
        start_state = time.perf_counter()
        include_screenshot = screenshot and not som
        try:
            if no_dom and not som:
                events_mod = importlib.import_module("browser_use.browser.events")
                event_cls = getattr(events_mod, "BrowserStateRequestEvent", None)
                if event_cls is None:
                    raise RuntimeError("BrowserStateRequestEvent not available")
                event = browser_session.event_bus.dispatch(
                    event_cls(include_dom=False, include_screenshot=include_screenshot)
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
            if "timeout" in str(e).lower():
                try:
                    if no_dom and not som and event is not None:
                        state_summary = await asyncio.wait_for(
                            event.event_result(raise_if_none=True, raise_if_any=True),
                            timeout=30.0,
                        )
                    elif include_screenshot:
                        state_summary = await asyncio.wait_for(
                            browser_session.get_browser_state_summary(
                                include_screenshot=include_screenshot
                            ),
                            timeout=30.0,
                        )
                    else:
                        raise RuntimeError("Failed to capture browser state")
                except Exception as e2:
                    if "timeout" in str(e2).lower():
                        raise RuntimeError("Timeout (timed out twice).")
                    raise RuntimeError(f"Failed to capture browser state: {e2}")
            else:
                raise RuntimeError(f"Failed to capture browser state: {e}")
        state_ms = (time.perf_counter() - start_state) * 1000.0
        if not no_dom:
            _selector_cache[instance_id] = time.time()
        focused_id = browser_session.agent_focus_target_id
        screenshot_data = state_summary.screenshot
        cdp_screenshot_error = None
        needs_screenshot = screenshot or omniparser or som
        omniparser_shot_dir = None
        omniparser_image_data = None
        som_path = None
        som_labels_count = None
        som_labels_skipped = None
        semantic_snapshot_text = None
        semantic_snapshot_truncated = None
        diagnostics_data = None
        viewport_data = None
        try:
            cdp_session_res = browser_session.get_or_create_cdp_session()
            cdp_session = (
                await cdp_session_res
                if inspect.isawaitable(cdp_session_res)
                else cdp_session_res
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to access browser via CDP (CDP access failed): {e}"
            )
        if semantic:
            start_semantic = time.perf_counter()
            selector_map_semantic = await browser_session.get_selector_map()
            ax_nodes = await inspection.get_accessibility_tree(cdp_session)
            resolved_max_chars = inspection.resolve_semantic_max_chars(mode, max_chars)
            root_backend_id = None
            refs_out: dict[str, dict[str, Any]] = {}
            if selector or frame:
                root_backend_id = await inspection.resolve_backend_node_id_for_selector(
                    cdp_session, selector=selector, frame_selector=frame
                )
                if root_backend_id is None:
                    raise ValueError("Selector/frame scope not found or inaccessible.")
            semantic_snapshot_text, semantic_snapshot_truncated = (
                inspection.format_semantic_snapshot(
                    ax_nodes,
                    selector_map_semantic,
                    mode=mode,
                    max_chars=resolved_max_chars,
                    root_backend_id=root_backend_id,
                    refs_out=refs_out,
                    include_refs=True,
                )
            )
            inspection.store_semantic_refs(
                focused_id,
                refs_out,
                instance_id=instance_id,
                persist_path=str(_semantic_refs_path(session_info)),
            )
            profile_extra["semantic_snapshot_ms"] = (
                time.perf_counter() - start_semantic
            ) * 1000.0
        if diagnostics:
            start_diag = time.perf_counter()
            diagnostics_data = await inspection.get_diagnostics(cdp_session)
            profile_extra["diagnostics_ms"] = (
                time.perf_counter() - start_diag
            ) * 1000.0
        try:
            viewport_res = await asyncio.wait_for(
                cdp_session.cdp_client.send.Runtime.evaluate(
                    params={
                        "expression": "({width: window.innerWidth, height: window.innerHeight, device_pixel_ratio: window.devicePixelRatio, scrollX: window.scrollX, scrollY: window.scrollY})",
                        "returnByValue": True,
                    },
                    session_id=cdp_session.session_id,
                ),
                timeout=5.0,
            )
            viewport_data = viewport_res.get("result", {}).get("value")
            if needs_screenshot and not screenshot_data:
                if som:
                    start_som = time.perf_counter()
                    selector_map = await browser_session.get_selector_map()
                    img_data, count, skipped = await inspection.screenshot_with_labels(
                        cdp_session,
                        selector_map,
                        viewport=viewport_data,
                        max_labels=max_labels if max_labels is not None else 150,
                    )
                    screenshot_data = (
                        base64.b64encode(img_data).decode("utf-8")
                        if isinstance(img_data, bytes)
                        else img_data
                    )
                    som_labels_count = count
                    som_labels_skipped = skipped
                    profile_extra["som_ms"] = (time.perf_counter() - start_som) * 1000.0
                else:
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

            viewport = ViewportInfo(
                width=viewport_data.get("width"),
                height=viewport_data.get("height"),
                device_pixel_ratio=viewport_data.get("device_pixel_ratio"),
            )
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
            if not omniparser_image_data:
                raise RuntimeError("OmniParser image preprocessing failed")
            analysis, som_image_base64 = await asyncio.wait_for(
                client.analyze(omniparser_image_data, viewport), timeout=60.0
            )
            if not analysis.elements:
                raise RuntimeError("OmniParser returned no elements")
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
        start_tabs = time.perf_counter()
        tabs = [
            TabInfo(id=t.target_id, title=t.title, url=t.url)
            for t in await browser_session.get_tabs()
        ]
        tabs_ms = (time.perf_counter() - start_tabs) * 1000.0
        screenshot_path = None
        screenshot_ms = 0.0
        if (screenshot or omniparser or som) and screenshot_data:
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
                    shot_dir.mkdir(exist_ok=True, parents=True)
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
            observed_at=time.time(),
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
            semantic_snapshot=semantic_snapshot_text,
            semantic_truncated=semantic_snapshot_truncated,
            diagnostics=diagnostics_data,
            som_labels=som_labels_count,
            som_labels_skipped=som_labels_skipped,
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
    browser_session: Any = BrowserSession(cdp_url=session_info.cdp_url)
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


def _output_human(data: dict) -> None:
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.tree import Tree
        from rich.markup import escape
        import sys

        console = Console(file=sys.stdout, highlight=False)
        console.print(
            f"\n[bold blue]🌐 {escape(data.get('title', 'Unknown Title'))}[/bold blue]"
        )
        console.print(f"[dim]🔗 {escape(data.get('url', 'Unknown URL'))}[/dim]\n")
        snapshot = data.get("semantic_snapshot", "")
        if snapshot:
            lines = [line for line in snapshot.split("\n") if line.strip()]
            if lines:
                root_text = lines[0].strip("- ")
                tree = Tree(f"[bold green]{escape(root_text)}[/bold green]")
                stack = {0: tree}
                for line in lines[1:]:
                    stripped = line.lstrip()
                    indent_len = len(line) - len(stripped)
                    depth = indent_len // 2
                    content = stripped.strip("- ")
                    if not content:
                        continue
                    if content.startswith("..."):
                        parent = stack.get(depth - 1, tree)
                        parent.add(f"[dim]{escape(content)}[/dim]")
                        continue
                    from rich.text import Text

                    node_text = Text()
                    if content.startswith("[") and "]" in content:
                        end_bracket = content.find("]")
                        role_info = content[1:end_bracket]
                        text_info = content[end_bracket + 1 :].strip()
                        if "#" in role_info:
                            node_text.append(f"[{role_info}]", style="bold yellow")
                        else:
                            node_text.append(f"[{role_info}]", style="cyan")
                        if text_info:
                            node_text.append(f" {text_info}")
                    else:
                        node_text.append(content)
                    parent_depth = depth - 1
                    while parent_depth >= 0 and parent_depth not in stack:
                        parent_depth -= 1
                    parent = stack.get(parent_depth, tree)
                    node = parent.add(node_text)
                    stack[depth] = node
                console.print(
                    Panel(
                        tree,
                        title="[bold]Semantic AXTree[/bold]",
                        border_style="blue",
                        expand=False,
                    )
                )
        diag = data.get("diagnostics", {})
        if diag:
            console.print("\n[bold]🛠 Diagnostics:[/bold]")
            has_errors = False
            if diag.get("console"):
                console.print(
                    f"  ❌ [red]{len(diag['console'])} Console Messages[/red]"
                )
                has_errors = True
            if diag.get("network_errors"):
                console.print(
                    f"  ⚠️ [yellow]{len(diag['network_errors'])} Network Resource Errors[/yellow]"
                )
                has_errors = True
            if diag.get("network_requests"):
                console.print(
                    f"  📡 [cyan]{len(diag['network_requests'])} Network Requests[/cyan]"
                )
            if diag.get("page_errors"):
                console.print(f"  💥 [red]{len(diag['page_errors'])} Page Errors[/red]")
                has_errors = True
            if not has_errors:
                console.print("  ✅ [green]No errors detected.[/green]")
        if data.get("screenshot_path"):
            console.print(f"\n[bold]📸 Screenshot:[/bold] {data['screenshot_path']}")
        if data.get("som_labels") is not None:
            skipped = data.get("som_labels_skipped")
            if skipped:
                console.print(
                    f"[bold]🏷 SoM Labels:[/bold] [bold yellow]{data['som_labels']}[/bold yellow] marked, [bold yellow]{skipped}[/bold yellow] skipped."
                )
            else:
                console.print(
                    f"[bold]🏷 SoM Labels:[/bold] [bold yellow]{data['som_labels']}[/bold yellow] elements marked."
                )
        if data.get("semantic_truncated"):
            console.print(
                "[bold]✂️ Semantic Snapshot:[/bold] [yellow]Truncated[/yellow]"
            )
        console.print("")
        sys.stdout.flush()
    except Exception as e:
        print(f"Error rendering human output: {e}")
        output_data(data)


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
        if action_name == "click":
            index = kwargs.get("index")
            selector = kwargs.get("selector")
            x = kwargs.get("x")
            y = kwargs.get("y")
            if x is None and y is None:
                x = kwargs.get("coordinate_x")
                y = kwargs.get("coordinate_y")
            element_id = kwargs.get("element_id")
            element_class = kwargs.get("element_class")
            right = kwargs.get("right", False)
            middle = kwargs.get("middle", False)
            double = kwargs.get("double", False)
            ctrl = kwargs.get("ctrl", False)
            shift = kwargs.get("shift", False)
            alt = kwargs.get("alt", False)
            meta = kwargs.get("meta", False)
            force = kwargs.get("force", False)
            if (
                x is None
                and y is None
                and index is None
                and selector is None
                and element_id is None
                and element_class is None
            ):
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
            session_info = session_manager.get_session(instance_id)
            if not session_info:
                result = await execute_tool(
                    instance_id,
                    "click",
                    {
                        "index": index,
                        "selector": selector,
                        "element_id": element_id,
                        "element_class": element_class,
                        "coordinate_x": x,
                        "coordinate_y": y,
                    },
                    return_result=True,
                    action_label=action_label,
                )
                if hasattr(result, "model_dump"):
                    result = result.model_dump()
                return _format_mcp_output(result)
            browser_session, _ = await _get_browser_session(instance_id, session_info)
            cdp_session = await browser_session.get_or_create_cdp_session()
            button = "right" if right else "middle" if middle else "left"
            click_count = 2 if double else 1
            modifiers = []
            if ctrl:
                modifiers.append("ctrl")
            if shift:
                modifiers.append("shift")
            if alt:
                modifiers.append("alt")
            if meta:
                modifiers.append("meta")
            try:
                if x is not None and y is not None:
                    mask = 0
                    if alt:
                        mask |= 1
                    if ctrl:
                        mask |= 2
                    if meta:
                        mask |= 4
                    if shift:
                        mask |= 8
                    await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
                        params={
                            "x": x,
                            "y": y,
                            "button": button,
                            "clickCount": click_count,
                            "modifiers": mask,
                            "type": "mousePressed",
                        },
                        session_id=cdp_session.session_id,
                    )
                    await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
                        params={
                            "x": x,
                            "y": y,
                            "button": button,
                            "clickCount": click_count,
                            "modifiers": mask,
                            "type": "mouseReleased",
                        },
                        session_id=cdp_session.session_id,
                    )
                    result = {"success": True, "action": "click", "x": x, "y": y}
                else:
                    target = index if index is not None else selector
                    if target is None:
                        if element_id:
                            target = f"#{element_id}"
                        elif element_class:
                            target = f".{element_class}"
                    if target is None:
                        raise ValueError("Provide a click target.")
                    await _ensure_selector_map(browser_session, instance_id)
                    selector_map = await browser_session.get_selector_map()
                    target_id = browser_session.agent_focus_target_id

                    async def refresh_selector_map():
                        await _ensure_selector_map(
                            browser_session, instance_id, force=True
                        )
                        return await browser_session.get_selector_map()

                    try:
                        coords = await interaction.click_with_retry(
                            cdp_session,
                            target,
                            selector_map=selector_map,
                            refresh_selector_map=refresh_selector_map,
                            modifiers=modifiers,
                            button=button,
                            click_count=click_count,
                            force=force,
                            timeout_ms=5000,
                            debug=False,
                            target_id=target_id,
                            instance_id=instance_id,
                            persist_path=str(_semantic_refs_path(session_info)),
                        )
                        result = {
                            "success": True,
                            "action": "click",
                            "target": target,
                            "coords": coords,
                        }
                    except Exception:
                        result = await execute_tool(
                            instance_id,
                            "click",
                            {
                                "index": index,
                                "selector": selector,
                                "element_id": element_id,
                                "element_class": element_class,
                                "coordinate_x": x,
                                "coordinate_y": y,
                            },
                            return_result=True,
                            action_label=action_label,
                        )
                        if hasattr(result, "model_dump"):
                            result = result.model_dump()
                        return _format_mcp_output(result)
            finally:
                if not _should_keep_session():
                    await _stop_cached_browser_session(instance_id)
            return _format_mcp_output(result)
        if action_name == "input":
            text = kwargs.get("text")
            index = kwargs.get("index")
            element_id = kwargs.get("element_id")
            element_class = kwargs.get("element_class")
            slowly = bool(kwargs.get("slowly", False))
            submit = bool(kwargs.get("submit", False))
            append = bool(kwargs.get("append", False))
            if text is None:
                raise ValueError("text is required")
            if index is None and element_id is None and element_class is None:
                raise ValueError(
                    "Provide an index or element_id/element_class for input."
                )
            if append or slowly or submit:
                session_info = session_manager.get_session(instance_id)
                if not session_info:
                    raise ValueError(f"Instance {instance_id} not found.")
                browser_session, _ = await _get_browser_session(
                    instance_id, session_info
                )
                cdp_session = await browser_session.get_or_create_cdp_session()
                await _ensure_selector_map(browser_session, instance_id)
                selector_map = await browser_session.get_selector_map()
                target_id = browser_session.agent_focus_target_id
                target = None
                if index is not None:
                    target = index
                elif element_id:
                    target = f"#{element_id}"
                elif element_class:
                    target = f".{element_class}"
                if target is None:
                    raise ValueError("Provide an input target.")
                try:
                    result = await interaction.type_text(
                        cdp_session,
                        target,
                        text,
                        selector_map=selector_map,
                        slowly=slowly,
                        submit=submit,
                        append=append,
                        target_id=target_id,
                        instance_id=instance_id,
                        persist_path=str(_semantic_refs_path(session_info)),
                    )
                finally:
                    if not _should_keep_session():
                        await _stop_cached_browser_session(instance_id)
                return _format_mcp_output(
                    {"success": True, "action": "input", "result": result}
                )
        if action_name == "wait":
            text = kwargs.get("text")
            selector = kwargs.get("selector")
            network_idle = kwargs.get("network_idle")
            timeout = kwargs.get("timeout")
            seconds = kwargs.get("seconds")
            has_condition = any(v is not None for v in [text, selector, network_idle])
            if has_condition:
                session_info = session_manager.get_session(instance_id)
                if not session_info:
                    raise ValueError(f"Instance {instance_id} not found.")
                browser_session, _ = await _get_browser_session(
                    instance_id, session_info
                )
                cdp_session = await browser_session.get_or_create_cdp_session()
                try:
                    result = await interaction.wait_for_condition(
                        cdp_session,
                        text=text,
                        selector=selector,
                        network_idle_ms=network_idle,
                        timeout_ms=int((timeout or 10) * 1000),
                    )
                finally:
                    if not _should_keep_session():
                        await _stop_cached_browser_session(instance_id)
                return _format_mcp_output(
                    {"success": True, "action": "wait", "result": result}
                )
            if seconds is None:
                raise ValueError("Provide seconds or a wait condition.")
        if action_name == "fill":
            fields = kwargs.get("fields")
            if not isinstance(fields, list):
                raise ValueError("fields must be a list")
            session_info = session_manager.get_session(instance_id)
            if not session_info:
                params = {
                    "selector": selector,
                    "element_id": element_id,
                    "element_class": element_class,
                }
                if index is not None:
                    params["index"] = (
                        int(str(index)) if str(index).isdigit() else str(index)
                    )
                await execute_tool(
                    instance_id,
                    "click",
                    params,
                    needs_selector_map=(index is not None),
                    action_label="click",
                )
                return
            browser_session, _ = await _get_browser_session(instance_id, session_info)
            cdp_session = await browser_session.get_or_create_cdp_session()
            await _ensure_selector_map(browser_session, instance_id)
            selector_map = await browser_session.get_selector_map()
            target_id = browser_session.agent_focus_target_id
            try:
                result = await interaction.fill_form_atomic(
                    cdp_session,
                    fields,
                    selector_map=selector_map,
                    target_id=target_id,
                    instance_id=instance_id,
                    persist_path=str(_semantic_refs_path(session_info)),
                )
            finally:
                if not _should_keep_session():
                    await _stop_cached_browser_session(instance_id)
            return _format_mcp_output(result)
        if action_name == "drag":
            start = kwargs.get("start")
            end = kwargs.get("end")
            if start is None or end is None:
                raise ValueError("start and end are required")
            session_info = session_manager.get_session(instance_id)
            if not session_info:
                raise ValueError(f"Instance {instance_id} not found.")
            browser_session, _ = await _get_browser_session(instance_id, session_info)
            cdp_session = await browser_session.get_or_create_cdp_session()
            await _ensure_selector_map(browser_session, instance_id)
            selector_map = await browser_session.get_selector_map()
            target_id = browser_session.agent_focus_target_id
            try:
                result = await interaction.drag_and_drop(
                    cdp_session,
                    start,
                    end,
                    selector_map=selector_map,
                    html5=kwargs.get("html5", True),
                    target_id=target_id,
                    instance_id=instance_id,
                    persist_path=str(_semantic_refs_path(session_info)),
                )
            finally:
                if not _should_keep_session():
                    await _stop_cached_browser_session(instance_id)
            return _format_mcp_output(
                {"success": True, "action": "drag", "result": result}
            )
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
        visual: str = "som",
        text: str = "ai",
        mode: str = "efficient",
        max_chars: Optional[int] = None,
        max_labels: Optional[int] = None,
        selector: Optional[str] = None,
        frame: Optional[str] = None,
        screenshot: Optional[bool] = None,
        no_dom: Optional[bool] = None,
        omniparser: Optional[bool] = None,
    ):
        do_screenshot = visual != "none"
        do_som = visual == "som"
        do_omni = visual == "omni"
        do_semantic = text == "ai"
        do_no_dom = text != "dom"
        if screenshot is not None:
            do_screenshot = bool(screenshot)
        if omniparser is not None:
            do_omni = bool(omniparser)
            if do_omni:
                do_screenshot = True
        if no_dom is not None:
            do_no_dom = bool(no_dom)
        result = await get_observation(
            instance_id,
            screenshot=do_screenshot,
            som=do_som,
            omniparser=do_omni,
            semantic=do_semantic,
            no_dom=do_no_dom,
            diagnostics=True,
            mode=mode,
            max_chars=max_chars,
            max_labels=max_labels,
            selector=selector,
            frame=frame,
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


def _semantic_refs_path(session_info) -> Path:
    return Path(session_info.user_data_dir) / "semantic_refs.json"


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


_DEFAULT_BROWSER_SESSION = BrowserSession
_filesystem_mod = importlib.import_module("browser_use.filesystem.file_system")
FileSystem = getattr(_filesystem_mod, "FileSystem")


async def _get_browser_session(instance_id: str, session_info):
    browser_session = _browser_sessions.get(instance_id)
    file_system = _file_systems.get(instance_id)
    if browser_session is None:
        browser_session_cls = BrowserSession
        try:
            browser_mod = importlib.import_module("browser_use.browser")
            alt_cls = getattr(browser_mod, "BrowserSession", None)
            if alt_cls is not None and alt_cls is not _DEFAULT_BROWSER_SESSION:
                browser_session_cls = alt_cls
        except Exception:
            pass
        browser_session: Any = browser_session_cls(cdp_url=session_info.cdp_url)
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
    lowered_error = message.lower()
    if action_name == "click" and "Could not resolve target" in message:
        hint = 'Run "buse <id> observe" to refresh indices and keep the same tab.'
    if action_name == "click" and "did not become actionable" in lowered_error:
        hint = (
            'Run "buse <id> observe" to refresh indices, then retry or close overlays.'
        )
    if action_name == "fill" and "Invalid JSON" in message:
        hint = 'Use a JSON list, e.g. \'[{"ref":"e1","value":"user"}]\'.'
    if "Could not resolve element index" in error:
        hint = "Run `buse <id> observe` and use an index or pass the actual <select> --id/--class."
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
    browser_use_mod = importlib.import_module("browser_use")
    Controller = getattr(browser_use_mod, "Controller")

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


class VisualMode(str, Enum):
    som = "som"
    omni = "omni"
    none = "none"


class TextMode(str, Enum):
    ai = "ai"
    dom = "dom"
    none = "none"


class ObserveMode(str, Enum):
    efficient = "efficient"
    full = "full"


@instance_app.command(
    help="Observe current state. Defaults to AI (compact semantic) + SoM context.",
    epilog="Example: buse b1 observe --mode efficient --human",
)
def observe(
    ctx: typer.Context,
    path: Optional[str] = typer.Option(
        None, "--path", help="Custom path for screenshot."
    ),
    max_chars: Optional[int] = typer.Option(
        None,
        "--max-chars",
        help="Max characters for semantic snapshot (0 disables truncation).",
    ),
    max_labels: Optional[int] = typer.Option(
        None, "--max-labels", help="Max SoM labels to draw (default: 150)."
    ),
    selector: Optional[str] = typer.Option(
        None, "--selector", help="Scope semantic snapshot to a CSS selector."
    ),
    frame: Optional[str] = typer.Option(
        None, "--frame", help="Scope semantic snapshot to a frame/iframe CSS selector."
    ),
    visual: VisualMode = typer.Option(
        VisualMode.som, "--visual", help="Visual grounding mode."
    ),
    text: TextMode = typer.Option(
        TextMode.ai,
        "--text",
        help="Text/Structure extraction mode (ai=compact semantic).",
    ),
    mode: ObserveMode = typer.Option(
        ObserveMode.efficient, "--mode", help="Heuristic compaction mode."
    ),
    human: bool = typer.Option(
        False, "--human", help="Print in a human-friendly format."
    ),
    screenshot: Optional[bool] = typer.Option(
        None,
        "--screenshot/--no-screenshot",
        help="(Deprecated) Force screenshot on/off.",
    ),
    omniparser: Optional[bool] = typer.Option(
        None,
        "--omniparser/--no-omniparser",
        help="(Deprecated) Use OmniParser visual mode.",
    ),
    no_dom: Optional[bool] = typer.Option(
        None,
        "--no-dom/--dom",
        help="(Deprecated) Skip DOM snapshot.",
    ),
    som: Optional[bool] = typer.Option(
        None,
        "--som/--no-som",
        help="(Deprecated) Draw SoM labels.",
    ),
    diagnostics: Optional[bool] = typer.Option(
        None,
        "--diagnostics/--no-diagnostics",
        help="(Deprecated) Include diagnostics.",
    ),
    semantic: Optional[bool] = typer.Option(
        None,
        "--semantic/--no-semantic",
        help="(Deprecated) Include semantic snapshot.",
    ),
):
    instance_id = ctx.obj["instance_id"]
    if omniparser and not os.getenv("BUSE_OMNIPARSER_URL"):
        raise typer.BadParameter(
            "BUSE_OMNIPARSER_URL environment variable is required when using --omniparser."
        )

    async def run():
        try:
            visual_value = visual.default if hasattr(visual, "default") else visual
            text_value = text.default if hasattr(text, "default") else text
            mode_value = mode.default if hasattr(mode, "default") else mode
            no_dom_value = _resolve_option_default(no_dom, None)
            som_value = _resolve_option_default(som, None)
            diagnostics_value = _resolve_option_default(diagnostics, None)
            semantic_value = _resolve_option_default(semantic, None)
            max_chars_value = _resolve_option_default(max_chars, None)
            max_labels_value = _resolve_option_default(max_labels, None)
            selector_value = _resolve_option_default(selector, None)
            frame_value = _resolve_option_default(frame, None)
            human_value = _resolve_option_default(human, False)
            legacy_flags = any(
                v is not None
                for v in [screenshot, omniparser, no_dom, som, diagnostics, semantic]
            )
            if legacy_flags:
                do_screenshot = bool(screenshot) if screenshot is not None else False
                do_omni = bool(omniparser) if omniparser is not None else False
                do_som = bool(som_value) if som_value is not None else False
                do_semantic = (
                    bool(semantic_value) if semantic_value is not None else False
                )
                do_no_dom = bool(no_dom_value) if no_dom_value is not None else False
                do_diagnostics = (
                    bool(diagnostics_value) if diagnostics_value is not None else False
                )
            else:
                do_screenshot = visual_value != VisualMode.none
                do_som = visual_value == VisualMode.som
                do_omni = visual_value == VisualMode.omni
                do_semantic = text_value == TextMode.ai
                do_no_dom = text_value != TextMode.dom
                do_diagnostics = True
            if do_omni:
                do_screenshot = True
            data = await get_observation(
                instance_id,
                screenshot=do_screenshot,
                path=path,
                omniparser=do_omni,
                no_dom=do_no_dom,
                som=do_som,
                diagnostics=do_diagnostics,
                semantic=do_semantic,
                mode=mode_value.value
                if hasattr(mode_value, "value")
                else str(mode_value),
                max_chars=max_chars_value,
                max_labels=max_labels_value,
                selector=selector_value,
                frame=frame_value,
            )
            if human_value:
                _output_human(data)
            else:
                output_data(data)
        except Exception as e:
            _output_error(
                "observe",
                {
                    "visual": visual_value,
                    "text": text_value,
                    "mode": mode_value,
                },
                str(e),
                error_details=_build_error_details(
                    "observe", exception_type=type(e).__name__
                ),
            )
            sys.exit(1)

    asyncio.run(run())


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
    help="Click by index, selector, id, or coordinates.",
    epilog="Examples: buse b1 click 12 | buse b1 click e3 | buse b1 click --selector '#submit' | buse b1 click --x 200 --y 300 --ctrl",
)
def click(
    ctx: typer.Context,
    index: Optional[str] = typer.Argument(
        None, help="Element index or ref (eN) from the observe command."
    ),
    selector: Optional[str] = typer.Option(
        None, "--selector", help="CSS selector to click."
    ),
    x: Optional[int] = typer.Option(None, "--x", help="X coordinate for raw click."),
    y: Optional[int] = typer.Option(None, "--y", help="Y coordinate for raw click."),
    element_id: Optional[str] = typer.Option(
        None, "--id", help="The HTML 'id' attribute of the element."
    ),
    element_class: Optional[str] = typer.Option(
        None, "--class", help="The HTML 'class' attribute of the element."
    ),
    right: bool = typer.Option(False, "--right", help="Perform a right-click."),
    middle: bool = typer.Option(False, "--middle", help="Perform a middle-click."),
    double: bool = typer.Option(False, "--double", help="Perform a double-click."),
    ctrl: bool = typer.Option(False, "--ctrl", help="Hold Control/Cmd key."),
    shift: bool = typer.Option(False, "--shift", help="Hold Shift key."),
    alt: bool = typer.Option(False, "--alt", help="Hold Alt key."),
    meta: bool = typer.Option(False, "--meta", help="Hold Meta/Command key."),
    force: bool = typer.Option(False, "--force", help="Bypass actionability checks."),
    debug: bool = typer.Option(
        False, "--debug", help="Include actionability debug info on failures."
    ),
):
    instance_id = ctx.obj["instance_id"]
    selector = _resolve_option_default(selector, None)
    x = _resolve_option_default(x, None)
    y = _resolve_option_default(y, None)
    element_id = _resolve_option_default(element_id, None)
    element_class = _resolve_option_default(element_class, None)
    right = _resolve_option_default(right, False)
    middle = _resolve_option_default(middle, False)
    double = _resolve_option_default(double, False)
    ctrl = _resolve_option_default(ctrl, False)
    shift = _resolve_option_default(shift, False)
    alt = _resolve_option_default(alt, False)
    meta = _resolve_option_default(meta, False)
    force = _resolve_option_default(force, False)
    debug = _resolve_option_default(debug, False)
    if index is not None and not str(index).strip():
        index = None
    if all(v is None for v in [index, selector, x, y, element_id, element_class]):
        raise typer.BadParameter(
            "Provide an index, selector, --id/--class, or --x/--y."
        )
    if (x is None) != (y is None):
        raise typer.BadParameter("Provide both --x and --y for coordinate clicks.")

    async def run():
        try:
            session_info = session_manager.get_session(instance_id)
            if not session_info:
                params = {
                    "selector": selector,
                    "element_id": element_id,
                    "element_class": element_class,
                    "coordinate_x": x,
                    "coordinate_y": y,
                }
                if index is not None:
                    params["index"] = (
                        int(str(index)) if str(index).isdigit() else str(index)
                    )
                await execute_tool(
                    instance_id,
                    "click",
                    params,
                    needs_selector_map=(index is not None),
                    action_label="click",
                )
                return
            browser_session, _ = await _get_browser_session(instance_id, session_info)
            cdp_session = await browser_session.get_or_create_cdp_session()
            button = "right" if right else "middle" if middle else "left"
            click_count = 2 if double else 1
            modifiers = []
            if ctrl:
                modifiers.append("ctrl")
            if shift:
                modifiers.append("shift")
            if alt:
                modifiers.append("alt")
            if meta:
                modifiers.append("meta")
            if x is not None and y is not None:
                mask = 0
                if alt:
                    mask |= 1
                if ctrl:
                    mask |= 2
                if meta:
                    mask |= 4
                if shift:
                    mask |= 8
                params = {
                    "x": x,
                    "y": y,
                    "button": button,
                    "clickCount": click_count,
                    "modifiers": mask,
                }
                try:
                    await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
                        params={**params, "type": "mousePressed"},
                        session_id=cdp_session.session_id,
                    )
                    await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
                        params={**params, "type": "mouseReleased"},
                        session_id=cdp_session.session_id,
                    )
                    output_data({"success": True, "action": "click", "x": x, "y": y})
                    return
                except AttributeError:
                    await execute_tool(
                        instance_id,
                        "click",
                        {
                            "coordinate_x": x,
                            "coordinate_y": y,
                            "element_id": element_id,
                            "element_class": element_class,
                        },
                        action_label="click",
                    )
                    return
            target: Optional[object] = None
            if index is not None:
                if str(index).isdigit():
                    target = int(str(index))
                else:
                    target = str(index)
            else:
                target = selector
            if target is None:
                if element_id:
                    target = f"#{element_id}"
                elif element_class:
                    target = f".{element_class}"
            if target is None:
                raise ValueError("Provide a click target.")
            try:
                await _ensure_selector_map(browser_session, instance_id)
                selector_map = await browser_session.get_selector_map()
                target_id = browser_session.agent_focus_target_id

                async def refresh_selector_map():
                    await _ensure_selector_map(browser_session, instance_id, force=True)
                    return await browser_session.get_selector_map()

                result = await interaction.click_with_retry(
                    cdp_session,
                    target,
                    selector_map=selector_map,
                    refresh_selector_map=refresh_selector_map,
                    modifiers=modifiers,
                    button=button,
                    click_count=click_count,
                    force=force,
                    timeout_ms=5000,
                    debug=debug,
                    target_id=target_id,
                    instance_id=instance_id,
                    persist_path=str(_semantic_refs_path(session_info)),
                )
                output_data(
                    {
                        "success": True,
                        "action": "click",
                        "target": target,
                        "coords": result,
                    }
                )
            except AttributeError:
                params = {
                    "selector": selector,
                    "element_id": element_id,
                    "element_class": element_class,
                }
                if index is not None:
                    params["index"] = (
                        int(str(index)) if str(index).isdigit() else str(index)
                    )
                await execute_tool(
                    instance_id,
                    "click",
                    params,
                    needs_selector_map=(index is not None),
                    action_label="click",
                )
        finally:
            if not _should_keep_session():
                await _stop_cached_browser_session(instance_id)

    @handle_errors(action="click")
    async def wrapper():
        await run()

    asyncio.run(wrapper())


@instance_app.command(
    help="Input text into form fields.",
    epilog='Examples: buse b1 input 12 "hello" | buse b1 input e3 --slowly | buse b1 input --id email --text "a@b.com" --submit',
)
def input(
    ctx: typer.Context,
    index: Optional[str] = typer.Argument(
        None, help="Element index or ref (eN) from the observe command."
    ),
    text: Optional[str] = typer.Argument(None, help="Text to input."),
    text_opt: Optional[str] = typer.Option(
        None, "--text", help="Text to input (alternative to positional)."
    ),
    element_id: Optional[str] = typer.Option(
        None, "--id", help="The HTML 'id' attribute of the element."
    ),
    element_class: Optional[str] = typer.Option(
        None, "--class", help="The HTML 'class' attribute of the element."
    ),
    slowly: bool = typer.Option(False, "--slowly", help="Type slowly (key by key)."),
    submit: bool = typer.Option(False, "--submit", help="Press Enter after typing."),
    append: bool = typer.Option(
        False, "--append", help="Append to existing text instead of replacing."
    ),
):
    element_id = _resolve_option_default(element_id, None)
    element_class = _resolve_option_default(element_class, None)
    slowly = _resolve_option_default(slowly, False)
    submit = _resolve_option_default(submit, False)
    append = _resolve_option_default(append, False)
    resolved_text = text if text is not None else text_opt
    if resolved_text is None:
        raise typer.BadParameter(
            'Provide text as a positional arg or --text. Example: buse b1 input 12 "hello".'
        )
    if index is None and element_id is None and element_class is None:
        raise typer.BadParameter(
            'Provide an index or --id/--class. Example: buse b1 input 12 "hello".'
        )
    if index is not None and not str(index).strip():
        index = None
    if append or slowly or submit or (index is not None and not str(index).isdigit()):
        instance_id = ctx.obj["instance_id"]

        def _parse_target(value: Optional[str]) -> Optional[Union[int, str]]:
            if value is None:
                return None
            return int(value) if str(value).isdigit() else str(value)

        async def run():
            try:
                session_info = session_manager.get_session(instance_id)
                if not session_info:
                    raise ValueError(f"Instance {instance_id} not found.")
                browser_session, _ = await _get_browser_session(
                    instance_id, session_info
                )
                cdp_session = await browser_session.get_or_create_cdp_session()
                await _ensure_selector_map(browser_session, instance_id)
                selector_map = await browser_session.get_selector_map()
                target_id = browser_session.agent_focus_target_id
                target = _parse_target(index)
                if target is None:
                    if element_id:
                        target = f"#{element_id}"
                    elif element_class:
                        target = f".{element_class}"
                if target is None:
                    raise ValueError("Provide an index/ref or --id/--class.")
                result = await interaction.type_text(
                    cdp_session,
                    target,
                    resolved_text,
                    selector_map=selector_map,
                    slowly=slowly,
                    submit=submit,
                    append=append,
                    target_id=target_id,
                    instance_id=instance_id,
                    persist_path=str(_semantic_refs_path(session_info)),
                )
                output_data({"success": True, "action": "input", "result": result})
            finally:
                if not _should_keep_session():
                    await _stop_cached_browser_session(instance_id)

        @handle_errors(action="input")
        async def wrapper():
            await run()

        asyncio.run(wrapper())
        return
    params = {"text": resolved_text}
    if index is not None:
        params["index"] = int(str(index))
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
    help="Fill multiple fields in a single command.",
    epilog='Example: buse b1 fill \'[{"ref": "e1", "value": "user"}, {"ref": 2, "value": "pass", "type": "text"}]\'',
)
def fill(
    ctx: typer.Context,
    data: str = typer.Argument(..., help="JSON list of field updates."),
):
    instance_id = ctx.obj["instance_id"]
    try:
        fields = json.loads(data)
    except Exception as exc:
        raise typer.BadParameter("Invalid JSON for fill data.") from exc
    if not isinstance(fields, list):
        raise typer.BadParameter("Fill data must be a JSON list.")

    async def run():
        try:
            session_info = session_manager.get_session(instance_id)
            if not session_info:
                raise ValueError(f"Instance {instance_id} not found.")
            browser_session, _ = await _get_browser_session(instance_id, session_info)
            cdp_session = await browser_session.get_or_create_cdp_session()
            await _ensure_selector_map(browser_session, instance_id)
            selector_map = await browser_session.get_selector_map()
            target_id = browser_session.agent_focus_target_id
            result = await interaction.fill_form_atomic(
                cdp_session,
                fields,
                selector_map=selector_map,
                target_id=target_id,
                instance_id=instance_id,
                persist_path=str(_semantic_refs_path(session_info)),
            )
            output_data(result)
        finally:
            if not _should_keep_session():
                await _stop_cached_browser_session(instance_id)

    @handle_errors(action="fill")
    async def wrapper():
        await run()

    asyncio.run(wrapper())


@instance_app.command(
    help="Drag and drop between two elements.",
    epilog="Examples: buse b1 drag 12 13 | buse b1 drag e1 e2",
)
def drag(
    ctx: typer.Context,
    start: str = typer.Argument(..., help="Start element index or ref (eN)."),
    end: str = typer.Argument(..., help="End element index or ref (eN)."),
    html5: bool = typer.Option(
        True, "--html5/--no-html5", help="Dispatch DragEvent/DataTransfer fallback."
    ),
):
    instance_id = ctx.obj["instance_id"]

    def _parse_target(value: str) -> Union[int, str]:
        return int(value) if value.isdigit() else value

    async def run():
        try:
            session_info = session_manager.get_session(instance_id)
            if not session_info:
                raise ValueError(f"Instance {instance_id} not found.")
            browser_session, _ = await _get_browser_session(instance_id, session_info)
            cdp_session = await browser_session.get_or_create_cdp_session()
            await _ensure_selector_map(browser_session, instance_id)
            selector_map = await browser_session.get_selector_map()
            target_id = browser_session.agent_focus_target_id
            result = await interaction.drag_and_drop(
                cdp_session,
                _parse_target(start),
                _parse_target(end),
                selector_map=selector_map,
                html5=html5,
                target_id=target_id,
                instance_id=instance_id,
                persist_path=str(_semantic_refs_path(session_info)),
            )
            output_data({"success": True, "action": "drag", "result": result})
        finally:
            if not _should_keep_session():
                await _stop_cached_browser_session(instance_id)

    @handle_errors(action="drag")
    async def wrapper():
        await run()

    asyncio.run(wrapper())


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
    index: Optional[int] = typer.Argument(
        None, help="Element index from the observe command."
    ),
    element_id: Optional[str] = typer.Option(
        None, "--id", help="The HTML 'id' attribute of the element."
    ),
    element_class: Optional[str] = typer.Option(
        None, "--class", help="The HTML 'class' attribute of the element."
    ),
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
    index: Optional[int] = typer.Argument(
        None, help="Element index from the observe command."
    ),
    text: Optional[str] = typer.Argument(
        None, help="Visible text of the option to select."
    ),
    text_opt: Optional[str] = typer.Option(
        None, "--text", help="Visible text of the option (alternative to positional)."
    ),
    element_id: Optional[str] = typer.Option(
        None, "--id", help="The HTML 'id' attribute of the element."
    ),
    element_class: Optional[str] = typer.Option(
        None, "--class", help="The HTML 'class' attribute of the element."
    ),
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
            code = (
                "window.scrollBy({top: window.innerHeight * "
                + str(pages)
                + " * "
                + str(direction)
                + ", behavior: 'smooth'})"
            )
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
    index: Optional[int] = typer.Argument(
        None, help="Element index from the observe command."
    ),
    element_id: Optional[str] = typer.Option(
        None, "--id", help="The HTML 'id' attribute of the element."
    ),
    element_class: Optional[str] = typer.Option(
        None, "--class", help="The HTML 'class' attribute of the element."
    ),
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
    help="Wait for specified seconds or a condition.",
    epilog='Examples: buse b1 wait 2 | buse b1 wait --text "Done" --timeout 10',
)
def wait(
    ctx: typer.Context,
    seconds: Optional[float] = typer.Argument(
        None, help="Seconds to wait (optional if using conditions)."
    ),
    text: Optional[str] = typer.Option(None, "--text", help="Wait until text appears."),
    selector: Optional[str] = typer.Option(
        None, "--selector", help="Wait until selector appears."
    ),
    network_idle: Optional[int] = typer.Option(
        None, "--network-idle", help="Wait until network is idle for N ms."
    ),
    timeout: int = typer.Option(
        10, "--timeout", help="Timeout in seconds for condition waits."
    ),
):
    instance_id = ctx.obj["instance_id"]
    text = _resolve_option_default(text, None)
    selector = _resolve_option_default(selector, None)
    network_idle = _resolve_option_default(network_idle, None)
    timeout = _resolve_option_default(timeout, 10)
    has_condition = any(v is not None for v in [text, selector, network_idle])
    if not has_condition and seconds is None:
        raise typer.BadParameter("Provide seconds or a wait condition.")
    if seconds is not None and seconds < 0:
        raise typer.BadParameter("Use a non-negative number of seconds.")
    if not has_condition:
        asyncio.run(
            execute_tool(
                instance_id,
                "wait",
                {"seconds": seconds},
                action_label="wait",
            )
        )
        return

    async def run():
        try:
            session_info = session_manager.get_session(instance_id)
            if not session_info:
                raise ValueError(f"Instance {instance_id} not found.")
            browser_session, _ = await _get_browser_session(instance_id, session_info)
            cdp_session = await browser_session.get_or_create_cdp_session()
            result = await interaction.wait_for_condition(
                cdp_session,
                text=text,
                selector=selector,
                network_idle_ms=network_idle,
                timeout_ms=int(timeout * 1000),
            )
            output_data({"success": True, "action": "wait", "result": result})
        finally:
            if not _should_keep_session():
                await _stop_cached_browser_session(instance_id)

    @handle_errors(action="wait")
    async def wrapper():
        await run()

    asyncio.run(wrapper())


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
        browser_session: Any = BrowserSession(cdp_url=session_info.cdp_url)
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
            "  fill               Fill multiple form fields\n"
            "  drag               Drag and drop between elements\n"
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
