import asyncio
import base64
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import typer

from .models import ActionResult, Observation, TabInfo
from .session import SessionManager
from .utils import OutputFormat, handle_errors, output_data, state

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
_browser_sessions = {}
_file_systems = {}
_controllers = {}
_selector_cache = {}
_SELECTOR_CACHE_TTL_SECONDS = 2.0


def _should_keep_session() -> bool:
    return os.getenv("BUSE_KEEP_SESSION", "").lower() in {"1", "true", "yes"}


async def _stop_cached_browser_session(instance_id: str) -> None:
    browser_session = _browser_sessions.pop(instance_id, None)
    if browser_session is not None:
        await browser_session.stop()
    _file_systems.pop(instance_id, None)
    _controllers.pop(instance_id, None)
    _selector_cache.pop(instance_id, None)


async def _get_browser_session(instance_id: str, session_info):
    from browser_use.browser import BrowserSession
    from browser_use.filesystem.file_system import FileSystem

    browser_session = _browser_sessions.get(instance_id)
    file_system = _file_systems.get(instance_id)

    if browser_session is None:
        browser_session = BrowserSession(cdp_url=session_info.cdp_url)
        await browser_session.start()
        _browser_sessions[instance_id] = browser_session

    if file_system is None:
        file_system = FileSystem(base_dir=session_info.user_data_dir)
        _file_systems[instance_id] = file_system

    return browser_session, file_system


async def _ensure_selector_map(browser_session, instance_id: str, force: bool = False) -> None:
    now = time.time()
    last = _selector_cache.get(instance_id, 0.0)
    if not force and now - last < _SELECTOR_CACHE_TTL_SECONDS:
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

    def __init__(self, defer_output: bool, params_for_hint: dict, profile: dict) -> None:
        self.defer_output = defer_output
        self.params_for_hint = params_for_hint
        self.profile = profile
        self.result_payload = None
        self.exit_code = 0

    def _emit(self, payload, code: int = 0) -> None:
        if self.defer_output:
            self.result_payload = payload
            self.exit_code = code
            return
        output_data(payload)
        if code:
            sys.exit(code)

    def emit_error(self, action: str, message: str) -> None:
        payload = ActionResult(
            success=False,
            action=action,
            message=None,
            error=_augment_error(action, self.params_for_hint, message),
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

    def emit_result(self, action: str, message: Optional[str], error: Optional[str], extracted_content) -> None:
        payload = ActionResult(
            success=not error,
            action=action,
            message=message,
            error=error,
            extracted_content=extracted_content if not error else None,
            profile=self.profile if state.profile else None,
        )
        self._emit(payload, code=1 if error else 0)

    def fail(self, action: str, message: str) -> None:
        self.emit_error(action, message)
        if self.defer_output:
            raise ResultEmitter.EarlyExit()

    def finalize(self) -> None:
        if self.defer_output and isinstance(self.result_payload, ActionResult):
            self.result_payload.profile = self.profile if state.profile else None
        if self.defer_output and self.result_payload is not None:
            output_data(self.result_payload)
            if self.exit_code:
                sys.exit(self.exit_code)


async def _get_navigation_timings(browser_session) -> dict[str, float]:
    cdp_session = await browser_session.get_or_create_cdp_session()
    code = (
        "(function(){"
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
    value = result.get("result", {}).get("value", {}) if result else {}
    if not isinstance(value, dict):
        return {}
    return {
        k: float(v)
        for k, v in value.items()
        if isinstance(v, (int, float)) and v >= 0
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
        "(function(){"
        "const selector="
        + json.dumps(selector)
        + ";"
        "const findDeep=(root)=>{"
        "const el=root.querySelector(selector);"
        "if(el){return el;}"
        "const nodes=root.querySelectorAll('*');"
        "for(const node of nodes){"
        "if(node.shadowRoot){"
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
        code += (
            "el.click();"
            "return {ok:true,tag:tag};"
        )
    elif action_name == "hover":
        code += (
            "el.dispatchEvent(new MouseEvent('mouseover',{bubbles:true}));"
            "return {ok:true,tag:tag};"
        )
    else:
        code += (
            "const value="
            + json.dumps(text)
            + ";"
            "if('value' in el){"
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
    value = (
        result.get("result", {}).get("value", {})
        if result
        else {}
    )
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
            "(function(){"
            "const sel=document.querySelector("
            + json.dumps(selector)
            + ");"
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
        value = (
            result.get("result", {}).get("value", {})
            if result
            else {}
        )
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
            "(function(){"
            "const sel=document.querySelector("
            + json.dumps(selector)
            + ");"
            "let el=sel;"
            "if(el && el.tagName!=='SELECT'){el=el.querySelector('select')||el.closest('select');}"
            "if(!el||el.tagName!=='SELECT'){return {error:'Select element not found'};}"
            "const target="
            + json.dumps(text)
            + ";"
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
        value = (
            result.get("result", {}).get("value", {})
            if result
            else {}
        )
        if value.get("error"):
            return True, value.get("error"), None
        msg = (
            f'Selected option: {value.get("text")} (value: {value.get("value")})'
        )
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
    if action_name in {"click", "input", "dropdown_options", "select_dropdown", "hover"}:
        missing_index = params.get("index") is None
        missing_resolver = not (params.get("element_id") or params.get("element_class"))
        missing_coords = not (
            params.get("coordinate_x") is not None and params.get("coordinate_y") is not None
        )
        if missing_index and missing_resolver and missing_coords:
            hint = "Provide an index or use --id/--class, or use --x/--y for coordinates."
        elif missing_index and missing_resolver:
            hint = "Provide an index or use --id/--class (run observe to get indices)."
    if action_name == "click" and (
        params.get("coordinate_x") is None or params.get("coordinate_y") is None
    ):
        hint = "Provide both --x and --y for coordinate clicks."
    if "Could not resolve element index" in error:
        hint = "Run `buse <id> observe` and use an index or pass the actual <select> --id/--class."
    if "Element index" in error and "not available" in error:
        hint = "The page likely changed. Run observe to refresh indices."
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
    if hint:
        return f"{error} Hint: {hint}"
    return error


def _output_error(action: str, params: dict, message: str, profile: Optional[dict[str, float]] = None):
    output_data(
        ActionResult(
            success=False,
            action=action,
            message=None,
            error=_augment_error(action, params, message),
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
    **extra_kwargs,
):
    from browser_use import Controller

    params_for_hint = dict(params)
    label = action_label or action_name
    defer_output = state.profile
    profiler = Profiler()
    profile = profiler.data
    emitter = ResultEmitter(defer_output, params_for_hint, profile)
    start_total = time.perf_counter()
    with profiler.span("get_session_ms"):
        session_info = session_manager.get_session(instance_id)
    if session_info is None:
        await _stop_cached_browser_session(instance_id)
        emitter.fail(label, f"Instance {instance_id} not found.")

    profile["browser_session_cached"] = 1.0 if instance_id in _browser_sessions else 0.0
    profile["file_system_cached"] = 1.0 if instance_id in _file_systems else 0.0
    with profiler.span("get_browser_session_ms"):
        browser_session, file_system = await _get_browser_session(instance_id, session_info)

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
                # Fallback: resolve and operate on native <select> via JS when selector_map misses it.
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
                        profile["total_ms"] = (time.perf_counter() - start_total) * 1000.0
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
        if error:
            error = _augment_error(label, params_for_hint, error)
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
        emitter.fail(label, f"{label} failed: {type(e).__name__}: {e}")
    finally:
        if not _should_keep_session():
            cleanup_start = time.perf_counter()
            await _stop_cached_browser_session(instance_id)
            profile["cleanup_ms"] = (time.perf_counter() - cleanup_start) * 1000.0
    emitter.finalize()


@instance_app.command(
    help="Observe current state.",
    epilog="Example: buse b1 observe --screenshot",
)
def observe(
    ctx: typer.Context, screenshot: bool = typer.Option(False, help="Take a screenshot")
):
    instance_id = ctx.obj["instance_id"]

    async def run():
        start_session = time.perf_counter()
        session_info = session_manager.get_session(instance_id)
        get_session_ms = (time.perf_counter() - start_session) * 1000.0
        if not session_info:
            await _stop_cached_browser_session(instance_id)
            _output_error("observe", {}, f"Instance {instance_id} not found.")
            sys.exit(1)
        start_browser = time.perf_counter()
        browser_session, _ = await _get_browser_session(instance_id, session_info)
        get_browser_session_ms = (time.perf_counter() - start_browser) * 1000.0

        try:
            start_state = time.perf_counter()
            state_summary = await browser_session.get_browser_state_summary(
                include_screenshot=screenshot
            )
            state_ms = (time.perf_counter() - start_state) * 1000.0
            _selector_cache[instance_id] = time.time()
            focused_id = browser_session.agent_focus_target_id
            start_tabs = time.perf_counter()
            tabs = [
                TabInfo(id=t.target_id, title=t.title, url=t.url)
                for t in await browser_session.get_tabs()
            ]
            tabs_ms = (time.perf_counter() - start_tabs) * 1000.0

            screenshot_path = None
            screenshot_ms = 0.0
            if screenshot and state_summary.screenshot:
                start_shot = time.perf_counter()
                shot_dir = Path(session_info.user_data_dir) / "screenshots"
                shot_dir.mkdir(exist_ok=True)
                screenshot_path = str(shot_dir / "last_state.png")
                with open(screenshot_path, "wb") as f:
                    f.write(base64.b64decode(state_summary.screenshot))
                screenshot_ms = (time.perf_counter() - start_shot) * 1000.0

            obs = Observation(
                session_id=instance_id,
                url=state_summary.url,
                title=state_summary.title,
                tabs=tabs,
                screenshot_path=screenshot_path,
                dom_minified=state_summary.dom_state.llm_representation()
                if state_summary.dom_state
                else "",
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
                }
            output_data(data)

        finally:
            if not _should_keep_session():
                await _stop_cached_browser_session(instance_id)

    @handle_errors
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
        execute_tool(
            ctx.obj["instance_id"], "navigate", {"url": url, "new_tab": True}
        )
    )


@instance_app.command(
    help="Search the web (duckduckgo, google, bing).",
    epilog="Example: buse b1 search \"site:example.com\" --engine duckduckgo",
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
    if index is None and x is None and y is None and element_id is None and element_class is None:
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
    epilog="Examples: buse b1 input 12 \"hello\" | buse b1 input --id email --text \"a@b.com\"",
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
            "Provide text as a positional arg or --text. Example: buse b1 input 12 \"hello\"."
        )
    if index is None and element_id is None and element_class is None:
        raise typer.BadParameter(
            "Provide an index or --id/--class. Example: buse b1 input 12 \"hello\"."
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
    epilog="Example: buse b1 send-keys \"Enter\"",
)
def send_keys(
    ctx: typer.Context,
    keys: str = typer.Argument(..., help="Keys to send (e.g. Enter, Space, A)"),
):
    asyncio.run(
        execute_tool(
            ctx.obj["instance_id"],
            "send_keys",
            {"keys": keys},
        )
    )


@instance_app.command(
    help="Scroll to text on the page.",
    epilog="Example: buse b1 find-text \"Contact Us\"",
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
    epilog="Examples: buse b1 select-dropdown 5 \"Canada\" | buse b1 select-dropdown --id country --text \"Canada\"",
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
            "Provide text as a positional arg or --text. Example: buse b1 select-dropdown 5 \"Option\"."
        )
    if index is None and element_id is None and element_class is None:
        raise typer.BadParameter(
            "Provide an index or --id/--class. Example: buse b1 select-dropdown 5 \"Option\"."
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
    epilog="Examples: buse b1 scroll --pages 2 | buse b1 scroll --index 12 --pages 0.5",
)
def scroll(
    ctx: typer.Context,
    down: bool = True,
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
        from browser_use.browser import BrowserSession

        session_info = session_manager.get_session(instance_id)
        if session_info is None:
            _output_error("save_state", {"path": path}, f"Instance {instance_id} not found.")
            sys.exit(1)
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
    epilog="Example: buse b1 extract \"List all form fields\"",
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
    epilog="Example: buse b1 evaluate \"(function(){return document.title})()\"",
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
        _output_error("stop", {}, f"Instance {instance_id} not found.")
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
            from rich import print as rprint

            rprint(f"[bold red]Error:[/bold red] {e}")
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
