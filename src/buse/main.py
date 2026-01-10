import asyncio
import base64
import json
import logging
import os
import sys
import time
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
_selector_cache = {}
_SELECTOR_CACHE_TTL_SECONDS = 2.0


def _should_keep_session() -> bool:
    return os.getenv("BUSE_KEEP_SESSION", "").lower() in {"1", "true", "yes"}


async def _stop_cached_browser_session(instance_id: str) -> None:
    browser_session = _browser_sessions.pop(instance_id, None)
    if browser_session is not None:
        await browser_session.stop()
    _file_systems.pop(instance_id, None)
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

    profile = {}
    start_total = time.perf_counter()
    start_session = time.perf_counter()
    session_info = session_manager.get_session(instance_id)
    profile["get_session_ms"] = (time.perf_counter() - start_session) * 1000.0
    if session_info is None:
        await _stop_cached_browser_session(instance_id)
        _output_error(label, params_for_hint, f"Instance {instance_id} not found.", profile=profile)
        sys.exit(1)

    start_browser = time.perf_counter()
    browser_session, file_system = await _get_browser_session(instance_id, session_info)
    profile["get_browser_session_ms"] = (time.perf_counter() - start_browser) * 1000.0

    try:
        if needs_selector_map:
            start_selector = time.perf_counter()
            await _ensure_selector_map(browser_session, instance_id)
            profile["selector_map_ms"] = (time.perf_counter() - start_selector) * 1000.0

        element_id = params.pop("element_id", None)
        element_class = params.pop("element_class", None)
        if params.get("index") is None and (element_id or element_class):
            resolved_index = None
            if element_id:
                resolved_index = await browser_session.get_index_by_id(element_id)
            if resolved_index is None and element_class:
                resolved_index = await browser_session.get_index_by_class(element_class)
            if resolved_index is None:
                start_selector = time.perf_counter()
                await _ensure_selector_map(browser_session, instance_id, force=True)
                profile["selector_map_refresh_ms"] = (time.perf_counter() - start_selector) * 1000.0
                if element_id:
                    resolved_index = await browser_session.get_index_by_id(element_id)
                if resolved_index is None and element_class:
                    resolved_index = await browser_session.get_index_by_class(element_class)
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
                    cdp_session = await browser_session.get_or_create_cdp_session()
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
                        result = await cdp_session.cdp_client.send.Runtime.evaluate(
                            params={"expression": code, "returnByValue": True},
                            session_id=cdp_session.session_id,
                        )
                        value = (
                            result.get("result", {}).get("value", {})
                            if result
                            else {}
                        )
                        if value.get("error"):
                            _output_error(action_name, params_for_hint, value["error"], profile=profile)
                            sys.exit(1)
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
                        profile["total_ms"] = (time.perf_counter() - start_total) * 1000.0
                        output_data(
                            ActionResult(
                                success=True,
                                action=action_name,
                                message=msg,
                                error=None,
                                extracted_content=msg,
                                profile=profile if state.profile else None,
                            )
                        )
                        return
                    if action_name == "select_dropdown":
                        text = params.get("text", "")
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
                        result = await cdp_session.cdp_client.send.Runtime.evaluate(
                            params={"expression": code, "returnByValue": True},
                            session_id=cdp_session.session_id,
                        )
                        value = (
                            result.get("result", {}).get("value", {})
                            if result
                            else {}
                        )
                        if value.get("error"):
                            _output_error(action_name, params_for_hint, value["error"], profile=profile)
                            sys.exit(1)
                        msg = (
                            f'Selected option: {value.get("text")} (value: {value.get("value")})'
                        )
                        profile["total_ms"] = (time.perf_counter() - start_total) * 1000.0
                        output_data(
                            ActionResult(
                                success=True,
                                action=action_name,
                                message=msg,
                                error=None,
                                extracted_content=msg,
                                profile=profile if state.profile else None,
                            )
                        )
                        return

            if resolved_index is None:
                error_msg = "Could not resolve element index"
                _output_error(
                    action_name,
                    params_for_hint,
                    f"{error_msg} (id={element_id}, class={element_class})",
                    profile=profile,
                )
                sys.exit(1)
            params["index"] = resolved_index

        controller = Controller()
        if action_name == "click" and (
            params.get("coordinate_x") is not None
            or params.get("coordinate_y") is not None
        ):
            controller.set_coordinate_clicking(True)
        start_action = time.perf_counter()
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
        profile["total_ms"] = (time.perf_counter() - start_total) * 1000.0
        output_data(
            ActionResult(
                success=not error,
                action=label,
                message=result.extracted_content if not error else None,
                error=error,
                extracted_content=result.extracted_content
                if not error
                else None,
                profile=profile if state.profile else None,
            )
        )
        if error:
            sys.exit(1)

        if action_name in {"navigate", "go_back", "search"} or label == "refresh":
            _selector_cache.pop(instance_id, None)
    except Exception as e:
        profile["total_ms"] = (time.perf_counter() - start_total) * 1000.0
        _output_error(label, params_for_hint, f"{label} failed: {type(e).__name__}: {e}", profile=profile)
        sys.exit(1)
    finally:
        if not _should_keep_session():
            await _stop_cached_browser_session(instance_id)


@instance_app.command(
    help="Observe current state.",
    epilog="Example: buse b1 observe --screenshot",
)
def observe(
    ctx: typer.Context, screenshot: bool = typer.Option(False, help="Take a screenshot")
):
    instance_id = ctx.obj["instance_id"]

    async def run():
        session_info = session_manager.get_session(instance_id)
        if not session_info:
            await _stop_cached_browser_session(instance_id)
            _output_error("observe", {}, f"Instance {instance_id} not found.")
            sys.exit(1)
        browser_session, _ = await _get_browser_session(instance_id, session_info)

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
            ctx.obj["instance_id"], "input", params, needs_selector_map=True
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
    if index is not None:
        selector = f'[highlight_index="{index}"]'
    elif element_id is not None:
        selector = f'#{element_id}'
    else:
        selector = f'.{element_class}'
    code = (
        f'(function() {{ const el = document.querySelector("{selector}"); '
        'if (el) { el.dispatchEvent(new MouseEvent("mouseover", {bubbles: true})); return "Hovered"; } '
        'return "Not found"; }})()'
    )
    asyncio.run(
        execute_tool(
            ctx.obj["instance_id"],
            "evaluate",
            {"code": code},
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


def app():
    args = sys.argv[1:]
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
            "  buse [--format json|toon] [--profile] list           List active instances\n"
            "  buse [--format json|toon] [--profile] <id>           Start/initialize an instance\n"
            "  buse [--format json|toon] [--profile] <id> observe   Observe instance state\n"
            "  buse [--format json|toon] [--profile] <id> <command> Execute an action\n\n"
            "Instance commands:\n"
            "  observe            Observe current state\n"
            "  navigate           Navigate to URL\n"
            "  new-tab            Open a URL in a new tab\n"
            "  search             Search the web (duckduckgo, google, bing)\n"
            "  click              Click by index OR coordinates\n"
            "  input              Input text into form fields\n"
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


if __name__ == "__main__":
    app()
