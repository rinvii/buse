import asyncio
import logging
import time
import json
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


async def fill_form_atomic(
    cdp_session: Any,
    fields: List[Dict[str, Any]],
    selector_map: Optional[Dict[int, Any]] = None,
    target_id: Optional[str] = None,
    instance_id: Optional[str] = None,
    persist_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fill multiple form fields in sequence with input/change/blur events.
    """
    results = []
    for field in fields:
        ref = field.get("ref")
        if ref is None:
            results.append({"ref": None, "ok": False, "error": "Missing ref"})
            continue
        value = field.get("value")
        field_type = field.get("type") or "text"
        target: Union[int, str] = ref
        if isinstance(ref, str) and ref.isdigit():
            target = int(ref)
        try:
            object_id = await resolve_element(
                cdp_session,
                target,
                selector_map,
                target_id=target_id,
                instance_id=instance_id,
                persist_path=persist_path,
            )
            if not object_id:
                results.append(
                    {"ref": ref, "ok": False, "error": "Could not resolve target"}
                )
                continue
            js_set = """
            function(value, type) {
                const el = this;
                const t = String(type || "text").toLowerCase();
                if (t === "checkbox" || t === "radio") {
                    const checked = value === true || value === 1 || value === "1" || value === "true";
                    el.checked = checked;
                } else {
                    el.value = value == null ? "" : String(value);
                }
                el.dispatchEvent(new Event("input", { bubbles: true }));
                el.dispatchEvent(new Event("change", { bubbles: true }));
                el.dispatchEvent(new Event("blur", { bubbles: true }));
                return { ok: true };
            }
            """
            await cdp_session.cdp_client.send.Runtime.callFunctionOn(
                params={
                    "objectId": object_id,
                    "functionDeclaration": js_set,
                    "arguments": [{"value": value}, {"value": field_type}],
                    "returnByValue": True,
                },
                session_id=cdp_session.session_id,
            )
            results.append({"ref": ref, "ok": True})
        except Exception as e:
            results.append({"ref": ref, "ok": False, "error": str(e)})
    success = all(r.get("ok") for r in results) if results else False
    return {"success": success, "results": results}


async def type_text(
    cdp_session: Any,
    target: Union[int, str],
    text: str,
    selector_map: Optional[Dict[int, Any]] = None,
    slowly: bool = False,
    submit: bool = False,
    append: bool = False,
    timeout_ms: int = 5000,
    target_id: Optional[str] = None,
    instance_id: Optional[str] = None,
    persist_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Type text into a field, optionally slowly and/or submit with Enter.
    """
    object_id = await resolve_element(
        cdp_session,
        target,
        selector_map,
        target_id=target_id,
        instance_id=instance_id,
        persist_path=persist_path,
    )
    if not object_id:
        raise ValueError(f"Could not resolve target: {target}")
    backend_node_id: Optional[int] = None
    if isinstance(target, int):
        if selector_map and target in selector_map:
            backend_node_id = getattr(selector_map[target], "backend_node_id", None)
        else:
            backend_node_id = target
    else:
        from . import inspection

        if inspection.is_ref_token(str(target)):
            backend_node_id = inspection.resolve_ref_backend_id(
                target_id,
                str(target),
                instance_id=instance_id,
                persist_path=persist_path,
            )
    coords = await ensure_actionable(
        cdp_session,
        object_id,
        timeout_ms=timeout_ms,
        backend_node_id=backend_node_id,
        debug=False,
    )
    await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
        params={
            "type": "mousePressed",
            "x": coords["x"],
            "y": coords["y"],
            "button": "left",
            "clickCount": 1,
        },
        session_id=cdp_session.session_id,
    )
    await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
        params={
            "type": "mouseReleased",
            "x": coords["x"],
            "y": coords["y"],
            "button": "left",
            "clickCount": 1,
        },
        session_id=cdp_session.session_id,
    )
    if slowly:
        if not append:
            js_clear = """
            function() {
                const el = this;
                if ('value' in el) {
                    el.value = "";
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                }
                return { ok: true };
            }
            """
            await cdp_session.cdp_client.send.Runtime.callFunctionOn(
                params={
                    "objectId": object_id,
                    "functionDeclaration": js_clear,
                    "returnByValue": True,
                },
                session_id=cdp_session.session_id,
            )
        for ch in str(text):
            await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
                params={"type": "char", "text": ch},
                session_id=cdp_session.session_id,
            )
            await asyncio.sleep(0.05)
    else:
        js_set = """
        function(value, append) {
            const el = this;
            if ('value' in el) {
                const incoming = value == null ? "" : String(value);
                el.value = append ? String(el.value || "") + incoming : incoming;
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
                return { ok: true };
            }
            return { ok: false, error: 'Element not writable' };
        }
        """
        res = await cdp_session.cdp_client.send.Runtime.callFunctionOn(
            params={
                "objectId": object_id,
                "functionDeclaration": js_set,
                "arguments": [{"value": text}, {"value": append}],
                "returnByValue": True,
            },
            session_id=cdp_session.session_id,
        )
        value = res.get("result", {}).get("value", {}) if res else {}
        if not value.get("ok"):
            raise ValueError(value.get("error") or "Element not writable")
    if submit:
        await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
            params={
                "type": "keyDown",
                "key": "Enter",
                "code": "Enter",
                "windowsVirtualKeyCode": 13,
                "nativeVirtualKeyCode": 13,
            },
            session_id=cdp_session.session_id,
        )
        await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
            params={
                "type": "keyUp",
                "key": "Enter",
                "code": "Enter",
                "windowsVirtualKeyCode": 13,
                "nativeVirtualKeyCode": 13,
            },
            session_id=cdp_session.session_id,
        )
    return {
        "target": target,
        "text": text,
        "slowly": slowly,
        "submit": submit,
        "append": append,
    }


async def wait_for_condition(
    cdp_session: Any,
    *,
    text: Optional[str] = None,
    selector: Optional[str] = None,
    network_idle_ms: Optional[int] = None,
    timeout_ms: int = 10000,
    poll_ms: int = 200,
) -> Dict[str, Any]:
    """
    Wait for text/selector/network idle using simple polling.
    """
    start = time.time()
    if network_idle_ms is not None:
        js_hook = """
        (function() {
            if (window.__buseNetworkIdle) return;
            const state = { pending: 0, lastChange: Date.now() };
            const bump = () => { state.lastChange = Date.now(); };
            const wrapFetch = () => {
                if (!window.fetch) return;
                const orig = window.fetch;
                window.fetch = function(...args) {
                    state.pending += 1; bump();
                    return orig.apply(this, args).then((res) => {
                        state.pending = Math.max(0, state.pending - 1); bump();
                        return res;
                    }).catch((err) => {
                        state.pending = Math.max(0, state.pending - 1); bump();
                        throw err;
                    });
                };
            };
            const wrapXhr = () => {
                if (!window.XMLHttpRequest) return;
                const origOpen = XMLHttpRequest.prototype.open;
                const origSend = XMLHttpRequest.prototype.send;
                XMLHttpRequest.prototype.open = function() {
                    return origOpen.apply(this, arguments);
                };
                XMLHttpRequest.prototype.send = function() {
                    state.pending += 1; bump();
                    this.addEventListener("loadend", () => {
                        state.pending = Math.max(0, state.pending - 1); bump();
                    });
                    return origSend.apply(this, arguments);
                };
            };
            wrapFetch(); wrapXhr();
            window.__buseNetworkIdle = state;
        })()
        """
        await cdp_session.cdp_client.send.Runtime.evaluate(
            params={"expression": js_hook, "returnByValue": True},
            session_id=cdp_session.session_id,
        )
    while (time.time() - start) * 1000.0 < timeout_ms:
        if text:
            js_text = """
            (function() { 
                const body = document.body ? document.body.innerText || '' : '';
                return body.includes(TEXT_JSON);
            } )()
            """
            js_text = js_text.replace("TEXT_JSON", json.dumps(text))
            res = await cdp_session.cdp_client.send.Runtime.evaluate(
                params={"expression": js_text, "returnByValue": True},
                session_id=cdp_session.session_id,
            )
            if res.get("result", {}).get("value") is True:
                return {"ok": True, "condition": "text"}
        if selector:
            js_sel = """
            (function() { 
                return !!document.querySelector(SELECTOR_JSON);
            } )()
            """
            js_sel = js_sel.replace("SELECTOR_JSON", json.dumps(selector))
            res = await cdp_session.cdp_client.send.Runtime.evaluate(
                params={"expression": js_sel, "returnByValue": True},
                session_id=cdp_session.session_id,
            )
            if res.get("result", {}).get("value") is True:
                return {"ok": True, "condition": "selector"}
        if network_idle_ms is not None:
            js_idle = """
            (function() { 
                const s = window.__buseNetworkIdle;
                if (!s) return false;
                const idleFor = Date.now() - s.lastChange;
                return s.pending === 0 && idleFor >= NETWORK_IDLE_MS;
            } )()
            """
            js_idle = js_idle.replace("NETWORK_IDLE_MS", str(int(network_idle_ms)))
            res = await cdp_session.cdp_client.send.Runtime.evaluate(
                params={"expression": js_idle, "returnByValue": True},
                session_id=cdp_session.session_id,
            )
            if res.get("result", {}).get("value") is True:
                return {"ok": True, "condition": "network_idle"}
        await asyncio.sleep(max(0.01, poll_ms / 1000.0))
    raise TimeoutError("wait condition timed out")


async def resolve_element(
    cdp_session: Any,
    target: Union[int, str],
    selector_map: Optional[Dict[int, Any]] = None,
    target_id: Optional[str] = None,
    instance_id: Optional[str] = None,
    persist_path: Optional[str] = None,
) -> Optional[str]:
    """
    Resolves an index (int) or selector (str) to a CDP RemoteObjectId.
    """
    if isinstance(target, int):
        if not selector_map or target not in selector_map:
            try:
                res = await cdp_session.cdp_client.send.DOM.resolveNode(
                    params={"backendNodeId": target}, session_id=cdp_session.session_id
                )
                return res.get("object", {}).get("objectId")
            except Exception:
                raise ValueError(
                    f"Index {target} not found in selector map, and backendNodeId resolve failed."
                )
        element = selector_map[target]
        backend_id = getattr(element, "backend_node_id", None)
        if not backend_id:
            raise ValueError(f"Element {target} has no backend_node_id.")
        res = await cdp_session.cdp_client.send.DOM.resolveNode(
            params={"backendNodeId": backend_id}, session_id=cdp_session.session_id
        )
        return res.get("object", {}).get("objectId")
    else:
        from . import inspection

        if inspection.is_ref_token(target):
            backend_id = inspection.resolve_ref_backend_id(
                target_id,
                target,
                instance_id=instance_id,
                persist_path=persist_path,
            )
            if backend_id is not None:
                res = await cdp_session.cdp_client.send.DOM.resolveNode(
                    params={"backendNodeId": backend_id},
                    session_id=cdp_session.session_id,
                )
                return res.get("object", {}).get("objectId")
        js_resolve = """
        () => { 
            const el = document.querySelector(TARGET_JSON);
            return el;
        } 
        """
        js_resolve = js_resolve.replace("TARGET_JSON", json.dumps(target))
        res = await cdp_session.cdp_client.send.Runtime.evaluate(
            params={"expression": f"({js_resolve})()", "returnByValue": False},
            session_id=cdp_session.session_id,
        )
        return res.get("result", {}).get("objectId")


async def ensure_actionable(
    cdp_session: Any,
    object_id: str,
    timeout_ms: int = 5000,
    force: bool = False,
    backend_node_id: Optional[int] = None,
    debug: bool = False,
) -> Dict[str, float]:
    """
    Port of OpenClaw/Playwright actionability checks.
    Returns the center coordinates (x, y) if actionable.
    """
    start_time = time.time()
    last_hit_backend = None
    last_positions = None
    last_overlay_error = None
    last_moving = False
    while (time.time() - start_time) * 1000 < timeout_ms:
        if backend_node_id is not None:
            try:
                await cdp_session.cdp_client.send.DOM.scrollIntoViewIfNeeded(
                    params={"backendNodeId": backend_node_id},
                    session_id=cdp_session.session_id,
                )
            except Exception:
                pass
        await cdp_session.cdp_client.send.Runtime.callFunctionOn(
            params={
                "objectId": object_id,
                "functionDeclaration": "function() { this.scrollIntoViewIfNeeded(); }",
            },
            session_id=cdp_session.session_id,
        )
        js_check = """
        function() {
            const rect = this.getBoundingClientRect();
            const style = window.getComputedStyle(this);
            return {
                x: rect.left + rect.width / 2,
                y: rect.top + rect.height / 2,
                width: rect.width,
                height: rect.height,
                visible: style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0',
                enabled: !this.disabled
            };
        }
        """
        res = await cdp_session.cdp_client.send.Runtime.callFunctionOn(
            params={
                "objectId": object_id,
                "functionDeclaration": js_check,
                "returnByValue": True,
            },
            session_id=cdp_session.session_id,
        )
        info = res.get("result", {}).get("value", {})
        if not info.get("visible"):
            await asyncio.sleep(0.1)
            continue
        if not force and not info.get("enabled"):
            raise RuntimeError("Element is disabled")
        pos1 = (info["x"], info["y"])
        await asyncio.sleep(0.05)
        res2 = await cdp_session.cdp_client.send.Runtime.callFunctionOn(
            params={
                "objectId": object_id,
                "functionDeclaration": js_check,
                "returnByValue": True,
            },
            session_id=cdp_session.session_id,
        )
        info2 = res2.get("result", {}).get("value", {})
        pos2 = (info2["x"], info2["y"])
        if pos1 != pos2:
            last_moving = True
            last_positions = {"pos1": pos1, "pos2": pos2}
            continue
        last_moving = False
        if not force:
            try:
                hit_res = await cdp_session.cdp_client.send.Runtime.callFunctionOn(
                    params={
                        "objectId": object_id,
                        "functionDeclaration": """
                        function(x, y) {
                            const el = document.elementFromPoint(x, y);
                            if (!el) return { hit: false, reason: "no-element" };
                            return { hit: this === el || this.contains(el) };
                        }
                        """,
                        "arguments": [{"value": pos2[0]}, {"value": pos2[1]}],
                        "returnByValue": True,
                    },
                    session_id=cdp_session.session_id,
                )
                hit_val = hit_res.get("result", {}).get("value", {})
                last_hit_backend = hit_val
                if not hit_val.get("hit"):
                    await asyncio.sleep(0.2)
                    continue
            except Exception as e:
                last_overlay_error = str(e)
                await asyncio.sleep(0.2)
                continue
        return {"x": pos2[0], "y": pos2[1]}
    if debug:
        debug_info = None
        try:
            dbg = await cdp_session.cdp_client.send.Runtime.callFunctionOn(
                params={
                    "objectId": object_id,
                    "functionDeclaration": """
                    function() {
                        const rect = this.getBoundingClientRect();
                        const style = window.getComputedStyle(this);
                        return {
                            rect: {x: rect.x, y: rect.y, w: rect.width, h: rect.height},
                            visible: style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0',
                            enabled: !this.disabled,
                            display: style.display,
                            visibility: style.visibility,
                            opacity: style.opacity
                        };
                    }
                    """,
                    "returnByValue": True,
                },
                session_id=cdp_session.session_id,
            )
            debug_info = dbg.get("result", {}).get("value")
            debug_info["last_hit_backend"] = last_hit_backend
            debug_info["last_positions"] = last_positions
            debug_info["last_overlay_error"] = last_overlay_error
            debug_info["last_moving"] = last_moving
        except Exception as e:
            debug_info = {"error": f"debug_failed: {e}"}
        raise RuntimeError(
            f"Element did not become actionable (visible/stable/clickable) within {timeout_ms}ms | debug={debug_info}"
        )
    raise RuntimeError(
        f"Element did not become actionable (visible/stable/clickable) within {timeout_ms}ms"
    )


async def click_robust(
    cdp_session: Any,
    target: Union[int, str],
    selector_map: Optional[Dict[int, Any]] = None,
    modifiers: List[str] = [],
    button: str = "left",
    click_count: int = 1,
    force: bool = False,
    timeout_ms: int = 5000,
    debug: bool = False,
    target_id: Optional[str] = None,
    instance_id: Optional[str] = None,
    persist_path: Optional[str] = None,
):
    """
    Executes a high-reliability click matching OpenClaw's standards.
    """
    backend_node_id: Optional[int] = None
    if isinstance(target, int):
        if selector_map and target in selector_map:
            backend_node_id = getattr(selector_map[target], "backend_node_id", None)
        else:
            backend_node_id = target
    else:
        from . import inspection

        if inspection.is_ref_token(str(target)):
            backend_node_id = inspection.resolve_ref_backend_id(
                target_id,
                str(target),
                instance_id=instance_id,
                persist_path=persist_path,
            )
    object_id = await resolve_element(
        cdp_session,
        target,
        selector_map,
        target_id=target_id,
        instance_id=instance_id,
        persist_path=persist_path,
    )
    if not object_id:
        raise ValueError(f"Could not resolve target: {target}")
    coords = await ensure_actionable(
        cdp_session,
        object_id,
        timeout_ms=timeout_ms,
        force=force,
        backend_node_id=backend_node_id,
        debug=debug,
    )
    mask = 0
    if "alt" in modifiers:
        mask |= 1
    if "ctrl" in modifiers or "control" in modifiers:
        mask |= 2
    if "meta" in modifiers or "command" in modifiers:
        mask |= 4
    if "shift" in modifiers:
        mask |= 8
    common = {
        "x": coords["x"],
        "y": coords["y"],
        "modifiers": mask,
        "button": button,
        "clickCount": click_count,
    }
    await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
        params={**common, "type": "mousePressed"}, session_id=cdp_session.session_id
    )
    await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
        params={**common, "type": "mouseReleased"}, session_id=cdp_session.session_id
    )
    return {
        "x": coords["x"],
        "y": coords["y"],
        "button": button,
        "click_count": click_count,
    }


async def click_with_retry(
    cdp_session: Any,
    target: Union[int, str],
    selector_map: Optional[Dict[int, Any]],
    refresh_selector_map,
    modifiers: List[str],
    button: str,
    click_count: int,
    force: bool,
    timeout_ms: int,
    debug: bool,
    target_id: Optional[str] = None,
    instance_id: Optional[str] = None,
    persist_path: Optional[str] = None,
):
    """
    Run a robust click and refresh selector map on retryable failures (no retry).
    """
    try:
        return await click_robust(
            cdp_session,
            target,
            selector_map=selector_map,
            modifiers=modifiers,
            button=button,
            click_count=click_count,
            force=force,
            timeout_ms=timeout_ms,
            debug=debug,
            target_id=target_id,
            instance_id=instance_id,
            persist_path=persist_path,
        )
    except Exception as e:
        msg = str(e).lower()
        from . import inspection

        if inspection.is_ref_token(str(target)):
            raise
        retryable = (
            "did not become actionable" in msg or "could not resolve target" in msg
        )
        if retryable and refresh_selector_map:
            await refresh_selector_map()
        raise


async def drag_and_drop(
    cdp_session: Any,
    start: Union[int, str],
    end: Union[int, str],
    selector_map: Optional[Dict[int, Any]] = None,
    timeout_ms: int = 5000,
    html5: bool = True,
    target_id: Optional[str] = None,
    instance_id: Optional[str] = None,
    persist_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Drag from one element to another using CDP mouse events.
    """
    start_obj = await resolve_element(
        cdp_session,
        start,
        selector_map,
        target_id=target_id,
        instance_id=instance_id,
        persist_path=persist_path,
    )
    end_obj = await resolve_element(
        cdp_session,
        end,
        selector_map,
        target_id=target_id,
        instance_id=instance_id,
        persist_path=persist_path,
    )
    if not start_obj:
        raise ValueError(f"Could not resolve start target: {start}")
    if not end_obj:
        raise ValueError(f"Could not resolve end target: {end}")
    start_coords = await ensure_actionable(
        cdp_session,
        start_obj,
        timeout_ms=timeout_ms,
        backend_node_id=None,
        debug=False,
    )
    end_coords = await ensure_actionable(
        cdp_session,
        end_obj,
        timeout_ms=timeout_ms,
        backend_node_id=None,
        debug=False,
    )
    x1, y1 = start_coords["x"], start_coords["y"]
    x2, y2 = end_coords["x"], end_coords["y"]
    await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
        params={"type": "mouseMoved", "x": x1, "y": y1, "buttons": 0},
        session_id=cdp_session.session_id,
    )
    await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
        params={
            "type": "mousePressed",
            "x": x1,
            "y": y1,
            "button": "left",
            "buttons": 1,
            "clickCount": 1,
        },
        session_id=cdp_session.session_id,
    )
    await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
        params={"type": "mouseMoved", "x": x2, "y": y2, "buttons": 1},
        session_id=cdp_session.session_id,
    )
    await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
        params={
            "type": "mouseReleased",
            "x": x2,
            "y": y2,
            "button": "left",
            "buttons": 0,
            "clickCount": 1,
        },
        session_id=cdp_session.session_id,
    )
    if html5:
        js_dnd = """
        function(target) {
            try {
                const data = new DataTransfer();
                const dragStart = new DragEvent('dragstart', { bubbles: true, dataTransfer: data });
                const dragOver = new DragEvent('dragover', { bubbles: true, dataTransfer: data });
                const drop = new DragEvent('drop', { bubbles: true, dataTransfer: data });
                const dragEnd = new DragEvent('dragend', { bubbles: true, dataTransfer: data });
                this.dispatchEvent(dragStart);
                target.dispatchEvent(dragOver);
                target.dispatchEvent(drop);
                this.dispatchEvent(dragEnd);
                return { ok: true };
            } catch (e) {
                return { ok: false, error: String(e) };
            }
        }
        """
        await cdp_session.cdp_client.send.Runtime.callFunctionOn(
            params={
                "objectId": start_obj,
                "functionDeclaration": js_dnd,
                "arguments": [{"objectId": end_obj}],
                "returnByValue": True,
            },
            session_id=cdp_session.session_id,
        )
    return {"start": {"x": x1, "y": y1}, "end": {"x": x2, "y": y2}}
