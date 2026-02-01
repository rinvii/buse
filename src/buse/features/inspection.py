"""
Browser Inspection Features for Buse.
Provenance & References:
- Logic for Set-of-Mark (SoM): Ported from OpenClaw's screenshotWithLabelsViaPlaywright
  https://github.com/openclaw/openclaw/blob/main/src/browser/pw-tools-core.interactions.ts
- Logic for Semantic Snapshot & Pruning: Ported/Adapted from OpenClaw's pw-role-snapshot.ts
  https://github.com/openclaw/openclaw/blob/main/src/browser/pw-role-snapshot.ts
- Data Source: Chrome DevTools Protocol (CDP) Accessibility Domain
  https://chromedevtools.github.io/devtools-protocol/tot/Accessibility/
"""

import logging
import json
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)
INTERACTIVE_ROLES = {
    "button",
    "link",
    "textbox",
    "checkbox",
    "radio",
    "combobox",
    "listbox",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "option",
    "searchbox",
    "slider",
    "spinbutton",
    "switch",
    "tab",
    "treeitem",
}
CONTENT_ROLES = {
    "heading",
    "cell",
    "gridcell",
    "columnheader",
    "rowheader",
    "listitem",
    "article",
    "region",
    "main",
    "navigation",
    "sectionheader",
}
STRUCTURAL_ROLES = {
    "generic",
    "group",
    "list",
    "table",
    "row",
    "rowgroup",
    "grid",
    "treegrid",
    "menu",
    "menubar",
    "toolbar",
    "tablist",
    "tree",
    "directory",
    "document",
    "application",
    "presentation",
    "none",
    "Section",
    "paragraph",
}
_REF_CACHE: Dict[str, Dict[str, Any]] = {}
_REF_PATTERN = re.compile(r"^e\d+$", re.IGNORECASE)


def resolve_semantic_max_chars(mode: str, max_chars: Optional[int]) -> Optional[int]:
    if max_chars is None:
        return 10_000 if mode == "efficient" else 80_000
    if max_chars <= 0:
        return None
    return max_chars


def _load_refs_from_path(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text())
    except Exception:
        return None
    if isinstance(data, dict) and isinstance(data.get("refs"), dict):
        return data
    return None


def store_semantic_refs(
    target_id: Optional[str],
    refs: Dict[str, Dict[str, Any]],
    instance_id: Optional[str] = None,
    persist_path: Optional[str] = None,
) -> None:
    if not refs:
        return
    if target_id:
        _REF_CACHE[target_id] = {"refs": refs, "timestamp": time.time()}
    if instance_id:
        _REF_CACHE[f"instance:{instance_id}"] = {"refs": refs, "timestamp": time.time()}
    _REF_CACHE["last"] = {"refs": refs, "timestamp": time.time()}
    if persist_path:
        try:
            path = Path(persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"refs": refs, "timestamp": time.time()}
            path.write_text(json.dumps(payload))
        except Exception:
            pass


def resolve_ref_backend_id(
    target_id: Optional[str],
    ref: str,
    instance_id: Optional[str] = None,
    persist_path: Optional[str] = None,
) -> Optional[int]:
    if not target_id or not ref:
        if not ref:
            return None
    if not _REF_PATTERN.match(ref):
        return None
    entry = None
    if target_id:
        entry = _REF_CACHE.get(target_id)
    if entry is None and instance_id:
        entry = _REF_CACHE.get(f"instance:{instance_id}")
    if entry is None:
        entry = _REF_CACHE.get("last")
    if entry is None and persist_path:
        loaded = _load_refs_from_path(persist_path)
        if loaded:
            entry = loaded
            if target_id:
                _REF_CACHE[target_id] = loaded
            if instance_id:
                _REF_CACHE[f"instance:{instance_id}"] = loaded
            _REF_CACHE["last"] = loaded
    if not entry:
        return None
    data = entry.get("refs", {})
    info = data.get(ref)
    if not info:
        return None
    return info.get("backend_id")


def is_ref_token(value: str) -> bool:
    return bool(value) and bool(_REF_PATTERN.match(value))


async def screenshot_with_labels(
    cdp_session: Any,
    selector_map: Dict[int, Any],
    viewport: Optional[dict] = None,
    quality: int = 75,
    max_labels: int = 150,
) -> Tuple[str, int, int]:
    """
    Injects visual overlays (Set-of-Mark) for elements in the selector_map,
    takes a screenshot, and returns the base64 image, label count, and skipped count.
    """
    if max_labels is None:
        max_labels = 150
    max_labels = int(max_labels)
    if max_labels < 0:
        max_labels = 0
    boxes = []
    skipped = 0
    scroll_x = viewport.get("scrollX", 0) if viewport else 0
    scroll_y = viewport.get("scrollY", 0) if viewport else 0
    viewport_width = viewport.get("width", 0) if viewport else 0
    viewport_height = viewport.get("height", 0) if viewport else 0
    for index, element in selector_map.items():
        if max_labels > 0 and len(boxes) >= max_labels:
            skipped += 1
            continue
        box_data = None
        pos = getattr(element, "absolute_position", None)
        if pos:
            if hasattr(pos, "x"):
                box_data = {"x": pos.x, "y": pos.y, "w": pos.width, "h": pos.height}
            elif isinstance(pos, dict):
                box_data = {
                    "x": pos.get("x"),
                    "y": pos.get("y"),
                    "w": pos.get("width"),
                    "h": pos.get("height"),
                }
        if not box_data:
            vc = getattr(element, "viewport_coordinates", None)
            if vc:
                if isinstance(vc, (list, tuple)) and len(vc) >= 4:
                    box_data = {
                        "x": vc[0],
                        "y": vc[1],
                        "w": vc[2] - vc[0],
                        "h": vc[3] - vc[1],
                    }
                elif hasattr(vc, "x"):
                    box_data = {"x": vc.x, "y": vc.y, "w": vc.width, "h": vc.height}
        if not box_data:
            b = getattr(element, "bbox", None)
            if b and isinstance(b, dict):
                box_data = {
                    "x": b.get("x"),
                    "y": b.get("y"),
                    "w": b.get("width"),
                    "h": b.get("height"),
                }
        if box_data and box_data["x"] is not None:
            x0 = box_data["x"] - scroll_x
            y0 = box_data["y"] - scroll_y
            x1 = x0 + (box_data["w"] or 0)
            y1 = y0 + (box_data["h"] or 0)
            if viewport_width and viewport_height:
                if x1 < 0 or y1 < 0 or x0 > viewport_width or y0 > viewport_height:
                    skipped += 1
                    continue
            boxes.append(
                {
                    "ref": str(index),
                    "x": x0,
                    "y": y0,
                    "w": box_data["w"],
                    "h": box_data["h"],
                }
            )
    if not boxes:
        logger.warning("No interactive elements with valid coordinates found to label.")
        try:
            res = await cdp_session.cdp_client.send.Page.captureScreenshot(
                params={"format": "jpeg", "quality": quality},
                session_id=cdp_session.session_id,
            )
            return res.get("data", ""), 0, skipped
        except Exception as e:
            logger.error(f"Failed to take fallback screenshot: {e}")
            return "", 0, skipped
    labels_json = json.dumps(boxes)
    js_code = """
    (function(labels) { 
        const existing = document.querySelectorAll("[data-buse-labels]");
        existing.forEach((el) => el.remove());
        const root = document.createElement("div");
        root.setAttribute("data-buse-labels", "1");
        root.style.position = "fixed";
        root.style.left = "0";
        root.style.top = "0";
        root.style.zIndex = "2147483647";
        root.style.pointerEvents = "none";
        root.style.fontFamily = '"SF Mono","SFMono-Regular",Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace';
        const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
        for (const label of labels) { 
            const box = document.createElement("div");
            box.setAttribute("data-buse-labels", "1");
            box.style.position = "absolute";
            box.style.left = `${ label.x} px`;
            box.style.top = `${ label.y} px`;
            box.style.width = `${ label.w} px`;
            box.style.height = `${ label.h} px`;
            box.style.border = "2px solid #ffb020";
            box.style.boxSizing = "border-box";
            const tag = document.createElement("div");
            tag.setAttribute("data-buse-labels", "1");
            tag.textContent = label.ref;
            tag.style.position = "absolute";
            tag.style.left = `${ label.x} px`;
            tag.style.top = `${ clamp(label.y - 18, 0, 20000)} px`;
            tag.style.background = "#ffb020";
            tag.style.color = "#1a1a1a";
            tag.style.fontSize = "12px";
            tag.style.lineHeight = "14px";
            tag.style.padding = "1px 4px";
            tag.style.borderRadius = "3px";
            tag.style.boxShadow = "0 1px 2px rgba(0,0,0,0.35)";
            tag.style.whiteSpace = "nowrap";
            root.appendChild(box);
            root.appendChild(tag);
        } 
        document.documentElement.appendChild(root);
    } )(LABELS_JSON)
    """
    js_code = js_code.replace("LABELS_JSON", labels_json)
    cleanup_js = """
    (function() {{
        const existing = document.querySelectorAll("[data-buse-labels]");
        existing.forEach((el) => el.remove());
    }})()
    """
    try:
        await cdp_session.cdp_client.send.Runtime.evaluate(
            params={"expression": js_code, "returnByValue": True},
            session_id=cdp_session.session_id,
        )
        res = await cdp_session.cdp_client.send.Page.captureScreenshot(
            params={"format": "jpeg", "quality": quality},
            session_id=cdp_session.session_id,
        )
        return res.get("data", ""), len(boxes), skipped
    except Exception as e:
        logger.error(f"Visual grounding failed: {e}")
        try:
            res = await cdp_session.cdp_client.send.Page.captureScreenshot(
                params={"format": "jpeg", "quality": quality},
                session_id=cdp_session.session_id,
            )
            return res.get("data", ""), 0, skipped
        except Exception:
            return "", 0, skipped
    finally:
        try:
            await cdp_session.cdp_client.send.Runtime.evaluate(
                params={"expression": cleanup_js, "returnByValue": True},
                session_id=cdp_session.session_id,
            )
        except Exception:
            pass


async def get_accessibility_tree(cdp_session: Any) -> List[Dict]:
    """
    Fetches the full Accessibility Tree (AXTree) via CDP.
    """
    try:
        await cdp_session.cdp_client.send.Accessibility.enable(
            session_id=cdp_session.session_id
        )
        res = await cdp_session.cdp_client.send.Accessibility.getFullAXTree(
            session_id=cdp_session.session_id
        )
        return res.get("nodes", [])
    except Exception as e:
        logger.error(f"Failed to fetch AXTree: {e}")
        return []


async def get_diagnostics(
    cdp_session: Any,
) -> Dict[str, List[Dict[str, Optional[str]]]]:
    """
    Collects console messages and basic network errors via an in-page diagnostics hook.
    Note: This captures events after the hook is installed. It persists on the page
    for subsequent observations within the same session.
    """
    js_code = """
    (function() {
        if (!window.__buseDiagnostics) {
            const maxEntries = 200;
            const diag = { console: [], network_errors: [], network_requests: [], page_errors: [] };
            const push = (arr, entry) => {
                try {
                    arr.push(entry);
                    if (arr.length > maxEntries) arr.shift();
                } catch (e) {}
            };
            const normalizeArgs = (args) => args.map((a) => {
                if (typeof a === "string") return a;
                try { return JSON.stringify(a); } catch (e) { return String(a); }
            }).join(" ");
            ["log", "info", "warn", "error", "debug"].forEach((level) => {
                const orig = console[level];
                console[level] = function(...args) {
                    try {
                        push(diag.console, { level, message: normalizeArgs(args) });
                    } catch (e) {}
                    if (orig) return orig.apply(this, args);
                };
            });
            window.addEventListener("error", (event) => {
                try {
                    const target = event && event.target;
                    const url = target && (target.src || target.href) ? (target.src || target.href) : event.filename;
                    push(diag.network_errors, { message: event.message || "Resource error", url: url || null });
                    if (!target || !(target.src || target.href)) {
                        push(diag.page_errors, {
                            message: event.message || "Script error",
                            url: event.filename || null,
                            line: event.lineno || null,
                            column: event.colno || null
                        });
                    }
                } catch (e) {}
            }, true);
            window.addEventListener("unhandledrejection", (event) => {
                try {
                    const reason = event && event.reason;
                    const msg = reason && reason.message ? reason.message : String(reason);
                    push(diag.console, { level: "error", message: "Unhandled rejection: " + msg });
                    push(diag.page_errors, { message: "Unhandled rejection: " + msg, url: null, line: null, column: null });
                } catch (e) {}
            });
            if (window.fetch) {
                const origFetch = window.fetch;
                window.fetch = function(...args) {
                    return origFetch.apply(this, args).then((res) => {
                        try {
                            const method = (args[1] && args[1].method) ? String(args[1].method) : "GET";
                            push(diag.network_requests, {
                                method,
                                url: res.url || null,
                                status: String(res.status || ""),
                                ok: Boolean(res.ok)
                            });
                            if (!res.ok) {
                                push(diag.network_errors, { message: "fetch " + res.status, url: res.url || null });
                            }
                        } catch (e) {}
                        return res;
                    }).catch((err) => {
                        try {
                            const first = args && args.length ? args[0] : null;
                            const url = first && typeof first === "string" ? first : (first && first.url ? first.url : null);
                            const method = (args[1] && args[1].method) ? String(args[1].method) : "GET";
                            push(diag.network_requests, {
                                method,
                                url: url || null,
                                status: "error",
                                ok: false
                            });
                            push(diag.network_errors, { message: "fetch error", url: url });
                        } catch (e) {}
                        throw err;
                    });
                };
            }
            if (window.XMLHttpRequest) {
                const origOpen = XMLHttpRequest.prototype.open;
                const origSend = XMLHttpRequest.prototype.send;
                XMLHttpRequest.prototype.open = function(method, url) {
                    this.__buseUrl = url;
                    this.__buseMethod = method;
                    return origOpen.apply(this, arguments);
                };
                XMLHttpRequest.prototype.send = function() {
                    this.addEventListener("error", () => {
                        try { push(diag.network_errors, { message: "xhr error", url: this.__buseUrl || null }); } catch (e) {}
                    });
                    this.addEventListener("loadend", () => {
                        try {
                            push(diag.network_requests, {
                                method: String(this.__buseMethod || "GET"),
                                url: this.__buseUrl || null,
                                status: String(this.status || ""),
                                ok: this.status >= 200 && this.status < 400
                            });
                            if (this.status === 0) {
                                push(diag.network_errors, { message: "xhr status 0", url: this.__buseUrl || null });
                            }
                        } catch (e) {}
                    });
                    return origSend.apply(this, arguments);
                };
            }
            window.__buseDiagnostics = diag;
        }
        return window.__buseDiagnostics;
    })()
    """
    try:
        res = await cdp_session.cdp_client.send.Runtime.evaluate(
            params={"expression": js_code, "returnByValue": True},
            session_id=cdp_session.session_id,
        )
        value = res.get("result", {}).get("value") if isinstance(res, dict) else None
        if isinstance(value, dict):
            value.setdefault("console", [])
            value.setdefault("network_errors", [])
            value.setdefault("network_requests", [])
            value.setdefault("page_errors", [])
            return value
    except Exception as e:
        logger.error(f"Diagnostics collection failed: {e}")
    return {
        "console": [],
        "network_errors": [],
        "network_requests": [],
        "page_errors": [],
    }


async def resolve_backend_node_id_for_selector(
    cdp_session: Any,
    selector: Optional[str] = None,
    frame_selector: Optional[str] = None,
) -> Optional[int]:
    selector = (selector or "").strip()
    frame_selector = (frame_selector or "").strip()
    js_code = """
    (function(selector, frameSelector) {
        let root = document;
        if (frameSelector) {
            const frame = document.querySelector(frameSelector);
            if (!frame) return null;
            const doc = frame.contentDocument;
            if (!doc) return null;
            root = doc;
        }
        let el = null;
        if (selector) {
            el = root.querySelector(selector);
        } else {
            el = root.documentElement;
        }
        return el;
    })
    """
    try:
        expression = (
            f"({js_code})({json.dumps(selector)}, {json.dumps(frame_selector)})"
        )
        eval_res = await cdp_session.cdp_client.send.Runtime.evaluate(
            params={
                "expression": expression,
                "returnByValue": False,
            },
            session_id=cdp_session.session_id,
        )
        object_id = (
            eval_res.get("result", {}).get("objectId")
            if isinstance(eval_res, dict)
            else None
        )
        if not object_id:
            return None
        desc = await cdp_session.cdp_client.send.DOM.describeNode(
            params={"objectId": object_id},
            session_id=cdp_session.session_id,
        )
        node = desc.get("node", {}) if isinstance(desc, dict) else {}
        return node.get("backendNodeId")
    except Exception:
        return None


def format_semantic_snapshot(
    nodes: List[Dict],
    selector_map: Optional[Dict[int, Any]] = None,
    mode: str = "efficient",
    max_chars: Optional[int] = None,
    root_backend_id: Optional[int] = None,
    refs_out: Optional[Dict[str, Dict[str, Any]]] = None,
    include_refs: bool = True,
) -> Tuple[str, bool]:
    """
    Converts raw AXTree nodes into a semantic snapshot with tree structure.
    Format: [Role#ID] "Content" with 2-space indents per depth.
    mode: "efficient" applies pruning, "full" includes more structure.
    """
    if not nodes:
        return "", False
    is_efficient = mode == "efficient"
    interactive_only = False
    compact = is_efficient
    max_depth = 6 if is_efficient else None
    max_children_head = None
    max_children_tail = None
    backend_id_map: Dict[int, int] = {}
    if selector_map:
        for index, element in selector_map.items():
            bid = getattr(element, "backend_node_id", None)
            if bid:
                backend_id_map[bid] = index
    node_map = {n["nodeId"]: n for n in nodes if "nodeId" in n}
    all_child_ids = {cid for n in nodes for cid in n.get("childIds", [])}
    roots = [n for n in nodes if n.get("nodeId") not in all_child_ids]
    if root_backend_id is not None:
        scoped = next(
            (n for n in nodes if n.get("backendDOMNodeId") == root_backend_id),
            None,
        )
        roots = [scoped] if scoped else []

    def _role_name(node: Dict[str, Any]) -> Tuple[str, str, str]:
        role = node.get("role", {}).get("value", "unknown")
        name = node.get("name", {}).get("value", "")
        value = node.get("value", {}).get("value", "")
        content = name.strip() if isinstance(name, str) else ""
        extra = value.strip() if isinstance(value, str) else ""
        return role, content, extra

    ref_counts: Dict[str, int] = {}
    ref_counter = 0

    def _next_ref(role: str, name: str) -> Tuple[str, int]:
        nonlocal ref_counter
        key = f"{role}:{name}"
        count = ref_counts.get(key, 0)
        ref_counts[key] = count + 1
        ref_counter += 1
        return f"e{ref_counter}", count

    def _buse_index(node: Dict[str, Any]) -> Optional[int]:
        backend_id = node.get("backendDOMNodeId")
        return backend_id_map.get(backend_id) if backend_id else None

    def _is_relevant(
        node: Dict[str, Any],
        parent_role: Optional[str] = None,
        parent_content: str = "",
    ) -> bool:
        role, content, _ = _role_name(node)
        role_l = role.lower() if isinstance(role, str) else "unknown"
        parent_role_l = parent_role.lower() if isinstance(parent_role, str) else ""
        if role == "InlineTextBox" or role_l == "inlinetextbox":
            return False
        if is_efficient and role_l == "image":
            return False
        if is_efficient and role_l == "statictext":
            if parent_role_l in {"link", "heading", "button"}:
                return False
            if parent_content and content == parent_content:
                return False
        if role == "StaticText" and not content:
            return False
        if _buse_index(node) is not None:
            return True
        if role_l in INTERACTIVE_ROLES:
            return True
        if interactive_only:
            return False
        if role_l in CONTENT_ROLES and content:
            return True
        if is_efficient and role_l in CONTENT_ROLES:
            return True
        if content or role_l == "statictext":
            return True
        return False

    def _format_line(
        node: Dict[str, Any],
        force_ids: bool = False,
        omit_content: bool = False,
    ) -> str:
        role, content, extra = _role_name(node)
        if omit_content:
            content = ""
            extra = ""
        role_l = role.lower() if isinstance(role, str) else "unknown"
        idx = _buse_index(node)
        backend_id = node.get("backendDOMNodeId")
        semantic_id = backend_id or node.get("nodeId")
        ref = None
        if role_l in INTERACTIVE_ROLES and backend_id is not None:
            ref, nth = _next_ref(role, content)
            if refs_out is not None:
                refs_out[ref] = {
                    "backend_id": backend_id,
                    "role": role,
                    "name": content or None,
                    "nth": nth,
                }
        if role_l not in INTERACTIVE_ROLES:
            idx = None
        if idx is not None:
            label = f"[{role}#{idx}]"
        elif semantic_id is not None:
            label = f"[{role}@{semantic_id}]"
        else:
            label = f"[{role}]"
        ref_suffix = f" [ref={ref}]" if include_refs and ref else ""
        if content and extra and extra != content and not is_efficient:
            return f'{label} "{content}" ({extra}){ref_suffix}'
        if content:
            return f'{label} "{content}"{ref_suffix}'
        return f"{label}{ref_suffix}"

    def _render(
        node: Dict[str, Any],
        depth: int,
        parent_role: Optional[str] = None,
        parent_content: str = "",
        force_ids: bool = False,
    ) -> Tuple[List[str], bool]:
        lines: List[str] = []
        if max_depth is not None and depth > max_depth:
            return lines, False
        role, content, _ = _role_name(node)
        role_l = role.lower() if isinstance(role, str) else "unknown"
        is_structural = role_l in STRUCTURAL_ROLES
        skippable_structural = (
            is_efficient and is_structural and not content and _buse_index(node) is None
        )
        children = [
            node_map[cid] for cid in node.get("childIds", []) if cid in node_map
        ]
        rendered_children: List[Tuple[Dict[str, Any], List[str], bool]] = []
        for child in children:
            child_depth = depth if skippable_structural else depth + 1
            child_out, child_relevant = _render(
                child, child_depth, role, content, force_ids=False
            )
            rendered_children.append((child, child_out, child_relevant))
        relevant_children = [c for c in rendered_children if c[2]]
        child_lines: List[str] = []
        if (
            max_children_head is not None
            and max_children_tail is not None
            and len(relevant_children) > (max_children_head + max_children_tail)
        ):
            head = relevant_children[:max_children_head]
            tail = relevant_children[-max_children_tail:]
            skipped = len(relevant_children) - (len(head) + len(tail))
            for _, out, _ in head:
                child_lines.extend(out)
            indent = "  " * (depth + (0 if skippable_structural else 1))
            child_lines.append(f"{indent}... (skipped {skipped} relevant items)")
            for child_node, _, _ in tail:
                child_depth = depth if skippable_structural else depth + 1
                tail_out, _ = _render(
                    child_node, child_depth, role, content, force_ids=True
                )
                child_lines.extend(tail_out)
        else:
            for _, out, _ in rendered_children:
                child_lines.extend(out)
        include_current = True
        if compact:
            include_current = _is_relevant(
                node, parent_role=parent_role, parent_content=parent_content
            ) or bool(child_lines)
        has_text_child = False
        if not is_efficient:
            for child in children:
                child_role = child.get("role", {}).get("value", "")
                child_name = child.get("name", {}).get("value", "")
                if str(child_role).lower() == "statictext" and str(child_name).strip():
                    has_text_child = True
                    break
        if skippable_structural:
            lines.extend(child_lines)
        elif include_current:
            indent = "  " * depth
            omit_content = (
                not is_efficient
                and has_text_child
                and role_l
                in {"link", "heading", "button", "checkbox", "radio", "textbox"}
            )
            lines.append(
                f"{indent}{_format_line(node, force_ids=force_ids, omit_content=omit_content)}"
            )
            if not (is_efficient and role_l in {"link", "heading"}):
                lines.extend(child_lines)
        else:
            lines.extend(child_lines)
        return lines, _is_relevant(
            node, parent_role=parent_role, parent_content=parent_content
        ) or bool(child_lines)

    output_lines: List[str] = []
    for root in roots:
        root_lines, _ = _render(root, 0)
        output_lines.extend(root_lines)
    snapshot = "\n".join(output_lines).strip()
    truncated = False
    if max_chars is not None and max_chars >= 0 and len(snapshot) > max_chars:
        truncated = True
        suffix = "\n\n[...TRUNCATED - page too large]"
        available = max(0, max_chars - len(suffix))
        snapshot = snapshot[:available].rstrip() + suffix
    return snapshot, truncated
