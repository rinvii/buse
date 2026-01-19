import base64
import json
from types import SimpleNamespace

import pytest
import typer

import buse.main as main


@pytest.fixture(autouse=True)
def reset_caches(monkeypatch):
    main._browser_sessions.clear()
    main._file_systems.clear()
    main._selector_cache.clear()
    monkeypatch.delenv("BUSE_SELECTOR_CACHE_TTL", raising=False)

    monkeypatch.setattr(main, "BrowserSession", FakeBrowserSession)

    class FakeFileSystem:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(main, "FileSystem", FakeFileSystem)
    yield


class DummyCtx:
    def __init__(self, instance_id="b1"):
        self.obj = {"instance_id": instance_id}


class FakeDomState:
    def __init__(self, value="dom"):
        self._value = value

    def llm_representation(self):
        return self._value


class FakeTab:
    def __init__(self, target_id="t1", title="T", url="U"):
        self.target_id = target_id
        self.title = title
        self.url = url


class FakeEvent:
    def __init__(self, session, include_screenshot: bool):
        self._session = session
        self._include_screenshot = include_screenshot

    async def event_result(self, **_kwargs):
        return await self._session.get_browser_state_summary(
            include_screenshot=self._include_screenshot
        )


class FakeEventBus:
    def __init__(self, session):
        self._session = session

    def dispatch(self, event):
        include_screenshot = bool(getattr(event, "include_screenshot", False))
        return FakeEvent(self._session, include_screenshot)


class FakeBrowserSession:
    def __init__(self, cdp_url=None):
        self.cdp_url = cdp_url
        self.agent_focus_target_id = "focus"
        self.event_bus = FakeEventBus(self)

    async def start(self):
        return None

    async def stop(self):
        return None

    async def get_or_create_cdp_session(self):
        async def evaluate(params, session_id):
            return {
                "result": {
                    "value": {"width": 1280, "height": 720, "device_pixel_ratio": 1.0}
                }
            }

        return SimpleNamespace(
            session_id="s1",
            cdp_client=SimpleNamespace(
                send=SimpleNamespace(Runtime=SimpleNamespace(evaluate=evaluate))
            ),
        )

    async def get_browser_state_summary(self, include_screenshot=False, cached=False):
        screenshot = (
            base64.b64encode(b"img").decode("ascii") if include_screenshot else None
        )
        return SimpleNamespace(
            url="http://x",
            title="Title",
            screenshot=screenshot,
            dom_state=FakeDomState(),
        )

    async def get_tabs(self):
        return [FakeTab()]

    async def export_storage_state(self, output_path=None):
        return {"cookies": [{"a": 1}]}


def patch_execute_tool(monkeypatch):
    calls = []

    async def fake_execute_tool(instance_id, action_name, params, **kwargs):
        calls.append((instance_id, action_name, params, kwargs))

    monkeypatch.setattr(main, "execute_tool", fake_execute_tool)
    return calls


def test_observe_no_session(monkeypatch):
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: None)
    with pytest.raises(SystemExit):
        main.observe(DummyCtx(), screenshot=False, path=None, omniparser=False)


def test_observe_no_session_outputs_error(monkeypatch, capsys):
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: None)
    with pytest.raises(SystemExit):
        main.observe(DummyCtx(), screenshot=False, path=None, omniparser=False)
    out = json.loads(capsys.readouterr().out)
    assert out["success"] is False


def test_observe_with_screenshot(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        main.session_manager,
        "get_session",
        lambda _: SimpleNamespace(cdp_url="x", user_data_dir=str(tmp_path)),
    )
    monkeypatch.setattr("browser_use.browser.BrowserSession", FakeBrowserSession)
    main.observe(DummyCtx(), screenshot=True, path=None, omniparser=False)
    out = json.loads(capsys.readouterr().out)
    assert out["screenshot_path"] is not None


def test_observe_without_dom_state(monkeypatch, tmp_path, capsys):
    class NoDomSession(FakeBrowserSession):
        async def get_browser_state_summary(
            self, include_screenshot=False, cached=False
        ):
            return SimpleNamespace(
                url="http://x", title="Title", screenshot=None, dom_state=None
            )

    monkeypatch.setattr(
        main.session_manager,
        "get_session",
        lambda _: SimpleNamespace(cdp_url="x", user_data_dir=str(tmp_path)),
    )
    monkeypatch.setattr("browser_use.browser.BrowserSession", NoDomSession)
    main.observe(DummyCtx(), screenshot=False, path=None, omniparser=False)
    out = json.loads(capsys.readouterr().out)
    assert out["dom_minified"] == ""


def test_observe_no_dom_flag(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        main.session_manager,
        "get_session",
        lambda _: SimpleNamespace(cdp_url="x", user_data_dir=str(tmp_path)),
    )
    monkeypatch.setattr("browser_use.browser.BrowserSession", FakeBrowserSession)
    main.observe(
        DummyCtx(),
        screenshot=False,
        path=None,
        omniparser=False,
        no_dom=True,
    )
    out = json.loads(capsys.readouterr().out)
    assert out["dom_minified"] == ""


def test_observe_profile(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        main.session_manager,
        "get_session",
        lambda _: SimpleNamespace(cdp_url="x", user_data_dir=str(tmp_path)),
    )
    monkeypatch.setattr("browser_use.browser.BrowserSession", FakeBrowserSession)
    main.state.profile = True
    main.observe(DummyCtx(), screenshot=False, path=None, omniparser=False)
    out = json.loads(capsys.readouterr().out)
    assert "profile" in out
    main.state.profile = False


def test_basic_commands(monkeypatch, tmp_path):
    calls = patch_execute_tool(monkeypatch)
    ctx = DummyCtx()
    main.navigate(ctx, "http://x", new_tab=True)
    main.new_tab(ctx, "http://new")
    main.search(ctx, "q", engine="bing")
    main.click(ctx, index=None, x=1, y=2, element_id=None, element_class="cls")
    main.click(ctx, index=7, x=None, y=None, element_id="id", element_class=None)
    main.go_back(ctx)
    main.refresh(ctx)
    main.wait(ctx, 1.0)
    main.switch_tab(ctx, "ABCD")
    main.close_tab(ctx, "ABCD")
    main.evaluate(ctx, "1+1")

    f = tmp_path / "file.txt"
    f.write_text("content")
    main.upload_file(ctx, index=1, path=str(f))
    main.send_keys(ctx, "Enter")
    main.find_text(ctx, "hello")

    assert [c[1] for c in calls] == [
        "navigate",
        "navigate",
        "search",
        "click",
        "click",
        "go_back",
        "evaluate",
        "wait",
        "switch",
        "close",
        "evaluate",
        "upload_file",
        "send_keys",
        "find_text",
    ]

    assert calls[11][2]["index"] == 1
    assert calls[11][2]["path"] == str(f)
    assert calls[12][2]["keys"] == "Enter"
    assert calls[13][2]["text"] == "hello"


def test_send_keys_list(monkeypatch, capsys):
    calls = patch_execute_tool(monkeypatch)
    ctx = DummyCtx()
    main.send_keys(ctx, keys=None, list_keys=True)
    out = capsys.readouterr().out
    assert "Named keys" in out
    assert "Navigation" in out
    assert calls == []


def test_send_keys_missing_keys():
    ctx = DummyCtx()
    with pytest.raises(typer.BadParameter):
        main.send_keys(ctx, keys=None, list_keys=False)


def test_send_keys_with_index(monkeypatch):
    calls = patch_execute_tool(monkeypatch)
    ctx = DummyCtx()
    main.send_keys(ctx, "Enter", index=7, element_id=None, element_class=None)
    assert calls[0][1] == "send_keys"
    assert calls[0][2]["keys"] == "Enter"
    assert calls[0][2]["index"] == 7
    assert calls[0][3]["needs_selector_map"] is True


def test_send_keys_with_id(monkeypatch):
    calls = patch_execute_tool(monkeypatch)
    ctx = DummyCtx()
    main.send_keys(ctx, "Enter", index=None, element_id="field", element_class=None)
    assert calls[0][2]["element_id"] == "field"
    assert calls[0][3]["needs_selector_map"] is True


def test_send_keys_with_class(monkeypatch):
    calls = patch_execute_tool(monkeypatch)
    ctx = DummyCtx()
    main.send_keys(ctx, "Enter", index=None, element_id=None, element_class="field")
    assert calls[0][2]["element_class"] == "field"
    assert calls[0][3]["needs_selector_map"] is True


def test_click_bad_params():
    ctx = DummyCtx()
    with pytest.raises(typer.BadParameter):
        main.click(ctx, index=None, x=None, y=None, element_id=None, element_class=None)
    with pytest.raises(typer.BadParameter):
        main.click(ctx, index=None, x=1, y=None, element_id=None, element_class=None)


def test_input_text_variants(monkeypatch):
    calls = patch_execute_tool(monkeypatch)
    ctx = DummyCtx()
    main.input(
        ctx, index=2, text="hi", text_opt=None, element_id=None, element_class=None
    )
    main.input(
        ctx, index=None, text=None, text_opt="opt", element_id="id", element_class="cls"
    )
    assert calls[0][1] == "input"
    assert calls[1][2]["text"] == "opt"

    with pytest.raises(typer.BadParameter):
        main.input(
            ctx,
            index=None,
            text=None,
            text_opt=None,
            element_id=None,
            element_class=None,
        )


def test_input_missing_selector():
    ctx = DummyCtx()
    with pytest.raises(typer.BadParameter):
        main.input(
            ctx,
            index=None,
            text="hi",
            text_opt=None,
            element_id=None,
            element_class=None,
        )


def test_dropdown_commands(monkeypatch):
    calls = patch_execute_tool(monkeypatch)
    ctx = DummyCtx()
    main.dropdown_options(ctx, index=1, element_id="id", element_class="cls")
    main.select_dropdown(
        ctx, index=1, text="A", text_opt=None, element_id=None, element_class="cls"
    )
    main.select_dropdown(
        ctx, index=None, text=None, text_opt="B", element_id="id", element_class="cls"
    )
    assert [c[1] for c in calls] == [
        "dropdown_options",
        "select_dropdown",
        "select_dropdown",
    ]

    with pytest.raises(typer.BadParameter):
        main.select_dropdown(
            ctx, index=1, text=None, text_opt=None, element_id=None, element_class=None
        )


def test_dropdown_options_missing_selector():
    ctx = DummyCtx()
    with pytest.raises(typer.BadParameter):
        main.dropdown_options(ctx, index=None, element_id=None, element_class=None)


def test_select_dropdown_missing_selector():
    ctx = DummyCtx()
    with pytest.raises(typer.BadParameter):
        main.select_dropdown(
            ctx,
            index=None,
            text="A",
            text_opt=None,
            element_id=None,
            element_class=None,
        )


def test_hover(monkeypatch):
    calls = patch_execute_tool(monkeypatch)
    ctx = DummyCtx()
    main.hover(ctx, index=5, element_id=None, element_class=None)
    main.hover(ctx, index=None, element_id="x", element_class=None)
    main.hover(ctx, index=None, element_id=None, element_class="c")
    assert [c[1] for c in calls] == ["hover", "hover", "hover"]
    assert calls[0][2] == {"index": 5}
    assert calls[1][2] == {"element_id": "x"}
    assert calls[2][2] == {"element_class": "c"}

    with pytest.raises(typer.BadParameter):
        main.hover(ctx, index=None, element_id=None, element_class=None)


def test_scroll(monkeypatch):
    calls = patch_execute_tool(monkeypatch)
    ctx = DummyCtx()
    main.scroll(ctx, down=False, pages=2.0, index=3)
    main.scroll(ctx, down=True, pages=0.5, index=None)
    assert calls[0][1] == "scroll"
    assert calls[1][1] == "evaluate"


def test_scroll_invalid_pages():
    ctx = DummyCtx()
    with pytest.raises(typer.BadParameter):
        main.scroll(ctx, down=True, pages=0, index=None)


def test_save_state(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        main.session_manager,
        "get_session",
        lambda _: SimpleNamespace(cdp_url="x", user_data_dir=str(tmp_path)),
    )
    monkeypatch.setattr("browser_use.browser.BrowserSession", FakeBrowserSession)
    monkeypatch.setattr(main, "BrowserSession", FakeBrowserSession)
    main.save_state(DummyCtx(), "state.json")
    out = json.loads(capsys.readouterr().out)
    assert out["cookies_count"] == 1

    monkeypatch.setattr(main.session_manager, "get_session", lambda _: None)
    with pytest.raises(SystemExit):
        main.save_state(DummyCtx(), "state.json")
    out = json.loads(capsys.readouterr().out)
    assert out["success"] is False


def test_extract(monkeypatch):
    calls = patch_execute_tool(monkeypatch)

    class FakeChat:
        def __init__(self, model=None):
            self.model = model

    monkeypatch.setattr("browser_use.llm.openai.chat.ChatOpenAI", FakeChat)
    monkeypatch.setenv("BUSE_EXTRACT_MODEL", "x")

    main.extract(DummyCtx(), "q")
    assert calls[0][1] == "extract"


def test_stop(monkeypatch, capsys):
    monkeypatch.setattr(
        main.session_manager, "get_session", lambda _: SimpleNamespace()
    )
    monkeypatch.setattr(main.session_manager, "stop_session", lambda _: None)
    main.stop(DummyCtx("b1"))
    out = json.loads(capsys.readouterr().out)
    assert out["message"] == "Stopped b1"


def test_stop_no_session(monkeypatch, capsys):
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: None)
    with pytest.raises(SystemExit):
        main.stop(DummyCtx("b1"))
    out = json.loads(capsys.readouterr().out)
    assert out["success"] is False


def test_wait_negative():
    ctx = DummyCtx()
    with pytest.raises(typer.BadParameter):
        main.wait(ctx, -1)


def test_switch_close_tab_short_id():
    ctx = DummyCtx()
    with pytest.raises(typer.BadParameter):
        main.switch_tab(ctx, "a")
    with pytest.raises(typer.BadParameter):
        main.close_tab(ctx, "a")


def test_app_paths(monkeypatch, capsys):
    monkeypatch.setattr(main.session_manager, "list_sessions", lambda: {"a": 1})
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: None)
    monkeypatch.setattr(main.session_manager, "start_session", lambda _: None)

    monkeypatch.setattr(main.sys, "argv", ["buse", "list"])
    main.app()
    assert json.loads(capsys.readouterr().out) == {"a": 1}

    monkeypatch.setattr(main.sys, "argv", ["buse", "--help"])
    main.app()
    capsys.readouterr()

    monkeypatch.setattr(main.sys, "argv", ["buse", "--format", "toon"])
    main.app()
    assert main.state.format == main.OutputFormat.toon
    capsys.readouterr()

    monkeypatch.setattr(main.sys, "argv", ["buse", "-f", "json"])
    main.app()
    assert main.state.format == main.OutputFormat.json
    capsys.readouterr()


def test_run_cli_success(monkeypatch):
    monkeypatch.setattr(main.session_manager, "list_sessions", lambda: {"a": 1})
    code, out = main.run_cli(["list"])
    assert code == 0
    assert json.loads(out) == {"a": 1}


def test_run_cli_system_exit(monkeypatch):
    monkeypatch.setattr(main, "_run", lambda *_: (_ for _ in ()).throw(SystemExit(2)))
    code, out = main.run_cli(["anything"])
    assert code == 2
    assert out == ""
    monkeypatch.setattr(main, "_run", main._run)


def test_app_profile_flags(monkeypatch, capsys):
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: None)
    monkeypatch.setattr(main.session_manager, "start_session", lambda _: None)

    monkeypatch.setattr(main.sys, "argv", ["buse", "--profile"])
    main.app()
    assert main.state.profile is True
    capsys.readouterr()
    main.state.profile = False

    monkeypatch.setattr(main.sys, "argv", ["buse", "-p"])
    main.app()
    assert main.state.profile is True
    capsys.readouterr()
    main.state.profile = False


def test_app_instance_start(monkeypatch, capsys):
    monkeypatch.setattr(main.session_manager, "get_session", lambda _: None)
    monkeypatch.setattr(main.session_manager, "start_session", lambda _: None)
    monkeypatch.setattr(main.sys, "argv", ["buse", "b1"])
    main.app()
    out = json.loads(capsys.readouterr().out)
    assert out["message"] == "Initialized b1"


def test_app_instance_existing(monkeypatch, capsys):
    monkeypatch.setattr(
        main.session_manager, "get_session", lambda _: SimpleNamespace()
    )
    monkeypatch.setattr(main.sys, "argv", ["buse", "b1"])
    main.app()
    out = json.loads(capsys.readouterr().out)
    assert out["already_running"] is True


def test_app_instance_command(monkeypatch):
    def ok_instance_app(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "instance_app", ok_instance_app)
    monkeypatch.setattr(main.sys, "argv", ["buse", "b1", "observe"])
    main.app()


def test_main_module_runs(monkeypatch):
    import runpy

    monkeypatch.setattr(main.sys, "argv", ["buse", "--help"])
    runpy.run_module("buse.main", run_name="__main__")


def test_app_instance_error(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(main, "instance_app", boom)
    monkeypatch.setattr(main.sys, "argv", ["buse", "b1", "observe"])
    with pytest.raises(SystemExit):
        main.app()


def test_app_wait_negative_rewrites_args(monkeypatch):
    seen = {}

    def capture_instance_app(*args, **kwargs):
        seen["args"] = kwargs.get("args")
        return None

    monkeypatch.setattr(main, "instance_app", capture_instance_app)
    monkeypatch.setattr(main.sys, "argv", ["buse", "b1", "wait", "-1"])
    main.app()
    assert seen["args"] == ["wait", "--", "-1"]


def test_app_wait_invalid_negative_not_rewritten(monkeypatch):
    seen = {}

    def capture_instance_app(*args, **kwargs):
        seen["args"] = kwargs.get("args")
        return None

    monkeypatch.setattr(main, "instance_app", capture_instance_app)
    monkeypatch.setattr(main.sys, "argv", ["buse", "b1", "wait", "-x"])
    main.app()
    assert seen["args"] == ["wait", "-x"]


def test_upload_file_validation(tmp_path):
    ctx = DummyCtx()

    with pytest.raises(typer.BadParameter, match="Path is not a file"):
        main.upload_file(ctx, index=1, path=str(tmp_path))

    non_existent = tmp_path / "fake.txt"
    with pytest.raises(typer.BadParameter, match="File not found"):
        main.upload_file(ctx, index=1, path=str(non_existent))
