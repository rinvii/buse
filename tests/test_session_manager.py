import json
import types
from pathlib import Path

import pytest
import psutil

from buse.session import SessionInfo, SessionManager


def make_manager(tmp_path: Path):
    return SessionManager(config_dir=tmp_path)


def test_load_sessions_empty_file(tmp_path):
    sessions_file = tmp_path / "sessions.json"
    sessions_file.write_text("   ")
    manager = make_manager(tmp_path)
    assert manager.sessions == {}


def test_load_sessions_invalid_json(tmp_path):
    sessions_file = tmp_path / "sessions.json"
    sessions_file.write_text("{not-json}")
    manager = make_manager(tmp_path)
    assert manager.sessions == {}


def test_load_sessions_valid_json(tmp_path):
    sessions_file = tmp_path / "sessions.json"
    sessions_file.write_text(
        json.dumps(
            {
                "b1": {
                    "instance_id": "b1",
                    "cdp_url": "http://x",
                    "pid": 1,
                    "user_data_dir": "/tmp",
                }
            }
        )
    )
    manager = make_manager(tmp_path)
    assert "b1" in manager.sessions


def test_save_sessions(tmp_path):
    manager = make_manager(tmp_path)
    manager.sessions["b1"] = SessionInfo(
        instance_id="b1", cdp_url="http://x:1", pid=1, user_data_dir="/tmp"
    )
    manager._save_sessions()
    data = json.loads((tmp_path / "sessions.json").read_text())
    assert data["b1"]["cdp_url"] == "http://x:1"


def test_cdp_ready_true(monkeypatch, tmp_path):
    manager = make_manager(tmp_path)

    class Resp:
        status_code = 200

    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: Resp())
    assert manager._cdp_ready("http://x") is True


def test_cdp_ready_false(monkeypatch, tmp_path):
    manager = make_manager(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr("httpx.get", boom)
    assert manager._cdp_ready("http://x") is False


def test_is_alive(monkeypatch, tmp_path):
    manager = make_manager(tmp_path)
    session = SessionInfo(
        instance_id="b1", cdp_url="http://x", pid=1, user_data_dir="/tmp"
    )
    monkeypatch.setattr("psutil.pid_exists", lambda pid: True)
    monkeypatch.setattr(manager, "_cdp_ready", lambda url: True)
    assert manager.is_alive(session) is True

    monkeypatch.setattr("psutil.pid_exists", lambda pid: True)
    monkeypatch.setattr(manager, "_cdp_ready", lambda url: False)
    assert manager.is_alive(session) is False

    monkeypatch.setattr("psutil.pid_exists", lambda pid: False)
    monkeypatch.setattr(manager, "_cdp_ready", lambda url: False)
    assert manager.is_alive(session) is False


def test_get_session_removes_stale(monkeypatch, tmp_path):
    manager = make_manager(tmp_path)
    manager.sessions["b1"] = SessionInfo(
        instance_id="b1", cdp_url="http://x", pid=1, user_data_dir="/tmp"
    )
    monkeypatch.setattr(manager, "is_alive", lambda s: False)
    assert manager.get_session("b1") is None
    assert "b1" not in manager.sessions


def test_get_session_alive(monkeypatch, tmp_path):
    manager = make_manager(tmp_path)
    manager.sessions["b1"] = SessionInfo(
        instance_id="b1", cdp_url="http://x", pid=1, user_data_dir="/tmp"
    )
    monkeypatch.setattr(manager, "is_alive", lambda s: True)
    assert manager.get_session("b1") is manager.sessions["b1"]


def test_start_session_reuses_existing(monkeypatch, tmp_path):
    manager = make_manager(tmp_path)
    existing = SessionInfo(
        instance_id="b1", cdp_url="http://x:1", pid=1, user_data_dir="/tmp"
    )
    monkeypatch.setattr(manager, "get_session", lambda _: existing)
    assert manager.start_session("b1") == existing


def test_start_session_new(monkeypatch, tmp_path):
    manager = make_manager(tmp_path)
    manager.sessions["b0"] = SessionInfo(
        instance_id="b0", cdp_url="http://localhost:9222", pid=2, user_data_dir="/tmp"
    )
    manager.sessions["b_bad"] = SessionInfo(
        instance_id="b_bad", cdp_url="http://localhost:notaport", pid=3, user_data_dir="/tmp"
    )

    captured = {}

    class FakeProcess:
        pid = 999

        def kill(self):
            captured["killed"] = True

    def fake_popen(args, stdout=None, stderr=None, start_new_session=None):
        captured["args"] = args
        return FakeProcess()

    monkeypatch.setattr(manager, "_find_chrome_executable", lambda: "/bin/chrome")
    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr(manager, "_cdp_ready", lambda url: True)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    info = manager.start_session("b1", headless=True)
    assert info.instance_id == "b1"
    assert info.cdp_url.endswith(":9223")
    assert "--headless=new" in captured["args"]


def test_start_session_cdp_not_ready(monkeypatch, tmp_path):
    manager = make_manager(tmp_path)

    class FakeProcess:
        pid = 999
        killed = False

        def kill(self):
            self.killed = True
            raise RuntimeError("kill failed")

    proc = FakeProcess()

    monkeypatch.setattr(manager, "_find_chrome_executable", lambda: "/bin/chrome")
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: proc)
    monkeypatch.setattr(manager, "_cdp_ready", lambda url: False)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    with pytest.raises(RuntimeError):
        manager.start_session("b1")
    assert proc.killed is True


def test_find_chrome_executable(monkeypatch, tmp_path):
    manager = make_manager(tmp_path)

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    assert manager._find_chrome_executable().endswith("Google Chrome")

    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: None)
    assert manager._find_chrome_executable() in {
        "google-chrome",
        "chromium-browser",
        "chromium",
        "google-chrome-stable",
    }

    def raise_fn(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("subprocess.run", raise_fn)
    with pytest.raises(RuntimeError):
        manager._find_chrome_executable()

    monkeypatch.setattr("platform.system", lambda: "Windows")
    assert manager._find_chrome_executable().endswith("chrome.exe")

    monkeypatch.setattr("platform.system", lambda: "Other")
    with pytest.raises(RuntimeError):
        manager._find_chrome_executable()


def test_stop_session(monkeypatch, tmp_path):
    manager = make_manager(tmp_path)
    manager.sessions["b1"] = SessionInfo(
        instance_id="b1", cdp_url="http://x", pid=1, user_data_dir="/tmp"
    )

    class FakeChild:
        killed = False

        def kill(self):
            self.killed = True

    class FakeProc:
        killed = False

        def children(self, recursive=False):
            return [FakeChild()]

        def kill(self):
            self.killed = True

    monkeypatch.setattr("psutil.Process", lambda pid: FakeProc())
    manager.stop_session("b1")
    assert "b1" not in manager.sessions

    manager.sessions["b2"] = SessionInfo(
        instance_id="b2", cdp_url="http://x", pid=2, user_data_dir="/tmp"
    )
    monkeypatch.setattr("psutil.Process", lambda pid: (_ for _ in ()).throw(psutil.NoSuchProcess(pid)))
    manager.stop_session("b2")
    assert "b2" not in manager.sessions


def test_list_sessions(monkeypatch, tmp_path):
    manager = make_manager(tmp_path)
    manager.sessions["b1"] = SessionInfo(
        instance_id="b1", cdp_url="http://x", pid=1, user_data_dir="/tmp"
    )
    monkeypatch.setattr(manager, "get_session", lambda instance_id: manager.sessions.get(instance_id))
    assert "b1" in manager.list_sessions()
