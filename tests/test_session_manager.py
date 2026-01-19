import json
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

    class DummySock:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("socket.create_connection", lambda *args, **kwargs: DummySock())
    assert manager._cdp_ready("http://x:9222") is True


def test_cdp_ready_false(monkeypatch, tmp_path):
    manager = make_manager(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr("socket.create_connection", boom)
    assert manager._cdp_ready("http://x:9222") is False


def test_cdp_ready_missing_port(tmp_path):
    manager = make_manager(tmp_path)
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
        instance_id="b_bad",
        cdp_url="http://localhost:notaport",
        pid=3,
        user_data_dir="/tmp",
    )

    captured = {}

    class FakeProcess:
        pid = 999

        def kill(self):
            captured["killed"] = True

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(manager, "_find_chrome_executable", lambda: "/bin/chrome")
    monkeypatch.setattr(manager, "_find_free_port", lambda reserved_ports: 9223)
    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr(manager, "_cdp_ready", lambda url: True)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    info = manager.start_session("b1", headless=True)
    assert info.instance_id == "b1"
    assert info.cdp_url.endswith(":9223")
    assert "--headless=new" in captured["args"]


def test_start_session_windows_creationflags(monkeypatch, tmp_path):
    manager = make_manager(tmp_path)
    captured = {}

    class FakeProcess:
        pid = 999

        def kill(self):
            captured["killed"] = True

    def fake_popen(args, **kwargs):
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr(
        manager, "_find_chrome_executable", lambda: "C:\\\\Chrome\\\\chrome.exe"
    )
    monkeypatch.setattr(manager, "_find_free_port", lambda reserved_ports: 9222)
    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr(manager, "_cdp_ready", lambda url: True)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    manager.start_session("b1", headless=False)
    assert "creationflags" in captured["kwargs"]


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
    monkeypatch.setattr(manager, "_find_free_port", lambda reserved_ports: 9222)
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: proc)
    monkeypatch.setattr(manager, "_cdp_ready", lambda url: False)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    with pytest.raises(RuntimeError):
        manager.start_session("b1")
    assert proc.killed is True


def test_find_free_port_falls_back(monkeypatch, tmp_path):
    manager = make_manager(tmp_path)
    monkeypatch.setattr(manager, "_is_port_free", lambda port: False)
    monkeypatch.setattr(manager, "_find_ephemeral_port", lambda: 9999)
    assert manager._find_free_port(set(), start=9222, max_tries=2) == 9999


def test_find_free_port_returns_available(monkeypatch, tmp_path):
    manager = make_manager(tmp_path)
    monkeypatch.setattr(manager, "_is_port_free", lambda port: port == 9223)
    assert manager._find_free_port(set(), start=9222, max_tries=5) == 9223


def test_is_port_free_false(monkeypatch, tmp_path):
    manager = make_manager(tmp_path)

    class FakeSock:
        def __init__(self, *args, **kwargs):
            pass

        def setsockopt(self, *args, **kwargs):
            return None

        def bind(self, *args, **kwargs):
            raise OSError("in use")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("socket.socket", lambda *args, **kwargs: FakeSock())
    assert manager._is_port_free(1234) is False


def test_is_port_free_true(monkeypatch, tmp_path):
    manager = make_manager(tmp_path)

    class FakeSock:
        def __init__(self, *args, **kwargs):
            pass

        def setsockopt(self, *args, **kwargs):
            return None

        def bind(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("socket.socket", lambda *args, **kwargs: FakeSock())
    assert manager._is_port_free(1234) is True


def test_find_ephemeral_port(monkeypatch, tmp_path):
    manager = make_manager(tmp_path)

    class FakeSock:
        def __init__(self, *args, **kwargs):
            pass

        def bind(self, *args, **kwargs):
            return None

        def getsockname(self):
            return ("127.0.0.1", 5555)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("socket.socket", lambda *args, **kwargs: FakeSock())
    assert manager._find_ephemeral_port() == 5555


def test_find_chrome_executable(monkeypatch, tmp_path):
    manager = make_manager(tmp_path)

    def fake_exists(path: Path):
        return str(path).endswith("Google Chrome")

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr(Path, "exists", fake_exists)
    assert manager._find_chrome_executable().endswith("Google Chrome")

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr(Path, "exists", lambda p: False)
    with pytest.raises(RuntimeError):
        manager._find_chrome_executable()

    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")
    assert manager._find_chrome_executable().startswith("/usr/bin/")

    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    with pytest.raises(RuntimeError):
        manager._find_chrome_executable()

    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr("shutil.which", lambda cmd: "C:\\\\Chrome\\\\chrome.exe")
    monkeypatch.setattr(Path, "exists", lambda p: False)
    assert manager._find_chrome_executable().endswith("chrome.exe")

    monkeypatch.setattr("platform.system", lambda: "Other")
    with pytest.raises(RuntimeError):
        manager._find_chrome_executable()


def test_find_chrome_override(monkeypatch, tmp_path):
    manager = make_manager(tmp_path)
    monkeypatch.setenv("BUSE_CHROME_PATH", "/tmp/chrome")
    monkeypatch.setattr(Path, "exists", lambda p: str(p) == "/tmp/chrome")
    assert manager._find_chrome_executable() == "/tmp/chrome"


def test_find_chrome_windows_candidate(monkeypatch, tmp_path):
    manager = make_manager(tmp_path)
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setenv("PROGRAMFILES", "C:\\\\Program Files")
    monkeypatch.setattr(
        Path,
        "exists",
        lambda p: str(p).endswith("Google/Chrome/Application/chrome.exe"),
    )
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    assert manager._find_chrome_executable().endswith("chrome.exe")


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
    monkeypatch.setattr(
        "psutil.Process", lambda pid: (_ for _ in ()).throw(psutil.NoSuchProcess(pid))
    )
    manager.stop_session("b2")
    assert "b2" not in manager.sessions


def test_list_sessions(monkeypatch, tmp_path):
    manager = make_manager(tmp_path)
    manager.sessions["b1"] = SessionInfo(
        instance_id="b1", cdp_url="http://x", pid=1, user_data_dir="/tmp"
    )
    monkeypatch.setattr(
        manager, "get_session", lambda instance_id: manager.sessions.get(instance_id)
    )
    assert "b1" in manager.list_sessions()
