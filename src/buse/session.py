import json
import os
import platform
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel


class SessionInfo(BaseModel):
    instance_id: str
    cdp_url: str
    pid: int
    user_data_dir: str


class SessionManager:
    def __init__(self, config_dir: Path = Path.home() / ".config" / "buse"):
        self.config_dir = config_dir
        self.sessions_file = config_dir / "sessions.json"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "profiles").mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, SessionInfo] = {}
        self._load_sessions()

    def _load_sessions(self):
        if self.sessions_file.exists():
            try:
                content = self.sessions_file.read_text()
                if not content.strip():
                    self.sessions = {}
                    return
                data = json.loads(content)
                self.sessions = {k: SessionInfo(**v) for k, v in data.items()}
            except Exception:
                self.sessions = {}
        else:
            self.sessions = {}

    def _save_sessions(self):
        with open(self.sessions_file, "w") as f:
            json.dump(
                {k: v.model_dump() for k, v in self.sessions.items()}, f, indent=2
            )

    def _cdp_ready(self, cdp_url: str) -> bool:
        from urllib.parse import urlparse

        try:
            parsed = urlparse(cdp_url)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port
            if port is None:
                return False
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except Exception:
            return False

    def is_alive(self, session: SessionInfo) -> bool:
        import psutil

        if psutil.pid_exists(session.pid):
            return self._cdp_ready(session.cdp_url)

        return self._cdp_ready(session.cdp_url)

    def get_session(self, instance_id: str) -> Optional[SessionInfo]:
        session = self.sessions.get(instance_id)
        if session:
            if self.is_alive(session):
                return session
            else:
                del self.sessions[instance_id]
                self._save_sessions()
        return None

    def start_session(self, instance_id: str, headless: bool = False) -> SessionInfo:
        session = self.get_session(instance_id)
        if session:
            return session

        user_data_dir = self.config_dir / "profiles" / instance_id
        user_data_dir.mkdir(parents=True, exist_ok=True)

        existing_ports = []
        for s in self.sessions.values():
            try:
                existing_ports.append(int(s.cdp_url.split(":")[-1]))
            except Exception:
                pass
        port = self._find_free_port(set(existing_ports))

        chrome_path = self._find_chrome_executable()

        allow_origins = os.getenv("BUSE_REMOTE_ALLOW_ORIGINS")
        if not allow_origins:
            allow_origins = f"http://localhost:{port},http://127.0.0.1:{port}"

        args = [
            chrome_path,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={user_data_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            f"--remote-allow-origins={allow_origins}",
        ]
        if headless:
            args.append("--headless=new")

        sys_name = platform.system()
        popen_kwargs: dict[str, object] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if sys_name == "Windows":
            creationflags = 0
            creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            creationflags |= getattr(subprocess, "DETACHED_PROCESS", 0)
            popen_kwargs["creationflags"] = creationflags
        else:
            popen_kwargs["start_new_session"] = True

        process = subprocess.Popen(args, **popen_kwargs)  # type: ignore[call-overload]

        cdp_url = f"http://localhost:{port}"

        max_retries = 30
        ready = False
        for _ in range(max_retries):
            if self._cdp_ready(cdp_url):
                ready = True
                break
            time.sleep(0.5)
        if not ready:
            try:
                process.kill()
            except Exception:
                pass
            raise RuntimeError("Chrome started but CDP never became ready.")

        session_info = SessionInfo(
            instance_id=instance_id,
            cdp_url=cdp_url,
            pid=process.pid,
            user_data_dir=str(user_data_dir),
        )
        self.sessions[instance_id] = session_info
        self._save_sessions()
        return session_info

    def _find_free_port(
        self, reserved_ports: set[int], start: int = 9222, max_tries: int = 100
    ) -> int:
        port = start
        tries = 0
        while tries < max_tries:
            if port not in reserved_ports and self._is_port_free(port):
                return port
            port += 1
            tries += 1
        return self._find_ephemeral_port()

    def _is_port_free(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                return False
            return True

    def _find_ephemeral_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _find_chrome_executable(self) -> str:
        override = os.getenv("BUSE_CHROME_PATH") or os.getenv("CHROME_PATH")
        if override and Path(override).exists():
            return override

        sys_name = platform.system()
        if sys_name == "Darwin":
            candidates = [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Chromium.app/Contents/MacOS/Chromium",
                "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
            ]
            for candidate in candidates:
                if Path(candidate).exists():
                    return candidate
            raise RuntimeError(
                "Chrome executable not found. Please install Google Chrome."
            )
        elif sys_name == "Linux":
            for cmd in [
                "google-chrome",
                "chromium-browser",
                "chromium",
                "google-chrome-stable",
            ]:
                path = shutil.which(cmd)
                if path:
                    return path
            raise RuntimeError(
                "Chrome executable not found. Please install Google Chrome."
            )
        elif sys_name == "Windows":
            candidates = [
                Path(os.environ.get("PROGRAMFILES", ""))
                / "Google/Chrome/Application/chrome.exe",
                Path(os.environ.get("PROGRAMFILES(X86)", ""))
                / "Google/Chrome/Application/chrome.exe",
                Path(os.environ.get("LOCALAPPDATA", ""))
                / "Google/Chrome/Application/chrome.exe",
            ]
            for candidate in candidates:
                if candidate and candidate.exists():
                    return str(candidate)
            path = shutil.which("chrome") or shutil.which("chrome.exe")
            if path:
                return path

        raise RuntimeError(
            "Chrome executable not found. Please ensure Google Chrome is installed."
        )

    def stop_session(self, instance_id: str):
        import psutil

        session = self.sessions.get(instance_id)
        if session:
            try:
                proc = psutil.Process(session.pid)
                for child in proc.children(recursive=True):
                    child.kill()
                proc.kill()
            except psutil.NoSuchProcess:
                pass

            if instance_id in self.sessions:
                del self.sessions[instance_id]
                self._save_sessions()

    def list_sessions(self) -> Dict[str, SessionInfo]:
        for instance_id in list(self.sessions.keys()):
            self.get_session(instance_id)
        return self.sessions
