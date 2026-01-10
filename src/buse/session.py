import json
import platform
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
        import httpx

        try:
            resp = httpx.get(f"{cdp_url}/json/version", timeout=0.5)
            return resp.status_code == 200
        except Exception:
            return False

    def is_alive(self, session: SessionInfo) -> bool:
        import psutil

        # First check PID
        if psutil.pid_exists(session.pid):
            # Prefer verifying CDP to avoid stale PID reuse
            return self._cdp_ready(session.cdp_url)
        # If PID is gone, try CDP port
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

        port = 9222
        existing_ports = []
        for s in self.sessions.values():
            try:
                existing_ports.append(int(s.cdp_url.split(":")[-1]))
            except Exception:
                pass

        while port in existing_ports:
            port += 1

        chrome_path = self._find_chrome_executable()

        args = [
            chrome_path,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={user_data_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--remote-allow-origins=*",
        ]
        if headless:
            args.append("--headless=new")

        # Use start_new_session=True to detach properly
        process = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        cdp_url = f"http://localhost:{port}"

        # Wait for CDP to be ready
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

    def _find_chrome_executable(self) -> str:
        sys_name = platform.system()
        if sys_name == "Darwin":
            return "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        elif sys_name == "Linux":
            for cmd in [
                "google-chrome",
                "chromium-browser",
                "chromium",
                "google-chrome-stable",
            ]:
                try:
                    subprocess.run(
                        [cmd, "--version"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return cmd
                except FileNotFoundError:
                    continue
        elif sys_name == "Windows":
            return (
                "C:\\\\Program Files\\\\Google\\\\Chrome\\\\Application\\\\chrome.exe"
            )

        raise RuntimeError(
            "Chrome executable not found. Please ensure Google Chrome is installed."
        )

    def stop_session(self, instance_id: str):
        import psutil

        session = self.sessions.get(
            instance_id
        )  # Don't use get_session to avoid cleanup if PID is gone but CDP is alive
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
