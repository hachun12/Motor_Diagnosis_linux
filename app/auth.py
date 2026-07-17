"""認證授權：users.yaml + bcrypt 密碼雜湊 + 簽章 session cookie。

角色：
- admin  ：啟動/停止、存檔、標籤與使用者管理
- viewer ：僅觀看波形與狀態

首次啟動若 users.yaml 不存在，會以環境變數 ADMIN_PASSWORD 建立 admin；
未設定該環境變數時產生隨機密碼並寫入日誌（請立即登入修改）。
"""
import os
import secrets
import threading
import time

import bcrypt
import yaml
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from .config import Config, USERS_FILE, SECRET_FILE

COOKIE_NAME = "motor_session"
ROLES = ("admin", "viewer")


class AuthManager:
    def __init__(self, cfg: Config, log_fn):
        self.cfg = cfg
        self._log = log_fn
        self._lock = threading.Lock()
        self._failed = {}  # ip -> [count, lockout_until]
        self.users = self._load_or_bootstrap()
        self.serializer = URLSafeTimedSerializer(self._load_secret(), salt="motor-session")

    # ── 使用者存放 ──────────────────────────────────────────
    def _load_or_bootstrap(self) -> dict:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("users", {})

        password = os.environ.get("ADMIN_PASSWORD")
        if not password:
            password = secrets.token_urlsafe(9)
            self._log(f"🔑 未設定 ADMIN_PASSWORD，已產生隨機管理員密碼：{password}"
                      "（請登入後盡快修改）")
        users = {"admin": {"password_hash": self._hash(password), "role": "admin"}}
        self._persist(users)
        self._log("已建立預設管理員帳號 admin（users.yaml）")
        return users

    def _persist(self, users=None):
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            yaml.safe_dump({"users": users or self.users}, f, allow_unicode=True)
        try:
            os.chmod(USERS_FILE, 0o600)
        except OSError:
            pass

    def _load_secret(self) -> str:
        secret = os.environ.get("SESSION_SECRET")
        if secret:
            return secret
        if os.path.exists(SECRET_FILE):
            with open(SECRET_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        secret = secrets.token_urlsafe(32)
        with open(SECRET_FILE, "w", encoding="utf-8") as f:
            f.write(secret)
        try:
            os.chmod(SECRET_FILE, 0o600)
        except OSError:
            pass
        return secret

    @staticmethod
    def _hash(password: str) -> str:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    # ── 登入 / 節流 ─────────────────────────────────────────
    def check_lockout(self, ip: str) -> float:
        """回傳剩餘鎖定秒數，0 表示未鎖定。"""
        with self._lock:
            entry = self._failed.get(ip)
            if entry and entry[1] > time.monotonic():
                return entry[1] - time.monotonic()
            return 0.0

    def login(self, username: str, password: str, ip: str):
        """成功回傳 session token，失敗回傳 None。"""
        user = self.users.get(username)
        ok = bool(user) and bcrypt.checkpw(password.encode(),
                                           user["password_hash"].encode())
        with self._lock:
            if ok:
                self._failed.pop(ip, None)
            else:
                entry = self._failed.setdefault(ip, [0, 0.0])
                entry[0] += 1
                if entry[0] >= self.cfg.max_failed_attempts:
                    entry[1] = time.monotonic() + self.cfg.lockout_seconds
                    entry[0] = 0
        if not ok:
            return None
        return self.serializer.dumps({"u": username, "r": user["role"]})

    def read_session(self, token: str):
        if not token:
            return None
        try:
            data = self.serializer.loads(token, max_age=int(self.cfg.session_hours * 3600))
        except (BadSignature, SignatureExpired):
            return None
        # 使用者可能已被刪除或改角色，以現況為準
        user = self.users.get(data.get("u"))
        if not user or user["role"] != data.get("r"):
            return None
        return {"username": data["u"], "role": data["r"]}

    def verify(self, username: str, password: str) -> bool:
        """單純驗證密碼（不動失敗計數），改密碼前確認舊密碼用。"""
        user = self.users.get(username)
        return bool(user) and bcrypt.checkpw(password.encode(),
                                             user["password_hash"].encode())

    # ── 使用者管理（admin）─────────────────────────────────
    def list_users(self):
        return [{"username": u, "role": info["role"]} for u, info in self.users.items()]

    def add_user(self, username: str, password: str, role: str):
        if role not in ROLES:
            raise ValueError(f"角色須為 {ROLES}")
        if not username or not username.isidentifier():
            raise ValueError("帳號僅能使用英數與底線")
        if len(password) < 6:
            raise ValueError("密碼至少 6 碼")
        with self._lock:
            if username in self.users:
                raise ValueError("帳號已存在")
            self.users[username] = {"password_hash": self._hash(password), "role": role}
            self._persist()

    def delete_user(self, username: str, operator: str):
        with self._lock:
            if username == operator:
                raise ValueError("不能刪除自己")
            if username not in self.users:
                raise ValueError("帳號不存在")
            remaining_admins = [u for u, i in self.users.items()
                                if i["role"] == "admin" and u != username]
            if not remaining_admins:
                raise ValueError("至少須保留一名管理員")
            del self.users[username]
            self._persist()

    def change_password(self, username: str, new_password: str):
        if len(new_password) < 6:
            raise ValueError("密碼至少 6 碼")
        with self._lock:
            if username not in self.users:
                raise ValueError("帳號不存在")
            self.users[username]["password_hash"] = self._hash(new_password)
            self._persist()
