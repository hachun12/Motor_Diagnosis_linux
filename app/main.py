"""馬達即時診斷系統 WebUI — FastAPI 入口。

REST 控制 + WebSocket 即時推播（波形 / 頻譜 / ISO / AI / 日誌）。
"""
import asyncio
import os
import threading
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import dsp
from .acquisition import AcquisitionService
from .auth import COOKIE_NAME, AuthManager
from .config import STATIC_DIR, load_config
from .datasource.replay import load_csv_channels
from .inference import InferenceService
from .recorder import Recorder

cfg = load_config()


# ══════════════════════════════════════════════════════════════
#  日誌環形緩衝 + WebSocket Hub
# ══════════════════════════════════════════════════════════════
class LogBuffer:
    def __init__(self, hub, maxlen=500):
        self._hub = hub
        self._lock = threading.Lock()
        self.entries = deque(maxlen=maxlen)

    def log(self, message: str):
        entry = {"time": datetime.now().strftime("%H:%M:%S"), "message": message}
        with self._lock:
            self.entries.append(entry)
        print(f"[{entry['time']}] {message}", flush=True)
        self._hub.publish_threadsafe({"type": "log", **entry})

    def recent(self):
        with self._lock:
            return list(self.entries)

    def clear(self):
        with self._lock:
            self.entries.clear()


class Hub:
    """WebSocket 廣播中心。執行緒可用 publish_threadsafe 丟事件進 event loop。"""

    def __init__(self):
        self.clients = set()
        self.loop = None

    async def register(self, ws: WebSocket):
        self.clients.add(ws)

    def unregister(self, ws: WebSocket):
        self.clients.discard(ws)

    async def broadcast(self, payload: dict):
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.unregister(ws)

    def publish_threadsafe(self, payload: dict):
        if self.loop is None or self.loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(payload), self.loop)


hub = Hub()
logbuf = LogBuffer(hub)
acq = AcquisitionService(cfg, logbuf.log)
recorder = Recorder(cfg, acq, logbuf.log)
auth = AuthManager(cfg, logbuf.log)
inference = None  # 模型於 lifespan 載入（啟動失敗時給出明確訊息）


# ══════════════════════════════════════════════════════════════
#  背景推播任務
# ══════════════════════════════════════════════════════════════
_last_iso_level = None


async def wave_loop():
    """每 50ms 推送波形幀（含 RMS / ISO 狀態）。"""
    global _last_iso_level
    while True:
        await asyncio.sleep(0.05)
        if not acq.is_running or not hub.clients:
            continue
        snap = acq.snapshot_display()
        iso = dsp.iso_zone(dsp.rms(snap.get("X", [])), cfg.iso_zones)
        if iso["level"] != _last_iso_level:
            if _last_iso_level is not None:
                logbuf.log(f"[ISO] RMS={iso['rms']:.3f} → {iso['text']}")
            _last_iso_level = iso["level"]
        frame = {
            "type": "wave",
            "cur": {k: [round(v, 4) for v in snap[k]] for k in cfg.current_keys},
            "vib": {k: [round(v, 4) for v in snap[k]] for k in cfg.vibration_keys},
            "iso": iso,
        }
        await hub.broadcast(frame)


async def spectrum_loop():
    """每 spectrum_interval 秒推送 6 通道頻譜。"""
    while True:
        await asyncio.sleep(cfg.spectrum_interval)
        if not acq.is_running or not hub.clients:
            continue
        window = acq.latest_window(cfg.spectrum_nfft)
        if window is None:
            continue
        payload = {"type": "spectrum", "channels": {}}
        freqs = None
        for k in cfg.channel_keys:
            f, m = dsp.spectrum(np.asarray(window[k]), cfg.sample_rate,
                                cfg.spectrum_max_bins)
            freqs = freqs or f
            payload["channels"][k] = m
        payload["freqs"] = freqs
        await hub.broadcast(payload)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference
    hub.loop = asyncio.get_running_loop()
    logbuf.log("系統初始化完成，等待啟動...")
    inference = InferenceService(
        cfg, acq, logbuf.log,
        on_result=lambda r: hub.publish_threadsafe({"type": "ai", **r}))
    inference.start()  # 常駐推論執行緒（未啟動採樣時待命）
    logbuf.log(f"AI 模型載入完成（運行於 {inference.model.device}）")
    logbuf.log(f"資料源設定：{cfg.driver}")
    tasks = [asyncio.create_task(wave_loop()), asyncio.create_task(spectrum_loop())]
    yield
    for t in tasks:
        t.cancel()
    acq.stop()


app = FastAPI(title="馬達即時診斷系統", lifespan=lifespan)


# ══════════════════════════════════════════════════════════════
#  認證
# ══════════════════════════════════════════════════════════════
def current_user(request: Request):
    return auth.read_session(request.cookies.get(COOKIE_NAME))


def require_user(request: Request):
    user = current_user(request)
    if not user:
        raise HTTPException(401, "請先登入")
    return user


def require_admin(user=Depends(require_user)):
    if user["role"] != "admin":
        raise HTTPException(403, "需要管理員權限")
    return user


class LoginBody(BaseModel):
    username: str
    password: str


class UserBody(BaseModel):
    username: str
    password: str
    role: str


class PasswordBody(BaseModel):
    old_password: str
    new_password: str


class SaveBody(BaseModel):
    label: str


@app.post("/api/login")
def login(body: LoginBody, request: Request):
    ip = request.client.host if request.client else "?"
    wait = auth.check_lockout(ip)
    if wait > 0:
        raise HTTPException(429, f"登入失敗次數過多，請 {int(wait) + 1} 秒後再試")
    token = auth.login(body.username.strip(), body.password, ip)
    if not token:
        raise HTTPException(401, "帳號或密碼錯誤")
    resp = JSONResponse({"ok": True})
    resp.set_cookie(COOKIE_NAME, token, httponly=True, samesite="lax",
                    max_age=int(cfg.session_hours * 3600))
    logbuf.log(f"🔓 使用者 {body.username} 登入")
    return resp


@app.post("/api/logout")
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE_NAME)
    return resp


@app.get("/api/me")
def me(user=Depends(require_user)):
    return user


@app.post("/api/password")
def change_password(body: PasswordBody, user=Depends(require_user)):
    if not auth.verify(user["username"], body.old_password):
        raise HTTPException(401, "舊密碼錯誤")
    try:
        auth.change_password(user["username"], body.new_password)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.get("/api/users")
def list_users(user=Depends(require_admin)):
    return auth.list_users()


@app.post("/api/users")
def add_user(body: UserBody, user=Depends(require_admin)):
    try:
        auth.add_user(body.username.strip(), body.password, body.role)
    except ValueError as e:
        raise HTTPException(400, str(e))
    logbuf.log(f"👤 管理員 {user['username']} 新增使用者 {body.username}（{body.role}）")
    return {"ok": True}


@app.delete("/api/users/{username}")
def delete_user(username: str, user=Depends(require_admin)):
    try:
        auth.delete_user(username, operator=user["username"])
    except ValueError as e:
        raise HTTPException(400, str(e))
    logbuf.log(f"👤 管理員 {user['username']} 刪除使用者 {username}")
    return {"ok": True}


# ══════════════════════════════════════════════════════════════
#  控制與狀態
# ══════════════════════════════════════════════════════════════
@app.post("/api/start")
def start_system(user=Depends(require_admin)):
    try:
        acq.start()
    except Exception as e:
        logbuf.log(f"❌ DAQ 啟動失敗: {e}")
        raise HTTPException(400, f"啟動失敗：{e}")
    global _last_iso_level
    _last_iso_level = None
    hub.publish_threadsafe({"type": "status", "running": True})
    return {"ok": True}


@app.post("/api/stop")
def stop_system(user=Depends(require_admin)):
    acq.stop()
    hub.publish_threadsafe({"type": "status", "running": False})
    return {"ok": True}


@app.get("/api/status")
def status(user=Depends(require_user)):
    return {
        "running": acq.is_running,
        "driver": cfg.driver,
        "sample_rate": cfg.sample_rate,
        "history_seconds": cfg.history_seconds,
        "buffered_seconds": round(acq.history_length() / cfg.sample_rate, 1),
        "current_keys": cfg.current_keys,
        "vibration_keys": cfg.vibration_keys,
        "ai": inference.latest if inference else None,
        "error": acq.error,
    }


@app.get("/api/logs")
def logs(user=Depends(require_user)):
    return logbuf.recent()


# ══════════════════════════════════════════════════════════════
#  標籤與存檔
# ══════════════════════════════════════════════════════════════
@app.get("/api/labels")
def get_labels(user=Depends(require_user)):
    return recorder.labels


@app.post("/api/save")
def save(body: SaveBody, user=Depends(require_admin)):
    try:
        result = recorder.save(body.label)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return result


# ══════════════════════════════════════════════════════════════
#  記錄清單 / 下載 / 回放
# ══════════════════════════════════════════════════════════════
_replay_cache = {"name": None, "data": None}


def _load_recording(name: str) -> np.ndarray:
    if _replay_cache["name"] != name:
        _replay_cache["data"] = load_csv_channels(Recorder.resolve(name))
        _replay_cache["name"] = name
    return _replay_cache["data"]


@app.get("/api/recordings")
def recordings(user=Depends(require_user)):
    return recorder.list_recordings()


@app.get("/api/recordings/{name}")
def download_recording(name: str, user=Depends(require_user)):
    try:
        path = Recorder.resolve(name)
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, "找不到檔案")
    return FileResponse(path, filename=os.path.basename(path), media_type="text/csv")


@app.get("/api/recordings/{name}/data")
def recording_data(name: str, start: float = 0.0, window: float = 2.0,
                   user=Depends(require_user)):
    """回放：取記錄中 [start, start+window] 秒的波形（降採樣）與頻譜。"""
    try:
        data = _load_recording(name)
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, "找不到檔案")
    except Exception as e:
        raise HTTPException(400, f"檔案讀取失敗：{e}")

    fs = cfg.sample_rate
    total = data.shape[1]
    total_seconds = total / fs
    window = max(0.1, min(float(window), 10.0))
    start = max(0.0, min(float(start), max(0.0, total_seconds - window)))
    i0, i1 = int(start * fs), min(int((start + float(window)) * fs), total)
    seg = data[:, i0:i1]

    stride = max(1, seg.shape[1] // 2000)
    t = [round(start + i / fs, 5) for i in range(0, seg.shape[1], stride)]
    wave, spec = {}, {}
    freqs = None
    for i, k in enumerate(cfg.channel_keys):
        wave[k] = [round(float(v), 4) for v in seg[i, ::stride]]
        f, m = dsp.spectrum(seg[i], fs, cfg.spectrum_max_bins)
        freqs = freqs or f
        spec[k] = m

    # 對本視窗最後 model_window（2048）點跑模型判斷
    ai = None
    if seg.shape[1] >= cfg.model_window:
        win = {k: seg[i, -cfg.model_window:].tolist()
               for i, k in enumerate(cfg.channel_keys)}
        try:
            pred, conf = inference.model.predict(win, cfg.channel_keys)
            ai = {"class": pred, "confidence": round(conf, 4),
                  "normal": pred in cfg.normal_classes}
        except Exception as e:
            logbuf.log(f"❌ 回放推論失敗: {e}")
    else:
        ai = {"insufficient": True, "needed": cfg.model_window,
              "got": int(seg.shape[1])}

    return {"total_seconds": round(total_seconds, 2), "fs": fs,
            "start": round(start, 3), "window": window,
            "t": t, "wave": wave, "freqs": freqs, "spectrum": spec, "ai": ai}


# ══════════════════════════════════════════════════════════════
#  WebSocket
# ══════════════════════════════════════════════════════════════
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    user = auth.read_session(ws.cookies.get(COOKIE_NAME))
    if not user:
        await ws.close(code=4401)
        return
    await ws.accept()
    await hub.register(ws)
    try:
        while True:
            await ws.receive_text()  # 目前僅用於偵測斷線
    except WebSocketDisconnect:
        pass
    finally:
        hub.unregister(ws)


# ══════════════════════════════════════════════════════════════
#  頁面與靜態資源
# ══════════════════════════════════════════════════════════════
# HTML 一律要求瀏覽器重新驗證，避免改版後拿到舊頁（資源檔另以 ?v= 參數控管）
_NO_CACHE = {"Cache-Control": "no-cache"}


@app.get("/")
def index(request: Request):
    if not current_user(request):
        return RedirectResponse("/login")
    return FileResponse(os.path.join(STATIC_DIR, "index.html"), headers=_NO_CACHE)


@app.get("/login")
def login_page(request: Request):
    if current_user(request):
        return RedirectResponse("/")
    return FileResponse(os.path.join(STATIC_DIR, "login.html"), headers=_NO_CACHE)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=cfg.host, port=cfg.port)
