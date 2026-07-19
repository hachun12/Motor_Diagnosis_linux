"""馬達即時診斷系統 WebUI — FastAPI 入口。

REST 控制 + WebSocket 即時推播（波形 / 頻譜 / ISO / AI / 日誌）。
"""
import asyncio
import os
import tempfile
import threading
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
from fastapi import (Depends, FastAPI, File, Form, HTTPException, Request,
                     UploadFile, WebSocket, WebSocketDisconnect)
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import dsp
from .acquisition import AcquisitionService
from .auth import COOKIE_NAME, AuthManager
from .config import STATIC_DIR, load_config
from .datasource.replay import load_csv_channels
from .inference import InferenceService
from .model_registry import MAX_WEIGHTS_BYTES, ModelRegistry
from .recorder import Recorder
from .training import TrainingService

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
registry = ModelRegistry(cfg, logbuf.log)
inference = None   # 模型於 lifespan 載入（啟動失敗時給出明確訊息）
training = None


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
    global inference, training
    hub.loop = asyncio.get_running_loop()
    logbuf.log("系統初始化完成，等待啟動...")
    inference = InferenceService(
        cfg, acq, logbuf.log,
        on_result=lambda r: hub.publish_threadsafe({"type": "ai", **r}),
        registry=registry)
    inference.start()  # 常駐推論執行緒（未啟動採樣時待命）
    training = TrainingService(cfg, registry, logbuf.log, hub.publish_threadsafe)
    logbuf.log(f"AI 模型「{inference.current.name}」載入完成"
               f"（{len(inference.current.classes)} 類，運行於 {inference.current.device}）")
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
        "model": {"name": inference.current.name,
                  "num_classes": len(inference.current.classes)} if inference else None,
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

    # 對本視窗最後 model_window 點跑模型判斷（使用當前啟用模型）
    ai = None
    model = inference.current
    if seg.shape[1] >= model.window:
        win = {k: seg[i, -model.window:].tolist()
               for i, k in enumerate(cfg.channel_keys)}
        try:
            pred, conf = model.predict(win, cfg.channel_keys)
            ai = inference.build_result(pred, conf)
        except Exception as e:
            logbuf.log(f"❌ 回放推論失敗: {e}")
    else:
        ai = {"insufficient": True, "needed": model.window,
              "got": int(seg.shape[1])}

    return {"total_seconds": round(total_seconds, 2), "fs": fs,
            "start": round(start, 3), "window": window,
            "t": t, "wave": wave, "freqs": freqs, "spectrum": spec, "ai": ai}


# ══════════════════════════════════════════════════════════════
#  模型管理
# ══════════════════════════════════════════════════════════════
@app.get("/api/models")
def list_models(user=Depends(require_admin)):
    return registry.list_models()


@app.post("/api/models/import")
async def import_model(user=Depends(require_admin),
                       file: UploadFile = File(...),
                       name: str = Form(...),
                       classes: str = Form("")):
    """匯入 .pth 權重。classes 為逗號分隔的類別清單；留空沿用當前模型的類別。"""
    data = await file.read()
    if len(data) > MAX_WEIGHTS_BYTES:
        raise HTTPException(400, "權重檔超過 50MB 上限")
    class_list = ([c.strip() for c in classes.split(",") if c.strip()]
                  if classes.strip() else list(inference.current.classes))
    meta = {
        "classes": class_list,
        "normal_classes": [c for c in class_list if c in ("H", "N")],
        "window": inference.current.window,
        "source": f"匯入（{file.filename}，由 {user['username']} 上傳）",
        "val_accuracy": None,
    }
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        saved = registry.save_model(name, tmp_path, meta)
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        if tmp_path:
            os.unlink(tmp_path)
    logbuf.log(f"📥 管理員 {user['username']} 匯入模型「{saved}」（{len(class_list)} 類）")
    return {"ok": True, "name": saved}


@app.post("/api/models/{name}/activate")
def activate_model(name: str, user=Depends(require_admin)):
    try:
        inference.activate(name)
    except FileNotFoundError:
        raise HTTPException(404, "模型不存在")
    except Exception as e:
        raise HTTPException(400, f"啟用失敗：{e}")
    hub.publish_threadsafe({"type": "model", "name": inference.current.name})
    return {"ok": True}


@app.delete("/api/models/{name}")
def delete_model(name: str, user=Depends(require_admin)):
    try:
        registry.delete(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    logbuf.log(f"🗑️ 管理員 {user['username']} 刪除模型「{name}」")
    return {"ok": True}


@app.get("/api/models/{name}/download")
def download_model(name: str, user=Depends(require_admin)):
    try:
        path = registry.weights_path(name)
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "模型不存在")
    return FileResponse(path, filename=f"{name}.pth",
                        media_type="application/octet-stream")


# ══════════════════════════════════════════════════════════════
#  模型訓練
# ══════════════════════════════════════════════════════════════
class TrainBody(BaseModel):
    name: str
    mapping: dict
    params: dict = {}


@app.get("/api/training/overview")
def training_overview(user=Depends(require_admin)):
    known = list(dict.fromkeys(
        [v.get("code") for v in cfg.class_info.values() if v.get("code")]
        + list(inference.current.classes)))
    return {"groups": training.dataset_overview(), "known_codes": known}


@app.post("/api/training/start")
def training_start(body: TrainBody, user=Depends(require_admin)):
    try:
        training.start(body.name, body.mapping, body.params)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.post("/api/training/cancel")
def training_cancel(user=Depends(require_admin)):
    training.cancel()
    return {"ok": True}


@app.get("/api/training/status")
def training_status(user=Depends(require_admin)):
    return training.status


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
