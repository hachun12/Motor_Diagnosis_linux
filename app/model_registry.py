"""模型庫：weights + meta 的版本管理、匯入驗證、啟用狀態。

目錄結構（MODELS_DIR，Docker 掛載 volume）：
    models/
    ├─ active.json            ← {"active": "<模型名>"}
    ├─ default/               ← 首次啟動自動由內建權重遷移
    │  ├─ weights.pth
    │  └─ meta.json           ← classes / normal_classes / window / 訓練資訊
    └─ <其他模型>/

安全注意：.pth 為 pickle 格式，一律以 torch.load(weights_only=True) 載入，
匯入時先做形狀驗證與假資料試推論，失敗即拒絕。
"""
import json
import os
import re
import shutil
import threading
from datetime import datetime

import torch

from .config import Config, MODELS_DIR
from .inference import MotorFaultCNN

_SAFE_NAME = re.compile(r"[^\w一-鿿.-]+")
WEIGHTS_FILE = "weights.pth"
META_FILE = "meta.json"
ACTIVE_FILE = os.path.join(MODELS_DIR, "active.json")
MAX_WEIGHTS_BYTES = 50 * 1024 * 1024


def sanitize_name(name: str) -> str:
    name = _SAFE_NAME.sub("_", (name or "").strip())
    if not name or name in (".", ".."):
        raise ValueError("請提供有效的模型名稱")
    return name


class ModelRegistry:
    def __init__(self, cfg: Config, log_fn):
        self.cfg = cfg
        self._log = log_fn
        self._lock = threading.Lock()
        os.makedirs(MODELS_DIR, exist_ok=True)
        self._bootstrap()

    # ── 初始遷移：把內建權重收編為 default 模型 ──────────────
    def _bootstrap(self):
        if self.list_models():
            return
        if not os.path.exists(self.cfg.model_path):
            self._log("⚠️ 模型庫為空且找不到內建權重檔")
            return
        meta = {
            "classes": self.cfg.model_classes,
            "normal_classes": sorted(self.cfg.normal_classes),
            "window": self.cfg.model_window,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "內建初始模型（best_motor_cnn_weights.pth）",
            "val_accuracy": None,
        }
        d = os.path.join(MODELS_DIR, "default")
        os.makedirs(d, exist_ok=True)
        shutil.copy2(self.cfg.model_path, os.path.join(d, WEIGHTS_FILE))
        with open(os.path.join(d, META_FILE), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        self._set_active("default")
        self._log("📦 已將內建權重遷移為模型庫項目「default」")

    # ── 查詢 ────────────────────────────────────────────────
    def list_models(self):
        items = []
        active = self.active_name()
        if not os.path.isdir(MODELS_DIR):
            return items
        for name in sorted(os.listdir(MODELS_DIR)):
            d = os.path.join(MODELS_DIR, name)
            wpath = os.path.join(d, WEIGHTS_FILE)
            if not os.path.isdir(d) or not os.path.exists(wpath):
                continue
            meta = self.get_meta(name) or {}
            items.append({
                "name": name,
                "classes": meta.get("classes", []),
                "num_classes": len(meta.get("classes", [])),
                "window": meta.get("window"),
                "val_accuracy": meta.get("val_accuracy"),
                "created": meta.get("created", ""),
                "source": meta.get("source", ""),
                "size": os.path.getsize(wpath),
                "active": name == active,
            })
        return items

    def get_meta(self, name):
        try:
            with open(os.path.join(MODELS_DIR, name, META_FILE), "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def weights_path(self, name):
        path = os.path.join(MODELS_DIR, sanitize_name(name), WEIGHTS_FILE)
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型不存在：{name}")
        return path

    def active_name(self):
        try:
            with open(ACTIVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("active")
        except Exception:
            return None

    def _set_active(self, name):
        with open(ACTIVE_FILE, "w", encoding="utf-8") as f:
            json.dump({"active": name}, f, ensure_ascii=False)

    def mark_active(self, name):
        """僅更新啟用記錄（實際熱切換由 InferenceService 完成後呼叫）。"""
        with self._lock:
            self._set_active(sanitize_name(name))

    # ── 驗證與載入 ──────────────────────────────────────────
    @staticmethod
    def validate_weights(weights_path: str, classes: list, window: int):
        """載入權重、形狀驗證、假資料試推論。失敗 raise ValueError。"""
        if len(classes) < 2:
            raise ValueError("classes 至少需 2 類")
        try:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
        except Exception as e:
            raise ValueError(f"權重檔無法以安全模式載入（weights_only）：{e}")
        model = MotorFaultCNN(input_dim=6, num_classes=len(classes))
        try:
            model.load_state_dict(state)
        except Exception as e:
            raise ValueError(f"權重與 MotorFaultCNN 結構不符：{e}")
        model.eval()
        try:
            with torch.no_grad():
                out = model(torch.zeros(1, int(window), 6))
            if out.shape != (1, len(classes)):
                raise ValueError(f"輸出形狀異常：{tuple(out.shape)}")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"試推論失敗：{e}")
        return model

    # ── 匯入 / 新增 ─────────────────────────────────────────
    def save_model(self, name: str, weights_src_path: str, meta: dict,
                   overwrite: bool = False) -> str:
        """驗證後把權重與 meta 寫入模型庫，回傳正式名稱。"""
        name = sanitize_name(name)
        meta = dict(meta)
        classes = meta.get("classes") or []
        window = int(meta.get("window") or self.cfg.model_window)
        if os.path.getsize(weights_src_path) > MAX_WEIGHTS_BYTES:
            raise ValueError("權重檔超過 50MB 上限")
        self.validate_weights(weights_src_path, classes, window)

        with self._lock:
            d = os.path.join(MODELS_DIR, name)
            if os.path.exists(d) and not overwrite:
                raise ValueError(f"模型「{name}」已存在")
            os.makedirs(d, exist_ok=True)
            shutil.copy2(weights_src_path, os.path.join(d, WEIGHTS_FILE))
            meta.setdefault("created", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            meta["window"] = window
            meta.setdefault("normal_classes",
                            [c for c in classes if c in ("H", "N")])
            with open(os.path.join(d, META_FILE), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        return name

    def delete(self, name: str):
        name = sanitize_name(name)
        with self._lock:
            if name == self.active_name():
                raise ValueError("啟用中的模型不可刪除，請先啟用其他模型")
            d = os.path.join(MODELS_DIR, name)
            if not os.path.isdir(d):
                raise ValueError(f"模型不存在：{name}")
            shutil.rmtree(d)
