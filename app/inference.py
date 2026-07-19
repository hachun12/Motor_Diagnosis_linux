"""AI 推論：MotorFaultCNN + 背景推論服務（模型自模型庫載入，支援熱切換）。"""
import threading
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from .config import Config


class MotorFaultCNN(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)


class LoadedModel:
    """一組可推論的模型 + 其 meta（classes、normal、視窗）。"""

    def __init__(self, name: str, weights_path: str, meta: dict):
        self.name = name
        self.classes = list(meta.get("classes", []))
        self.normal_classes = set(meta.get("normal_classes", []))
        self.window = int(meta.get("window", 2048))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model = MotorFaultCNN(input_dim=6, num_classes=len(self.classes))
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, window: dict, keys: list):
        """window: {通道key: list}，每通道至少 self.window 筆。回傳 (類別, 信心)。"""
        sliced = [window[k][-self.window:] for k in keys]
        combined = np.vstack(sliced).T                      # (window, 6)
        tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, idx = torch.max(probs, dim=1)
        return self.classes[idx.item()], float(conf.item())


class InferenceService:
    """背景執行緒：每 interval 秒取最新視窗推論一次。模型可熱切換。"""

    def __init__(self, cfg: Config, acquisition, log_fn, on_result, registry):
        self.cfg = cfg
        self.acq = acquisition
        self._log = log_fn
        self._on_result = on_result
        self.registry = registry
        self._thread = None
        self.latest = None

        name = registry.active_name()
        if not name:
            raise RuntimeError("模型庫中沒有可用模型")
        self.current = LoadedModel(name, registry.weights_path(name),
                                   registry.get_meta(name) or {})

    # 便捷屬性（回放與狀態端點使用）
    @property
    def window(self):
        return self.current.window

    @property
    def normal_classes(self):
        return self.current.normal_classes

    def activate(self, name: str):
        """熱切換：載入新模型成功後才替換引用，失敗不影響現役模型。"""
        loaded = LoadedModel(name, self.registry.weights_path(name),
                             self.registry.get_meta(name) or {})
        self.current = loaded          # 引用替換為原子操作，推論端取區域引用使用
        self.registry.mark_active(name)
        self.latest = None
        self._log(f"🧠 已啟用模型「{name}」（{len(loaded.classes)} 類，運行於 {loaded.device}）")

    def build_result(self, pred: str, conf: float) -> dict:
        is_normal = pred in self.current.normal_classes
        return {
            "class": pred,
            "confidence": round(conf, 4),
            "normal": is_normal,
            "time": datetime.now().strftime("%H:%M:%S"),
            **self.cfg.describe_class(pred),
        }

    def start(self):
        """常駐背景執行緒：系統未啟動時待命，啟動後每 interval 秒推論一次。"""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        keys = self.cfg.channel_keys
        while True:
            time.sleep(self.cfg.model_interval)
            if not self.acq.is_running:
                continue
            model = self.current               # 取區域引用，避免切換途中換模型
            window = self.acq.latest_window(model.window)
            if window is None:
                continue
            try:
                pred, conf = model.predict(window, keys)
            except Exception as e:
                self._log(f"❌ AI 推論發生錯誤: {e}")
                continue
            result = self.build_result(pred, conf)
            self.latest = result
            if not result["normal"]:
                self._log(f"⚠️ [AI 警報] 偵測到異常: {result['name'] or pred}"
                          f"（{result['code']}，{conf * 100:.1f}%）")
            self._on_result(result)
