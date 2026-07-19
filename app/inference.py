"""AI 推論：MotorFaultCNN（自桌面版原樣移植）+ 背景推論服務。"""
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


class MotorDiagnosisModel:
    def __init__(self, cfg: Config):
        self.classes = cfg.model_classes
        self.window = cfg.model_window
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MotorFaultCNN(input_dim=6, num_classes=len(self.classes))
        weights = torch.load(cfg.model_path, map_location=self.device)
        self.model.load_state_dict(weights)
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
    """背景執行緒：每 interval 秒取最新視窗推論一次。"""

    def __init__(self, cfg: Config, acquisition, log_fn, on_result):
        self.cfg = cfg
        self.acq = acquisition
        self._log = log_fn
        self._on_result = on_result       # callback(result_dict)，執行緒安全由呼叫端處理
        self.model = MotorDiagnosisModel(cfg)
        self._thread = None
        self.latest = None                # 最近一次推論結果

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
            window = self.acq.latest_window(self.cfg.model_window)
            if window is None:
                continue
            try:
                pred, conf = self.model.predict(window, keys)
            except Exception as e:
                self._log(f"❌ AI 推論發生錯誤: {e}")
                continue
            is_normal = pred in self.cfg.normal_classes
            desc = self.cfg.describe_class(pred)
            result = {
                "class": pred,
                "confidence": round(conf, 4),
                "normal": is_normal,
                "time": datetime.now().strftime("%H:%M:%S"),
                **desc,
            }
            self.latest = result
            if not is_normal:
                self._log(f"⚠️ [AI 警報] 偵測到異常: {desc['name'] or pred}"
                          f"（{desc['code']}，{conf * 100:.1f}%）")
            self._on_result(result)
