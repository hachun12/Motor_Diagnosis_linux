"""採樣服務：背景執行緒從 DataSource 讀資料，維護顯示/歷史緩衝。

沿用桌面版的執行緒模型：採樣執行緒只寫緩衝，不碰任何 I/O 之外的東西；
其他人透過 snapshot 方法在鎖內快速複製。
"""
import threading
from collections import deque
from datetime import datetime

import numpy as np

from .config import Config
from .datasource import create_datasource


class AcquisitionService:
    def __init__(self, cfg: Config, log_fn):
        self.cfg = cfg
        self._log = log_fn
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self.is_running = False
        self.source = None
        self._thread = None
        self._error = None

        keys = cfg.channel_keys
        self.display = {k: deque([0.0] * cfg.display_points, maxlen=cfg.display_points)
                        for k in keys}
        self.history = {k: deque(maxlen=cfg.history_len) for k in keys}
        # 歷史起點資訊：用於還原每個樣本的時間戳
        self.start_time = None          # datetime，本次啟動時間
        self.total_samples = 0          # 本次啟動以來累計樣本數

        self._offsets = np.array(cfg.offsets).reshape(-1, 1)
        self._gains = np.array(cfg.gains).reshape(-1, 1)

    # ── 控制 ────────────────────────────────────────────────
    def start(self):
        with self._state_lock:
            if self.is_running:
                raise RuntimeError("系統已在執行中")
            source = create_datasource(self.cfg.driver, self.cfg.sample_rate,
                                       self.cfg.driver_options)
            source.start()          # 失敗會 raise，狀態不變
            self.source = source
            self._error = None
            with self._lock:
                for k in self.cfg.channel_keys:
                    self.history[k].clear()
                    self.display[k].extend([0.0] * self.cfg.display_points)
                self.start_time = datetime.now()
                self.total_samples = 0
            self.is_running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            self._log(f"▶ 系統啟動，資料源：{source.describe()}")

    def stop(self):
        with self._state_lock:
            if not self.is_running:
                return
            self.is_running = False
            if self.source:
                try:
                    self.source.stop()
                except Exception as e:
                    self._log(f"⚠️ 資料源停止時發生錯誤: {e}")
                self.source = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._log("⏹ 系統已停止。")

    # ── 採樣執行緒 ──────────────────────────────────────────
    def _run(self):
        batch = self.cfg.batch_size
        while self.is_running:
            try:
                data = self.source.read(batch)          # (6, n)
                if data.shape[1] == 0:
                    continue
                if not self.source.calibrated:
                    data = (data + self._offsets) * self._gains
                with self._lock:
                    for i, k in enumerate(self.cfg.channel_keys):
                        samples = data[i].tolist()
                        self.display[k].extend(samples)
                        self.history[k].extend(samples)
                    self.total_samples += data.shape[1]
            except Exception as e:
                if self.is_running:
                    self._error = str(e)
                    self._log(f"❌ 背景讀取錯誤: {e}")
                break

    # ── 快照 ────────────────────────────────────────────────
    def snapshot_display(self) -> dict:
        with self._lock:
            return {k: list(v) for k, v in self.display.items()}

    def latest_window(self, n: int):
        """取每通道最新 n 筆歷史，回傳 {key: list}。不足時回傳 None。"""
        with self._lock:
            first = self.cfg.channel_keys[0]
            if len(self.history[first]) < n:
                return None
            return {k: list(self.history[k])[-n:] for k in self.cfg.channel_keys}

    def history_snapshot(self):
        """完整歷史複本 + 時間資訊（存檔用）。

        回傳 (channels_dict, start_time, first_sample_index)。
        first_sample_index 是緩衝區第一筆樣本自啟動起算的序號，
        用於計算每筆樣本的時間戳。
        """
        with self._lock:
            channels = {k: list(v) for k, v in self.history.items()}
            n = len(channels[self.cfg.channel_keys[0]])
            first_index = self.total_samples - n
            return channels, self.start_time, first_index

    def history_length(self) -> int:
        with self._lock:
            return len(self.history[self.cfg.channel_keys[0]])

    @property
    def error(self):
        return self._error
