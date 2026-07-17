"""CSV 回放資料源：把 saved_data/ 的記錄當即時訊號重播。"""
import os
import time

import numpy as np

from .base import DataSource
from ..config import DATA_DIR

CSV_COLUMNS = ["cur_A", "cur_B", "cur_C", "vib_X", "vib_Y", "vib_Z"]


def load_csv_channels(path: str) -> np.ndarray:
    """讀取記錄 CSV，回傳 shape=(6, N) 的工程單位陣列。"""
    data = np.genfromtxt(path, delimiter=",", names=True,
                         usecols=CSV_COLUMNS, encoding="utf-8")
    if data.shape == ():  # 只有一列
        data = data.reshape(1)
    return np.vstack([data[c] for c in CSV_COLUMNS])


class ReplaySource(DataSource):
    calibrated = True  # CSV 內已是工程單位

    def start(self):
        opts = self.options.get("replay", {})
        filename = opts.get("file", "")
        if not filename:
            raise RuntimeError("replay 模式需在 config.yaml 設定 datasource.replay.file")
        path = os.path.join(DATA_DIR, os.path.basename(filename))
        if not os.path.exists(path):
            raise RuntimeError(f"回放檔不存在：{path}")
        self._data = load_csv_channels(path)
        self._loop = bool(opts.get("loop", True))
        self._pos = 0
        self._count = 0
        self._t0 = time.monotonic()
        self._running = True

    def stop(self):
        self._running = False

    def read(self, n: int) -> np.ndarray:
        if not self._running:
            raise RuntimeError("replay 已停止")

        target = self._t0 + (self._count + n) / self.sample_rate
        delay = target - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._count += n

        total = self._data.shape[1]
        out = np.empty((6, n))
        filled = 0
        while filled < n:
            take = min(n - filled, total - self._pos)
            out[:, filled:filled + take] = self._data[:, self._pos:self._pos + take]
            filled += take
            self._pos += take
            if self._pos >= total:
                if not self._loop:
                    self._running = False
                    return out[:, :filled]
                self._pos = 0
        return out

    def describe(self):
        filename = self.options.get("replay", {}).get("file", "?")
        return f"Replay（{filename}）"
