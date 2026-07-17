"""模擬資料源：3 相 60Hz 電流 + 3 軸振動，依牆鐘時間節流。"""
import time

import numpy as np

from .base import DataSource

TWO_PI = 2.0 * np.pi


class SimulatorSource(DataSource):
    calibrated = True

    def start(self):
        self._count = 0
        self._t0 = time.monotonic()
        self._rng = np.random.default_rng()
        self._running = True

    def stop(self):
        self._running = False

    def read(self, n: int) -> np.ndarray:
        if not self._running:
            raise RuntimeError("simulator 已停止")

        # 節流：等到牆鐘追上樣本數
        target = self._t0 + (self._count + n) / self.sample_rate
        delay = target - time.monotonic()
        if delay > 0:
            time.sleep(delay)

        t = (self._count + np.arange(n)) / self.sample_rate
        self._count += n

        noise = lambda s: self._rng.normal(0.0, s, n)

        # 3 相電流：60Hz、相位差 120°、含 5 次諧波與雜訊
        cur_phase = TWO_PI * 60.0 * t
        data = np.empty((6, n))
        for i, ph in enumerate((0.0, -TWO_PI / 3, TWO_PI / 3)):
            data[i] = (1.0 * np.sin(cur_phase + ph)
                       + 0.05 * np.sin(5 * (cur_phase + ph))
                       + noise(0.02))

        # 3 軸振動：25Hz 轉頻 + 120Hz 電磁分量 + 雜訊（RMS 約 0.05g → Zone A）
        vib_phase = TWO_PI * 25.0 * t
        for j, (amp, ph) in enumerate(((0.06, 0.0), (0.05, 1.0), (0.04, 2.0))):
            data[3 + j] = (amp * np.sin(vib_phase + ph)
                           + 0.015 * np.sin(TWO_PI * 120.0 * t + ph)
                           + noise(0.03))
        return data

    def describe(self):
        return "Simulator（模擬訊號）"
