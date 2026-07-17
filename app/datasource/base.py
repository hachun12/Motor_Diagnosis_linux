"""DataSource 驅動介面。

新增硬體支援時：
1. 在本目錄新增 <driver>.py，實作 DataSource 子類別
2. 在 __init__.py 的 DRIVERS 註冊
3. config.yaml 設 datasource.driver: <driver>
"""
from abc import ABC, abstractmethod

import numpy as np


class DataSource(ABC):
    """6 通道連續採樣來源。

    calibrated = False 時，read() 回傳原始電壓，
    由 AcquisitionService 依 config channels 套用 (raw + offset) * gain；
    calibrated = True 時直接回傳工程單位（simulator / replay）。
    """

    #: 是否已是工程單位（毋須再套通道校正）
    calibrated = False

    def __init__(self, sample_rate: int, num_channels: int = 6, options: dict = None):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.options = options or {}

    @abstractmethod
    def start(self):
        """初始化並開始採樣。失敗時 raise RuntimeError（附可讀訊息）。"""

    @abstractmethod
    def stop(self):
        """停止並釋放資源。可重複呼叫。"""

    @abstractmethod
    def read(self, n: int) -> np.ndarray:
        """阻塞讀取 n 筆樣本，回傳 shape=(num_channels, n) 的 float 陣列。"""

    def describe(self) -> str:
        return type(self).__name__
