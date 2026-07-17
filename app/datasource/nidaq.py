"""NI-DAQmx 資料源（僅 x86_64 Linux / Windows；NI 未提供 ARM64 Linux 驅動）。"""
import numpy as np

from .base import DataSource

try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, TerminalConfiguration
    NIDAQMX_AVAILABLE = True
    NIDAQMX_IMPORT_ERROR = None
except Exception as _e:  # ImportError 或底層驅動未安裝
    nidaqmx = None
    NIDAQMX_AVAILABLE = False
    NIDAQMX_IMPORT_ERROR = _e


class NIDAQSource(DataSource):
    calibrated = False  # 回傳原始電壓，由上層套用通道校正

    def start(self):
        if not NIDAQMX_AVAILABLE:
            raise RuntimeError(
                "未偵測到 NI-DAQmx 驅動或 nidaqmx 套件，無法連線硬體。"
                f"（{NIDAQMX_IMPORT_ERROR}）"
            )
        channels = self.options.get("nidaq", {}).get("channels", "Dev1/ai0:5")
        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_voltage_chan(
            channels, terminal_config=TerminalConfiguration.RSE)
        self.task.timing.cfg_samp_clk_timing(
            rate=self.sample_rate, sample_mode=AcquisitionType.CONTINUOUS)
        self.task.start()

    def stop(self):
        task = getattr(self, "task", None)
        if task is not None:
            self.task = None
            task.stop()
            task.close()

    def read(self, n: int) -> np.ndarray:
        task = getattr(self, "task", None)
        if task is None:
            raise RuntimeError("NI-DAQ 尚未啟動")
        data = task.read(number_of_samples_per_channel=n, timeout=5.0)
        return np.asarray(data, dtype=float)

    def describe(self):
        channels = self.options.get("nidaq", {}).get("channels", "Dev1/ai0:5")
        return f"NI-DAQmx（{channels}）"
