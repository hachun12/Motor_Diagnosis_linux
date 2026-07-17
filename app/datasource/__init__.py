"""驅動註冊表與工廠。"""
from .base import DataSource
from .simulator import SimulatorSource
from .nidaq import NIDAQSource
from .replay import ReplaySource

DRIVERS = {
    "simulator": SimulatorSource,
    "nidaq": NIDAQSource,
    "replay": ReplaySource,
}


def create_datasource(driver: str, sample_rate: int, options: dict) -> DataSource:
    if driver not in DRIVERS:
        raise ValueError(f"未知的 datasource driver: {driver}（可用：{', '.join(DRIVERS)}）")
    return DRIVERS[driver](sample_rate=sample_rate, num_channels=6, options=options)
