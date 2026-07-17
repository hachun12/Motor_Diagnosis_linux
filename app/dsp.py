"""訊號處理：ISO 10816 分區判斷與 FFT 頻譜。"""
import numpy as np

ISO_LEVELS = [
    ("正常 (Zone A)", "normal"),
    ("注意 (Zone B)", "warning"),
    ("警告 (Zone C)", "alert"),
    ("危險 (Zone D)", "danger"),
]


def iso_zone(rms: float, zones) -> dict:
    """zones = (zone_a, zone_b, zone_c) 門檻，回傳顯示文字與等級。"""
    idx = 3
    for i, threshold in enumerate(zones):
        if rms < threshold:
            idx = i
            break
    text, level = ISO_LEVELS[idx]
    return {"text": text, "level": level, "rms": round(float(rms), 4)}


def rms(samples) -> float:
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr ** 2)))


def spectrum(samples: np.ndarray, fs: int, max_bins: int = 512):
    """單通道振幅頻譜（Hanning 窗），峰值保留降採樣到 max_bins。

    回傳 (freqs, mags)，皆為 list（方便 JSON 序列化）。
    """
    x = np.asarray(samples, dtype=float)
    n = x.size
    if n < 16:
        return [], []
    x = x - x.mean()
    win = np.hanning(n)
    mag = np.abs(np.fft.rfft(x * win)) * (2.0 / win.sum())
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    if mag.size > max_bins:
        # 每組取最大值，保留頻譜峰（診斷關注的特徵頻率）
        usable = (mag.size // max_bins) * max_bins
        group = mag.size // max_bins
        mag = mag[:usable].reshape(max_bins, group).max(axis=1)
        freqs = freqs[:usable:group]

    return [round(float(f), 2) for f in freqs], [float(f"{v:.5g}") for v in mag]
