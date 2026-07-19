"""設定載入：config.yaml + 環境變數覆寫。"""
import os

import yaml

# 專案根目錄（app/ 的上一層）
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_FILE = os.environ.get("CONFIG_FILE", os.path.join(ROOT_DIR, "config.yaml"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT_DIR, "saved_data"))
LABELS_FILE = os.environ.get("LABELS_FILE", os.path.join(ROOT_DIR, "saved_labels.json"))
USERS_FILE = os.environ.get("USERS_FILE", os.path.join(ROOT_DIR, "users.yaml"))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(ROOT_DIR, "models"))
SECRET_FILE = os.environ.get("SECRET_FILE", os.path.join(ROOT_DIR, ".session_secret"))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class Config:
    """薄包裝：dict 內容 + 常用屬性。"""

    def __init__(self, raw: dict):
        self.raw = raw

        ds = raw.get("datasource", {})
        # 環境變數 DATA_SOURCE 優先於 config.yaml（方便 Docker 切換）
        self.driver = os.environ.get("DATA_SOURCE", ds.get("driver", "simulator"))
        self.sample_rate = int(ds.get("sample_rate", 5000))
        self.batch_size = int(ds.get("batch_size", 50))
        self.driver_options = {k: v for k, v in ds.items()
                               if k not in ("driver", "sample_rate", "batch_size")}

        self.channels = raw.get("channels", [])
        if len(self.channels) != 6:
            raise ValueError("config.yaml 的 channels 必須恰好定義 6 個通道")
        self.channel_keys = [c["key"] for c in self.channels]
        self.current_keys = [c["key"] for c in self.channels if c["group"] == "current"]
        self.vibration_keys = [c["key"] for c in self.channels if c["group"] == "vibration"]
        self.offsets = [float(c.get("offset", 0.0)) for c in self.channels]
        self.gains = [float(c.get("gain", 1.0)) for c in self.channels]

        self.history_seconds = int(raw.get("history_seconds", 60))
        self.history_len = self.sample_rate * self.history_seconds
        self.display_points = int(raw.get("display_points", 200))

        m = raw.get("model", {})
        self.model_path = os.path.join(ROOT_DIR, m.get("path", "best_motor_cnn_weights.pth"))
        self.model_window = int(m.get("window", 2048))
        self.model_interval = float(m.get("interval_seconds", 1.0))
        self.model_classes = list(m.get("classes", []))
        self.normal_classes = set(m.get("normal_classes", ["N"]))
        self.class_info = m.get("class_info", {}) or {}

        iso = raw.get("iso10816", {})
        self.iso_zones = (float(iso.get("zone_a", 0.28)),
                          float(iso.get("zone_b", 0.45)),
                          float(iso.get("zone_c", 0.71)))

        sp = raw.get("spectrum", {})
        self.spectrum_nfft = int(sp.get("nfft", 4096))
        self.spectrum_interval = float(sp.get("interval_seconds", 0.5))
        self.spectrum_max_bins = int(sp.get("max_bins", 512))

        au = raw.get("auth", {})
        self.session_hours = float(au.get("session_hours", 12))
        self.max_failed_attempts = int(au.get("max_failed_attempts", 5))
        self.lockout_seconds = float(au.get("lockout_seconds", 60))

        srv = raw.get("server", {})
        self.host = os.environ.get("HOST", srv.get("host", "0.0.0.0"))
        self.port = int(os.environ.get("PORT", srv.get("port", 8000)))

    def describe_class(self, pred: str) -> dict:
        """模型類別 → 顯示用說明（代號、名稱、故障程度）。未定義時回傳空欄位。

        同時支援以內部類別（N、RBS1…）或實驗代號（H、RBS-2…）查找——
        新訓練的模型會直接以實驗代號作為類別名稱。
        """
        info = self.class_info.get(pred)
        if not info:  # 以實驗代號反查
            info = next((v for v in self.class_info.values()
                         if v.get("code") == pred), None)
        if not info:
            return {"code": pred, "name": "", "detail": ""}
        if pred in self.normal_classes:
            detail = "無故障"
        else:
            detail = (f"轉子斷條 {info.get('broken_bars', '?')} 支｜"
                      f"繞組短路 {info.get('shorted', '?')} 處｜"
                      f"掛載砝碼 {info.get('weight', '?')}")
        return {"code": info.get("code", pred), "name": info.get("name", ""), "detail": detail}


def load_config(path: str = None) -> Config:
    path = path or CONFIG_FILE
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config(raw)
