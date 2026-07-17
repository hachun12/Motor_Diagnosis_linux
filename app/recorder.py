"""存檔與標籤管理。

修正桌面版的既有問題：時間戳原本每批（50 筆）只記一筆，導致 CSV 只存到
一小段資料且時間對不上；改為以啟動時間 + 樣本序號還原每筆樣本的時間戳，
完整輸出整段歷史。
"""
import csv
import json
import os
import re
import threading
from datetime import datetime, timedelta

from .config import Config, DATA_DIR, LABELS_FILE

_SAFE_LABEL = re.compile(r"[^\w一-鿿.-]+")


class Recorder:
    def __init__(self, cfg: Config, acquisition, log_fn):
        self.cfg = cfg
        self.acq = acquisition
        self._log = log_fn
        self._lock = threading.Lock()
        os.makedirs(DATA_DIR, exist_ok=True)
        self.labels = self._load_labels()

    # ── 標籤 ────────────────────────────────────────────────
    def _load_labels(self):
        try:
            with open(LABELS_FILE, "r", encoding="utf-8") as f:
                labels = json.load(f)
                if isinstance(labels, list):
                    return list(dict.fromkeys(labels))
        except Exception:
            pass
        return []

    def _persist_labels(self):
        with open(LABELS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.labels, f, ensure_ascii=False, indent=2)

    def add_label(self, name: str):
        with self._lock:
            if name not in self.labels:
                self.labels.append(name)
                self._persist_labels()

    # ── 存檔 ────────────────────────────────────────────────
    def save(self, label: str) -> dict:
        label = _SAFE_LABEL.sub("_", label.strip())
        if not label:
            raise ValueError("請輸入有效的標籤名稱")

        channels, start_time, first_index = self.acq.history_snapshot()
        keys = self.cfg.channel_keys
        n = len(channels[keys[0]])
        if n == 0 or start_time is None:
            raise ValueError("尚無資料可存，請先啟動系統")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{label}_{timestamp}.csv"
        filepath = os.path.join(DATA_DIR, filename)
        dt = 1.0 / self.cfg.sample_rate

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp",
                             "cur_A", "cur_B", "cur_C",
                             "vib_X", "vib_Y", "vib_Z",
                             "label"])
            cols = [channels[k] for k in keys]
            for i in range(n):
                ts = start_time + timedelta(seconds=(first_index + i) * dt)
                writer.writerow([ts.isoformat(timespec="milliseconds"),
                                 *(col[i] for col in cols), label])

        self.add_label(label)
        secs = n / self.cfg.sample_rate
        self._log(f"💾 已存檔：{filename}（{n} 點 / {secs:.1f} 秒）")
        return {"filename": filename, "points": n, "seconds": round(secs, 1)}

    # ── 記錄清單 ────────────────────────────────────────────
    def list_recordings(self):
        items = []
        for name in sorted(os.listdir(DATA_DIR), reverse=True):
            if not name.endswith(".csv"):
                continue
            path = os.path.join(DATA_DIR, name)
            st = os.stat(path)
            items.append({
                "name": name,
                "size": st.st_size,
                "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            })
        return items

    @staticmethod
    def resolve(name: str) -> str:
        """檔名安全解析：限制在 DATA_DIR 內的 .csv。"""
        base = os.path.basename(name)
        if not base.endswith(".csv"):
            raise ValueError("僅支援 .csv 檔")
        path = os.path.join(DATA_DIR, base)
        if not os.path.exists(path):
            raise FileNotFoundError(base)
        return path
