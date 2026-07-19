"""訓練服務：用 saved_data/ 的標註記錄在部署機上訓練新模型。

流程：
1. dataset_overview()：掃描記錄檔，按標籤分組統計（供 UI 做標籤→類別對應）
2. start()：背景執行緒——切視窗、分層切分、CPU/GPU 訓練、
   每個 epoch 經 WebSocket 推播進度，完成後把最佳權重存入模型庫（不自動啟用）
"""
import os
import re
import threading
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from .config import Config, DATA_DIR
from .datasource.replay import load_csv_channels
from .inference import MotorFaultCNN

# 檔名格式：<標籤>_YYYYmmdd_HHMMSS.csv
_TS_SUFFIX = re.compile(r"_\d{8}_\d{6}\.csv$")

DEFAULT_PARAMS = {
    "epochs": 50,
    "lr": 1e-3,
    "batch_size": 64,
    "val_split": 0.2,
    "stride": 1024,
}
LIMITS = {"epochs": (1, 500), "lr": (1e-6, 1.0), "batch_size": (8, 512),
          "val_split": (0.05, 0.5), "stride": (256, 8192)}


def _label_of(filename: str):
    if not _TS_SUFFIX.search(filename):
        return None
    return _TS_SUFFIX.sub("", filename)


def _count_rows(path: str) -> int:
    """快速數資料列（扣掉表頭）。"""
    count = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            count += chunk.count(b"\n")
    return max(0, count - 1)


class TrainingService:
    def __init__(self, cfg: Config, registry, log_fn, publish):
        self.cfg = cfg
        self.registry = registry
        self._log = log_fn
        self._publish = publish          # hub.publish_threadsafe
        self._thread = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self.status = {"running": False}

    # ── 資料集彙整 ──────────────────────────────────────────
    def dataset_overview(self, window: int = None, stride: int = None):
        window = window or self.cfg.model_window
        stride = stride or DEFAULT_PARAMS["stride"]
        groups = {}
        for name in sorted(os.listdir(DATA_DIR)):
            if not name.endswith(".csv"):
                continue
            label = _label_of(name)
            if label is None:
                continue
            rows = _count_rows(os.path.join(DATA_DIR, name))
            g = groups.setdefault(label, {"label": label, "files": 0,
                                          "rows": 0, "windows": 0})
            g["files"] += 1
            g["rows"] += rows
            g["windows"] += max(0, (rows - window) // stride + 1) if rows >= window else 0
        for g in groups.values():
            g["seconds"] = round(g["rows"] / self.cfg.sample_rate, 1)
        return list(groups.values())

    # ── 訓練工作 ────────────────────────────────────────────
    def start(self, model_name: str, mapping: dict, params: dict):
        """mapping: {標籤: 類別代號}；params 見 DEFAULT_PARAMS。"""
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise ValueError("已有訓練工作進行中")
            mapping = {k: v.strip() for k, v in (mapping or {}).items() if v and v.strip()}
            if not mapping:
                raise ValueError("請至少對應一個標籤到類別")
            classes = sorted(set(mapping.values()))
            if len(classes) < 2:
                raise ValueError("至少需要 2 個不同類別才能訓練")

            p = dict(DEFAULT_PARAMS)
            for k, v in (params or {}).items():
                if k in p:
                    p[k] = type(p[k])(v)
            for k, (lo, hi) in LIMITS.items():
                if not (lo <= p[k] <= hi):
                    raise ValueError(f"參數 {k} 須在 {lo}–{hi} 之間")

            # 名稱先驗證可用（避免訓練完才發現重名）
            from .model_registry import sanitize_name, MODELS_DIR
            model_name = sanitize_name(model_name)
            if os.path.exists(os.path.join(MODELS_DIR, model_name)):
                raise ValueError(f"模型「{model_name}」已存在")

            self._cancel.clear()
            self.status = {"running": True, "name": model_name, "stage": "準備資料",
                           "epoch": 0, "epochs": p["epochs"], "history": [],
                           "started": datetime.now().strftime("%H:%M:%S")}
            self._thread = threading.Thread(
                target=self._run, args=(model_name, mapping, classes, p), daemon=True)
            self._thread.start()

    def cancel(self):
        if self.status.get("running"):
            self._cancel.set()

    def _emit(self, **kw):
        self.status.update(kw)
        self._publish({"type": "training", **self.status})

    # ── 主流程（背景執行緒）────────────────────────────────
    def _run(self, model_name, mapping, classes, p):
        try:
            self._log(f"🎓 開始訓練「{model_name}」：{len(mapping)} 個標籤 → {len(classes)} 類")
            X, y = self._build_dataset(mapping, classes, p["stride"])
            if self._cancel.is_set():
                raise InterruptedError
            counts = np.bincount(y, minlength=len(classes))
            if counts.min() < 10:
                weakest = classes[int(counts.argmin())]
                raise ValueError(f"類別「{weakest}」樣本數過少（{counts.min()}），"
                                 "請補收資料或調小 stride")
            self._train(model_name, X, y, classes, p)
        except InterruptedError:
            self._log(f"⏹ 訓練「{model_name}」已取消")
            self._emit(running=False, cancelled=True, stage="已取消")
        except Exception as e:
            self._log(f"❌ 訓練失敗: {e}")
            self._emit(running=False, error=str(e), stage="失敗")

    def _build_dataset(self, mapping, classes, stride):
        window = self.cfg.model_window
        cls_idx = {c: i for i, c in enumerate(classes)}
        xs, ys = [], []
        for name in sorted(os.listdir(DATA_DIR)):
            label = _label_of(name) if name.endswith(".csv") else None
            if label not in mapping:
                continue
            if self._cancel.is_set():
                raise InterruptedError
            data = load_csv_channels(os.path.join(DATA_DIR, name))  # (6, N)
            n = data.shape[1]
            for s in range(0, n - window + 1, stride):
                xs.append(data[:, s:s + window].T.astype(np.float32))  # (window, 6)
                ys.append(cls_idx[mapping[label]])
            self._emit(stage=f"準備資料（{len(xs)} 視窗）")
        if not xs:
            raise ValueError("沒有可用的訓練視窗（記錄長度是否足夠 2048 點？）")
        return np.stack(xs), np.array(ys, dtype=np.int64)

    def _train(self, model_name, X, y, classes, p):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rng = np.random.default_rng(42)

        # 分層切分：每類抽 val_split 當驗證集
        train_idx, val_idx = [], []
        for c in range(len(classes)):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            k = max(1, int(len(idx) * p["val_split"]))
            val_idx.extend(idx[:k])
            train_idx.extend(idx[k:])
        train_idx, val_idx = np.array(train_idx), np.array(val_idx)
        rng.shuffle(train_idx)

        Xt = torch.from_numpy(X)
        yt = torch.from_numpy(y)
        model = MotorFaultCNN(input_dim=6, num_classes=len(classes)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"])
        criterion = nn.CrossEntropyLoss()
        batch = p["batch_size"]

        self._log(f"  訓練樣本 {len(train_idx)}／驗證 {len(val_idx)}，裝置：{device}")
        best_acc, best_state = 0.0, None

        for epoch in range(1, p["epochs"] + 1):
            if self._cancel.is_set():
                raise InterruptedError
            model.train()
            perm = train_idx[np.random.permutation(len(train_idx))]
            total_loss = 0.0
            for i in range(0, len(perm), batch):
                b = perm[i:i + batch]
                xb, yb = Xt[b].to(device), yt[b].to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(b)
            train_loss = total_loss / len(perm)

            model.eval()
            correct = np.zeros(len(classes)); totals = np.zeros(len(classes))
            with torch.no_grad():
                for i in range(0, len(val_idx), batch):
                    b = val_idx[i:i + batch]
                    pred = model(Xt[b].to(device)).argmax(dim=1).cpu().numpy()
                    for t, pr in zip(y[b], pred):
                        totals[t] += 1
                        correct[t] += (t == pr)
            val_acc = float(correct.sum() / max(1, totals.sum()))

            if val_acc >= best_acc:
                best_acc = val_acc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                per_class = {classes[c]: round(float(correct[c] / totals[c]), 3)
                             for c in range(len(classes)) if totals[c] > 0}

            self.status["history"].append(
                {"epoch": epoch, "loss": round(train_loss, 5), "val_acc": round(val_acc, 4)})
            self._emit(stage="訓練中", epoch=epoch,
                       train_loss=round(train_loss, 5), val_acc=round(val_acc, 4),
                       best_acc=round(best_acc, 4))

        # 存入模型庫（最佳權重）
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            torch.save(best_state, tmp.name)
            tmp_path = tmp.name
        try:
            meta = {
                "classes": classes,
                "normal_classes": [c for c in classes if c in ("H", "N")],
                "window": self.cfg.model_window,
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": f"站上訓練（{len(train_idx)} 訓練/{len(val_idx)} 驗證視窗，"
                          f"{p['epochs']} epochs）",
                "val_accuracy": round(best_acc, 4),
                "per_class_accuracy": per_class,
                "params": p,
            }
            self.registry.save_model(model_name, tmp_path, meta)
        finally:
            os.unlink(tmp_path)

        self._log(f"✅ 訓練完成「{model_name}」：驗證準確率 {best_acc * 100:.1f}%"
                  "（已存入模型庫，請至模型管理啟用）")
        self._emit(running=False, done=True, stage="完成",
                   best_acc=round(best_acc, 4), per_class=per_class)
