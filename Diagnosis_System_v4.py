import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import threading
import time
import json
import os
import sys
import csv
from datetime import datetime
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import platform

# ── DAQ 驅動為可選匯入：未安裝 NI-DAQmx 時 GUI 仍可啟動，
#    只有實際按下「啟動」連線硬體時才會提示。
try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, TerminalConfiguration
    NIDAQMX_AVAILABLE = True
    NIDAQMX_IMPORT_ERROR = None
except Exception as _e:               # ImportError 或底層驅動未安裝
    nidaqmx = None
    NIDAQMX_AVAILABLE = False
    NIDAQMX_IMPORT_ERROR = _e

matplotlib.use("TkAgg")

# ── 跨平台等寬字型：Windows 用 Courier New，Linux/其他用 DejaVu Sans Mono ──
MONO_FONT = "Courier New" if platform.system() == "Windows" else "DejaVu Sans Mono"

# ─── 全域設定 ───────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

CURRENT_COLORS   = {"A": "#FF4C4C", "B": "#FF9900", "C": "#FFD700"}
VIBRATION_COLORS = {"X": "#00BFFF", "Y": "#00FF99", "Z": "#CC88FF"}


def _app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


APP_DIR = _app_dir()


def _resource_path(filename):
    for base in (APP_DIR, getattr(sys, "_MEIPASS", None), os.getcwd()):
        if not base:
            continue
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    return os.path.join(APP_DIR, filename)


SAVE_FILE        = os.path.join(APP_DIR, "saved_labels.json")
DATA_DIR         = os.path.join(APP_DIR, "saved_data")
MODEL_FILE       = _resource_path("best_motor_cnn_weights.pth")
DEFAULT_LABELS_FILE = _resource_path("saved_labels.json")

MPL_BG    = "#2b2b2b"
MPL_AX_BG = "#1e1e1e"
MPL_GRID  = "#444444"
MPL_SPINE = "#666666"
MPL_TICK  = "#aaaaaa"
MPL_LABEL = "#999999"

SAMPLE_RATE     = 5000
HISTORY_SECONDS = 60
HISTORY_LEN     = SAMPLE_RATE * HISTORY_SECONDS  # 60000
BATCH           = 50    # 每幀採樣點數：1000Hz / 20fps = 50
DISPLAY_POINTS  = 200   # 波形圖顯示點數

STATUS_COLORS = {
    "normal":  "#00FF88",
    "warning": "#FFD700",
    "danger":  "#FF4C4C",
}

# ==================================================================
# Data Simulator ===================================================
class DAQHardware:
    def __init__(self, channels="Dev1/ai0:5", sample_rate=SAMPLE_RATE):
        self.channels = channels
        self.sample_rate = sample_rate
        self.task = None
        self.running = False

    def start(self):
        """初始化並啟動 DAQ 硬體"""
        if not NIDAQMX_AVAILABLE:
            raise RuntimeError(
                "未偵測到 NI-DAQmx 驅動或 nidaqmx 套件，無法連線硬體。"
                f"（{NIDAQMX_IMPORT_ERROR}）"
            )
        self.task = nidaqmx.Task()
        # 一次加入 6 個通道 (ai0 到 ai5)
        self.task.ai_channels.add_ai_voltage_chan(self.channels,
            terminal_config=TerminalConfiguration.RSE)
        self.task.timing.cfg_samp_clk_timing(
            rate=self.sample_rate,
            sample_mode=AcquisitionType.CONTINUOUS
        )
        self.task.start()
        self.running = True

    def stop(self):
        """停止並釋放 DAQ 資源"""
        self.running = False
        if self.task:
            self.task.stop()
            self.task.close()
            self.task = None

    def get_batch(self, n):
        """從 DAQ 讀取 n 筆資料，並對應到 UI 需要的格式"""
        if not self.task:
            # 防呆機制：若硬體未啟動，回傳空陣列
            return {"A":[], "B":[], "C":[]}, {"X":[], "Y":[], "Z":[]}
        
        # data 會是一個包含 6 個 list 的二維陣列
        data = self.task.read(number_of_samples_per_channel=n, timeout=5.0)
    
        
        
        # 0:2 為電流 (A, B, C)
        cur = {
            "A": [(v + 0.4 ) * 10 for v in data[0]],
            "B": [(v + 0.69) * 10 for v in data[1]], 
            "C": [(v + 0.1 ) * 10 for v in data[2]],
        }
        # 3:5 為震動 (X, Y, Z)
        vib = {
            "X": [(v - 1.2 ) for v in data[3]],
            "Y": [(v - 1.26) for v in data[4]],
            "Z": [(v - 1.53) for v in data[5]],
        }
        return cur, vib

class MotorFaultCNN(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(MotorFaultCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) 
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = x.transpose(1, 2)  
        x = self.features(x)
        x = x.squeeze(-1) 
        out = self.classifier(x)
        return out
# ==================================================================
# AI Model Wrapper =================================================
class MotorDiagnosisModel:
    def __init__(self, model_path=MODEL_FILE):
        self.model_path = model_path
        
        # ⚠️ 請確保這裡的類別順序，與你訓練時資料夾 (01, 02, 03...) 的排序完全一致！
        self.classes = ["N", "RB", "RB2", "RBS", "RBS1", "RBS2", "RU", "RUB", "RUB2", "RUBS", "RUBS1", "RUBS2"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 實例化模型：輸入 6 個特徵 (Ia,Ib,Ic,X,Y,Z)，輸出 4 個類別
        self.model = MotorFaultCNN(input_dim=6, num_classes=len(self.classes))
        
        # 2. 載入權重並塞入模型
        weights = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(weights)
        
        self.model.to(self.device)
        self.model.eval()
        print(f"AI 模型載入完成: {model_path} (運行於 {self.device})")

    def predict(self, current_window, vibration_window):
        REQUIRED_LEN = 2048
        try:
            # 1. 檢查資料長度
            if len(current_window[0]) < REQUIRED_LEN:
                return "資料收集當中...", 0.0
                
            # 2. 擷取最新的 2048 點
            cur_sliced = [ch[-REQUIRED_LEN:] for ch in current_window]
            vib_sliced = [ch[-REQUIRED_LEN:] for ch in vibration_window]

            # 3. 合併資料
            # vstack 後形狀為 (6, 2048)
            combined_data = np.vstack((cur_sliced, vib_sliced))
            
            # ⚠️ 關鍵修正：轉置矩陣，讓形狀變成 (2048, 6)，以符合你訓練時的 Dataset 輸出！
            combined_data = combined_data.T 
            
            # 轉換為 Tensor 並增加 Batch 維度 -> 形狀變成 (1, 2048, 6)
            tensor_data = torch.tensor(combined_data, dtype=torch.float32).unsqueeze(0)
            tensor_data = tensor_data.to(self.device)
            
            # 4. 模型推論
            with torch.no_grad():
                # 若你的環境支援 AMP，也可以加上 autocast 加速推論
                with torch.amp.autocast('cuda' if 'cuda' in str(self.device) else 'cpu'):
                    outputs = self.model(tensor_data)
                    
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence_tensor, pred_idx_tensor = torch.max(probabilities, dim=1)
                
                pred_idx = pred_idx_tensor.item()
                confidence = confidence_tensor.item()
                
            return self.classes[pred_idx], confidence
            
        except Exception as e:
            print(f"AI 推論發生錯誤: {e}")
            return "診斷失敗", 0.0

# ==================================================================
# Main App =========================================================
class MotorDiagnosticApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("馬達即時診斷系統")
        self.geometry("1280x780")
        self.minsize(1100, 680)

        self.daq = DAQHardware(channels="Dev1/ai0:5", sample_rate=SAMPLE_RATE)
        self.is_running = False
        self._alive     = True

        self.ai_model = MotorDiagnosisModel()

        # ── 顯示用緩衝（用 deque 保持固定長度，避免 slice 問題）
        self.BUFFER    = DISPLAY_POINTS
        self.time_axis = list(range(self.BUFFER))
        self.current_data   = {k: deque([0.0]*self.BUFFER, maxlen=self.BUFFER)
                                for k in ["A","B","C"]}
        self.vibration_data = {k: deque([0.0]*self.BUFFER, maxlen=self.BUFFER)
                                for k in ["X","Y","Z"]}

        # ── 60秒歷史緩衝
        self.history_cur = {k: deque(maxlen=HISTORY_LEN) for k in ["A","B","C"]}
        self.history_vib = {k: deque(maxlen=HISTORY_LEN) for k in ["X","Y","Z"]}
        self.history_ts  = deque(maxlen=HISTORY_LEN)

        # ── 執行緒間資料交換用的 lock
        self._data_lock = threading.Lock()

        # ── 勾選狀態
        self.current_vars   = {k: ctk.BooleanVar(value=True) for k in ["A","B","C"]}
        self.vibration_vars = {k: ctk.BooleanVar(value=True) for k in ["X","Y","Z"]}

        self.saved_labels = self._load_labels()

        self.current_status_text  = "待機"
        self.current_status_color = "#888888"
        self._iso_counter = 0

        os.makedirs(DATA_DIR, exist_ok=True)

        self._build_ui()
        self._init_plots()
        self._log("系統初始化完成，等待啟動...")
        if not NIDAQMX_AVAILABLE:
            self._log("⚠️ 未偵測到 NI-DAQmx 驅動，GUI 為預覽模式，按「啟動」將無法連線硬體。")

    # ══════════════════════════════════════════════════════════════
    #  UI 建構
    # ══════════════════════════════════════════════════════════════
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.left_frame = ctk.CTkFrame(self, corner_radius=12)
        self.left_frame.grid(row=0, column=0, padx=(14,6), pady=14, sticky="nsew")
        self.left_frame.grid_columnconfigure(0, weight=1)
        self.left_frame.grid_rowconfigure(1, weight=1)
        self.left_frame.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(
            self.left_frame, text="📊  監控畫面",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, padx=16, pady=(12,4), sticky="w")

        self.current_block = self._build_waveform_block(
            parent=self.left_frame, title="⚡  電流 (A / B / C)",
            axes_keys=["A","B","C"], color_map=CURRENT_COLORS,
            var_map=self.current_vars, row=1,
        )

        ctk.CTkLabel(
            self.left_frame, text="📳  震動 (X / Y / Z)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("gray50","gray70"),
        ).grid(row=2, column=0, padx=16, pady=(8,2), sticky="w")

        self.vibration_block = self._build_waveform_block(
            parent=self.left_frame, title="",
            axes_keys=["X","Y","Z"], color_map=VIBRATION_COLORS,
            var_map=self.vibration_vars, row=3,
        )

        self.right_frame = ctk.CTkFrame(self, corner_radius=12)
        self.right_frame.grid(row=0, column=1, padx=(6,14), pady=14, sticky="nsew")
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(3, weight=1)

        self._build_control_panel()
        self._build_motor_status()
        self._build_save_panel()
        self._build_status_panel()

    def _build_waveform_block(self, parent, title, axes_keys, color_map, var_map, row):
        container = ctk.CTkFrame(parent, corner_radius=10, fg_color=("gray90","gray17"))
        container.grid(row=row, column=0, padx=12, pady=(0,6), sticky="nsew")
        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=0)
        container.grid_rowconfigure(1, weight=1)

        if title:
            ctk.CTkLabel(
                container, text=title,
                font=ctk.CTkFont(size=13, weight="bold"),
            ).grid(row=0, column=0, columnspan=2, padx=10, pady=(8,2), sticky="w")

        plot_frame = ctk.CTkFrame(container, fg_color="transparent")
        plot_frame.grid(row=1, column=0, padx=(8,0), pady=(0,8), sticky="nsew")

        check_outer = ctk.CTkFrame(container, fg_color="transparent")
        check_outer.grid(row=1, column=1, padx=(4,10), pady=(0,8), sticky="nsew")
        check_outer.grid_rowconfigure(0, weight=1)
        check_outer.grid_rowconfigure(1, weight=0)
        check_outer.grid_rowconfigure(2, weight=1)
        check_outer.grid_columnconfigure(0, weight=1)

        check_frame = ctk.CTkFrame(check_outer, fg_color="transparent")
        check_frame.grid(row=1, column=0)

        for key in axes_keys:
            cb = ctk.CTkCheckBox(
                check_frame, text=key, variable=var_map[key],
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=color_map[key], fg_color=color_map[key],
                hover_color=color_map[key], checkmark_color="white",
                width=60, command=self._refresh_lines,
            )
            cb.pack(pady=5, anchor="w")

        return {"container": container, "plot_frame": plot_frame}

    def _build_control_panel(self):
        panel = ctk.CTkFrame(self.right_frame, corner_radius=10, fg_color=("gray88","gray20"))
        panel.grid(row=0, column=0, padx=12, pady=(14,6), sticky="ew")
        panel.grid_columnconfigure((0,1), weight=1)

        ctk.CTkLabel(
            panel, text="🎛️  控制面板",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, padx=12, pady=(10,6), sticky="w")

        self.btn_on = ctk.CTkButton(
            panel, text="▶  啟動",
            fg_color="#1a7a3c", hover_color="#145e2e",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38, command=self._start_system,
        )
        self.btn_on.grid(row=1, column=0, padx=(10,4), pady=(0,12), sticky="ew")

        self.btn_off = ctk.CTkButton(
            panel, text="⏹  停止",
            fg_color="#8b1a1a", hover_color="#6b1212",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38, state="disabled", command=self._stop_system,
        )
        self.btn_off.grid(row=1, column=1, padx=(4,10), pady=(0,12), sticky="ew")

    def _build_motor_status(self):
        panel = ctk.CTkFrame(self.right_frame, corner_radius=10, fg_color=("gray88","gray20"))
        panel.grid(row=1, column=0, padx=12, pady=6, sticky="ew")
        panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            panel, text="🔍  馬達狀態",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, padx=12, pady=(10,6), sticky="w")

        status_box = ctk.CTkFrame(panel, corner_radius=8, fg_color=("#1a1a1a","#111111"))
        status_box.grid(row=1, column=0, padx=12, pady=(0,12), sticky="ew")
        status_box.grid_columnconfigure(2, weight=1)

        self.status_dot = ctk.CTkLabel(
            status_box, text="●",
            font=ctk.CTkFont(size=22),
            text_color="#888888",
        )
        self.status_dot.grid(row=0, column=0, padx=(14,4), pady=10, sticky="w")

        self.status_label = ctk.CTkLabel(
            status_box, text="待機",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#888888",
        )
        self.status_label.grid(row=0, column=1, padx=(0,14), pady=10, sticky="w")

        self.rms_label = ctk.CTkLabel(
            status_box, text="RMS: ---",
            font=ctk.CTkFont(size=10),
            text_color="#666666",
        )
        self.rms_label.grid(row=0, column=2, padx=(0,14), pady=10, sticky="e")

        ai_box = ctk.CTkFrame(panel, corner_radius=8, fg_color=("#1a1a1a","#111111"))
        ai_box.grid(row=2, column=0, padx=12, pady=(0,12), sticky="ew")
        ai_box.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            ai_box, text="模型結果",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#AAAAAA",
        ).grid(row=0, column=0, padx=(14,10), pady=10, sticky="w")

        self.ai_result_label = ctk.CTkLabel(
            ai_box, text="等待資料...",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#00BFFF",
        )
        self.ai_result_label.grid(row=0, column=1, padx=(0,14), pady=10, sticky="w")

    def _build_save_panel(self):
        panel = ctk.CTkFrame(self.right_frame, corner_radius=10, fg_color=("gray88","gray20"))
        panel.grid(row=2, column=0, padx=12, pady=6, sticky="ew")
        panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            panel, text="💾  存檔",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, padx=12, pady=(10,4), sticky="w")

        input_row = ctk.CTkFrame(panel, fg_color="transparent")
        input_row.grid(row=1, column=0, padx=10, pady=(0,6), sticky="ew")
        input_row.grid_columnconfigure(0, weight=1)

        self.label_entry = ctk.CTkEntry(
            input_row, placeholder_text="輸入標籤名稱...",
            font=ctk.CTkFont(size=12), height=34,
        )
        self.label_entry.grid(row=0, column=0, padx=(0,6), sticky="ew")

        self.btn_save = ctk.CTkButton(
            input_row, text="存檔", width=60, height=34,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._save_label,
        )
        self.btn_save.grid(row=0, column=1)

        # ── 下拉選單：有記錄才顯示標籤，否則顯示提示 ────────────
        unique_labels = self._get_unique_labels()

        if unique_labels:
            dropdown_values = ["-- 選擇已有標籤 --"] + unique_labels
            dropdown_state  = "normal"
        else:
            dropdown_values = ["（尚無記錄）"]
            dropdown_state  = "disabled"   # 沒有標籤時禁用，避免誤選

        self.label_dropdown = ctk.CTkOptionMenu(
            panel,
            values=dropdown_values,
            font=ctk.CTkFont(size=12), height=34,
            state=dropdown_state,
            command=self._on_label_selected,
        )
        self.label_dropdown.set(dropdown_values[0])
        self.label_dropdown.grid(row=2, column=0, padx=10, pady=(0,10), sticky="ew")

    def _build_status_panel(self):
        panel = ctk.CTkFrame(self.right_frame, corner_radius=10, fg_color=("gray88","gray20"))
        panel.grid(row=3, column=0, padx=12, pady=(6,14), sticky="nsew")
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(panel, fg_color="transparent")
        header.grid(row=0, column=0, padx=10, pady=(10,4), sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header, text="🖥️  系統狀態",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkButton(
            header, text="清除", width=50, height=26,
            font=ctk.CTkFont(size=11),
            fg_color="#555555", hover_color="#444444",
            command=self._clear_log,
        ).grid(row=0, column=1, sticky="e")

        self.terminal = ctk.CTkTextbox(
            panel,
            font=ctk.CTkFont(family=MONO_FONT, size=11),
            fg_color=("gray95","#0d0d0d"),
            text_color=("black","#00FF88"),
            wrap="word", state="disabled",
        )
        self.terminal.grid(row=1, column=0, padx=10, pady=(0,10), sticky="nsew")

    #  Matplotlib 初始化
    def _init_plots(self):
        self._setup_axes(
            fig_attr="fig_cur", ax_attr="ax_cur",
            lines_attr="cur_lines", canvas_attr="canvas_cur",
            plot_frame=self.current_block["plot_frame"],
            color_map=CURRENT_COLORS, ylim=(-2, 2), ylabel="A",
        )
        self._setup_axes(
            fig_attr="fig_vib", ax_attr="ax_vib",
            lines_attr="vib_lines", canvas_attr="canvas_vib",
            plot_frame=self.vibration_block["plot_frame"],
            color_map=VIBRATION_COLORS, ylim=(-0.3, 0.3), ylabel="g",
        )

    def _setup_axes(self, fig_attr, ax_attr, lines_attr,
                    canvas_attr, plot_frame, color_map, ylim, ylabel):
        fig, ax = plt.subplots(figsize=(5, 2.2), facecolor=MPL_BG)
        ax.set_facecolor(MPL_AX_BG)
        ax.tick_params(colors=MPL_TICK, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(MPL_SPINE)
        ax.set_ylim(*ylim)
        ax.set_xlim(0, self.BUFFER)
        ax.grid(True, color=MPL_GRID, linewidth=0.4)
        ax.set_ylabel(ylabel, color=MPL_LABEL, fontsize=8)

        lines = {}
        for key, color in color_map.items():
            (line,) = ax.plot([], [], color=color, linewidth=1.2, label=key)
            lines[key] = line

        fig.tight_layout(pad=0.5)
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        setattr(self, fig_attr, fig)
        setattr(self, ax_attr, ax)
        setattr(self, lines_attr, lines)
        setattr(self, canvas_attr, canvas)

    #  採樣執行緒：只負責採資料，不碰任何 UI / matplotlib
    def _sampling_thread(self):
        while self.is_running:
            try:
                # 這裡會自動阻塞，直到 DAQ 準備好 BATCH 數量的資料
                cur, vib = self.daq.get_batch(BATCH)
                ts_now   = datetime.now().isoformat(timespec="milliseconds")

                with self._data_lock:
                    for k in ["A","B","C"]:
                        self.current_data[k].extend(cur[k])
                        self.history_cur[k].extend(cur[k])
                    for k in ["X","Y","Z"]:
                        self.vibration_data[k].extend(vib[k])
                        self.history_vib[k].extend(vib[k])
                    self.history_ts.append(ts_now)
                    
            except Exception as e:
                if self.is_running:
                    print(f"背景讀取錯誤: {e}")
                break

    def _ai_inference_thread(self):
        """AI 專屬背景執行緒：定期抓取 Window 資料進行推論"""
        window_size = SAMPLE_RATE * 1  # 假設模型需要 1 秒的資料 (5000點)
        
        while self.is_running:
            time.sleep(1.0)  # 每 1 秒執行一次診斷
            
            # 檢查資料量是否足夠
            if len(self.history_cur["A"]) < window_size:
                continue

            # 1. 安全地從歷史緩衝區擷取最新的 window_size 資料
            with self._data_lock:
                cur_window = [list(self.history_cur[k])[-window_size:] for k in ["A", "B", "C"]]
                vib_window = [list(self.history_vib[k])[-window_size:] for k in ["X", "Y", "Z"]]

            # 2. 呼叫模型進行推論
            try:
                pred_class, conf = self.ai_model.predict(cur_window, vib_window)
                
                # 3. 透過主執行緒安全更新 UI
                self.after(0, self._update_ai_ui, pred_class, conf)
            except Exception as e:
                print(f"AI 推論發生錯誤: {e}")

    def _update_ai_ui(self, pred_class, conf):
        """在主執行緒更新 AI 診斷結果"""
        display_text = f"{pred_class} ({conf*100:.1f}%)"
        
        # 根據結果改變顏色
        if "正常" in pred_class:
            color = STATUS_COLORS["normal"]
        else:
            color = STATUS_COLORS["danger"]
            self._log(f"⚠️ [AI 警報] 偵測到異常: {display_text}")
            
        self.ai_result_label.configure(text=display_text, text_color=color)

    #  UI 刷新：由主執行緒的 after 驅動，安全更新 matplotlib
    def _ui_refresh(self):
        """每 50ms 由主執行緒呼叫一次，更新波形圖與狀態"""
        if not self._alive or not self.is_running:
            return

        # 複製顯示資料（短暫持鎖）
        with self._data_lock:
            cur_snap = {k: list(self.current_data[k])   for k in ["A","B","C"]}
            vib_snap = {k: list(self.vibration_data[k]) for k in ["X","Y","Z"]}

        # ── 更新電流折線
        for k, line in self.cur_lines.items():
            if self.current_vars[k].get() and len(cur_snap[k]) == self.BUFFER:
                line.set_data(self.time_axis, cur_snap[k])
                line.set_visible(True)
            else:
                line.set_visible(False)

        # ── 更新震動折線
        for k, line in self.vib_lines.items():
            if self.vibration_vars[k].get() and len(vib_snap[k]) == self.BUFFER:
                line.set_data(self.time_axis, vib_snap[k])
                line.set_visible(True)
            else:
                line.set_visible(False)

        # ── 重繪（在主執行緒，安全）
        self.canvas_cur.draw_idle()
        self.canvas_vib.draw_idle()

        # ── ISO 判斷
        if vib_snap["X"]:
            rms_x = float(np.sqrt(np.mean(np.array(vib_snap["X"]) ** 2)))
            self._iso_check(rms_x)

        # ── 排程下一幀
        if self._alive and self.is_running:
            self.after(50, self._ui_refresh)

    #  ISO 10816 判斷
    def _iso_check(self, rms_val):
        if rms_val < 0.28:
            text, color = "正常  (Zone A)", STATUS_COLORS["normal"]
        elif rms_val < 0.45:
            text, color = "注意  (Zone B)", STATUS_COLORS["warning"]
        elif rms_val < 0.71:
            text, color = "警告  (Zone C)", "#FF9900"
        else:
            text, color = "危險  (Zone D)", STATUS_COLORS["danger"]

        self.status_dot.configure(text_color=color)
        self.status_label.configure(text=text, text_color=color)
        self.rms_label.configure(text=f"RMS: {rms_val:.3f}")
        self.current_status_text  = text
        self.current_status_color = color

        self._iso_counter += 1
        if self._iso_counter % 20 == 0:   # 約每秒一次
            self._log(f"[ISO] RMS={rms_val:.3f}  →  {text}")

    # ══════════════════════════════════════════════════════════════
    #  控制邏輯
    # ══════════════════════════════════════════════════════════════
    def _start_system(self):
        if self.is_running:
            return
        
        try:
            self.daq.start() # 啟動硬體
        except Exception as e:
            self._log(f"❌ DAQ 啟動失敗: {e}")
            return

        self.is_running = True
        # self.simulator.running = True
        self.btn_on.configure(state="disabled")
        self.btn_off.configure(state="normal")
        self._log("▶ 系統啟動，開始擷取資料...")
        self._log(f"  時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 啟動採樣執行緒
        self.worker = threading.Thread(target=self._sampling_thread, daemon=True)
        self.worker.start()

        self.ai_worker = threading.Thread(target=self._ai_inference_thread, daemon=True)
        self.ai_worker.start()

        # 啟動 UI 刷新循環（主執行緒）
        self.after(50, self._ui_refresh)

    def _stop_system(self):
        self.is_running = False
        self.daq.stop() # 停止硬體

        # self.simulator.running = False
        self.btn_on.configure(state="normal")
        self.btn_off.configure(state="disabled")
        self.status_dot.configure(text_color="#888888")
        self.status_label.configure(text="待機", text_color="#888888")
        self.rms_label.configure(text="RMS: ---")
        self.ai_result_label.configure(text="等待資料...", text_color="#AAAAAA")

        self._log("⏹ 系統已停止。")

    def _refresh_lines(self):
        pass


    # 存檔邏輯
    def _save_label(self):
        name = self.label_entry.get().strip()
        if not name:
            self._log("⚠️  請輸入標籤名稱後再存檔。")
            return
        if not self.history_ts:
            self._log("⚠️  尚無資料可存，請先啟動系統。")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"{name}_{timestamp}.csv"
        filepath  = os.path.join(DATA_DIR, filename)

        # 複製歷史資料（持鎖）
        with self._data_lock:
            h_ts  = list(self.history_ts)
            h_cur = {k: list(self.history_cur[k]) for k in ["A","B","C"]}
            h_vib = {k: list(self.history_vib[k]) for k in ["X","Y","Z"]}

        try:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp",
                                  "cur_A","cur_B","cur_C",
                                  "vib_X","vib_Y","vib_Z",
                                  "label"])
                n = min(len(h_ts), len(h_cur["A"]))
                for i in range(n):
                    writer.writerow([
                        h_ts[i] if i < len(h_ts) else "",
                        h_cur["A"][i], h_cur["B"][i], h_cur["C"][i],
                        h_vib["X"][i], h_vib["Y"][i], h_vib["Z"][i],
                        name,
                    ])
            secs = len(h_cur["A"]) / SAMPLE_RATE
            self._log(f"💾 已存檔：{filename}  ({len(h_cur['A'])} 點 / {secs:.1f} 秒)")
        except Exception as e:
            self._log(f"❌ 存檔失敗：{e}")
            return

        # ── 更新下拉選單 ──────────────────────────────────────────
        if name not in self.saved_labels:
            self.saved_labels.append(name)
            self._persist_labels()

        # 無論新舊標籤，都重新整理一次下拉內容與狀態
        self.label_dropdown.configure(
            values=["-- 選擇已有標籤 --"] + self.saved_labels,
            state="normal",   # 確保從「尚無記錄」disabled 狀態恢復
        )
        self.label_dropdown.set("-- 選擇已有標籤 --")
        self.label_entry.delete(0, "end")

    def _on_label_selected(self, value):
        if value == "-- 選擇已有標籤 --":
            return
        # ✅ 清空輸入框，填入選取的標籤名稱
        self.label_entry.delete(0, "end")
        self.label_entry.insert(0, value)
        self._log(f"📂 已選取標籤：{value}，可直接按存檔")

    def _get_unique_labels(self):
        return list(dict.fromkeys(self.saved_labels))

    def _load_labels(self):
        for labels_file in (SAVE_FILE, DEFAULT_LABELS_FILE):
            if not labels_file or not os.path.exists(labels_file):
                continue
            try:
                with open(labels_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _persist_labels(self):
        with open(SAVE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.saved_labels, f, ensure_ascii=False, indent=2)

    #  Terminal 日誌
    def _log(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.terminal.configure(state="normal")
        self.terminal.insert("end", f"[{ts}]  {message}\n")
        self.terminal.see("end")
        self.terminal.configure(state="disabled")

    def _clear_log(self):
        self.terminal.configure(state="normal")
        self.terminal.delete("1.0", "end")
        self.terminal.configure(state="disabled")

    # ══════════════════════════════════════════════════════════════
    #  關閉
    # ══════════════════════════════════════════════════════════════
    def on_closing(self):
        self._alive     = False
        self.is_running = False
        
        if hasattr(self, 'daq'):
            self.daq.stop()

        if hasattr(self, "worker") and self.worker.is_alive():
            self.worker.join(timeout=0.5)

        plt.close("all")
        self.quit()
        self.after(100, self.destroy)


# ─── 入口 ────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = MotorDiagnosticApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
