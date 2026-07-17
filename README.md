# Motor Diagnosis System

馬達即時診斷系統：透過 DAQ 擷取 3 相電流與 3 軸振動訊號（6 通道、5 kHz），
以 PyTorch 1D-CNN 進行故障分類，並依 ISO 10816 振動 RMS 分區判斷。

提供兩種介面：

| 版本 | 進入點 | 適用情境 |
|---|---|---|
| **WebUI（建議）** | `app/`（FastAPI + WebSocket） | Docker 部署、ARM64/x64 皆可、多人瀏覽、含登入權限、FFT 頻譜、歷史回放 |
| 桌面版 | `Diagnosis_System_v4.py`（CustomTkinter） | 單機 GUI，原有流程 |

---

## WebUI 版

### 功能

- 即時波形（電流 A/B/C、震動 X/Y/Z，20 fps）與通道勾選
- FFT 頻譜（Hanning 窗、峰值保留降採樣、線性/dB 切換）
- ISO 10816 RMS 分區燈號 + AI 模型即時診斷（每秒一次）
- 標籤存檔：60 秒歷史 → CSV（每筆樣本含完整時間戳）
- 歷史記錄：清單、下載、**回放**（時間軸拉桿瀏覽波形與該視窗頻譜）
- 登入權限：`admin`（控制/存檔/使用者管理）與 `viewer`（僅觀看）
- 系統日誌即時推播

### 架構

```
瀏覽器 ── HTTP(REST) + WebSocket ──▶ FastAPI (uvicorn)
                                      ├─ datasource 驅動插件層（simulator│nidaq│replay│…）
                                      ├─ 採樣執行緒（歷史/顯示緩衝）
                                      ├─ AI 推論執行緒（PyTorch CNN）
                                      └─ FFT / ISO / CSV 存檔
```

**驅動插件層**（`app/datasource/`）：換 DAQ 硬體時新增一個 driver 檔並在
`__init__.py` 註冊，再改 `config.yaml` 的 `datasource.driver` 與 `channels`
校正值即可，核心程式零修改。內建：

- `simulator` — 模擬訊號（60 Hz 三相電流、25 Hz 振動），開發/展示用
- `nidaq` — NI-DAQmx（**僅 x86_64**；NI 未提供 ARM64 Linux 驅動）
- `replay` — 以 `saved_data/` 的 CSV 當即時訊號重播

### Docker 部署（建議）

```bash
# 建置 + 啟動（首次建置需下載 torch，較久）
ADMIN_PASSWORD=你的管理員密碼 docker compose up -d --build

# 瀏覽 http://<主機IP>:8000 ，帳號 admin
```

- 未設 `ADMIN_PASSWORD` 時會產生隨機密碼並寫入容器日誌（`docker logs motor-diagnosis`）。
- 資料持久化：`./saved_data`（記錄 CSV）、`./state`（使用者、標籤、session 金鑰）。
- 切換資料源：`DATA_SOURCE=simulator|nidaq|replay`（環境變數，優先於 config.yaml）。

#### Multi-arch 建置（x64 開發機 → ARM64 目標機）

```bash
# 一次性：安裝 QEMU binfmt 與 builder
docker run --privileged --rm tonistiigi/binfmt --install arm64
docker buildx create --name multiarch --use

# 建置 ARM64 映像並匯出部署
docker buildx build --platform linux/arm64 -t motor-diagnosis:arm64 --load .
docker save motor-diagnosis:arm64 | gzip > motor-diagnosis-arm64.tar.gz
# 目標機：docker load < motor-diagnosis-arm64.tar.gz && docker compose up -d
```

（有私有 registry 時可改用 `--platform linux/arm64,linux/amd64 --push` 一次出雙架構。）

### 本機開發（不經 Docker）

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-web.txt
ADMIN_PASSWORD=dev123 python -m app.main     # http://localhost:8000
```

### 設定：config.yaml

- `datasource.driver` / `sample_rate` / `batch_size`
- `channels`：6 通道的 `offset`/`gain` 校正（`value = (raw + offset) * gain`，
  僅套用於輸出原始電壓的驅動如 `nidaq`）
- `model`：權重路徑、推論視窗、類別清單
- `iso10816`：RMS 分區門檻；`spectrum`：FFT 參數
- `auth`：session 時效、登入失敗鎖定

### DAQ 硬體與平台支援

| 平台 | 可用資料源 |
|---|---|
| x86_64（裝 NI-DAQmx 驅動） | nidaq、simulator、replay |
| ARM64（Ubuntu 24.04 等） | simulator、replay、（未來：LabJack/MCC 等 ARM 相容硬體 driver） |

> **注意**：NI-DAQmx for Linux 僅支援 x86_64，ARM64 機器無法使用 NI 硬體。
> ARM64 選型門檻：6 通道 AI、每通道 ≥5 kHz、有 aarch64 Linux 函式庫與 Python API
>（例：LabJack T7/T8、MCC DAQ HAT）。選定後新增對應 driver 即可。
> 另外：模型是以 NI 訊號鏈訓練的，更換感測前端後建議重新收資料驗證/微調模型。

> `nidaq` 模式若要在容器內使用，容器內也需 NI-DAQmx 執行環境，建置複雜；
> x86_64 + NI 硬體的場景建議以「本機開發」方式直接跑在主機上。

---

## 桌面版（原 CustomTkinter 程式）

### 環境需求

- Python 3.11、`requirements.txt`
- DAQ 驅動：Windows 裝 NI-DAQmx；Ubuntu 裝 NI-DAQmx for Linux（僅特定
  LTS/核心版本，請查 NI 支援矩陣）。裝置名稱預設 `Dev1`（`Dev1/ai0:5`）。
- 未裝 DAQ 驅動時 GUI 仍可開啟（預覽模式）。

### Ubuntu 安裝與執行

```bash
bash setup_ubuntu.sh                  # python3-tk、字型
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install customtkinter matplotlib numpy nidaqmx
python Diagnosis_System_v4.py
```

### Windows 安裝與執行

```powershell
python -m pip install --user -r requirements.txt
python Diagnosis_System_v4.py
```

### 打包成執行檔（PyInstaller）

建議 one-folder 模式（含 PyTorch，單檔過大）。**無法跨平台打包**，需在目標 OS 上執行。

```bash
# Ubuntu（--add-data 分隔符為冒號）
python -m PyInstaller --noconfirm --clean --windowed \
  --name MotorDiagnosisSystem \
  --copy-metadata nidaqmx --copy-metadata nitypes \
  --add-data "best_motor_cnn_weights.pth:." \
  --add-data "saved_labels.json:." \
  --add-data "saved_data:saved_data" \
  Diagnosis_System_v4.py
# Windows 改用分號分隔符，或直接 python -m PyInstaller MotorDiagnosisSystem.spec
```

---

## 專案內容

```
app/                     WebUI（FastAPI 後端 + 靜態前端）
  datasource/            DAQ 驅動插件層（simulator / nidaq / replay）
  static/                前端頁面（uPlot 圖表、免建置工具鏈）
config.yaml              系統設定（資料源、通道校正、模型、門檻）
Dockerfile / docker-compose.yml / .dockerignore
requirements-web.txt     WebUI 相依
Diagnosis_System_v4.py   桌面版主程式
requirements.txt         桌面版相依
best_motor_cnn_weights.pth   CNN 模型權重（輸入 (2048,6)、12 類）
saved_labels.json        標籤清單（桌面版位置；WebUI 預設同檔，Docker 移至 state/）
saved_data/              標註輸出 CSV
setup_ubuntu.sh          桌面版 Ubuntu 系統相依
```

## 使用注意事項

- WebUI 為區網應用，如需對外開放請置於反向代理（TLS）之後。
- 啟動失敗顯示 DAQ 錯誤時，確認驅動已裝、裝置已接、名稱為 `Dev1`。
- WebUI 存檔的 CSV 已修正桌面版時間戳問題：每筆樣本都有依取樣率推算的時間戳。
