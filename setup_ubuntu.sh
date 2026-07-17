#!/usr/bin/env bash
# Ubuntu 系統相依安裝腳本
# 安裝 Tkinter GUI 後端與 emoji 字型。DAQ 驅動 (NI-DAQmx for Linux) 請依 NI 官方步驟另行安裝。
set -euo pipefail

echo "==> 更新套件索引"
sudo apt-get update

echo "==> 安裝 python3-tk (Tkinter / matplotlib TkAgg 後端)"
sudo apt-get install -y python3-tk

echo "==> 安裝 fonts-noto-color-emoji (介面 emoji 顯示)"
sudo apt-get install -y fonts-noto-color-emoji

# DejaVu Sans Mono（等寬字型，多數桌面環境已內建，保險起見一併安裝）
echo "==> 安裝 fonts-dejavu (等寬字型)"
sudo apt-get install -y fonts-dejavu || true

echo ""
echo "系統相依安裝完成。"
echo "接下來："
echo "  python3 -m venv .venv && source .venv/bin/activate"
echo "  pip install --upgrade pip"
echo "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
echo "  pip install customtkinter matplotlib numpy nidaqmx"
echo "  python Diagnosis_System_v4.py"
