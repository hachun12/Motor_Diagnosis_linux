/* 馬達即時診斷系統 — 前端邏輯 */
"use strict";

const COLORS = {
  A: "#FF4C4C", B: "#FF9900", C: "#FFD700",
  X: "#00BFFF", Y: "#00FF99", Z: "#CC88FF",
};
const CUR_KEYS = ["A", "B", "C"];
const VIB_KEYS = ["X", "Y", "Z"];
const ALL_KEYS = [...CUR_KEYS, ...VIB_KEYS];
const ISO_COLORS = { normal: "#00FF88", warning: "#FFD700", alert: "#FF9900", danger: "#FF4C4C" };

let me = null;
let running = false;

/* ── API 輔助：401 一律導回登入頁 ─────────────────────── */
async function api(path, opts = {}) {
  if (opts.body && !(opts.body instanceof FormData)
      && !(opts.headers || {})["Content-Type"]) {
    opts.headers = Object.assign({ "Content-Type": "application/json" }, opts.headers);
  }
  const res = await fetch(path, opts);
  if (res.status === 401) { location.href = "/login"; throw new Error("未登入"); }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

/* ── uPlot 圖表工廠 ───────────────────────────────────── */
const AXIS_STYLE = {
  stroke: "#9a9a9a",
  grid: { stroke: "#333333", width: 0.5 },
  ticks: { stroke: "#444444", width: 0.5 },
  font: "10px sans-serif",
};

function makeChart(el, keys, { yRange = null, xLabel = "", yLabel = "", height = null,
                               tipX = null, tipVal = null } = {}) {
  const series = [{}].concat(keys.map((k) => ({
    label: k, stroke: COLORS[k], width: 1.2, points: { show: false },
  })));
  const opts = {
    width: el.clientWidth || 600,
    height: height || el.clientHeight || 170,
    legend: { show: false },
    // 拖曳框選 = 放大（x 軸），雙擊 = 還原
    cursor: { drag: { x: true, y: false } },
    scales: {
      x: { time: false },
      y: yRange ? { auto: false, range: () => yRange } : { auto: true },
    },
    series,
    axes: [
      Object.assign({ label: xLabel, labelFont: "11px sans-serif", labelSize: xLabel ? 14 : 0 }, AXIS_STYLE),
      // space: y 刻度最小間距（px）。預設 50 在矮圖上只擠得下一個刻度，改 20
      Object.assign({ label: yLabel, labelFont: "11px sans-serif",
                      labelSize: yLabel ? 14 : 0, space: 20 }, AXIS_STYLE),
    ],
    hooks: {
      // 使用者框選後標記「已縮放」：串流 setData 改為保留縮放範圍
      setSelect: [(u) => { if (u.select && u.select.width > 0) u._userZoomed = true; }],
    },
  };

  // ── hover 數值提示 ──
  const tip = document.createElement("div");
  tip.className = "chart-tip";
  tip.style.display = "none";
  opts.hooks.setCursor = [(u) => {
    const idx = u.cursor.idx;
    if (idx == null || u.data[0] == null || u.data[0].length < 2) {
      tip.style.display = "none";
      return;
    }
    let html = tipX ? `<div class="tip-x">${tipX(u.data[0][idx])}</div>` : "";
    for (let i = 1; i < u.series.length; i++) {
      if (!u.series[i].show) continue;
      const v = u.data[i] && u.data[i][idx];
      if (v == null || Number.isNaN(v)) continue;
      const k = keys[i - 1];
      html += `<div><span style="color:${COLORS[k]}">●</span> ${k}：${tipVal ? tipVal(k, v) : v}</div>`;
    }
    tip.innerHTML = html;
    tip.style.display = "block";
    let left = u.cursor.left + 14, top = u.cursor.top + 12;
    if (left + tip.offsetWidth > u.over.clientWidth) left = u.cursor.left - tip.offsetWidth - 10;
    if (top + tip.offsetHeight > u.over.clientHeight) top = Math.max(0, u.cursor.top - tip.offsetHeight - 8);
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
  }];

  const chart = new uPlot(opts, [[0], ...keys.map(() => [0])], el);
  chart._userZoomed = false;
  chart.over.appendChild(tip);
  chart.over.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  // 雙擊還原（uPlot 內建會重設比例，這裡同步清除縮放旗標）
  chart.over.addEventListener("dblclick", () => { chart._userZoomed = false; });
  new ResizeObserver(() => {
    if (el.clientWidth > 0) chart.setSize({ width: el.clientWidth, height: height || el.clientHeight });
  }).observe(el);
  return chart;
}

/* 各通道工程單位 */
const UNIT = { A: "A", B: "A", C: "A", X: "g", Y: "g", Z: "g" };
const fmtWave = (k, v) => `${v.toFixed(4)} ${UNIT[k]}`;
const fmtSpec = (dbEl) => (k, v) =>
  dbEl.checked ? `${v.toFixed(1)} dB` : `${v.toPrecision(3)} ${UNIT[k]}`;

/* 串流更新：未縮放時全幅重繪，縮放中保留使用者視野 */
function streamData(chart, data) {
  chart.setData(data, !chart._userZoomed);
}

/* ── 即時圖表 ─────────────────────────────────────────── */
let sampleRate = 5000;  // init() 時以 /api/status 實際值更新
const liveX = Array.from({ length: 200 }, (_, i) => i);
const tipLiveX = (v) => `${(v * 1000 / sampleRate).toFixed(1)} ms`;
const curChart = makeChart(document.getElementById("chart-cur"), CUR_KEYS,
  { yLabel: "電流 (A)", tipX: tipLiveX, tipVal: fmtWave });
const vibChart = makeChart(document.getElementById("chart-vib"), VIB_KEYS,
  { yLabel: "振動 (g)", tipX: tipLiveX, tipVal: fmtWave });
const specChart = makeChart(document.getElementById("chart-spec"), ALL_KEYS,
  { xLabel: "頻率 (Hz)", yLabel: "振幅",
    tipX: (v) => `${v.toFixed(1)} Hz`,
    tipVal: fmtSpec(document.getElementById("spec-db")) });

const visible = {};          // 波形通道勾選
const specVisible = {};      // 頻譜通道勾選
ALL_KEYS.forEach((k) => { visible[k] = true; specVisible[k] = VIB_KEYS.includes(k); });

function buildChecks(containerId, keys, state, onChange) {
  const box = document.getElementById(containerId);
  keys.forEach((k) => {
    const label = document.createElement("label");
    label.style.color = COLORS[k];
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = state[k];
    cb.addEventListener("change", () => { state[k] = cb.checked; onChange(k); });
    label.append(cb, document.createTextNode(k));
    box.appendChild(label);
  });
}
buildChecks("checks-cur", CUR_KEYS, visible, (k) => curChart.setSeries(CUR_KEYS.indexOf(k) + 1, { show: visible[k] }));
buildChecks("checks-vib", VIB_KEYS, visible, (k) => vibChart.setSeries(VIB_KEYS.indexOf(k) + 1, { show: visible[k] }));
buildChecks("spec-controls", ALL_KEYS, specVisible, (k) => specChart.setSeries(ALL_KEYS.indexOf(k) + 1, { show: specVisible[k] }));
ALL_KEYS.forEach((k) => specChart.setSeries(ALL_KEYS.indexOf(k) + 1, { show: specVisible[k] }));

/* ── 頻譜 dB 轉換 ─────────────────────────────────────── */
let lastSpectrum = null;
const toDb = (m) => 20 * Math.log10(Math.max(m, 1e-6));

function renderSpectrum() {
  if (!lastSpectrum) return;
  const db = document.getElementById("spec-db").checked;
  const rows = ALL_KEYS.map((k) => {
    const mags = lastSpectrum.channels[k] || [];
    return db ? mags.map(toDb) : mags;
  });
  streamData(specChart, [lastSpectrum.freqs, ...rows]);
}
document.getElementById("spec-db").addEventListener("change", renderSpectrum);

/* ── WebSocket ────────────────────────────────────────── */
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (ev) => handleMessage(JSON.parse(ev.data));
  ws.onclose = (ev) => {
    if (ev.code === 4401) { location.href = "/login"; return; }
    setTimeout(connectWS, 2000);
  };
}

function handleMessage(msg) {
  if (msg.type === "wave") {
    streamData(curChart, [liveX, ...CUR_KEYS.map((k) => msg.cur[k])]);
    streamData(vibChart, [liveX, ...VIB_KEYS.map((k) => msg.vib[k])]);
    updateIso(msg.iso);
    setRunning(true);
  } else if (msg.type === "spectrum") {
    lastSpectrum = msg;
    renderSpectrum();
  } else if (msg.type === "ai") {
    const el = document.getElementById("ai-result");
    el.textContent = aiText(msg);
    el.style.color = msg.normal ? ISO_COLORS.normal : ISO_COLORS.danger;
    document.getElementById("ai-detail").textContent = msg.detail || "";
  } else if (msg.type === "log") {
    appendLog(msg.time, msg.message);
  } else if (msg.type === "status") {
    setRunning(msg.running);
    if (!msg.running) resetStatusUI();
  } else if (msg.type === "training") {
    if (me && me.role === "admin") renderTraining(msg);
  } else if (msg.type === "model") {
    activeModelName = msg.name;
    renderDriverInfo();
  }
}

let activeModelName = "";
let driverInfoText = "";
function renderDriverInfo() {
  document.getElementById("driver-info").textContent =
    driverInfoText + (activeModelName ? `｜模型：${activeModelName}` : "");
}

function updateIso(iso) {
  const color = ISO_COLORS[iso.level] || "#888";
  document.getElementById("iso-dot").style.color = color;
  const text = document.getElementById("iso-text");
  text.textContent = iso.text;
  text.style.color = color;
  document.getElementById("iso-rms").textContent = `RMS: ${iso.rms.toFixed(3)}`;
}

function resetStatusUI() {
  document.getElementById("iso-dot").style.color = "#888";
  const text = document.getElementById("iso-text");
  text.textContent = "待機"; text.style.color = "#888";
  document.getElementById("iso-rms").textContent = "RMS: ---";
  const ai = document.getElementById("ai-result");
  ai.textContent = "等待資料..."; ai.style.color = "#AAAAAA";
  document.getElementById("ai-detail").textContent = "";
}

/* 診斷結果顯示文字：中文名稱（實驗代號）信心% */
function aiText(ai) {
  const pct = `${(ai.confidence * 100).toFixed(1)}%`;
  return ai.name ? `${ai.name}（${ai.code}） ${pct}` : `${ai.class} (${pct})`;
}

/* ── 日誌 ─────────────────────────────────────────────── */
const logEl = document.getElementById("log-terminal");
function appendLog(time, message) {
  const line = document.createElement("div");
  line.textContent = `[${time}]  ${message}`;
  logEl.appendChild(line);
  while (logEl.childNodes.length > 500) logEl.removeChild(logEl.firstChild);
  logEl.scrollTop = logEl.scrollHeight;
}
document.getElementById("btn-clear-log").addEventListener("click", () => { logEl.innerHTML = ""; });

/* ── 控制 ─────────────────────────────────────────────── */
function setRunning(state) {
  if (running === state) return;
  running = state;
  const pill = document.getElementById("status-pill");
  pill.textContent = state ? "● 執行中" : "待機";
  pill.classList.toggle("running", state);
  updateButtons();
}

function updateButtons() {
  const isAdmin = me && me.role === "admin";
  document.getElementById("btn-start").disabled = !isAdmin || running;
  document.getElementById("btn-stop").disabled = !isAdmin || !running;
  document.getElementById("btn-save").disabled = !isAdmin;
}

document.getElementById("btn-start").addEventListener("click", async () => {
  try { await api("/api/start", { method: "POST" }); setRunning(true); }
  catch (e) { appendLog(now(), `❌ ${e.message}`); }
});
document.getElementById("btn-stop").addEventListener("click", async () => {
  try { await api("/api/stop", { method: "POST" }); setRunning(false); resetStatusUI(); }
  catch (e) { appendLog(now(), `❌ ${e.message}`); }
});
const now = () => new Date().toTimeString().slice(0, 8);

/* ── 存檔與標籤 ───────────────────────────────────────── */
async function loadLabels() {
  const labels = await api("/api/labels");
  const sel = document.getElementById("label-select");
  sel.innerHTML = '<option value="">-- 選擇已有標籤 --</option>';
  labels.forEach((l) => {
    const opt = document.createElement("option");
    opt.value = opt.textContent = l;
    sel.appendChild(opt);
  });
}
document.getElementById("label-select").addEventListener("change", (e) => {
  if (e.target.value) {
    document.getElementById("label-input").value = e.target.value;
    appendLog(now(), `📂 已選取標籤：${e.target.value}，可直接按存檔`);
  }
});
document.getElementById("btn-save").addEventListener("click", async () => {
  const label = document.getElementById("label-input").value.trim();
  if (!label) { appendLog(now(), "⚠️ 請輸入標籤名稱後再存檔。"); return; }
  try {
    await api("/api/save", { method: "POST", body: JSON.stringify({ label }) });
    document.getElementById("label-input").value = "";
    document.getElementById("label-select").value = "";
    await loadLabels();
    loadRecordings();
  } catch (e) { appendLog(now(), `❌ 存檔失敗：${e.message}`); }
});

/* ── 分頁切換 ─────────────────────────────────────────── */
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll("main").forEach((m) =>
      m.classList.toggle("active", m.id === `view-${btn.dataset.view}`));
    if (btn.dataset.view === "history") loadRecordings();
    if (btn.dataset.view === "models") { loadModels(); loadTrainingOverview(); }
  });
});

/* ── 歷史記錄與回放 ───────────────────────────────────── */
let playFile = null;
let playTotal = 0;
let playCharts = null;
let lastPlayData = null;

async function loadRecordings() {
  const items = await api("/api/recordings");
  const tbody = document.querySelector("#rec-table tbody");
  tbody.innerHTML = "";
  document.getElementById("rec-empty").hidden = items.length > 0;
  items.forEach((r) => {
    const tr = document.createElement("tr");
    if (r.name === playFile) tr.classList.add("selected");
    const sizeMB = (r.size / 1048576).toFixed(1);
    tr.innerHTML = `<td>${r.name}</td><td>${sizeMB} MB</td><td>${r.mtime}</td>`;
    const td = document.createElement("td");
    const dl = document.createElement("a");
    dl.href = `/api/recordings/${encodeURIComponent(r.name)}`;
    dl.className = "btn btn-sm btn-gray"; dl.textContent = "下載";
    dl.style.textDecoration = "none";
    const play = document.createElement("button");
    play.className = "btn btn-sm btn-primary"; play.textContent = "回放";
    play.addEventListener("click", () => startPlayback(r.name));
    td.append(dl, play);
    tr.appendChild(td);
    tbody.appendChild(tr);
  });
}
document.getElementById("btn-refresh-rec").addEventListener("click", loadRecordings);

function ensurePlayCharts() {
  if (playCharts) return;
  // 回放為靜態資料：y 軸依資料自動調整，避免超出固定刻度被裁切
  const tipPlayX = (v) => `${v.toFixed(4)} s`;
  playCharts = {
    cur: makeChart(document.getElementById("chart-play-cur"), CUR_KEYS,
      { xLabel: "時間 (秒)", yLabel: "電流 (A)", tipX: tipPlayX, tipVal: fmtWave }),
    vib: makeChart(document.getElementById("chart-play-vib"), VIB_KEYS,
      { xLabel: "時間 (秒)", yLabel: "振動 (g)", tipX: tipPlayX, tipVal: fmtWave }),
    spec: makeChart(document.getElementById("chart-play-spec"), ALL_KEYS,
      { xLabel: "頻率 (Hz)", yLabel: "振幅",
        tipX: (v) => `${v.toFixed(1)} Hz`,
        tipVal: fmtSpec(document.getElementById("play-spec-db")) }),
  };
  const state = {};
  ALL_KEYS.forEach((k) => { state[k] = VIB_KEYS.includes(k); });
  buildChecks("play-spec-controls", ALL_KEYS, state, (k) =>
    playCharts.spec.setSeries(ALL_KEYS.indexOf(k) + 1, { show: state[k] }));
  ALL_KEYS.forEach((k) => playCharts.spec.setSeries(ALL_KEYS.indexOf(k) + 1, { show: state[k] }));
  document.getElementById("play-spec-db").addEventListener("change", renderPlayback);
}

async function startPlayback(name) {
  playFile = name;
  document.getElementById("play-card").hidden = false;
  document.getElementById("play-placeholder").hidden = true;
  document.getElementById("play-name").textContent = name;
  ensurePlayCharts();
  document.getElementById("play-slider").value = 0;
  await fetchPlayData();
  loadRecordings();
}

let playFetchTimer = null;
function schedulePlayFetch() {
  clearTimeout(playFetchTimer);
  playFetchTimer = setTimeout(fetchPlayData, 150);
}

async function fetchPlayData() {
  if (!playFile) return;
  const start = parseFloat(document.getElementById("play-slider").value) || 0;
  const win = document.getElementById("play-window").value;
  try {
    const data = await api(`/api/recordings/${encodeURIComponent(playFile)}/data?start=${start}&window=${win}`);
    lastPlayData = data;
    playTotal = data.total_seconds;
    const slider = document.getElementById("play-slider");
    slider.max = Math.max(0, playTotal - data.window).toFixed(1);
    document.getElementById("play-pos").textContent =
      `${data.start.toFixed(1)}s – ${(data.start + data.window).toFixed(1)}s / 共 ${playTotal.toFixed(1)}s`;
    renderPlayAI(data.ai);
    renderPlayback();
  } catch (e) {
    document.getElementById("play-pos").textContent = `讀取失敗：${e.message}`;
  }
}

function renderPlayAI(ai) {
  const el = document.getElementById("play-ai");
  const detail = document.getElementById("play-ai-detail");
  detail.textContent = "";
  if (!ai) {
    el.textContent = "推論失敗"; el.style.color = "#AAAAAA";
  } else if (ai.insufficient) {
    el.textContent = `資料不足（${ai.got}/${ai.needed} 點）`;
    el.style.color = "#AAAAAA";
  } else {
    el.textContent = aiText(ai);
    el.style.color = ai.normal ? ISO_COLORS.normal : ISO_COLORS.danger;
    detail.textContent = ai.detail || "";
  }
}

function renderPlayback() {
  if (!lastPlayData || !playCharts) return;
  const d = lastPlayData;
  playCharts.cur.setData([d.t, ...CUR_KEYS.map((k) => d.wave[k])]);
  playCharts.vib.setData([d.t, ...VIB_KEYS.map((k) => d.wave[k])]);
  const db = document.getElementById("play-spec-db").checked;
  playCharts.spec.setData([d.freqs, ...ALL_KEYS.map((k) =>
    db ? d.spectrum[k].map(toDb) : d.spectrum[k])]);
}

document.getElementById("play-slider").addEventListener("input", schedulePlayFetch);
document.getElementById("play-window").addEventListener("change", fetchPlayData);

/* ── 模型管理（admin） ────────────────────────────────── */
async function loadModels() {
  const models = await api("/api/models");
  const tbody = document.querySelector("#models-table tbody");
  tbody.innerHTML = "";
  models.forEach((m) => {
    const tr = document.createElement("tr");
    const acc = m.val_accuracy != null ? `${(m.val_accuracy * 100).toFixed(1)}%` : "—";
    const nameCell = document.createElement("td");
    nameCell.textContent = m.name;
    if (m.active) {
      const badge = document.createElement("span");
      badge.className = "badge-active"; badge.textContent = "啟用中";
      nameCell.appendChild(badge);
    }
    tr.appendChild(nameCell);
    tr.insertAdjacentHTML("beforeend", `<td>${m.num_classes} 類</td><td>${acc}</td><td>${m.created}</td>`);
    const td = document.createElement("td");
    if (!m.active) {
      const act = document.createElement("button");
      act.className = "btn btn-sm btn-primary"; act.textContent = "啟用";
      act.addEventListener("click", async () => {
        try { await api(`/api/models/${encodeURIComponent(m.name)}/activate`, { method: "POST" }); await loadModels(); }
        catch (e) { alert(e.message); }
      });
      td.appendChild(act);
      const del = document.createElement("button");
      del.className = "btn btn-sm btn-gray"; del.textContent = "刪除";
      del.addEventListener("click", async () => {
        if (!confirm(`確定刪除模型「${m.name}」？`)) return;
        try { await api(`/api/models/${encodeURIComponent(m.name)}`, { method: "DELETE" }); await loadModels(); }
        catch (e) { alert(e.message); }
      });
      td.appendChild(del);
    }
    const dl = document.createElement("a");
    dl.href = `/api/models/${encodeURIComponent(m.name)}/download`;
    dl.className = "btn btn-sm btn-gray"; dl.textContent = "下載";
    dl.style.textDecoration = "none";
    td.appendChild(dl);
    tr.appendChild(td);
    tbody.appendChild(tr);
  });
}

document.getElementById("btn-import").addEventListener("click", async () => {
  const err = document.getElementById("import-error");
  err.hidden = true;
  const name = document.getElementById("import-name").value.trim();
  const file = document.getElementById("import-file").files[0];
  if (!name || !file) {
    err.textContent = "請填寫模型名稱並選擇 .pth 檔"; err.hidden = false; return;
  }
  const fd = new FormData();
  fd.append("file", file);
  fd.append("name", name);
  fd.append("classes", document.getElementById("import-classes").value.trim());
  try {
    await api("/api/models/import", { method: "POST", body: fd });
    document.getElementById("import-name").value = "";
    document.getElementById("import-file").value = "";
    document.getElementById("import-classes").value = "";
    await loadModels();
  } catch (e) { err.textContent = e.message; err.hidden = false; }
});

/* ── 訓練面板 ─────────────────────────────────────────── */
let knownCodes = [];

async function loadTrainingOverview() {
  const data = await api("/api/training/overview");
  knownCodes = data.known_codes || [];
  const tbody = document.querySelector("#train-table tbody");
  tbody.innerHTML = "";
  document.getElementById("train-empty").hidden = data.groups.length > 0;
  data.groups.forEach((g) => {
    const tr = document.createElement("tr");
    tr.insertAdjacentHTML("beforeend",
      `<td>${g.label}</td><td>${g.files}</td><td>${g.seconds}s</td><td>${g.windows}</td>`);
    const td = document.createElement("td");
    const sel = document.createElement("select");
    sel.dataset.label = g.label;
    sel.innerHTML = '<option value="">（不使用）</option>'
      + knownCodes.map((c) => `<option value="${c}">${c}</option>`).join("")
      + '<option value="__custom__">自訂…</option>';
    // 標籤與已知代號同名時自動對應
    const hit = knownCodes.find((c) => c.toLowerCase() === g.label.toLowerCase());
    if (hit) sel.value = hit;
    sel.addEventListener("change", () => {
      if (sel.value === "__custom__") {
        const custom = (prompt("輸入新類別代號（例如 RB-3）：") || "").trim();
        if (custom && !knownCodes.includes(custom)) {
          knownCodes.push(custom);
          document.querySelectorAll("#train-table select").forEach((s) => {
            const keep = s.value;
            s.insertAdjacentHTML("beforeend", `<option value="${custom}">${custom}</option>`);
            s.value = keep === "__custom__" && s === sel ? custom : keep;
          });
        }
        sel.value = custom && knownCodes.includes(custom) ? custom : "";
      }
    });
    td.appendChild(sel);
    tr.appendChild(td);
    tbody.appendChild(tr);
  });
}
document.getElementById("btn-train-rescan").addEventListener("click", loadTrainingOverview);

let trainChart = null;
function ensureTrainChart() {
  if (trainChart) return;
  const el = document.getElementById("chart-train");
  trainChart = new uPlot({
    width: el.clientWidth || 500, height: 200,
    legend: { show: true },
    scales: { x: { time: false }, loss: { auto: true }, acc: { range: () => [0, 1] } },
    series: [
      { label: "epoch" },
      { label: "train loss", stroke: "#FF9900", width: 1.5, scale: "loss", points: { show: false } },
      { label: "val acc", stroke: "#00FF88", width: 1.5, scale: "acc", points: { show: false } },
    ],
    axes: [
      Object.assign({}, AXIS_STYLE),
      Object.assign({ scale: "loss", space: 20 }, AXIS_STYLE),
      Object.assign({ scale: "acc", side: 1, space: 20,
                      values: (u, vs) => vs.map((v) => `${(v * 100).toFixed(0)}%`) }, AXIS_STYLE),
    ],
  }, [[], [], []], el);
  new ResizeObserver(() => {
    if (el.clientWidth > 0) trainChart.setSize({ width: el.clientWidth, height: 200 });
  }).observe(el);
}

function renderTraining(st) {
  const card = document.getElementById("train-progress-card");
  if (!st || st.epoch === undefined && !st.running && !st.done && !st.error) {
    card.hidden = true; return;
  }
  card.hidden = false;
  ensureTrainChart();
  document.getElementById("train-job-name").textContent = st.name || "";
  const pct = st.epochs ? Math.round(((st.epoch || 0) / st.epochs) * 100) : 0;
  document.getElementById("train-progress-bar").style.width = `${pct}%`;
  let stage = st.stage || "";
  if (st.running && st.epoch) {
    stage = `Epoch ${st.epoch}/${st.epochs}｜loss ${st.train_loss}｜驗證準確率 ${(st.val_acc * 100).toFixed(1)}%（最佳 ${(st.best_acc * 100).toFixed(1)}%）`;
  }
  document.getElementById("train-stage").textContent = stage;
  const h = st.history || [];
  trainChart.setData([h.map((r) => r.epoch), h.map((r) => r.loss), h.map((r) => r.val_acc)]);

  const result = document.getElementById("train-result");
  if (st.done && st.per_class) {
    result.innerHTML = "各類別驗證準確率：" + Object.entries(st.per_class)
      .map(([c, a]) => `${c} ${(a * 100).toFixed(0)}%`).join("｜")
      + "<br>模型已存入模型庫，請於左側啟用。";
  } else if (st.error) {
    result.textContent = `錯誤：${st.error}`;
  } else if (st.cancelled) {
    result.textContent = "已取消。";
  } else {
    result.textContent = "";
  }
  document.getElementById("btn-train-start").disabled = !!st.running;
  document.getElementById("btn-train-cancel").disabled = !st.running;
  if ((st.done || st.error || st.cancelled) && !st.running) loadModels();
}

document.getElementById("btn-train-start").addEventListener("click", async () => {
  const err = document.getElementById("train-error");
  err.hidden = true;
  const mapping = {};
  document.querySelectorAll("#train-table select").forEach((s) => {
    if (s.value && s.value !== "__custom__") mapping[s.dataset.label] = s.value;
  });
  const name = document.getElementById("train-name").value.trim();
  if (!name) { err.textContent = "請輸入新模型名稱"; err.hidden = false; return; }
  try {
    await api("/api/training/start", {
      method: "POST",
      body: JSON.stringify({
        name, mapping,
        params: {
          epochs: +document.getElementById("tp-epochs").value,
          lr: +document.getElementById("tp-lr").value,
          batch_size: +document.getElementById("tp-batch").value,
          val_split: +document.getElementById("tp-val").value,
          stride: +document.getElementById("tp-stride").value,
        },
      }),
    });
    if (trainChart) trainChart.setData([[], [], []]);
    document.getElementById("btn-train-start").disabled = true;
    document.getElementById("btn-train-cancel").disabled = false;
  } catch (e) { err.textContent = e.message; err.hidden = false; }
});

document.getElementById("btn-train-cancel").addEventListener("click", () =>
  api("/api/training/cancel", { method: "POST" }).catch(() => {}));

/* ── 帳號選單 ─────────────────────────────────────────── */
document.getElementById("btn-logout").addEventListener("click", async () => {
  await api("/api/logout", { method: "POST" });
  location.href = "/login";
});

const dlgPwd = document.getElementById("dlg-pwd");
document.getElementById("btn-pwd").addEventListener("click", () => {
  document.getElementById("pwd-old").value = "";
  document.getElementById("pwd-new").value = "";
  document.getElementById("pwd-error").hidden = true;
  dlgPwd.showModal();
});
document.getElementById("pwd-form").addEventListener("submit", async (e) => {
  if (e.submitter && e.submitter.value === "cancel") return;
  e.preventDefault();
  try {
    await api("/api/password", {
      method: "POST",
      body: JSON.stringify({
        old_password: document.getElementById("pwd-old").value,
        new_password: document.getElementById("pwd-new").value,
      }),
    });
    dlgPwd.close();
    appendLog(now(), "🔑 密碼已更新");
  } catch (err) {
    const el = document.getElementById("pwd-error");
    el.textContent = err.message; el.hidden = false;
  }
});

const dlgUsers = document.getElementById("dlg-users");
document.getElementById("btn-users").addEventListener("click", async () => {
  await refreshUsers();
  document.getElementById("users-error").hidden = true;
  dlgUsers.showModal();
});
document.getElementById("users-close").addEventListener("click", () => dlgUsers.close());

async function refreshUsers() {
  const users = await api("/api/users");
  const tbody = document.querySelector("#users-table tbody");
  tbody.innerHTML = "";
  users.forEach((u) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${u.username}</td><td>${u.role}</td>`;
    const td = document.createElement("td");
    if (u.username !== me.username) {
      const del = document.createElement("button");
      del.className = "btn btn-sm btn-gray"; del.textContent = "刪除";
      del.addEventListener("click", async () => {
        if (!confirm(`確定刪除使用者 ${u.username}？`)) return;
        try { await api(`/api/users/${encodeURIComponent(u.username)}`, { method: "DELETE" }); await refreshUsers(); }
        catch (e) { showUsersError(e.message); }
      });
      td.appendChild(del);
    }
    tr.appendChild(td);
    tbody.appendChild(tr);
  });
}
function showUsersError(msg) {
  const el = document.getElementById("users-error");
  el.textContent = msg; el.hidden = false;
}
document.getElementById("user-add-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    await api("/api/users", {
      method: "POST",
      body: JSON.stringify({
        username: document.getElementById("new-user").value.trim(),
        password: document.getElementById("new-pass").value,
        role: document.getElementById("new-role").value,
      }),
    });
    document.getElementById("new-user").value = "";
    document.getElementById("new-pass").value = "";
    document.getElementById("users-error").hidden = true;
    await refreshUsers();
  } catch (err) { showUsersError(err.message); }
});

/* ── 初始化 ───────────────────────────────────────────── */
async function init() {
  me = await api("/api/me");
  document.getElementById("user-name").textContent = `${me.username}（${me.role}）`;
  document.getElementById("btn-users").hidden = me.role !== "admin";

  const status = await api("/api/status");
  sampleRate = status.sample_rate || 5000;
  driverInfoText =
    `資料源：${status.driver}｜取樣率：${status.sample_rate} Hz｜歷史：${status.history_seconds} 秒`;
  activeModelName = status.model ? status.model.name : "";
  renderDriverInfo();
  setRunning(status.running);
  updateButtons();
  if (status.ai) handleMessage(Object.assign({ type: "ai" }, status.ai));

  if (me.role === "admin") {
    document.getElementById("tab-models").hidden = false;
    api("/api/training/status").then((st) => renderTraining(st)).catch(() => {});
  }

  (await api("/api/logs")).forEach((l) => appendLog(l.time, l.message));
  await loadLabels();
  connectWS();

  // 定期校正執行狀態（避免多分頁/斷線期間狀態漂移）
  setInterval(async () => {
    try { setRunning((await api("/api/status")).running); } catch {}
  }, 5000);
}
init();
