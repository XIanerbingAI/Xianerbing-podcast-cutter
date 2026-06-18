// 饼哥帮你剪播客(Xianerbing-podcast-cutter) 前端逻辑 —— 上传/审核/试听/导出
// 零框架,原生 JS + wavesurfer.js 波形

import WaveSurfer from "https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/wavesurfer.esm.js";
import RegionsPlugin from "https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/regions.esm.js";
import TimelinePlugin from "https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/timeline.esm.js";

const API = "/api";
let currentJob = null;
let analysisData = null;     // AnalysisResult
let wavesurfer = null;
let waveRegions = null;
let waveRegionMap = new Map();
let selectedEditId = null;
let activePreviewAudio = null;
let editState = {};          // id -> {keep, ...原始}

// ============ 初始化 ============
document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  bindUpload();
  bindReview();
  bindDone();
});

async function checkHealth() {
  const pill = document.getElementById("healthPill");
  try {
    const r = await fetch(`${API}/health`);
    const j = await r.json();
    pill.textContent = "● 在线";
    pill.classList.add("ok");
  } catch {
    pill.textContent = "● 离线";
  }
}

// ============ Step 1: 上传 ============
function bindUpload() {
  const dz = document.getElementById("dropzone");
  const input = document.getElementById("fileInput");
  const btn = document.getElementById("uploadBtn");
  let selectedFile = null;

  dz.addEventListener("click", () => input.click());
  ["dragover", "dragenter"].forEach(ev =>
    dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach(ev =>
    dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("dragover"); }));
  dz.addEventListener("drop", e => {
    if (e.dataTransfer.files.length) {
      input.files = e.dataTransfer.files;
      input.dispatchEvent(new Event("change"));
    }
  });

  input.addEventListener("change", () => {
    selectedFile = input.files[0];
    if (selectedFile) {
      btn.disabled = false;
      btn.textContent = `开始分析: ${selectedFile.name}`;
    }
  });

  btn.addEventListener("click", () => {
    if (!selectedFile) return;
    uploadFile(selectedFile);
  });
}

async function uploadFile(file) {
  const btn = document.getElementById("uploadBtn");
  const strength = document.getElementById("strengthSel").value;
  const prog = document.getElementById("uploadProgress");
  const bar = document.getElementById("barFill");
  const msg = document.getElementById("progressMsg");

  btn.disabled = true;
  prog.classList.remove("hidden");
  msg.textContent = "上传中…";

  const fd = new FormData();
  fd.append("file", file);
  fd.append("strength", strength);

  try {
    const r = await fetch(`${API}/upload`, { method: "POST", body: fd });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || `上传失败 (${r.status})`);
    }
    const { job_id } = await r.json();
    currentJob = job_id;
    pollStatus(job_id, bar, msg);
  } catch (e) {
    msg.textContent = `❌ ${e.message}`;
    btn.disabled = false;
  }
}

async function pollStatus(jobId, bar, msg) {
  const tick = async () => {
    try {
      const r = await fetch(`${API}/status/${jobId}`);
      const s = await r.json();
      bar.style.width = `${(s.progress * 100).toFixed(0)}%`;
      msg.textContent = s.message || s.stage;

      if (s.stage === "error") {
        msg.textContent = `❌ ${s.error || "分析失败"}`;
        document.getElementById("uploadBtn").disabled = false;
        return;
      }
      if (s.stage === "ready") {
        await loadAnalysis(jobId);
        return;
      }
      setTimeout(tick, 1000);
    } catch (e) {
      msg.textContent = `❌ ${e.message}`;
    }
  };
  tick();
}

// ============ Step 2: 审核 ============
async function loadAnalysis(jobId) {
  const r = await fetch(`${API}/analysis/${jobId}`);
  analysisData = await r.json();

  // 切换视图
  document.getElementById("step-upload").classList.add("hidden");
  document.getElementById("step-review").classList.remove("hidden");

  // 元信息
  const meta = document.getElementById("reviewMeta");
  const stats = analysisData.stats || {};
  meta.innerHTML = `
    <span>📄 <b>${analysisData.filename}</b></span>
    <span>⏱ <b>${analysisData.duration_sec.toFixed(1)}s</b></span>
    <span>📝 <b>${analysisData.segments_count}</b> 段</span>
    <span>✂ 共 <b>${stats.total_edits || analysisData.edits.length}</b> 处</span>
    <span>🔴 自动删 <b>${stats.auto_cut || 0}</b></span>
    <span>🟡 待复核 <b>${stats.need_review || 0}</b></span>
  `;

  // 初始化 editState
  editState = {};
  selectedEditId = null;
  analysisData.edits.forEach(e => {
    editState[e.id] = { keep: e.keep, cutMode: "standard", ...e };
  });

  renderEditList();
  initWaveform(jobId);
}

function renderEditList() {
  const list = document.getElementById("editList");
  const search = document.getElementById("searchBox").value.toLowerCase();
  const fAuto = document.getElementById("filtAuto").checked;
  const fReview = document.getElementById("filtReview").checked;
  const fKeep = document.getElementById("filtKeep").checked;

  // 计数
  let cnt = { auto: 0, review: 0, keep: 0 };
  analysisData.edits.forEach(e => {
    const st = editState[e.id];
    if (st.keep === false) cnt.auto++;
    else if (e.confidence === "medium" || e.confidence === "low") cnt.review++;
    else cnt.keep++;
  });
  document.getElementById("cntAuto").textContent = cnt.auto;
  document.getElementById("cntReview").textContent = cnt.review;
  document.getElementById("cntKeep").textContent = cnt.keep;

  list.innerHTML = "";
  analysisData.edits.forEach((e, idx) => {
    const st = editState[e.id];
    // 过滤
    const willCut = st.keep === false;
    const isReview = st.keep !== false && (e.confidence === "medium" || e.confidence === "low");
    if (willCut && !fAuto) return;
    if (isReview && !fReview) return;
    if (!willCut && !isReview && !fKeep) return;
    if (search && !(e.original_text || "").toLowerCase().includes(search)) return;

    const item = document.createElement("div");
    item.className = "edit-item " + (willCut ? "will-cut" : (isReview ? "need-review" : "will-keep"));
    if (selectedEditId === e.id) item.classList.add("active");
    item.dataset.id = e.id;
    const mode = st.cutMode || "standard";
    const ctxBefore = e.context_before ? `<span class="ctx-before">${escapeHtml(e.context_before)}</span> ` : "";
    const ctxAfter = e.context_after ? ` <span class="ctx-after">${escapeHtml(e.context_after)}</span>` : "";
    const cutWord = `<span class="ctx-cut">${escapeHtml(e.original_text || "")}</span>`;
    item.innerHTML = `
      <div class="edit-time">${fmt(e.start)} → ${fmt(e.end)}<br><span style="opacity:.6">${(e.duration||0).toFixed(2)}s</span></div>
      <div class="edit-text">
        <div class="ctx-line">${ctxBefore}${cutWord}${ctxAfter}</div>
        <div class="tag-line">
          <span class="reason-tag">${reasonLabel(e.reason)}</span>
          <span class="conf-tag">[${e.confidence}]</span>
          ${e.explanation ? `<span class="expl">${escapeHtml(e.explanation)}</span>` : ""}
        </div>
      </div>
      <div class="edit-actions">
        <button class="btn small play-btn" data-id="${e.id}" title="试听原文(前后2秒)">▶ 原文</button>
        <button class="btn small playafter-btn ${mode === "conservative" ? "active" : ""}" data-id="${e.id}" data-mode="conservative" title="少切一点,降低误删风险">▶ 保守</button>
        <button class="btn small playafter-btn ${mode === "standard" ? "active" : ""}" data-id="${e.id}" data-mode="standard" title="标准能量谷+零交叉方案">▶ 标准</button>
        <button class="btn small playafter-btn ${mode === "clean" ? "active" : ""}" data-id="${e.id}" data-mode="clean" title="多吃一点边界,降低黏连风险">▶ 干净</button>
        <button class="btn small keep-btn ${st.keep !== false ? "active" : ""}" data-id="${e.id}" data-act="keep">保留</button>
        <button class="btn small cut-btn ${st.keep === false ? "active" : ""}" data-id="${e.id}" data-act="cut">删除</button>
      </div>
    `;
    list.appendChild(item);
  });

  // 绑定按钮
  list.querySelectorAll(".keep-btn, .cut-btn").forEach(b => {
    b.addEventListener("click", () => {
      const id = b.dataset.id;
      editState[id].keep = (b.dataset.act === "keep");
      updateWaveRegion(id);
      renderEditList();
    });
  });
  list.querySelectorAll(".play-btn").forEach(b => {
    b.addEventListener("click", () => playClip(b.dataset.id));
  });
  list.querySelectorAll(".playafter-btn").forEach(b => {
    b.addEventListener("click", () => {
      const id = b.dataset.id;
      const mode = b.dataset.mode || "standard";
      editState[id].cutMode = mode;
      playAfter(id, mode);
      renderEditList();
    });
  });
  list.querySelectorAll(".edit-item").forEach(item => {
    item.addEventListener("click", (ev) => {
      if (ev.target.closest("button")) return;
      selectEdit(item.dataset.id, { seek: true, scroll: false });
    });
  });
}

function initWaveform(jobId) {
  if (wavesurfer) {
    wavesurfer.destroy();
    wavesurfer = null;
  }
  waveRegions = RegionsPlugin.create();
  const timeline = TimelinePlugin.create({
    container: "#wave-timeline",
    height: 24,
    timeInterval: 10,
    primaryLabelInterval: 6,
    secondaryLabelInterval: 1,
    style: {
      fontSize: "11px",
      color: "#9aa0aa",
    },
  });
  waveRegionMap = new Map();
  const ws = WaveSurfer.create({
    container: "#waveform",
    waveColor: "#5a6377",
    progressColor: "#4f8cff",
    cursorColor: "#fff",
    height: 90,
    barWidth: 2,
    barRadius: 2,
    url: `${API}/preview/${jobId}`,
    plugins: [waveRegions, timeline],
  });
  wavesurfer = ws;
  ws.on("decode", () => {
    // 高亮剪辑区间
    const duration = ws.getDuration();
    analysisData.edits.forEach(e => {
      if (e.start >= duration) return;
      const region = waveRegions.addRegion({
        start: e.start,
        end: Math.min(e.end, duration),
        color: regionColor(e.id),
        drag: false,
        resize: false,
      });
      waveRegionMap.set(e.id, region);
    });
    updateWaveReadout({ duration });
  });
  ws.on("timeupdate", (time) => {
    updateWaveReadout({ current: time });
  });
  ws.on("interaction", (time) => {
    updateWaveReadout({ current: time });
  });
  waveRegions.on("region-clicked", (region, ev) => {
    ev.stopPropagation();
    const id = findEditIdByRegion(region);
    if (id) selectEdit(id, { seek: true, scroll: true });
  });
  bindWaveHover(ws);
  document.getElementById("playBtn").onclick = () => {
    pausePreviewAudio();
    ws.play();
  };
  document.getElementById("pauseAllBtn").onclick = pauseAllPlayback;
  document.getElementById("origPlayBtn").onclick = () => {
    playManagedAudio(`${API}/preview/${jobId}`);
  };
}

function bindWaveHover(ws) {
  const el = document.getElementById("waveform");
  if (!el) return;
  el.onmousemove = (ev) => {
    const rect = el.getBoundingClientRect();
    const x = Math.min(Math.max(ev.clientX - rect.left, 0), rect.width);
    const duration = ws.getDuration() || 0;
    const hover = rect.width > 0 ? (x / rect.width) * duration : 0;
    updateWaveReadout({ hover });
  };
  el.onmouseleave = () => {
    updateWaveReadout({ hoverText: "--:--.-" });
  };
}

function findEditIdByRegion(region) {
  for (const [id, r] of waveRegionMap.entries()) {
    if (r === region) return id;
  }
  return null;
}

function selectEdit(id, opts = {}) {
  const e = analysisData?.edits?.find(x => x.id === id);
  if (!e) return;
  selectedEditId = id;
  updateWaveReadout({ selection: e });
  waveRegionMap.forEach((region, regionId) => {
    region.setOptions({ color: regionColor(regionId) });
  });

  if (wavesurfer && opts.seek) {
    const duration = wavesurfer.getDuration() || 0;
    if (duration > 0) {
      wavesurfer.seekTo(Math.min(Math.max(e.start / duration, 0), 1));
      updateWaveReadout({ current: e.start });
    }
  }

  if (opts.scroll) {
    const row = document.querySelector(`.edit-item[data-id="${CSS.escape(id)}"]`);
    if (row) row.scrollIntoView({ block: "center", behavior: "smooth" });
  }
  renderEditList();
}

function updateWaveReadout({ current, hover, hoverText, selection, duration } = {}) {
  if (typeof current === "number") {
    const el = document.getElementById("waveCurrentTime");
    if (el) el.textContent = fmt(current);
  }
  if (typeof hover === "number") {
    const el = document.getElementById("waveHoverTime");
    if (el) el.textContent = fmt(hover);
  } else if (hoverText) {
    const el = document.getElementById("waveHoverTime");
    if (el) el.textContent = hoverText;
  }
  if (selection) {
    const el = document.getElementById("waveSelectionTime");
    if (el) {
      const label = selection.original_text ? `${selection.original_text} ` : "";
      el.textContent = `${label}${formatRange(selection.start, selection.end)}`;
    }
  } else if (duration && !selectedEditId) {
    const el = document.getElementById("waveSelectionTime");
    if (el) el.textContent = "未选择";
  }
}

function regionColor(id) {
  if (selectedEditId === id) return "rgba(79,140,255,0.38)";
  return editState[id]?.keep === false ? "rgba(255,92,92,0.25)" : "rgba(245,166,35,0.18)";
}

function updateWaveRegion(id) {
  const region = waveRegionMap.get(id);
  if (region) {
    region.setOptions({ color: regionColor(id) });
  }
}

function pausePreviewAudio() {
  if (activePreviewAudio) {
    activePreviewAudio.pause();
    activePreviewAudio = null;
  }
}

function pauseWaveform() {
  if (wavesurfer && wavesurfer.isPlaying && wavesurfer.isPlaying()) {
    wavesurfer.pause();
  }
}

function pauseAllPlayback() {
  pausePreviewAudio();
  pauseWaveform();
  pauseResultAudio();
}

function playManagedAudio(url) {
  pausePreviewAudio();
  pauseWaveform();
  pauseResultAudio();
  const audio = new Audio(url);
  activePreviewAudio = audio;
  audio.onended = () => {
    if (activePreviewAudio === audio) activePreviewAudio = null;
  };
  audio.play().catch(err => {
    if (activePreviewAudio === audio) activePreviewAudio = null;
    console.warn("audio play failed", err);
  });
}

function resetRenderButton() {
  const btn = document.getElementById("renderBtn");
  if (!btn) return;
  btn.disabled = false;
  btn.textContent = "渲染导出";
}

function pauseResultAudio() {
  const audio = document.getElementById("resultAudio");
  if (audio) audio.pause();
}

async function playClip(id) {
  const e = analysisData.edits.find(x => x.id === id);
  if (!e) return;
  // 前后各 2 秒,听完整小语境
  const url = `${API}/clip/${currentJob}?start=${e.start}&end=${e.end}&padding=2.0`;
  selectEdit(id, { seek: true, scroll: false });
  playManagedAudio(url);
}

async function playAfter(id, mode = "standard") {
  const e = analysisData.edits.find(x => x.id === id);
  if (!e) return;
  const url = `${API}/preview_after/${currentJob}?start=${e.start}&end=${e.end}&mode=${encodeURIComponent(mode)}`;
  selectEdit(id, { seek: true, scroll: false });
  playManagedAudio(url);
}

function bindReview() {
  ["searchBox", "filtAuto", "filtReview", "filtKeep"].forEach(idc => {
    const el = document.getElementById(idc);
    el.addEventListener("input", renderEditList);
    el.addEventListener("change", renderEditList);
  });
  document.getElementById("keepAllBtn").onclick = () => {
    Object.values(editState).forEach(s => s.keep = true);
    renderEditList();
  };
  document.getElementById("cutAllBtn").onclick = () => {
    if (!confirm("确认把所有候选都标记为删除?语义守卫标记为'保留'的也会被删。")) return;
    Object.values(editState).forEach(s => s.keep = false);
    renderEditList();
  };
  document.getElementById("renderBtn").onclick = startRender;
}

// ============ Step 3: 渲染 ============
async function startRender() {
  const btn = document.getElementById("renderBtn");
  const loudnorm = document.getElementById("loudnormChk").checked;
  const decisions = Object.values(editState).map(s => ({
    id: s.id,
    keep: s.keep !== false,
    mode: s.cutMode || "standard",
  }));

  pausePreviewAudio();
  pauseWaveform();
  pauseResultAudio();
  btn.disabled = true;
  btn.textContent = "渲染中…";

  const r = await fetch(`${API}/render`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_id: currentJob, decisions, apply_loudnorm: loudnorm }),
  });
  if (!r.ok) { btn.disabled = false; btn.textContent = "渲染导出"; alert("提交失败"); return; }

  // 轮询
  const tick = async () => {
    const s = await (await fetch(`${API}/status/${currentJob}`)).json();
    btn.textContent = `渲染中… ${(s.progress * 100).toFixed(0)}%`;
    if (s.stage === "error") {
      btn.disabled = false; btn.textContent = "渲染导出";
      alert(`渲染失败: ${s.error}`); return;
    }
    if (s.stage === "done" && s.done) {
      showDone(); return;
    }
    setTimeout(tick, 1200);
  };
  tick();
}

async function showDone() {
  const r = await fetch(`${API}/status/${currentJob}`);
  const s = await r.json();
  document.getElementById("step-review").classList.add("hidden");
  document.getElementById("step-done").classList.remove("hidden");

  // 从后端拉渲染元信息(完整口径统计)
  let realCut = 0, realApplied = 0, userDecided = 0, merged = 0;
  try {
    const mr = await fetch(`${API}/cutlog/${currentJob}?fmt=json`);
    if (mr.ok) {
      const logData = await mr.json();
      realApplied = logData.filter(c => c.applied).length;
      realCut = logData.filter(c => c.applied)
                   .reduce((sum, c) => sum + (c.refined_duration_ms || 0), 0) / 1000;
    }
    const metaR = await fetch(`${API}/render_meta/${currentJob}`);
    if (metaR.ok) {
      const meta = await metaR.json();
      userDecided = meta.user_decided_cut;
      realApplied = meta.applied_count;
      realCut = meta.total_cut_sec;
      merged = meta.merged_or_filtered;
    }
  } catch (e) { /* 退化 */ }

  let statsText;
  if (realApplied > 0 || userDecided > 0) {
    statsText = `你选择删除 ${userDecided} 处 → 实际删除 ${realApplied} 处(剪掉 ${realCut.toFixed(1)}s)`;
    if (merged > 0) {
      statsText += ` · 其中 ${merged} 处因相邻/过短合并(内容已删,详见日志)`;
    }
  } else {
    let saved = 0;
    Object.values(editState).forEach(st => { if (st.keep === false) saved += (st.duration || 0); });
    statsText = `预计剪掉 ${saved.toFixed(1)}s`;
  }
  document.getElementById("doneStats").textContent = statsText;

  const audio = document.getElementById("resultAudio");
  audio.src = `${API}/download/${currentJob}?t=${Date.now()}`;
  audio.onplay = () => {
    pausePreviewAudio();
    pauseWaveform();
  };
  document.getElementById("downloadLink").href = `${API}/download/${currentJob}`;
  document.getElementById("downloadLink").download = s.output_path ? s.output_path.split(/[\\/]/).pop() : "edited.mp3";
  document.getElementById("cutlogLink").href = `${API}/cutlog/${currentJob}?fmt=txt`;
}

function bindDone() {
  document.getElementById("newJobBtn").onclick = () => location.reload();
  // 返回审核页重新调整(不重跑转写,analysisData/editState 都在内存)
  document.getElementById("backToReviewBtn").onclick = () => {
    pauseResultAudio();
    resetRenderButton();
    document.getElementById("step-done").classList.add("hidden");
    document.getElementById("step-review").classList.remove("hidden");
    // 滚动到顶部
    window.scrollTo({ top: 0, behavior: "smooth" });
    // 重新渲染审核列表(保留用户之前的勾选状态 editState)
    renderEditList();
  };
}

// ============ 工具 ============
function fmt(sec) {
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1);
  return `${m}:${s.padStart(4, "0")}`;
}
function formatRange(start, end) {
  return `${fmt(start)} - ${fmt(end)}`;
}
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function reasonLabel(r) {
  return { filler: "填充", discourse: "话语标记", repeat: "重复", stutter: "口吃", false_start: "废弃话头" }[r] || r;
}
