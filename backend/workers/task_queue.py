"""任务队列 + Job 状态管理。

本地:进程内字典 + 同步执行(适合开发/小规模)
服务器:预留 Celery 接口(USE_CELERY=true 时启用)

Job 生命周期:
  uploaded → transcribing → analyzing → ready(等待审核)
  → rendering → done
任何阶段失败 → error
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from backend.config import settings, UPLOAD_DIR, OUTPUT_DIR


@dataclass(slots=True)
class Job:
    job_id: str
    filename: str
    upload_path: str
    stage: str = "uploaded"
    progress: float = 0.0
    message: str = ""
    done: bool = False
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    # 分析结果(ready 后填充)
    duration_sec: float = 0.0
    language: str = "zh"
    transcript: str = ""
    edits_json_path: Optional[str] = None  # 序列化的 EditItem 列表(供前端读)
    output_path: Optional[str] = None      # 渲染产物
    # 内部缓存(pickle 序列化到磁盘以便跨请求保留)
    _state_file: Optional[str] = None


class JobStore:
    """线程安全的 job 仓库。"""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()

    def _load_from_disk(self, job_id: str) -> Optional[Job]:
        job_dir = OUTPUT_DIR / job_id
        if not job_dir.exists() or not job_dir.is_dir():
            return None

        upload_candidates = list(UPLOAD_DIR.glob(f"{job_id}.*"))
        upload_path = str(upload_candidates[0]) if upload_candidates else ""
        output_candidates = sorted(job_dir.glob("*_剪辑后.mp3"))
        output_path = str(output_candidates[0]) if output_candidates else None
        edits_json = job_dir / "edits.json"

        filename = Path(upload_path).name if upload_path else f"{job_id}.mp3"
        if output_path:
            filename = Path(output_path).name.replace("_剪辑后.mp3", ".mp3")

        stage = "uploaded"
        progress = 0.0
        done = False
        if output_path:
            stage = "done"
            progress = 1.0
            done = True
        elif edits_json.exists():
            stage = "ready"
            progress = 1.0

        transcript = ""
        duration_sec = 0.0
        if edits_json.exists():
            try:
                import json
                data = json.loads(edits_json.read_text(encoding="utf-8"))
                segs = data.get("segments", [])
                transcript = "".join(s.get("text", "") for s in segs)
                if segs:
                    duration_sec = max(float(s.get("end", 0.0) or 0.0) for s in segs)
            except Exception:
                pass

        job = Job(
            job_id=job_id,
            filename=filename,
            upload_path=upload_path,
            stage=stage,
            progress=progress,
            message="已从磁盘恢复任务",
            done=done,
            duration_sec=duration_sec,
            transcript=transcript,
            edits_json_path=str(edits_json) if edits_json.exists() else None,
            output_path=output_path,
        )
        job._state_file = str(job_dir / "state.txt")
        return job

    def create(self, filename: str, upload_path: str) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id=job_id, filename=filename, upload_path=str(upload_path))
        # 持久化目录
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job._state_file = str(job_dir / "state.txt")
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return job
            job = self._load_from_disk(job_id)
            if job is not None:
                self._jobs[job_id] = job
            return job

    def remove(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.pop(job_id, None)

    def update(self, job_id: str, **kwargs) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for k, v in kwargs.items():
                if hasattr(job, k):
                    setattr(job, k, v)
            return job

    def all_jobs(self) -> list[Job]:
        with self._lock:
            for job_dir in OUTPUT_DIR.iterdir():
                if job_dir.is_dir() and job_dir.name not in self._jobs:
                    job = self._load_from_disk(job_dir.name)
                    if job is not None:
                        self._jobs[job.job_id] = job
            return list(self._jobs.values())


# 全局单例
store = JobStore()


def build_llm_context_map(items) -> dict[str, str]:
    """给 LLM 复核构造真实文本上下文,避免用时间戳粗略映射字符位置。"""
    ctx_map: dict[str, str] = {}
    for it in items:
        before = (getattr(it, "context_before", "") or "").strip()
        after = (getattr(it, "context_after", "") or "").strip()
        meta = (
            f"时间 {it.start:.2f}-{it.end:.2f}s; "
            f"类型 {it.reason.value}; 置信度 {it.confidence.value}; "
            f"位置 {it.position.value}"
        )
        ctx_map[it.id] = f"{meta}\n上下文: ...{before}【{it.original_text}】{after}..."
    return ctx_map


# ============================================================
# 任务执行(进程内同步线程)
# ============================================================

_executor_lock = threading.Lock()


def run_analysis(job_id: str, strength: str, progress_cb: Callable | None = None) -> None:
    """执行:转写 → VAD → 检测 → 序列化 EditItem。后台线程调用。"""
    import json
    from dataclasses import asdict

    from backend.pipeline.transcribe import transcribe
    from backend.pipeline.vad import detect_pauses
    from backend.pipeline.detect import detect, DetectionConfig
    from backend.pipeline.render import probe_duration

    job = store.get(job_id)
    if job is None:
        return

    try:
        store.update(job_id, stage="transcribing", message="正在转写音频…", progress=0.05)
        audio_path = job.upload_path

        # 时长
        duration = probe_duration(audio_path)
        store.update(job_id, duration_sec=duration)

        def _cb(p, msg):
            store.update(job_id, stage="transcribing", progress=0.05 + 0.55 * p, message=msg)
            if progress_cb:
                progress_cb(p, msg)

        segments, full_text = transcribe(audio_path, progress_cb=_cb)
        store.update(job_id, transcript=full_text, language=settings.whisper_language)

        store.update(job_id, stage="analyzing", progress=0.65, message="正在分析停顿与口癖…")
        # VAD
        speech_segs = [(s.start, s.end) for s in segments]
        pauses, _audio, _sr = detect_pauses(audio_path, speech_segments=speech_segs)
        pause_tuples = [(p.start, p.end, p.is_breath) for p in pauses]

        # 检测
        items = detect(segments, DetectionConfig(strength=strength, pause_list=pause_tuples))

        # LLM 复核(可选):仅对歧义候选
        if items:
            from backend.pipeline import semantic_llm
            if semantic_llm.is_enabled():
                store.update(job_id, stage="analyzing", progress=0.85, message="LLM 复核歧义项…")
                # 构造上下文:用相邻词拼出前后文
                full_text = "".join(w.text for seg in segments for w in seg.words)
                ambiguous = [it for it in items
                             if it.confidence.value in ("medium", "low") and not it.llm_reviewed]
                if ambiguous:
                    ctx_map = build_llm_context_map(ambiguous)
                    try:
                        semantic_llm.review_batch(ambiguous, ctx_map)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"LLM 复核异常(忽略): {e}")

        # 序列化 EditItem 到磁盘(前端/渲染都要读)
        job_dir = OUTPUT_DIR / job_id
        edits_json = job_dir / "edits.json"
        edits_data = []
        for it in items:
            d = asdict(it) if hasattr(it, "__dataclass_fields__") else dict(it)
            # 枚举转字符串
            for k in ("reason", "confidence", "position", "llm_verdict"):
                if k in d and d[k] is not None and not isinstance(d[k], str):
                    d[k] = d[k].value if hasattr(d[k], "value") else str(d[k])
            d["duration"] = it.duration
            edits_data.append(d)
        edits_json.write_text(
            json.dumps({"job_id": job_id, "edits": edits_data,
                        "segments": [{"start": s.start, "end": s.end, "text": s.text}
                                     for s in segments]},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 统计
        stats = {
            "total_edits": len(items),
            "auto_cut": sum(1 for it in items if not it.keep),
            "need_review": sum(1 for it in items if it.keep and it.confidence.value in ("medium", "low")),
            "by_reason": {},
        }
        for it in items:
            stats["by_reason"][it.reason.value] = stats["by_reason"].get(it.reason.value, 0) + 1

        store.update(
            job_id,
            stage="ready",
            progress=1.0,
            message=f"分析完成,共 {len(items)} 处候选,自动删除 {stats['auto_cut']} 处",
            done=False,  # ready 不算 done,等待渲染
            edits_json_path=str(edits_json),
        )
        # 把统计也写到 state(供 status 接口返回)
        (job_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")
        logger.info(f"job {job_id} 分析完成")

    except Exception as e:  # noqa: BLE001
        logger.exception(f"job {job_id} 分析失败")
        store.update(job_id, stage="error", error=str(e), message=f"分析失败: {e}")


def run_render(job_id: str, decisions: list[dict], apply_loudnorm: bool,
               progress_cb: Callable | None = None) -> None:
    """执行:应用决策 → 平滑剪辑 → 渲染。"""
    import json
    from backend.models import CutPosition, EditItem, CutReason, Confidence
    from backend.pipeline.editplan import build_cut_regions
    from backend.pipeline.smooth import load_render_context
    from backend.pipeline.render import render
    from backend.pipeline.vad import detect_pauses
    from backend.pipeline.transcribe import transcribe

    job = store.get(job_id)
    if job is None:
        return

    try:
        store.update(job_id, stage="rendering", progress=0.1, message="加载音频与底噪…")

        # 读 edits.json
        edits_json = Path(job.edits_json_path) if job.edits_json_path else None
        if not edits_json or not edits_json.exists():
            raise RuntimeError("找不到分析结果,请先完成分析")
        data = json.loads(edits_json.read_text(encoding="utf-8"))

        # 重新转写拿 segments(用于 VAD)。为省时,从 segments 字段读
        # 但 VAD 需要重新跑;segments 时间范围足够
        from backend.models import Segment
        segments = [Segment(start=s["start"], end=s["end"], text=s["text"])
                    for s in data.get("segments", [])]
        speech_segs = [(s.start, s.end) for s in segments]
        pauses, _a, _sr = detect_pauses(job.upload_path, speech_segments=speech_segs)

        # 构造 EditItem 并应用用户决策
        decision_map = {d["id"]: d.get("keep", True) for d in decisions}
        mode_map = {d["id"]: d.get("mode", "standard") for d in decisions}
        # 调试日志:确认用户决策真的传到了
        dec_cut = sum(1 for v in decision_map.values() if v is False or v == "false")
        dec_keep = len(decision_map) - dec_cut
        logger.info(f"job {job_id} 收到用户决策:共 {len(decision_map)} 条 "
                    f"(删 {dec_cut} / 留 {dec_keep})")
        logger.debug(f"job {job_id} decision_map 样本: {list(decision_map.items())[:3]}")
        items: list[EditItem] = []
        for d in data["edits"]:
            it = EditItem(
                id=d["id"],
                start=d["start"],
                end=d["end"],
                original_text=d["original_text"],
                reason=CutReason(d["reason"]),
                confidence=Confidence(d["confidence"]),
                position=CutPosition(d["position"]),
                keep=decision_map.get(d["id"], d.get("keep", True)),
                llm_reviewed=d.get("llm_reviewed", False),
                llm_verdict=d.get("llm_verdict"),
                explanation=d.get("explanation", ""),
            )
            items.append(it)

        store.update(job_id, progress=0.3, message="生成剪辑方案…")
        regions = build_cut_regions(items)
        _apply_region_modes(regions, mode_map)
        # 调试日志:用户决策删 vs 实际生成区间数
        user_cut = sum(1 for it in items if not it.keep)
        logger.info(f"job {job_id} keep=False: {user_cut} → build_cut_regions: {len(regions)} 个区间")

        store.update(job_id, progress=0.4, message="平滑剪辑中…")
        ctx = load_render_context(job.upload_path, pauses)
        out_path = OUTPUT_DIR / job_id / f"{Path(job.filename).stem}_剪辑后"
        final, cut_log = render(ctx, regions, out_path, apply_loudnorm=apply_loudnorm, out_format="mp3")

        # 写 cut log(JSON 供程序读 + txt 供人读)
        job_dir = OUTPUT_DIR / job_id
        log_json = job_dir / "cut_log.json"
        log_txt = job_dir / "cut_log.txt"
        log_json.write_text(json.dumps(cut_log, ensure_ascii=False, indent=2), encoding="utf-8")
        log_txt.write_text(_cut_log_to_text(cut_log, job.filename), encoding="utf-8")

        # 写渲染元信息(供前端显示完整口径的统计)
        user_decided_cut = sum(1 for it in items if not it.keep)
        regions_count = len(regions)
        applied_count = len([c for c in cut_log if c.get("applied")])
        merged_or_filtered = user_decided_cut - regions_count
        render_meta = {
            "user_decided_cut": user_decided_cut,      # 用户勾选删除的词条数
            "regions_count": regions_count,            # 实际删除区间数(合并后)
            "applied_count": applied_count,            # apply_cuts 执行数(=regions)
            "merged_or_filtered": merged_or_filtered,  # 合并/过滤掉的(内容仍删了)
            "total_cut_sec": round(sum(c.get("refined_duration_ms", 0) for c in cut_log) / 1000, 2),
        }
        (job_dir / "render_meta.json").write_text(
            json.dumps(render_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"job {job_id} cut log: 候选删{user_decided_cut} → 区间{regions_count} "
                    f"(合并/过滤{merged_or_filtered}) → 执行{applied_count} -> {log_txt}")

        store.update(
            job_id, stage="done", progress=1.0, done=True,
            output_path=str(final),
            message=f"渲染完成: {final.name}",
        )
        logger.info(f"job {job_id} 渲染完成 → {final}")

    except Exception as e:  # noqa: BLE001
        logger.exception(f"job {job_id} 渲染失败")
        store.update(job_id, stage="error", error=str(e), message=f"渲染失败: {e}")


# ============================================================
# 线程池(进程内)
# ============================================================

def submit_analysis(job_id: str, strength: str) -> None:
    t = threading.Thread(target=run_analysis, args=(job_id, strength), daemon=True)
    t.start()


def submit_render(job_id: str, decisions: list[dict], apply_loudnorm: bool) -> None:
    t = threading.Thread(target=run_render, args=(job_id, decisions, apply_loudnorm), daemon=True)
    t.start()


# ============================================================
# cut log 可读化
# ============================================================

_REASON_CN = {
    "filler": "语气填充", "discourse": "话语标记", "repeat": "重复词",
    "stutter": "口吃", "false_start": "废弃话头",
}

_MODE_PRIORITY = {"conservative": 0, "standard": 1, "clean": 2}


def _apply_region_modes(regions, mode_map: dict[str, str]) -> None:
    """把用户选择的剪切方案应用到合并后的删除区间。

    多个候选被合并时,取更偏"干净"的方案,避免相邻碎切造成黏连。
    """
    for region in regions:
        modes = [mode_map.get(source_id, "standard") for source_id in region.source_ids]
        valid = [m for m in modes if m in _MODE_PRIORITY]
        region.mode = max(valid or ["standard"], key=lambda m: _MODE_PRIORITY[m])


def _cut_log_to_text(cut_log: list[dict], filename: str) -> str:
    """把 cut log 转成人类可读的 txt 报告,方便 debug。

    设计原则:本工具只做建议,删/不删完全由用户决策。
    本报告记录每处删除的实际执行情况(切点精确化、边界警告)。
    """
    lines = []
    lines.append("=" * 72)
    lines.append("PodcastZ 剪辑日志 (cut log)")
    lines.append(f"文件: {filename}")
    lines.append("原则:严格按用户选择的删除/保留执行,本工具不做删/不删的替用户决定")
    lines.append("=" * 72)

    applied = [c for c in cut_log if c.get("applied")]
    warned = [c for c in applied if c.get("boundary_warning")]
    risky = [c for c in applied if c.get("quality_label") == "risky"]
    review = [c for c in applied if c.get("quality_label") == "review"]
    total_cut_s = sum(c.get("refined_duration_ms", 0) for c in applied) / 1000.0

    lines.append("")
    lines.append(f"统计:已删除 {len(applied)} 处 | 实际剪掉 {total_cut_s:.2f}s | "
                 f"其中 {len(warned)} 处有边界警告 | {len(review)} 处建议试听 | "
                 f"{len(risky)} 处高风险")
    lines.append("")

    # 已删除项
    lines.append("-" * 72)
    lines.append(f"【已删除 {len(applied)} 处(全部按用户决策执行)】")
    lines.append("-" * 72)
    lines.append(f"{'#':>3}  {'原文':<8} {'类型':<8} {'方案':<12} {'原区间(s)':<18} {'精确化后(s)':<18} {'质量':<10} {'偏移(ms)':<14} {'警告':<4}")
    for i, c in enumerate(applied, 1):
        txt = (c.get("original_text") or "")[:6]
        reason = _REASON_CN.get(c.get("reason", ""), c.get("reason", ""))
        orig = f"{c['original_start']:.2f}-{c['original_end']:.2f}"
        ref = f"{c['refined_start']:.2f}-{c['refined_end']:.2f}"
        mode = c.get("selected_mode", "standard")
        quality = f"{c.get('quality_label','?')}:{c.get('quality_score','?')}"
        shift = f"起{c.get('boundary_shift_start_ms',0):+.0f}/止{c.get('boundary_shift_end_ms',0):+.0f}"
        warn = "⚠" if c.get("boundary_warning") else ""
        lines.append(f"{i:>3}  {txt:<8} {reason:<8} {mode:<12} {orig:<18} {ref:<18} {quality:<10} {shift:<14} {warn:<4}")

    lines.append("")
    lines.append("=" * 72)
    lines.append("字段说明:")
    lines.append("  - 原区间:Whisper 转写给出的词边界(常偏,落在有声处)")
    lines.append("  - 精确化后:声学边界精确化找到的能量谷(更接近真正词间隙)")
    lines.append("  - 偏移:精确化后相对原边界的位移(+后移,-前移)。仅微调切点,不改变删/不删")
    lines.append("  - 警告⚠:切点疑似靠近词边(可能影响听感)。仅提示,已按用户决策执行删除")
    lines.append("  - 质量:clean 通常较干净; review 建议试听; risky 应重点复核")
    lines.append("  - 方案:conservative 少切; standard 标准; clean 更重视去黏连")
    lines.append("  - 所有删除均严格按用户选择执行,工具未自行否决任何用户决策")
    lines.append("=" * 72)
    return "\n".join(lines)
