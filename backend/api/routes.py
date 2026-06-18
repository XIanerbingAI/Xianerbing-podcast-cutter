"""API routes —— 上传 / 分析 / 状态 / 决策 / 渲染 / 下载 / 试听片段。"""
from __future__ import annotations

import json
import shutil
from collections import OrderedDict
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from loguru import logger

from backend.config import settings, UPLOAD_DIR, OUTPUT_DIR
from backend.models import AnalysisResult, JobStatus, RenderRequest
from backend.workers.task_queue import store, submit_analysis, submit_render

router = APIRouter(prefix="/api")

ALLOWED_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma", ".opus"}
UPLOAD_CHUNK_BYTES = 1024 * 1024
RENDER_CTX_CACHE_LIMIT = 3


async def _save_upload_stream(
    file: UploadFile,
    upload_path: Path,
    *,
    max_bytes: int,
    chunk_bytes: int = UPLOAD_CHUNK_BYTES,
) -> int:
    """把上传文件分块写入磁盘,避免长音频一次性进入内存。"""
    total = 0
    too_large = False
    try:
        out = upload_path.open("wb")
        try:
            while True:
                chunk = await file.read(chunk_bytes)
                if not chunk:
                    break
                if total + len(chunk) > max_bytes:
                    too_large = True
                    break
                out.write(chunk)
                total += len(chunk)
        finally:
            out.close()
            del out
        if too_large:
            try:
                upload_path.unlink()
            except OSError:
                pass
            raise HTTPException(413, f"文件超过 {settings.max_upload_mb}MB 限制")
    except HTTPException:
        raise
    except Exception:
        try:
            upload_path.unlink()
        except OSError:
            pass
        raise
    return total


@router.post("/upload")
async def upload(
    file: UploadFile = File(...),
    strength: str = Form("balanced"),
):
    """上传音频文件,立即启动后台分析。返回 job_id。"""
    if not file.filename:
        raise HTTPException(400, "缺少文件名")
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"不支持的格式 {ext},允许: {', '.join(ALLOWED_EXT)}")

    job = store.create(file.filename, "")  # upload_path 先空
    upload_path = UPLOAD_DIR / f"{job.job_id}{ext}"
    try:
        size = await _save_upload_stream(
            file,
            upload_path,
            max_bytes=settings.max_upload_mb * 1024 * 1024,
        )
    except Exception:
        store.remove(job.job_id)
        try:
            shutil.rmtree(OUTPUT_DIR / job.job_id)
        except OSError:
            pass
        raise
    job.upload_path = str(upload_path)
    store.update(job.job_id, upload_path=str(upload_path),
                 stage="queued", message="已入队,等待分析…")

    logger.info(f"上传 {file.filename} → job {job.job_id} ({size/1e6:.1f}MB)")
    submit_analysis(job.job_id, strength)
    return {"job_id": job.job_id, "filename": file.filename}


@router.get("/status/{job_id}", response_model=JobStatus)
def status(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "job 不存在")
    return JobStatus(
        job_id=job.job_id,
        stage=job.stage,
        progress=job.progress,
        message=job.message,
        done=job.done,
        error=job.error,
    )


@router.get("/analysis/{job_id}", response_model=AnalysisResult)
def analysis(job_id: str):
    """获取分析结果(EditItem 列表 + 转写)。"""
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "job 不存在")
    if job.stage not in ("ready", "done", "rendering"):
        raise HTTPException(409, f"分析尚未完成(当前阶段:{job.stage})")

    edits_path = Path(job.edits_json_path) if job.edits_json_path else None
    if not edits_path or not edits_path.exists():
        raise HTTPException(404, "分析结果文件缺失")

    data = json.loads(edits_path.read_text(encoding="utf-8"))
    edits = data.get("edits", [])
    segs = data.get("segments", [])

    # stats
    stats_path = OUTPUT_DIR / job_id / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}

    return AnalysisResult(
        job_id=job_id,
        filename=job.filename,
        duration_sec=job.duration_sec,
        language=job.language,
        transcript=job.transcript,
        segments_count=len(segs),
        edits=edits,
        stats=stats,
    )


@router.post("/render")
def render_endpoint(req: RenderRequest):
    """根据用户决策渲染最终音频(后台执行)。"""
    job = store.get(req.job_id)
    if job is None:
        raise HTTPException(404, "job 不存在")
    if job.stage not in ("ready", "done"):
        raise HTTPException(409, f"必须先完成分析(当前:{job.stage})")

    decisions = [{"id": d.id, "keep": d.keep, "mode": d.mode} for d in req.decisions]
    submit_render(req.job_id, decisions, req.apply_loudnorm)
    return {"job_id": req.job_id, "message": "已提交渲染"}


@router.get("/download/{job_id}")
def download(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "job 不存在")
    if not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(404, "产物不存在,请先渲染")
    return FileResponse(
        job.output_path,
        media_type="audio/mpeg",
        filename=Path(job.output_path).name,
    )


@router.get("/preview/{job_id}")
def preview_original(job_id: str):
    """原始文件试听/下载。"""
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "job 不存在")
    p = Path(job.upload_path)
    if not p.exists():
        raise HTTPException(404, "原始文件缺失")
    return FileResponse(str(p), filename=p.name)


@router.get("/clip/{job_id}")
def clip_segment(job_id: str, start: float, end: float, padding: float = 2.0):
    """截取 [start-padding, end+padding] 区间用于试听单条剪辑。

    padding 默认 2.0s(前后各 2 秒),让用户听到完整小语境。
    用 ffmpeg 精确截取,避免加载整段。
    """
    import subprocess
    from backend.ffmpeg_util import ffmpeg_bin

    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "job 不存在")
    s = max(0.0, start - padding)
    e = end + padding
    if e - s < 0.05:
        raise HTTPException(400, "区间过短")

    out = OUTPUT_DIR / job_id / f"clip_{start:.2f}_{end:.2f}_{padding:.1f}.mp3"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{s}", "-to", f"{e}", "-i", job.upload_path,
        "-ac", "1", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "128k",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return FileResponse(str(out), media_type="audio/mpeg", filename=out.name)


# 渲染上下文缓存(供 preview_after 复用解码结果)。job_id -> RenderContext
_RENDER_CTX_CACHE: OrderedDict[str, object] = OrderedDict()


def _cache_render_context(job_id: str, ctx: object) -> None:
    _RENDER_CTX_CACHE[job_id] = ctx
    _RENDER_CTX_CACHE.move_to_end(job_id)
    while len(_RENDER_CTX_CACHE) > RENDER_CTX_CACHE_LIMIT:
        _RENDER_CTX_CACHE.popitem(last=False)


@router.get("/preview_after/{job_id}")
def preview_after(job_id: str, start: float, end: float, mode: str = "standard"):
    """预览单个剪辑点删除后的听感(平滑缝合后的音频)。

    取剪辑点前后各 ~1.5s,删掉中间,用与正式渲染相同的平滑技术
    (零交叉吸附 + 等功率交叉淡化)缝合,返回 MP3 供 A/B 对比。
    """
    import subprocess
    import tempfile
    import soundfile as sf
    from backend.ffmpeg_util import ffmpeg_bin
    from backend.pipeline.smooth import load_render_context, preview_single_cut
    from backend.pipeline.vad import detect_pauses
    from backend.models import Segment

    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "job 不存在")

    # 复用渲染上下文(避免每次预览都重解码+重跑VAD)
    ctx = _RENDER_CTX_CACHE.get(job_id)
    if ctx is not None:
        _RENDER_CTX_CACHE.move_to_end(job_id)
    if ctx is None:
        # 从 edits.json 取 segments 做 VAD 输入
        edits_json = Path(job.edits_json_path) if job.edits_json_path else None
        if not edits_json or not edits_json.exists():
            raise HTTPException(409, "分析结果未就绪")
        data = json.loads(edits_json.read_text(encoding="utf-8"))
        segs = [Segment(start=s["start"], end=s["end"], text=s["text"]) for s in data.get("segments", [])]
        speech = [(s.start, s.end) for s in segs]
        pauses, _a, _sr = detect_pauses(job.upload_path, speech_segments=speech)
        ctx = load_render_context(job.upload_path, pauses)
        _cache_render_context(job_id, ctx)

    try:
        pcm, sr = preview_single_cut(ctx, start, end, mode=mode)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if pcm.size == 0:
        raise HTTPException(400, "剪辑点区间过短,无法预览")

    # 写临时 wav 再转 mp3
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)
    out = OUTPUT_DIR / job_id / f"preview_after_{mode}_{start:.2f}_{end:.2f}.mp3"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        sf.write(str(tmp_wav), pcm, sr, subtype="FLOAT")
        cmd = [
            ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(tmp_wav), "-ac", "1", "-ar", "44100",
            "-c:a", "libmp3lame", "-b:a", "128k", str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        try:
            tmp_wav.unlink()
        except OSError:
            pass
    return FileResponse(str(out), media_type="audio/mpeg", filename=out.name)


@router.get("/jobs")
def list_jobs():
    """列出所有 job(简易管理)。"""
    return [
        {
            "job_id": j.job_id,
            "filename": j.filename,
            "stage": j.stage,
            "progress": j.progress,
            "done": j.done,
            "created_at": j.created_at,
        }
        for j in store.all_jobs()
    ]


@router.get("/health")
def health():
    from backend.ffmpeg_util import ffmpeg_bin
    return {"status": "ok", "ffmpeg": ffmpeg_bin()}


@router.get("/cutlog/{job_id}")
def cutlog(job_id: str, fmt: str = "txt"):
    """下载剪辑日志(txt 人类可读 / json 程序读)。"""
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "job 不存在")
    job_dir = OUTPUT_DIR / job_id
    if fmt == "json":
        p = job_dir / "cut_log.json"
        if not p.exists():
            raise HTTPException(404, "log 未生成(请先渲染)")
        return FileResponse(str(p), media_type="application/json",
                            filename=f"{Path(job.filename).stem}_cutlog.json")
    p = job_dir / "cut_log.txt"
    if not p.exists():
        raise HTTPException(404, "log 未生成(请先渲染)")
    return FileResponse(str(p), media_type="text/plain; charset=utf-8",
                        filename=f"{Path(job.filename).stem}_cutlog.txt")


@router.get("/render_meta/{job_id}")
def render_meta(job_id: str):
    """返回渲染统计的完整口径(供前端显示)。"""
    import json as _json
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "job 不存在")
    p = OUTPUT_DIR / job_id / "render_meta.json"
    if not p.exists():
        raise HTTPException(404, "未渲染")
    return _json.loads(p.read_text(encoding="utf-8"))
