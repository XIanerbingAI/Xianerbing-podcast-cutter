"""转写模块 —— 音频 → 中文文本 + 词级时间戳。

基于 faster-whisper(CTranslate2 加速),内置 VAD 去静音。
输出带 word-level timestamps,供下游检测/剪辑精确定位。

设备自适应:
- 有 CUDA → GPU(int8_float16)
- 否则 → CPU(int8)
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from loguru import logger

from backend.config import MODELS_DIR, settings
from backend.models import Segment, WordToken

# 模型单例,避免反复加载
_MODEL = None


def _resolve_device() -> str:
    if settings.whisper_device != "auto":
        return settings.whisper_device
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _resolve_compute(device: str) -> str:
    if device == "cuda":
        # 用户显式指定时尊重之
        if settings.whisper_compute in ("float16", "int8_float16", "int8"):
            return settings.whisper_compute
        return "int8_float16"
    return "int8"


def get_model():
    """惰性加载 faster-whisper 模型(单例)。"""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    from faster_whisper import WhisperModel  # type: ignore

    device = _resolve_device()
    compute = _resolve_compute(device)
    logger.info(f"加载 Whisper 模型: {settings.whisper_model} / {device} / {compute}")
    _MODEL = WhisperModel(
        settings.whisper_model,
        device=device,
        compute_type=compute,
        download_root=str(MODELS_DIR) if MODELS_DIR else None,
    )
    return _MODEL


def transcribe(
    audio_path: str | Path,
    *,
    progress_cb=None,
) -> tuple[list[Segment], str]:
    """转写并返回 (segments, full_text)。

    每个 Segment 含 words(WordToken 列表,带词级时间戳)。

    Args:
        audio_path: 输入音频路径
        progress_cb: 可选回调 (0.0~1.0, message)
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    model = get_model()

    def _emit(p, msg):
        if progress_cb:
            try:
                progress_cb(p, msg)
            except Exception:  # noqa: BLE001
                pass

    _emit(0.05, "开始转写…")

    # faster-whisper segments 是惰性迭代器,逐段返回
    fw_segments, info = model.transcribe(
        str(audio_path),
        language=settings.whisper_language,
        beam_size=settings.beam_size,
        vad_filter=settings.vad_filter,
        vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200),
        word_timestamps=True,
        initial_prompt=settings.whisper_initial_prompt,
    )

    language = info.language or settings.whisper_language
    duration = getattr(info, "duration", 0.0) or 0.0
    logger.info(f"检测语言={language}, 时长={duration:.1f}s")

    segments: list[Segment] = []
    full_text_parts: list[str] = []
    last_end = 0.0

    for i, seg in enumerate(fw_segments):
        words: list[WordToken] = []
        if seg.words:
            for w in seg.words:
                txt = (w.word or "").strip()
                if not txt:
                    continue
                words.append(
                    WordToken(
                        text=txt,
                        start=float(w.start or 0.0),
                        end=float(w.end or 0.0),
                        probability=float(getattr(w, "probability", 1.0) or 1.0),
                    )
                )
        seg_start = float(seg.start)
        seg_end = float(seg.end)
        segments.append(
            Segment(start=seg_start, end=seg_end, text=(seg.text or "").strip(), words=words)
        )
        full_text_parts.append(seg.text or "")
        last_end = max(last_end, seg_end)
        if progress_cb and duration > 0:
            _emit(min(0.9, 0.1 + 0.8 * (last_end / duration)), f"已转写 {last_end:.0f}s")

    _emit(0.95, "转写完成,合并词级时间戳")
    # 后处理:合并相邻极近的字(Whisper 中文常按字切)
    segments = _merge_close_words(segments)
    _emit(1.0, "转写完成")
    return segments, "".join(full_text_parts)


def _merge_close_words(segments: list[Segment]) -> list[Segment]:
    """中文 Whisper 经常按"字"给出 word。这里不做强制合并,
    保留字级粒度(检测重复词需要),但会规范化明显错误的时间戳:
    - 不允许词越出片段
    - 不允许同片段内词时间倒退
    - end <= start 时补一个很小的安全时长
    """
    min_word_s = 0.02
    fallback_word_s = 0.05
    for seg in segments:
        fixed: list[WordToken] = []
        seg.start = max(0.0, float(seg.start))
        seg.end = max(seg.start, float(seg.end))
        cursor = seg.start
        for w in seg.words:
            if not w.text:
                continue
            start = max(seg.start, min(float(w.start), seg.end))
            end = max(seg.start, min(float(w.end), seg.end))
            start = max(start, cursor)
            if end <= start:
                end = min(seg.end, start + fallback_word_s)
            if end - start < min_word_s:
                end = min(seg.end, start + min_word_s)
            if end <= start:
                continue
            prob = max(0.0, min(1.0, float(w.probability)))
            fixed.append(WordToken(text=w.text, start=start, end=end, probability=prob))
            cursor = end
        seg.words = fixed
    return segments


def save_segments_json(segments: list[Segment], path: str | Path) -> None:
    """序列化为 JSON,供前端/调试用。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "start": s.start,
            "end": s.end,
            "text": s.text,
            "words": [asdict(w) for w in s.words],
        }
        for s in segments
    ]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
