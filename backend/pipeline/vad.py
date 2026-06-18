"""VAD / 停顿检测 —— 找出语音段之间的静音与呼吸区间。

这部分独立于 Whisper 内置 VAD:后者用于转写去静音,
这里在原始音频上做能量/过零率分析,产出 Pause 列表,
供 smooth.py 判断"句间 vs 句内"、以及"呼吸 vs 纯静音"。

依赖:ffmpeg 解码为 PCM + numpy/scipy 分析。不引入 silero/webrtcvad
(它们对中文播客的呼吸检测反而不如能量+频谱规则稳定)。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from loguru import logger

from backend.config import settings
from backend.ffmpeg_util import ffmpeg_bin
from backend.models import Pause


def decode_mono_pcm(
    audio_path: str | Path,
    *,
    sample_rate: int = 16000,
) -> tuple[np.ndarray, int]:
    """用 ffmpeg 把任意音频解码为单声道 float32 PCM。"""
    audio_path = str(audio_path)
    cmd = [
        ffmpeg_bin(), "-hide_banner", "-loglevel", "error",
        "-i", audio_path,
        "-ac", "1",                       # 单声道
        "-ar", str(sample_rate),
        "-f", "f32le",                    # 32-bit float little-endian PCM
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    raw = proc.stdout
    audio = np.frombuffer(raw, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError(f"ffmpeg 解码失败(空数据): {audio_path}")
    return audio, sample_rate


def _rms_frame(audio: np.ndarray, sr: int, frame_ms: int = 20) -> tuple[np.ndarray, int]:
    """分帧计算 RMS,返回 (rms_per_frame, hop_samples)。"""
    hop = int(sr * frame_ms / 1000)
    n_frames = audio.size // hop
    if n_frames == 0:
        return np.array([float(np.sqrt(np.mean(audio**2)))]), hop
    trimmed = audio[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt(np.mean(trimmed**2, axis=1))
    return rms, hop


def detect_pauses(
    audio_path: str | Path,
    *,
    speech_segments: list[tuple[float, float]] | None = None,
) -> tuple[list[Pause], np.ndarray, int]:
    """检测停顿/呼吸。

    Args:
        audio_path: 音频
        speech_segments: [(start,end),...] Whisper 给出的语音段。若提供,
            只在这些段之间找 gap(更准);否则全程能量检测。

    Returns:
        pauses: Pause 列表(含 is_breath 标记)
        audio: 原始 float32 PCM
        sr: 采样率
    """
    sr = 16000
    audio, sr = decode_mono_pcm(audio_path, sample_rate=sr)
    total = audio.size / sr

    rms, hop = _rms_frame(audio, sr, frame_ms=20)
    # 帧时间轴
    frame_times = np.arange(rms.size) * (hop / sr)
    # 动态阈值:用整体 RMS 的一个比例 + 最小绝对值,鲁棒于不同录音电平
    base = float(np.percentile(rms, 30))
    silence_thr = max(base * 0.6, 1e-4)
    breath_thr = silence_thr * 2.5  # 呼吸比纯静音响一点

    # 找低能量连续区间
    low_mask = rms < silence_thr
    pauses: list[Pause] = []

    def _add_run(s_idx: int, e_idx: int):
        s_t = float(frame_times[s_idx])
        e_t = float(frame_times[min(e_idx, frame_times.size - 1)]) + (hop / sr)
        dur_ms = (e_t - s_t) * 1000
        if dur_ms < 40:
            return
        # 在该区间内取最大 RMS 判断是否呼吸
        seg_rms = rms[s_idx:e_idx] if e_idx > s_idx else rms[s_idx:s_idx+1]
        peak = float(np.max(seg_rms))
        is_breath = (peak > breath_thr) and (dur_ms >= settings.breath_min_ms)
        pauses.append(Pause(start=s_t, end=e_t, is_breath=is_breath))

    # 连续低能量段
    i = 0
    n = low_mask.size
    while i < n:
        if low_mask[i]:
            j = i
            while j < n and low_mask[j]:
                j += 1
            _add_run(i, j)
            i = j
        else:
            i += 1

    # 若提供了 speech_segments,只保留落在 gap 内的 pause
    if speech_segments:
        gaps = _gaps(speech_segments, total)
        pauses = [p for p in pauses if any(g[0] <= p.start and p.end <= g[1] + 0.05 for g in gaps)]

    pauses.sort(key=lambda p: p.start)
    logger.info(f"VAD 检测到 {len(pauses)} 个停顿/呼吸区间")
    return pauses, audio, sr


def _gaps(seg: list[tuple[float, float]], total: float) -> list[tuple[float, float]]:
    gaps: list[tuple[float, float]] = []
    prev_end = 0.0
    for s, e in seg:
        if s > prev_end:
            gaps.append((prev_end, s))
        prev_end = max(prev_end, e)
    if prev_end < total:
        gaps.append((prev_end, total))
    return gaps


def estimate_speaker_pause_stats(pauses: list[Pause]) -> dict:
    """统计说话人句间停顿的典型长度,用于 smooth.py 节奏保持。"""
    pure = [p.duration for p in pauses if not p.is_breath and p.duration > 0.1]
    if not pure:
        return {"median_pause_s": 0.28, "p90_pause_s": 0.6, "count": 0}
    arr = np.array(pure)
    return {
        "median_pause_s": float(np.median(arr)),
        "p90_pause_s": float(np.percentile(arr, 90)),
        "count": int(arr.size),
    }
