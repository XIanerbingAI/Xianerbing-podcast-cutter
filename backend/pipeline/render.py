"""渲染 —— PCM → 音频文件 + EBU R128 响度归一。

流程:
1. smooth.py 产出平滑 PCM(float32, 44.1k, mono)
2. 写入临时 WAV
3. (可选)ffmpeg loudnorm 两遍处理:第一遍测 I/LRA/TP,第二遍精确归一
4. 输出最终 mp3/m4a/wav

loudnorm 两遍是关键:单遍模式只是动态压缩,两遍才精确达到目标 LUFS。
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import soundfile as sf
from loguru import logger

from backend.config import settings
from backend.ffmpeg_util import ffmpeg_bin, ffprobe_bin
from backend.pipeline.smooth import RenderContext, apply_cuts
from backend.pipeline.editplan import CutRegion


def render(
    ctx: RenderContext,
    regions: list[CutRegion],
    out_path: str | Path,
    *,
    apply_loudnorm: bool = True,
    out_format: str = "mp3",
) -> tuple[Path, list[dict]]:
    """主渲染入口。

    Args:
        ctx: 渲染上下文(原始 PCM + 底噪)
        regions: 删除区间
        out_path: 输出路径(无扩展名,自动加)
        apply_loudnorm: 是否做 LUFS 归一
        out_format: mp3 / m4a / wav

    Returns:
        (final_path, cut_log)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) 平滑剪辑
    pcm, cut_log = apply_cuts(ctx, regions)

    # 2) 写临时 WAV(32-bit float,无损失)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)
    final = out_path.with_suffix(f".{out_format}")
    try:
        sf.write(str(tmp_wav), pcm, ctx.sr, subtype="FLOAT")
        logger.info(f"平滑音频写入临时 WAV: {tmp_wav} ({pcm.size/ctx.sr:.1f}s)")

        # 3) ffmpeg 处理
        if apply_loudnorm:
            final = _loudnorm_two_pass(tmp_wav, final, out_format)
        else:
            _encode(tmp_wav, final, out_format)
    finally:
        try:
            tmp_wav.unlink()
        except OSError:
            pass

    logger.info(f"渲染完成: {final}")
    return final, cut_log


# ============================================================
# 编码
# ============================================================

def _encode(in_wav: Path, out_path: Path, fmt: str, extra_args: list[str] | None = None) -> None:
    """WAV → 目标格式。"""
    if fmt == "wav":
        # 直接复制(已是无损)
        import shutil
        shutil.copy2(in_wav, out_path)
        return

    codec, bitrate = {
        "mp3": ("libmp3lame", "192k"),
        "m4a": ("aac", "192k"),
        "aac": ("aac", "192k"),
        "ogg": ("libvorbis", "192k"),
        "flac": ("flac", None),
    }.get(fmt, ("libmp3lame", "192k"))

    cmd = [
        ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(in_wav),
        "-ac", "1",
        "-ar", "44100",
        "-c:a", codec,
    ]
    if bitrate:
        cmd += ["-b:a", bitrate]
    if extra_args:
        cmd += extra_args
    cmd.append(str(out_path))

    subprocess.run(cmd, check=True, capture_output=True)


# ============================================================
# EBU R128 两遍响度归一
# ============================================================

def _loudnorm_two_pass(in_wav: Path, out_path: Path, fmt: str) -> Path:
    """两遍 loudnorm,精确达到目标 LUFS。

    第一遍:动态模式测量,得到 measured_I / measured_LRA / measured_TP / measured_thresh
    第二遍:用测量值做线性模式归一,无动态压缩痕迹。
    """
    target = settings.loudnorm_target_lufs
    lra = settings.loudnorm_lra
    tp = settings.loudnorm_true_peak

    # ---- Pass 1: 测量 ----
    measure_cmd = [
        ffmpeg_bin(), "-hide_banner", "-i", str(in_wav),
        "-af", f"loudnorm=I={target}:LRA={lra}:TP={tp}:print_format=json",
        "-f", "null", "-",
    ]
    proc = subprocess.run(measure_cmd, capture_output=True, text=True)
    stats = _parse_loudnorm_json(proc.stderr)
    if not stats:
        logger.warning("loudnorm 测量失败,退化为单遍动态归一")
        _encode(in_wav, out_path, fmt,
                extra_args=["-af", f"loudnorm=I={target}:LRA={lra}:TP={tp}"])
        return out_path

    logger.info(f"loudnorm 测量: I={stats['input_i']} LRA={stats['input_lra']} TP={stats['input_tp']}")

    # ---- Pass 2: 线性归一 ----
    af = (
        f"loudnorm="
        f"I={target}:LRA={lra}:TP={tp}:"
        f"measured_I={stats['input_i']}:"
        f"measured_LRA={stats['input_lra']}:"
        f"measured_TP={stats['input_tp']}:"
        f"measured_thresh={stats['input_thresh']}:"
        f"offset={stats['target_offset']}:"
        f"linear=true:print_format=summary"
    )
    _encode(in_wav, out_path, fmt, extra_args=["-af", af])
    return out_path


def _parse_loudnorm_json(stderr: str) -> dict | None:
    """从 ffmpeg stderr 提取 loudnorm JSON 输出。"""
    import json
    import re
    # 找最后一个 {...} 块
    matches = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr, re.DOTALL)
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        return None


# ============================================================
# 探测音频元信息
# ============================================================

def probe_duration(path: str | Path) -> float:
    """获取音频时长(秒)。优先 ffprobe,退化用 ffmpeg。"""
    path = str(path)
    fp = ffprobe_bin()
    if fp:
        try:
            out = subprocess.run(
                [fp, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, check=True,
            )
            return float(out.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            pass

    # 退化:ffmpeg 解析
    out = subprocess.run(
        [ffmpeg_bin(), "-hide_banner", "-i", path],
        capture_output=True, text=True,
    )
    import re
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", out.stderr)
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        return h * 3600 + mi * 60 + s
    return 0.0
