"""统一获取 ffmpeg/ffprobe 二进制路径。

优先使用 imageio-ffmpeg 打包的静态二进制,这样用户无需手动安装 ffmpeg。
若系统 PATH 上有原生 ffmpeg(可能带更多编解码器),优先使用之。
"""
from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path

# 让 ffmpeg-python / pydub 都能拿到二进制
_FFMPEG_BIN: str | None = None
_FFPROBE_BIN: str | None = None


def _find_ffmpeg() -> str:
    # 1) 系统 PATH
    sys_bin = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if sys_bin:
        return sys_bin
    # 2) imageio-ffmpeg 打包二进制
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "未找到 ffmpeg。请 pip install imageio-ffmpeg,或手动安装 ffmpeg 并加入 PATH。"
        ) from e


def _find_ffprobe() -> str | None:
    # ffprobe 一般随 ffmpeg 分发;imageio-ffmpeg 不带 ffprobe,需系统提供。
    # 若缺失,我们用 ffmpeg 自身探测(见 audio_meta),功能不致全损。
    return shutil.which("ffprobe") or shutil.which("ffprobe.exe")


@lru_cache(maxsize=1)
def ffmpeg_bin() -> str:
    global _FFMPEG_BIN
    if _FFMPEG_BIN is None:
        _FFMPEG_BIN = _find_ffmpeg()
        # 同步设置给 pydub
        os.environ.setdefault("FFMPEG_BINARY", _FFMPEG_BIN)
    return _FFMPEG_BIN


@lru_cache(maxsize=1)
def ffprobe_bin() -> str | None:
    global _FFPROBE_BIN
    if _FFPROBE_BIN is None:
        _FFPROBE_BIN = _find_ffprobe()
    return _FFPROBE_BIN


def ensure_ffmpeg() -> None:
    """启动期检查。"""
    p = Path(ffmpeg_bin())
    if not p.exists():
        raise RuntimeError(f"ffmpeg 二进制不存在: {p}")
