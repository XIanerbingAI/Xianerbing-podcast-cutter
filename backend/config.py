"""饼哥帮你剪播客(Xianerbing-podcast-cutter) 全局配置。

通过环境变量 / .env 文件覆盖默认值。所有运行期可调参数都集中在这里。
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

for _d in (UPLOAD_DIR, OUTPUT_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# 把 .env 里的 HuggingFace 镜像变量注入 os.environ,
# 让 huggingface_hub 下载模型时走国内镜像(hf-mirror.com),解决国内下载卡住。
# python-dotenv 只把变量塞进 os.environ 而不覆盖已存在的系统变量。
def _load_env_to_environ() -> None:
    import os
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import dotenv_values
        for k, v in dotenv_values(env_file).items():
            if v is None:
                continue
            # HF 镜像相关 + 其它可能需要的传递变量
            if k.startswith("HF_") or k.startswith("HUGGINGFACE_") or k in ("HF_HUB_ENABLE_HF_TRANSFER",):
                os.environ.setdefault(k, v)
    except Exception:
        pass


_load_env_to_environ()


class EditStrength(str, Enum):
    """剪辑力度。"""
    CONSERVATIVE = "conservative"  # 只删最确定的口癖/明显重复
    BALANCED = "balanced"          # 默认:删明确口癖+重复,语义守卫保含义
    AGGRESSIVE = "aggressive"      # 删所有检测到的填充词/口癖


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ===== 基础 =====
    app_name: str = "饼哥帮你剪播客"
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_mb: int = 2048

    # ===== 转写 =====
    whisper_model: str = "medium"            # tiny/base/small/medium/large-v3
    whisper_device: str = "auto"             # auto / cpu / cuda
    whisper_compute: str = "int8"            # int8 / int8_float16 / float16
    whisper_language: str = "zh"
    beam_size: int = 5
    vad_filter: bool = True                  # faster-whisper 内置 VAD 去静音
    # 简体中文引导 prompt:让 Whisper 直接输出简体(而非繁体),
    # 同时给出口语场景的词汇提示,提升词边界稳定性。
    whisper_initial_prompt: str = (
        "以下是普通话的句子,使用简体中文输出。"
        "这是一段播客对话,讨论健康、医疗、生活方式等话题。"
        "说话自然,有口语化的表达,比如然后、就是、这个、那个、其实。"
    )

    # ===== 剪辑力度(默认) =====
    default_strength: EditStrength = EditStrength.BALANCED

    # ===== 平滑剪辑参数(smooth.py 核心)=====
    crossfade_ms: int = 40                   # 等功率交叉淡化长度(毫秒,中文连读需更长)
    zero_cross_search_ms: int = 5            # 零交叉吸附最大搜索半径
    acoustic_valley_radius_ms: int = 100     # 声学边界精确化:能量谷搜索半径
    energy_guard_threshold: float = 0.6      # 能量守卫:内/外比超此值判误删
    internal_gap_ms: int = 40                # 句内删除后插入的短静音(模拟自然停顿)
    breath_min_ms: int = 80                  # ≥此长度视为呼吸(保留)
    silence_min_ms: int = 200                # ≥此长度视为自然停顿
    fill_floor_noise: bool = True            # 用房间底噪填充而非纯零
    target_pause_ms: int = 280               # 句间停顿目标长度(节奏保持)
    min_cut_ms: int = 60                     # 短于此不剪(避免碎切)

    # ===== 响度归一 =====
    loudnorm_target_lufs: float = -16.0      # EBU R128 播客标准
    loudnorm_true_peak: float = -1.5
    loudnorm_lra: float = 11.0

    # ===== LLM 复核(可选)=====
    llm_base_url: Optional[str] = None       # OpenAI 兼容,如 https://api.deepseek.com/v1
    llm_api_key: Optional[str] = None
    llm_model: str = "deepseek-v4-flash"
    llm_timeout_sec: int = 20
    llm_enabled: bool = False                # 仅当 base_url+key 都配置才生效

    # ===== 任务队列 =====
    use_celery: bool = False                 # False=进程内 BackgroundTasks
    celery_broker: str = "redis://localhost:6379/0"


settings = Settings()
