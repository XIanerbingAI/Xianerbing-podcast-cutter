"""数据模型(供整个 pipeline 与 API 共用)。

用纯 dataclass / pydantic 双轨:内部计算用 dataclass(零开销),
API 边界用 pydantic v2 做校验。EditItem 是贯穿全流程的核心数据结构。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class CutReason(str, Enum):
    """删除原因分类。"""
    FILLER = "filler"               # 语气词填充(嗯/啊/呃)
    DISCOURSE = "discourse"         # 口语化话语标记(然后/就是/那个)→口癖用法
    REPEAT = "repeat"               # 连续重复词(我我我)
    STUTTER = "stutter"             # 口吃式卡顿
    FALSE_START = "false_start"     # 话头废弃(我想...我觉得)


class Confidence(str, Enum):
    """置信度等级,决定是否需要 LLM 复核。"""
    HIGH = "high"       # 规则可定,直接删
    MEDIUM = "medium"   # 有歧义,可删但建议复核
    LOW = "low"         # 仅激进模式删


class CutPosition(str, Enum):
    """删除项在句子中的位置 —— 决定平滑策略。"""
    SENTENCE_INTERNAL = "internal"  # 句中 → 直接拼接 + 交叉淡化
    SENTENCE_BOUNDARY = "boundary"  # 句间(前后是停顿)→ 底噪填充保节奏
    UTTERANCE_EDGE = "edge"         # 句首/句末边缘 → 删除+缩短停顿


@dataclass(slots=True)
class WordToken:
    """转写出的一个词/字,带时间戳。"""
    text: str
    start: float          # 秒
    end: float            # 秒
    probability: float = 1.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class Segment:
    """转写片段。"""
    start: float
    end: float
    text: str
    words: list[WordToken] = field(default_factory=list)


@dataclass(slots=True)
class Pause:
    """两个语音段之间的停顿/呼吸区间。"""
    start: float
    end: float
    is_breath: bool = False   # True=换气(保留),False=纯静音停顿

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class EditItem:
    """一条剪辑建议 —— 贯穿全流程的核心结构。"""
    id: str
    start: float                      # 删除区间起(秒)
    end: float                        # 删除区间止(秒)
    original_text: str                # 被删原文
    reason: CutReason
    confidence: Confidence
    position: CutPosition
    keep: bool = True                 # 用户审核状态:True=保留(默认不删),False=确认删除
    llm_reviewed: bool = False        # 是否经过 LLM 复核
    llm_verdict: Optional[Literal["keep", "cut"]] = None
    explanation: str = ""             # 人类可读原因
    # 上下文(人工审核用):被删词前后各若干字
    context_before: str = ""
    context_after: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


# ============ API Schemas (pydantic v2) ============

class EditItemOut(BaseModel):
    id: str
    start: float
    end: float
    original_text: str
    reason: CutReason
    confidence: Confidence
    position: CutPosition
    keep: bool
    llm_reviewed: bool
    llm_verdict: Optional[Literal["keep", "cut"]] = None
    explanation: str
    context_before: str = ""
    context_after: str = ""
    duration: float = Field(description="删除区间时长(秒)")


class AnalysisResult(BaseModel):
    job_id: str
    filename: str
    duration_sec: float
    language: str
    transcript: str
    segments_count: int
    edits: list[EditItemOut]
    stats: dict


class EditDecision(BaseModel):
    """用户对单条剪辑的勾选。"""
    id: str
    keep: bool   # True=保留(撤销删除),False=删除
    mode: Literal["conservative", "standard", "clean"] = "standard"


class RenderRequest(BaseModel):
    job_id: str
    decisions: list[EditDecision]
    apply_loudnorm: bool = False   # 默认不动响度/音质,用户可手动开启


class JobStatus(BaseModel):
    job_id: str
    stage: str
    progress: float = 0.0
    message: str = ""
    done: bool = False
    error: Optional[str] = None
