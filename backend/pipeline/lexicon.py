"""中文口癖/填充词词典 + 力度阈值配置。

设计原则(严格按中文语义):
- FILLER(纯填充,无语义):嗯/啊/呃/额/唉/诶 → 任何力度都可删
- DISCOURSE(话语标记,有歧义):然后/就是/那么/这个/那个/基本上/其实/对吧/你知道吗
  → 需语义守卫判定:作连接词或承担语义时保留;作填充时删
- REPEAT(连续重复):"我 我 我 想说" → 保留一个,删其余
- STUTTER(口吃卡顿):"我 我 我想" 同上但置信度略低
- FALSE_START(话头废弃):"我想... 我觉得应该是 X" → 删前半句

每个词带"歧义度":HIGH=无歧义,LOW=有歧义需守卫/LLM 复核。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from backend.models import CutReason, Confidence


@dataclass(frozen=True, slots=True)
class FillerDef:
    word: str
    reason: CutReason
    confidence: Confidence           # 该词作为口癖的默认置信度
    ambiguous: bool = False          # True=需语义守卫复核
    aliases: tuple[str, ...] = ()


# ============================================================
# 词典
# ============================================================

# 纯语气词填充 —— 几乎不承担语义,任何力度可删
PURE_FILLERS: list[FillerDef] = [
    FillerDef("嗯", CutReason.FILLER, Confidence.HIGH),
    FillerDef("呃", CutReason.FILLER, Confidence.HIGH),
    FillerDef("啊", CutReason.FILLER, Confidence.MEDIUM, ambiguous=True),   # "啊?"反问/感叹 保留
    FillerDef("额", CutReason.FILLER, Confidence.HIGH),
    FillerDef("唉", CutReason.FILLER, Confidence.MEDIUM, ambiguous=True),   # 叹气表情绪
    FillerDef("诶", CutReason.FILLER, Confidence.HIGH),
    FillerDef("哦", CutReason.FILLER, Confidence.MEDIUM, ambiguous=True),   # "哦?"恍然/应答
    FillerDef("噢", CutReason.FILLER, Confidence.HIGH),
    FillerDef("哈", CutReason.FILLER, Confidence.LOW,    ambiguous=True),   # "哈哈"笑声保留
    FillerDef("嘛", CutReason.FILLER, Confidence.LOW,    ambiguous=True),
    FillerDef("呢", CutReason.FILLER, Confidence.LOW,    ambiguous=True),   # 疑问语气词
]

# 话语标记 —— 口语高频口癖,但承语义可能性高,必走语义守卫
DISCOURSE_MARKERS: list[FillerDef] = [
    FillerDef("然后", CutReason.DISCOURSE, Confidence.MEDIUM, ambiguous=True),
    FillerDef("就是", CutReason.DISCOURSE, Confidence.MEDIUM, ambiguous=True),
    FillerDef("那么", CutReason.DISCOURSE, Confidence.MEDIUM, ambiguous=True),
    FillerDef("这个", CutReason.DISCOURSE, Confidence.MEDIUM, ambiguous=True),
    FillerDef("那个", CutReason.DISCOURSE, Confidence.MEDIUM, ambiguous=True),
    FillerDef("基本上", CutReason.DISCOURSE, Confidence.MEDIUM, ambiguous=True),
    FillerDef("其实", CutReason.DISCOURSE, Confidence.LOW, ambiguous=True),
    FillerDef("对吧", CutReason.DISCOURSE, Confidence.LOW, ambiguous=True),
    FillerDef("你知道吗", CutReason.DISCOURSE, Confidence.LOW, ambiguous=True),
    FillerDef("怎么说呢", CutReason.DISCOURSE, Confidence.LOW, ambiguous=True),
    FillerDef("之类的", CutReason.DISCOURSE, Confidence.LOW, ambiguous=True),
    FillerDef("反正", CutReason.DISCOURSE, Confidence.LOW, ambiguous=True),
]

ALL_FILLERS: dict[str, FillerDef] = {
    f.word: f for f in (PURE_FILLERS + DISCOURSE_MARKERS)
}
# 加上 alias(常见转写变体)
_VARIANTS = {
    "嗯嗯": "嗯", "嗯...": "嗯", "呃呃": "呃", "啊啊": "啊",
    "然后然后": "然后", "就是就是": "就是", "那那那个": "那个",
}
for v, base in _VARIANTS.items():
    if base in ALL_FILLERS:
        ALL_FILLERS[v] = ALL_FILLERS[base]


# ============================================================
# 力度配置 —— 决定哪些 reason/confidence 会被检出/默认删除
# ============================================================

@dataclass(slots=True)
class StrengthPolicy:
    name: str
    # 是否检测某 reason
    detect_filler: bool = True
    detect_discourse: bool = True
    detect_repeat: bool = True
    detect_false_start: bool = True
    # 默认删除的最低置信度阈值(高于等于此值才默认删,其余 keep=True 待人工/LLM)
    cut_threshold: Confidence = Confidence.HIGH
    # 重复词:连续重复 N 次及以上才处理
    repeat_min_count: int = 2
    # 单次口癖最小持续 ms(更短视为噪声不处理)
    min_filler_ms: int = 80


# 力度 → 策略
POLICIES: dict[str, StrengthPolicy] = {
    "conservative": StrengthPolicy(
        name="conservative",
        detect_discourse=False,            # 保守:不动话语标记(歧义大)
        detect_false_start=False,
        cut_threshold=Confidence.HIGH,     # 只删 HIGH
        repeat_min_count=3,
        min_filler_ms=100,
    ),
    "balanced": StrengthPolicy(
        name="balanced",
        detect_discourse=True,             # 开启,语义守卫判定后由能量守卫兜底
        detect_false_start=True,
        cut_threshold=Confidence.MEDIUM,   # MEDIUM 及以上删(HIGH 连接词守卫会拦,AMBIGUOUS 默认删由能量守卫兜底)
        repeat_min_count=2,
        min_filler_ms=80,
    ),
    "aggressive": StrengthPolicy(
        name="aggressive",
        detect_discourse=True,
        detect_false_start=True,
        cut_threshold=Confidence.LOW,      # 连 LOW 都删
        repeat_min_count=2,
        min_filler_ms=60,
    ),
}


def get_policy(strength: str) -> StrengthPolicy:
    return POLICIES.get(strength, POLICIES["balanced"])


# 置信度排序便于比较
_CONF_ORDER = {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}


def confidence_ge(a: Confidence, b: Confidence) -> bool:
    """a 是否 >= b。"""
    return _CONF_ORDER[a] >= _CONF_ORDER[b]
