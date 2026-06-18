"""语义守卫 —— 严格按中文语义判定歧义话语标记是"口癖"还是"承担语义"。

这是"不误删有意义词语"的关键。规则层即可定绝大多数情况,
LLM 复核(semantic_llm.py)仅处理规则无法裁决的少数歧义。

判定逻辑(以"然后"为例):
1. 上下文窗口:取该词前后各 N 字
2. 规则集:
   a. 句中重复出现 → 口癖(如"然后然后")→ CUT
   b. 独立成句(该词前后是标点/停顿,本身构成一个语调单位)→ 口癖 → CUT
   c. 承接:其后紧跟动词/动作,且前文是完整子句 → 连接词 → KEEP
   d. 起句:位于句首且后接完整陈述 → 多为时间/逻辑承接 → KEEP
   e. 仅作填充拖音(时长异常长,如 >0.4s)→ 口癖 → CUT
3. 规则无法裁决 → 返回 AMBIGUOUS,交给 LLM 或按力度保守处理

"就是":接"是 X"(判断/定义)→ KEEP;接名词作强调"就是这个人"→ KEEP;
作填充"就是说那个"→ CUT。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from backend.models import WordToken

# 判定结果
class Verdict(str, Enum):
    CUT = "cut"        # 口癖,可删
    KEEP = "keep"      # 承担语义,保留
    AMBIGUOUS = "ambiguous"  # 规则无法定,需 LLM 或保守处理


# 词性提示词集合(粗粒度,无需精确分词器也能工作)
# 这些字符常出现在动词/动作前
VERB_HINTS = set("去来是说看听想做会能给让叫用买吃喝玩走跑写读买打"
                 "开始继续决定认为觉得认为认为发现觉得考虑准备打算")
# 名词/实体提示
NOUN_HINTS = set("的人事地物时间年月日号分秒块元件本书种类样"
                 "个只条台部篇场")
# 时间承接词后常接的动词
TEMPORAL_VERBS = {"开始", "继续", "接着", "马上", "就", "去", "来", "做"}

# 标点(转写后如无标点,我们用停顿/语调近似)
SOFT_PUNCT = set("。!?;,.!?;,")


@dataclass(slots=True)
class GuardContext:
    """守卫上下文。"""
    word: str
    word_start: float
    word_end: float
    word_duration: float
    prev_text: str       # 该词之前的文本(同段内,已截断)
    next_text: str       # 该词之后的文本(同段内,已截断)
    prev_is_pause: bool  # 前面是否是较长停顿(句首)
    next_is_pause: bool  # 后面是否是较长停顿(句末)
    sentence_start: bool  # 是否在句首
    sentence_end: bool    # 是否在句末


def judge_discourse_marker(ctx: GuardContext) -> Verdict:
    """话语标记(然后/就是/那个…)的语义守卫主入口。

    优先级从高到低,命中即返回。
    """
    w = ctx.word
    dur = ctx.word_duration

    # --- 规则 1:重复(如"然后然后")→ 口癖 ---
    if w in ("然后", "就是", "那个", "这个", "那么"):
        # 看 prev/next 文本是否以本词开头/结尾(去标点空格)
        p = ctx.prev_text.strip().rstrip("。!?,.;!? ")
        n = ctx.next_text.strip().lstrip("。!?,.;!? ")
        if p.endswith(w) or n.startswith(w):
            return Verdict.CUT

    # --- 规则 2:独立成句(前后都是停顿)→ 口癖拖音 ---
    if ctx.prev_is_pause and ctx.next_is_pause:
        # "嗯然后。" 这种独立成调的,且本身是话语标记 → 多为口癖
        return Verdict.CUT

    # --- 规则 3:异常拖音(>0.4s)且非句首承接 → 口癖 ---
    if dur > 0.45 and not ctx.sentence_start:
        # 拖长的话语标记多为填充
        if w in ("然后", "就是", "那个", "这个", "那么", "其实"):
            return Verdict.CUT

    # 词专属规则
    if w == "然后":
        return _judge_ranhao(ctx)
    if w == "就是":
        return _judge_jiushi(ctx)
    if w in ("那个", "这个"):
        return _judge_zhege_nage(ctx)
    if w == "那么":
        return _judge_name(ctx)
    if w in ("其实", "基本上", "反正"):
        return _judge_hedge(ctx)
    if w in ("对吧", "你知道吗", "怎么说呢", "之类的"):
        return _judge_tag(ctx)

    return Verdict.AMBIGUOUS


# ============================================================
# 词专属规则
# ============================================================

def _first_non_space(s: str) -> str:
    for ch in s:
        if not ch.isspace() and ch not in SOFT_PUNCT:
            return ch
    return ""


def _judge_ranhao(ctx: GuardContext) -> Verdict:
    """然后:时间/逻辑承接 → KEEP;纯填充 → CUT。"""
    nxt = _first_non_space(ctx.next_text)
    next_strip = ctx.next_text.lstrip()
    # 承接副词:然后就/然后才/然后又/然后便... → 明确叙事承接 → KEEP
    CONNECTIVE_ADVERBS = set("就才又也还便才能会")
    if ctx.sentence_start:
        # 句首"然后..." 后接动作/陈述 → 多为承接叙事 → KEEP
        if nxt and nxt not in SOFT_PUNCT:
            return Verdict.KEEP
    # 后接动词/动作 → 承接
    if nxt in VERB_HINTS or next_strip[:2] in TEMPORAL_VERBS:
        return Verdict.KEEP
    # 后接承接副词(然后就/然后才...)→ KEEP
    if nxt in CONNECTIVE_ADVERBS:
        return Verdict.KEEP
    # 后接"就是说/我觉得"这类 → 填充链
    if next_strip.startswith(("就是说", "我觉得", "我想", "我认为")):
        return Verdict.CUT
    # 前文是完整子句(以标点结尾 或 前面是停顿)且后接实词 → 连接词 KEEP
    prev_ends_clause = ctx.prev_is_pause or bool(ctx.prev_text.rstrip() and ctx.prev_text.rstrip()[-1] in SOFT_PUNCT)
    if prev_ends_clause and nxt and nxt not in SOFT_PUNCT:
        return Verdict.KEEP
    return Verdict.AMBIGUOUS


def _judge_jiushi(ctx: GuardContext) -> Verdict:
    """就是:判断/定义 → KEEP;填充 → CUT。"""
    n = ctx.next_text.lstrip()
    # "就是 X" 判断/强调 → KEEP
    if n.startswith("是"):
        return Verdict.KEEP
    # "就是说" → 填充链 → CUT
    if n.startswith("说"):
        return Verdict.CUT
    # "就是那个/就是这个" → 填充链 → CUT
    if n.startswith(("那个", "这个", "嘛", "啊", "嗯")):
        return Verdict.CUT
    # 句末独立 → 口癖
    if ctx.next_is_pause and not n:
        return Verdict.CUT
    # 后接名词/数量 → 强调 → KEEP
    nxt = _first_non_space(n)
    if nxt in NOUN_HINTS:
        return Verdict.KEEP
    return Verdict.AMBIGUOUS


def _judge_zhege_nage(ctx: GuardContext) -> Verdict:
    """这个/那个:指示 → KEEP;填充 → CUT。"""
    n = ctx.next_text.lstrip()
    # "那个 人/时候/东西" → 指示 → KEEP
    nxt = _first_non_space(n)
    if nxt in NOUN_HINTS:
        return Verdict.KEEP
    # "那个那个" → 已在重复规则处理
    # 后接"就是说/我觉得" → 填充链
    if n.startswith(("就是说", "我觉得", "我想")):
        return Verdict.CUT
    # 句首且后接名词短语 → 指示 KEEP
    if ctx.sentence_start and nxt and nxt not in SOFT_PUNCT and nxt not in VERB_HINTS:
        return Verdict.KEEP
    # 其余多为口癖填充
    if ctx.word_duration > 0.3:
        return Verdict.CUT
    return Verdict.AMBIGUOUS


def _judge_name(ctx: GuardContext) -> Verdict:
    """那么:逻辑推导 → KEEP;填充 → CUT。"""
    n = ctx.next_text.lstrip()
    nxt = _first_non_space(n)
    # "那么 X 就是/我们/你" → 推导承接 → KEEP
    if n.startswith(("我们", "你们", "他们", "现在", "接下来", "问题")):
        return Verdict.KEEP
    if nxt in VERB_HINTS and ctx.sentence_start:
        return Verdict.KEEP
    # 拖音/独立 → 填充
    if ctx.word_duration > 0.35 or (ctx.prev_is_pause and ctx.next_is_pause):
        return Verdict.CUT
    return Verdict.AMBIGUOUS


def _judge_hedge(ctx: GuardContext) -> Verdict:
    """其实/基本上/反正:语气副词,常承语义但口语高频作填充。"""
    n = ctx.next_text.lstrip()
    nxt = _first_non_space(n)
    # 后接陈述内容 → 多为副词修饰,承语义 → KEEP
    if nxt and nxt not in SOFT_PUNCT:
        # 但若紧跟另一个话语标记 → 填充链
        if n.startswith(("然后", "就是", "那个", "这个")):
            return Verdict.CUT
        return Verdict.KEEP
    return Verdict.CUT


def _judge_tag(ctx: GuardContext) -> Verdict:
    """对吧/你知道吗:句末语气标签。"""
    # 句末独立出现 → 多为口癖 → CUT(但有人用作互动,所以 AMBIGUOUS 更稳)
    if ctx.next_is_pause:
        return Verdict.AMBIGUOUS
    return Verdict.KEEP


# ============================================================
# 上下文构建
# ============================================================

def build_context(
    word: WordToken,
    seg_words: list[WordToken],
    word_idx: int,
    pauses: list[tuple[float, float, bool]],
    pause_threshold_s: float = 0.25,
) -> GuardContext:
    """从词序列与停顿列表构建守卫上下文。

    Args:
        pauses: [(start,end,is_breath), ...] 已排序
        pause_threshold_s: 视为"句间停顿"的最小长度
    """
    # 前后文本(最多各 12 字)
    prev_words = seg_words[max(0, word_idx - 8): word_idx]
    next_words = seg_words[word_idx + 1: word_idx + 9]
    prev_text = "".join(w.text for w in prev_words)[-12:]
    next_text = "".join(w.text for w in next_words)[:12]

    # 前后是否是较长停顿
    def _near_pause(t: float, before: bool) -> bool:
        for ps, pe, _isb in pauses:
            if before and abs(pe - word.start) < 0.05 and (word.start - pe) >= 0:
                return (word.start - pe) >= pause_threshold_s * 0.5  # 紧邻停顿
            if not before and abs(ps - word.end) < 0.05 and (ps - word.end) >= 0:
                return (ps - word.end) >= pause_threshold_s * 0.5
        return False

    prev_is_pause = _near_pause(word.start, before=True)
    next_is_pause = _near_pause(word.end, before=False)

    # 句首:本词是段内第一个词,或前面是停顿
    sentence_start = (word_idx == 0) or prev_is_pause
    sentence_end = (word_idx == len(seg_words) - 1) or next_is_pause

    dur = max(0.0, word.end - word.start)

    return GuardContext(
        word=word.text,
        word_start=word.start,
        word_end=word.end,
        word_duration=dur,
        prev_text=prev_text,
        next_text=next_text,
        prev_is_pause=prev_is_pause,
        next_is_pause=next_is_pause,
        sentence_start=sentence_start,
        sentence_end=sentence_end,
    )
