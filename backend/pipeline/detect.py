"""检测主模块 —— 把转写词序列 + 词典 + 语义守卫 → EditItem 候选清单。

输出:未应用力度的"全量候选"(含所有 reason/confidence),
由 editplan.py 再按力度 + 用户决策裁剪。这样切换力度无需重跑检测。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from loguru import logger

from backend.models import (
    Confidence,
    CutPosition,
    CutReason,
    EditItem,
    Segment,
    WordToken,
)
from backend.pipeline.lexicon import (
    ALL_FILLERS,
    confidence_ge,
    get_policy,
)
from backend.pipeline.semantic import (
    Verdict,
    build_context,
    judge_discourse_marker,
)


@dataclass(slots=True)
class DetectionConfig:
    strength: str = "balanced"
    pause_list: list[tuple[float, float, bool]] | None = None  # [(s,e,is_breath)]


def detect(segments: list[Segment], cfg: DetectionConfig) -> list[EditItem]:
    """主检测入口。返回所有候选 EditItem(keep 默认按力度策略设定)。"""
    policy = get_policy(cfg.strength)
    pauses = cfg.pause_list or []
    items: list[EditItem] = []

    for seg in segments:
        words = seg.words
        if not words:
            continue
        items.extend(_detect_fillers(seg, words, policy, pauses))
        items.extend(_detect_repeats(seg, words, policy))

    # 去重/合并重叠区间(同一时间段可能被多个规则命中,取最严)
    items = _merge_overlapping(items)
    logger.info(f"检测完成:力度={cfg.strength}, 候选={len(items)} 条")
    return items


# ============================================================
# 口癖/话语标记检测
# ============================================================

def _detect_fillers(
    seg: Segment,
    words: list[WordToken],
    policy,
    pauses: list[tuple[float, float, bool]],
) -> list[EditItem]:
    items: list[EditItem] = []
    for idx, w in enumerate(words):
        text = _normalize(w.text)
        fdef = ALL_FILLERS.get(text)
        if fdef is None:
            # 处理"嗯。" 这类带尾标点的
            bare = text.rstrip("。!?,.;!? ")
            if bare and bare in ALL_FILLERS:
                fdef = ALL_FILLERS[bare]
        if fdef is None:
            continue

        # 力度过滤:话语标记在保守模式不检测
        if fdef.reason == CutReason.DISCOURSE and not policy.detect_discourse:
            continue
        if fdef.reason == CutReason.FILLER and not policy.detect_filler:
            continue

        # 太短(噪声)跳过
        dur_ms = w.duration * 1000
        if dur_ms < policy.min_filler_ms and fdef.reason == CutReason.FILLER:
            continue

        # 语义守卫:歧义词必走
        confidence = fdef.confidence
        keep = False
        explanation = ""

        if fdef.ambiguous:
            ctx = build_context(w, words, idx, pauses)
            verdict = judge_discourse_marker(ctx)
            if verdict == Verdict.KEEP:
                keep = True
                confidence = Confidence.HIGH
                explanation = f"语义守卫:{text} 此处承担语义(连接/强调),保留"
            elif verdict == Verdict.CUT:
                keep = False
                explanation = f"语义守卫:{text} 此处为口癖填充,建议删除"
            else:  # AMBIGUOUS
                explanation = f"语义守卫:无法确定,{text} 疑为口癖"
                # 关键:均衡力度下话语标记 AMBIGUOUS 默认删(交给能量守卫兜底)
                # 能量守卫会在渲染时拦截切到有声处的误删,所以这里删风险可控。
                # 若开了 LLM 复核,会先复核再决定;否则直接删。
                keep = False
                confidence = Confidence.MEDIUM
        else:
            keep = False
            explanation = f"{text} 为语气填充词"

        # 力度阈值:话语标记在保守力度下要求 HIGH 才删;
        # 均衡/激进允许 MEDIUM(由上面的 AMBIGUOUS 逻辑 + 能量守卫共同决定)
        if not keep and not confidence_ge(confidence, policy.cut_threshold):
            # 保守力度:话语标记达不到 HIGH 就保留
            keep = True

        position = _decide_position(w, idx, words, pauses)
        ctx_before, ctx_after = _build_text_context(words, idx)
        items.append(
            EditItem(
                id=f"ed_{uuid.uuid4().hex[:10]}",
                start=w.start,
                end=w.end,
                original_text=w.text,
                reason=fdef.reason,
                confidence=confidence,
                position=position,
                keep=keep,
                explanation=explanation,
                context_before=ctx_before,
                context_after=ctx_after,
            )
        )
    return items


# ============================================================
# 重复词 / 口吃检测
# ============================================================

def _detect_repeats(seg: Segment, words: list[WordToken], policy) -> list[EditItem]:
    if not policy.detect_repeat:
        return []
    items: list[EditItem] = []
    n = len(words)
    i = 0
    while i < n:
        w = words[i]
        text = _normalize(w.text)
        if not text:
            i += 1
            continue
        # 向后数连续相同的词
        j = i + 1
        while j < n and _normalize(words[j].text) == text:
            # 间隔不能太大(>0.6s 视为不同次重复)
            if words[j].start - words[j - 1].end > 0.6:
                break
            j += 1
        run_len = j - i
        if run_len >= policy.repeat_min_count and len(text) >= 1:
            # 保留第一个,删 i+1..j-1
            keep_explanation = "保留首个词,删除其后重复"
            reason = CutReason.REPEAT if run_len <= 3 else CutReason.STUTTER
            conf = Confidence.HIGH if run_len >= 3 else Confidence.MEDIUM
            for k in range(i + 1, j):
                # 重复词之间的微小停顿也一并纳入删除区间
                start = words[k - 1].end if k > i + 1 else words[k].start
                # 区间为当前词本身 + 与前一词的间隔
                start = words[k].start
                end = words[k].end
                position = _decide_position(words[k], k, words, [])
                ctx_before, ctx_after = _build_text_context(words, k)
                items.append(
                    EditItem(
                        id=f"ed_{uuid.uuid4().hex[:10]}",
                        start=start,
                        end=end,
                        original_text=words[k].text,
                        reason=reason,
                        confidence=conf,
                        position=position,
                        keep=False,
                        explanation=f"重复词“{text}”(第{k - i + 1}次),{keep_explanation}",
                        context_before=ctx_before,
                        context_after=ctx_after,
                    )
                )
            i = j
        else:
            i += 1
    return items


# ============================================================
# 辅助
# ============================================================

# OpenCC 繁→简转换器(惰性加载,兼容转换失败)
_T2S_CONVERTER = None


def _to_simplified(s: str) -> str:
    """繁体转简体。Whisper 中文常输出繁体(然後/濫用),词典是简体,需归一化。"""
    global _T2S_CONVERTER
    if _T2S_CONVERTER is None:
        try:
            from opencc import OpenCC
            _T2S_CONVERTER = OpenCC("t2s")  # 繁→简
        except Exception:
            _T2S_CONVERTER = False  # 标记不可用,后续直接返回原文
    if _T2S_CONVERTER is False:
        return s
    try:
        return _T2S_CONVERTER.convert(s)
    except Exception:
        return s


def _normalize(s: str) -> str:
    """归一化:去标点空格 + 繁转简。"""
    s = s.strip().strip("。!?,.;!? 、~ ")
    return _to_simplified(s)


def _build_text_context(words: list[WordToken], idx: int, span: int = 6) -> tuple[str, str]:
    """取第 idx 个词前后各 span 个词的文本,作为人工审核上下文。

    用时间戳进一步剔除过长停顿(避免上下文跨越很远的句子)。
    """
    # 前 6 个词(若与前一词间隔 >1.5s 视为跨句,截断)
    before_parts: list[str] = []
    i = idx - 1
    while i >= 0 and len(before_parts) < span:
        if words[idx].start - words[i].end > 1.5:
            break
        before_parts.append(words[i].text)
        i -= 1
    before_parts.reverse()
    ctx_before = "".join(before_parts)

    # 后 6 个词
    after_parts: list[str] = []
    j = idx + 1
    while j < len(words) and len(after_parts) < span:
        if words[j].start - words[idx].end > 1.5:
            break
        after_parts.append(words[j].text)
        j += 1
    ctx_after = "".join(after_parts)

    return ctx_before, ctx_after


def _decide_position(
    w: WordToken, idx: int, words: list[WordToken], pauses
) -> CutPosition:
    """决定删除项在句中位置,供 smooth.py 选择平滑策略。"""
    prev_pause = idx == 0 or _is_after_pause(w.start, pauses)
    next_pause = idx == len(words) - 1 or _is_before_pause(w.end, pauses)
    if prev_pause and next_pause:
        return CutPosition.UTTERANCE_EDGE
    if prev_pause or next_pause:
        return CutPosition.SENTENCE_BOUNDARY
    return CutPosition.SENTENCE_INTERNAL


def _is_after_pause(t: float, pauses, thr: float = 0.12) -> bool:
    for ps, pe, _ in pauses or []:
        if pe <= t <= pe + thr + 0.3 and (t - pe) <= 0.3:
            return (t - pe) >= 0.05
    return False


def _is_before_pause(t: float, pauses, thr: float = 0.12) -> bool:
    for ps, pe, _ in pauses or []:
        if ps >= t >= ps - thr - 0.3 and (ps - t) <= 0.3:
            return (ps - t) >= 0.05
    return False


def _merge_overlapping(items: list[EditItem]) -> list[EditItem]:
    """合并时间重叠的候选。优先保留 reason 优先级高的;若都"删"则合并区间。"""
    if not items:
        return []
    items = sorted(items, key=lambda it: it.start)
    # 简化:对重叠的,若两者都 keep=False 则合并成一个(取并集区间,reason 取前者);
    # 若一方 keep=True(保留)则压制同区间其它删除项(保留优先,避免误删)。
    merged: list[EditItem] = []
    for it in items:
        if not merged:
            merged.append(it)
            continue
        last = merged[-1]
        if it.start < last.end - 0.02:  # 重叠
            if last.keep or it.keep:
                # 保留优先:把删除项标记为 keep(被保护)
                if not last.keep and it.keep:
                    last.keep = True
                    last.explanation = (last.explanation + " | 与保留项重叠,降级为保留").strip(" |")
                continue
            else:
                # 都删 → 合并
                last.end = max(last.end, it.end)
                last.explanation += f"; 合并[{it.reason.value}]"
        else:
            merged.append(it)
    return merged
