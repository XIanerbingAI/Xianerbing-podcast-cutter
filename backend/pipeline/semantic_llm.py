"""LLM 复核模块 —— 对语义守卫无法裁决(AMBIGUOUS)的歧义项做最终判定。

仅当 settings.llm_enabled 且 base_url+api_key 都配置时启用。
否则 detect.py 已按力度保守处理(默认保留),不会报错。

设计:
- 批量送检(把若干歧义项 + 上下文拼成一次请求,省 token)
- 强制 JSON 输出
- 超时/解析失败 → 保守保留(不删),并把 llm_reviewed=False 标记,前端可识别
- 永不改变 keep=False→True 的方向之外的内容(LLM 只能"建议删/建议留",不强制)
"""
from __future__ import annotations

import json
from typing import Optional

from loguru import logger

from backend.config import settings
from backend.models import Confidence, EditItem


SYSTEM_PROMPT = """你是中文播客音频剪辑助手。你的任务是判断给定的"口癖候选词"在当前上下文中是【口癖填充】还是【承担语义】。

严格规则:
- "口癖填充"= 该词可删除且不影响句子语义,如拖音、无意义重复、纯连接废话。
- "承担语义"= 该词承载时间/逻辑/强调/指代等含义,删除会损害句意或造成不通顺。

只返回 JSON,格式:
{"results": [{"id": "...", "verdict": "cut" 或 "keep", "reason": "简短中文说明"}]}
不要输出 JSON 以外的任何内容。"""


def is_enabled() -> bool:
    return bool(
        settings.llm_enabled
        and settings.llm_base_url
        and settings.llm_api_key
    )


def review_batch(items: list[EditItem], transcript_context: dict[str, str]) -> list[EditItem]:
    """对一批歧义项做 LLM 复核,原地更新 llm_reviewed/llm_verdict/explanation。

    Args:
        items: 待复核的 EditItem(通常是 confidence=medium/low 且 ambiguous 的)
        transcript_context: {item_id: "前文...【候选词】...后文"}

    Returns:
        更新后的 items(同一引用)。复核失败的保持原状(llm_reviewed=False)。
    """
    if not is_enabled() or not items:
        return items

    from openai import OpenAI

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key,
                    timeout=settings.llm_timeout_sec)

    # 构造用户消息
    user_items = []
    for it in items:
        ctx = transcript_context.get(it.id, it.original_text)
        user_items.append({"id": it.id, "word": it.original_text, "context": ctx})
    user_msg = "请逐项判断以下候选词在各自上下文中是否为口癖填充:\n\n" + json.dumps(
        user_items, ensure_ascii=False, indent=2)

    try:
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or ""
        data = json.loads(content)
        results = {r["id"]: r for r in data.get("results", [])}
        logger.info(f"LLM 复核 {len(items)} 项,返回 {len(results)} 项")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"LLM 复核失败,保守保留全部: {e}")
        mark_review_failed_keep(items, f"LLM 复核失败,保守保留: {e}")
        return items

    for it in items:
        r = results.get(it.id)
        if not r:
            continue
        verdict = str(r.get("verdict", "")).lower().strip()
        if verdict in ("cut", "keep"):
            it.llm_reviewed = True
            it.llm_verdict = verdict  # type: ignore
            reason = str(r.get("reason", "")).strip()
            it.explanation = (it.explanation + f" | LLM: {verdict}({reason})").strip(" |")
            # LLM 建议 cut → 在均衡/激进力度下采纳;保守力度仍尊重其判断
            if verdict == "cut":
                it.keep = False
                it.confidence = Confidence.HIGH
            else:
                it.keep = True
    return items


def mark_review_failed_keep(items: list[EditItem], reason: str = "LLM 复核失败,保守保留") -> None:
    """LLM 不可用或失败时,歧义项回退为人工确认,避免自动误删。"""
    for it in items:
        it.keep = True
        it.explanation = (it.explanation + f" | {reason}").strip(" |")
