"""编辑方案 —— 把候选 EditItem 应用力度/用户决策 → 最终剪辑区间。

输出"删除区间列表"(sorted by start),供 smooth.py 渲染。
核心原则:keep=True 的项绝不进删除列表(保护有意义词语)。
"""
from __future__ import annotations

from dataclasses import dataclass

from backend.config import settings
from backend.models import CutPosition, EditItem


@dataclass(slots=True)
class CutRegion:
    """最终将被剪除的音频区间。"""
    start: float
    end: float
    position: CutPosition
    reason: str
    original_text: str
    source_ids: list[str]
    mode: str = "standard"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def build_cut_regions(items: list[EditItem]) -> list[CutRegion]:
    """根据 keep 标志生成删除区间。

    keep=False(确认删除)→ 进入删除列表
    keep=True(保留/撤销)→ 不处理
    """
    to_cut = [it for it in items if not it.keep]
    to_cut.sort(key=lambda it: it.start)

    regions: list[CutRegion] = []
    min_cut_s = settings.min_cut_ms / 1000.0

    for it in to_cut:
        dur = it.duration
        if dur < min_cut_s:
            # 过短的碎切跳过(避免咔哒),除非是重复词的一部分
            continue
        # 与上一区间紧邻(间隔 < 80ms)则合并,避免连续碎切造成多次淡化
        if regions:
            last = regions[-1]
            gap = it.start - last.end
            if 0 <= gap < 0.08:
                last.end = max(last.end, it.end)
                last.source_ids.append(it.id)
                last.reason += f"+{it.reason.value}"
                continue
        regions.append(
            CutRegion(
                start=it.start,
                end=it.end,
                position=it.position,
                reason=it.reason.value,
                original_text=it.original_text,
                source_ids=[it.id],
            )
        )
    return regions


def compute_time_saved(regions: list[CutRegion]) -> dict:
    total = sum(r.duration for r in regions)
    return {
        "regions_count": len(regions),
        "total_cut_sec": round(total, 2),
        "by_position": {
            pos.value: round(sum(r.duration for r in regions if r.position == pos), 2)
            for pos in CutPosition
        },
    }
