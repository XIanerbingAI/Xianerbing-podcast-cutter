"""平滑剪辑 —— 核心模块,解决"剪辑后突兀"。

管线:原始 PCM(高采样率,保证零交叉精度)
      + CutRegion 列表
      + Pause 列表(呼吸/停顿)
      -> 平滑后的 PCM

平滑技术:
1. 声学边界精确化:删除边界吸附到能量谷(真正的词间隙),而非 Whisper 词时间戳
2. 零交叉吸附    :谷点附近再对齐零交叉,消除咔哒爆音
3. 等功率交叉淡化:拼接点 40ms equal-power crossfade
4. 节奏保持      :句间口癖删后用房间底噪填充(长度=说话人停顿中位数)
5. 句内短停顿    :句内口癖删后插入 40ms 短静音,避免两个音黏一起
6. 底噪一致性    :从录音最安静段采样房间底噪填充删除区
7. 能量守卫      :删除区间能量过高则判误删,保留不删
8. 呼吸声保留    :呼吸是自然的,绝不当噪声删
9. 软限幅        :tanh 防拼接瞬时过冲爆音
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger

from backend.config import settings
from backend.models import CutPosition, Pause
from backend.pipeline.editplan import CutRegion
from backend.pipeline.vad import decode_mono_pcm

SUPPORTED_CUT_MODES = ("conservative", "standard", "clean")


@dataclass(slots=True)
class RenderContext:
    audio: np.ndarray              # 原始 PCM (mono, float32)
    sr: int                        # 采样率(高精度用 44.1k 或 48k)
    pauses: list[Pause]
    floor_noise: np.ndarray        # 房间底噪样本(用于填充)
    speaker_pause_s: float         # 说话人典型停顿时长


def load_render_context(audio_path: str, pauses: list[Pause]) -> RenderContext:
    """加载高精度 PCM + 准备底噪 profile。"""
    audio, sr = decode_mono_pcm(audio_path, sample_rate=44100)
    floor = _extract_floor_noise(audio, pauses)
    pure = [p.duration for p in pauses if not p.is_breath and p.duration > 0.1]
    speaker_pause = float(np.median(pure)) if pure else settings.target_pause_ms / 1000.0
    speaker_pause = float(np.clip(speaker_pause, 0.15, 0.6))
    logger.info(f"渲染上下文:sr={sr}, 底噪len={floor.size}, 典型停顿={speaker_pause:.2f}s")
    return RenderContext(audio=audio, sr=sr, pauses=pauses,
                         floor_noise=floor, speaker_pause_s=speaker_pause)


# ============================================================
# 房间底噪采样
# ============================================================

def _extract_floor_noise(audio: np.ndarray, pauses: list[Pause]) -> np.ndarray:
    """从录音最安静的 ~500ms 提取房间底噪 profile,用于填充删除区。"""
    target_len = int(0.5 * 44100)  # 500ms
    candidates: list[np.ndarray] = []
    for p in pauses:
        if p.is_breath:
            continue
        if p.duration < 0.2:
            continue
        s = int(p.start * 44100)
        e = int(p.end * 44100)
        seg = _safe_slice(audio, s, e)
        if seg.size > 100:
            candidates.append(seg)

    if candidates:
        candidates.sort(key=lambda x: float(np.sqrt(np.mean(x**2))))
        best = candidates[0]
        if best.size >= target_len:
            return best[:target_len].copy()
        reps = int(np.ceil(target_len / best.size))
        return np.tile(best, reps)[:target_len].copy()

    if audio.size <= target_len:
        return np.zeros(target_len, dtype=np.float32)
    n_win = audio.size // target_len
    windows = audio[: n_win * target_len].reshape(n_win, target_len)
    rms = np.sqrt(np.mean(windows**2, axis=1))
    idx = int(np.argmin(rms))
    return windows[idx].copy()


# ============================================================
# 零交叉吸附
# ============================================================

def _snap_to_zero_crossing(audio: np.ndarray, sample: int, sr: int,
                           search_ms: Optional[int] = None) -> int:
    """把样本索引吸附到最近的零交叉点,消除咔哒声。"""
    search_ms = settings.zero_cross_search_ms if search_ms is None else search_ms
    radius = max(1, int(sr * search_ms / 1000))
    lo = max(0, sample - radius)
    hi = min(audio.size, sample + radius + 1)
    if hi <= lo:
        return sample
    window = np.abs(audio[lo:hi])
    sign_changes = np.where(np.diff(np.signbit(audio[lo:hi])))[0]
    if sign_changes.size > 0:
        rel_center = sample - lo
        nearest = sign_changes[np.argmin(np.abs(sign_changes - rel_center))]
        return int(lo + nearest)
    return int(lo + int(np.argmin(window)))


# ============================================================
# 等功率交叉淡化
# ============================================================

def _equal_power_fade(length: int) -> tuple[np.ndarray, np.ndarray]:
    """生成等功率淡入/淡出窗(cos^2 曲线,保证总功率恒定)。"""
    t = np.arange(length, dtype=np.float32) / max(1, length - 1)
    fade_out = np.cos(t * np.pi / 2) ** 2   # 1->0
    fade_in = np.sin(t * np.pi / 2) ** 2    # 0->1
    return fade_out.astype(np.float32), fade_in.astype(np.float32)


def _crossfade(left: np.ndarray, right: np.ndarray, sr: int) -> np.ndarray:
    """对 left 尾部与 right 头部做等功率交叉淡化并拼接。"""
    xf = max(1, int(sr * settings.crossfade_ms / 1000))
    xf = min(xf, left.size // 2, right.size // 2)
    if xf < 4:
        return np.concatenate([left, right])

    fo, fi = _equal_power_fade(xf)
    left_tail = left[-xf:].copy()
    right_head = right[:xf].copy()
    blended = left_tail * fo + right_head * fi

    out = np.empty(left.size + right.size - xf, dtype=np.float32)
    out[: left.size - xf] = left[: left.size - xf]
    out[left.size - xf: left.size] = blended
    out[left.size:] = right[xf:]
    return out


# ============================================================
# 底噪填充
# ============================================================

def _make_floor_fill(length: int, floor: np.ndarray) -> np.ndarray:
    """从房间底噪 profile 生成指定长度的填充段(循环+轻微抖动避免机械感)。"""
    if floor.size == 0:
        return np.zeros(length, dtype=np.float32)
    reps = int(np.ceil(length / floor.size))
    base = np.tile(floor, reps)[:length].astype(np.float32)
    rng = np.random.default_rng(42)
    jitter = 1.0 + (rng.random(length).astype(np.float32) - 0.5) * 0.2
    return (base * jitter * 0.7).astype(np.float32)


# ============================================================
# 声学边界精确化 + 能量守卫
# ============================================================

def _find_energy_valley(audio: np.ndarray, sr: int, anchor_sample: int,
                        search_radius_ms: float | None = None,
                        frame_ms: float = 8) -> int:
    """从锚点出发,在 +/- search_radius_ms 范围内找能量最低点(真正的词间隙)。

    Whisper 边界常落在词中间(有声处),而真正的词间隙(静音谷)在附近几十毫秒。
    切到谷点才能避免切半字。
    """
    if search_radius_ms is None:
        search_radius_ms = settings.acoustic_valley_radius_ms
    radius = int(sr * search_radius_ms / 1000)
    frame = max(1, int(sr * frame_ms / 1000))
    lo = max(frame, anchor_sample - radius)
    hi = min(audio.size - frame, anchor_sample + radius)
    if hi <= lo:
        return anchor_sample

    best_sample = anchor_sample
    best_rms = float("inf")
    for c in range(lo, hi, max(1, frame // 2)):
        seg = audio[c:c + frame]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms < best_rms:
            best_rms = rms
            best_sample = c
    return _snap_to_zero_crossing(audio, best_sample + frame // 2, sr, search_ms=3)


def _segment_energy(audio: np.ndarray, sr: int, start_s: float, end_s: float) -> float:
    """计算 [start_s, end_s] 区间内音频的 RMS。"""
    s = max(0, int(start_s * sr))
    e = min(audio.size, int(end_s * sr))
    if e <= s:
        return 0.0
    seg = audio[s:e]
    return float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0


def _text_units(text: str) -> int:
    """Count pronounceable units roughly enough for duration sanity checks."""
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff" or ch.isalnum())


def _expected_token_duration_s(region: "CutRegion") -> float:
    """Estimate a lower-bound-ish spoken duration for short Mandarin fillers."""
    units = max(1, _text_units(region.original_text))
    return float(np.clip(units * 0.105, 0.12, 0.55))


def _find_voiced_island(audio: np.ndarray, sr: int, region: "CutRegion",
                        mode: str = "standard") -> tuple[int, int, dict] | None:
    """Find the nearby voiced island when Whisper's token timestamp is early/short."""
    audio_dur = audio.size / sr
    expected = _expected_token_duration_s(region)
    inner = _segment_energy(audio, sr, region.start, region.end)
    before = _segment_energy(audio, sr, max(0, region.start - 0.15), region.start)
    after = _segment_energy(audio, sr, region.end, region.end + 0.22)
    silent_forward = inner < 0.018 and after > max(0.04, inner * 8)
    silent_backward = inner < 0.018 and before > max(0.04, inner * 8) and not silent_forward

    search_pre = 0.18 if mode == "clean" else 0.14
    search_post = max(0.36 if mode == "clean" else 0.30, expected * 2.1)
    search_start = max(0.0, region.start - search_pre)
    search_end = min(audio_dur, region.end + search_post)
    if search_end <= search_start + 0.05:
        return None

    frame = max(1, int(sr * 0.020))
    hop = max(1, int(sr * 0.010))
    lo = int(search_start * sr)
    hi = int(search_end * sr)
    if hi - lo <= frame:
        return None

    starts: list[int] = []
    rms_values: list[float] = []
    for s in range(lo, hi - frame + 1, hop):
        seg = audio[s:s + frame]
        rms_values.append(float(np.sqrt(np.mean(seg ** 2))))
        starts.append(s)
    if not rms_values:
        return None

    peak = max(rms_values)
    if peak < 0.025:
        return None
    threshold = max(0.025, peak * 0.24)

    islands: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end: int | None = None
    max_gap = int(sr * 0.085)
    for s, rms in zip(starts, rms_values):
        if rms >= threshold:
            if cur_start is None:
                cur_start = s
                cur_end = s + frame
            elif s - (cur_end or s) <= max_gap:
                cur_end = s + frame
            else:
                islands.append((cur_start, cur_end or cur_start + frame))
                cur_start = s
                cur_end = s + frame
    if cur_start is not None:
        islands.append((cur_start, cur_end or cur_start + frame))
    islands = [(s, e) for s, e in islands if e - s >= int(0.035 * sr)]
    if not islands:
        return None

    center = (region.start + region.end) * 0.5 * sr
    candidates = islands
    if silent_forward:
        forward = [
            (s, e) for s, e in islands
            if e >= int(region.end * sr) and s >= int((region.start - 0.02) * sr)
        ]
        candidates = forward or islands
    elif silent_backward:
        backward = [(s, e) for s, e in islands if s <= int(region.start * sr)]
        candidates = backward or islands

    def _score(island: tuple[int, int]) -> float:
        s, e = island
        island_center = (s + e) * 0.5
        overlap = max(0, min(e, int(region.end * sr)) - max(s, int(region.start * sr)))
        overlap_bonus = -overlap / sr * 0.35
        forward_bonus = -0.12 if silent_forward and s >= int(region.start * sr) else 0.0
        return abs(island_center - center) / sr + overlap_bonus + forward_bonus

    s, e = min(candidates, key=_score)
    max_token_s = max(region.duration + 0.08, expected * (1.35 if mode == "clean" else 1.25), 0.16)
    max_token_s = min(max_token_s, 0.68)
    max_samples = int(max_token_s * sr)
    if e - s > max_samples:
        if silent_forward:
            e = s + max_samples
        elif silent_backward:
            s = e - max_samples
        else:
            preferred = int(region.start * sr)
            s = min(max(s, preferred), e - max_samples)
            e = s + max_samples

    pad = 0.012 if mode == "standard" else 0.025 if mode == "clean" else 0.0
    s_anchor = max(0, s - int(pad * sr))
    e_anchor = min(audio.size, e + int(pad * sr))
    refined_s = _find_energy_valley(audio, sr, s_anchor, search_radius_ms=45)
    refined_e = _find_energy_valley(audio, sr, e_anchor, search_radius_ms=45)
    if refined_e <= refined_s + int(0.04 * sr):
        refined_s, refined_e = s_anchor, e_anchor
    if refined_e <= refined_s + 2:
        return None

    return refined_s, refined_e, {
        "alignment_note": "voiced_island",
        "voiced_island_start": round(s / sr, 3),
        "voiced_island_end": round(e / sr, 3),
        "voiced_threshold": round(threshold, 4),
        "expected_token_s": round(expected, 3),
    }


def _energy_guard_passes(audio: np.ndarray, sr: int, region: "CutRegion",
                         context_radius_s: float = 0.15) -> bool:
    """能量守卫:删除区间内 RMS 应明显低于前后上下文,否则判误删。

    注意:调用方应传入【精确化后】的边界,而非 Whisper 原始边界。
    原始边界常落在有声处,用它算能量会误拦;精确化后的区间才是真正口癖所在。
    """
    inner = _segment_energy(audio, sr, region.start, region.end)
    before = _segment_energy(audio, sr, max(0, region.start - context_radius_s), region.start)
    after = _segment_energy(audio, sr, region.end, region.end + context_radius_s)
    ref = max(before, after, 1e-6)
    ratio = inner / ref
    return ratio < settings.energy_guard_threshold


def _boundary_is_clean_cut(audio: np.ndarray, sr: int, cut_sample: int,
                           window_ms: float = 8) -> bool:
    """判断剪切边界是否"干净"(没切到词中间)。

    原理:如果切点落在一个词的中间,切点两侧极短窗口(8ms)的能量会都很高
    (都是同一个词的发声);如果切点落在词间隙,切点两侧能量会都很低(静音)。
    只有"一侧高一侧低"(切到词边界)或"两侧都低"(切到静音)才是干净的。

    返回 False 表示切点切在了词中间(脏切,应拦截)。
    """
    w = max(1, int(sr * window_ms / 1000))
    left = audio[max(0, cut_sample - w):cut_sample]
    right = audio[cut_sample:min(audio.size, cut_sample + w)]
    if left.size < 2 or right.size < 2:
        return True  # 边界太近文件头尾,放行
    left_rms = float(np.sqrt(np.mean(left ** 2)))
    right_rms = float(np.sqrt(np.mean(right ** 2)))
    # 动态阈值:用全局中位能量的一个比例
    # 如果两侧都明显高于静音水平 → 切在了发声中间(脏切)
    floor = 0.04  # 静音判定阈值
    if left_rms > floor and right_rms > floor:
        # 两侧都有声,可能是切在词中间。再确认:看是不是真在同一词内
        # (两侧能量接近 = 同一词;差异大 = 词边界,算干净)
        ratio = min(left_rms, right_rms) / max(left_rms, right_rms, 1e-6)
        if ratio > 0.4:  # 两侧能量接近 → 同一词中间 → 脏切
            return False
    return True


def _diagnose_bounds(
    audio: np.ndarray,
    sr: int,
    region: "CutRegion",
    s: int,
    e: int,
    *,
    collapsed: bool = False,
) -> dict:
    """对一组候选切点做统一诊断。"""
    s_clean = _boundary_is_clean_cut(audio, sr, s)
    e_clean = _boundary_is_clean_cut(audio, sr, e)
    refined_inner = _segment_energy(audio, sr, s / sr, e / sr)
    before = _segment_energy(audio, sr, max(0, s / sr - 0.15), s / sr)
    after = _segment_energy(audio, sr, e / sr, e / sr + 0.15)
    ratio = refined_inner / max(before, after, 1e-6)
    return {
        "refined_start": round(s / sr, 3),
        "refined_end": round(e / sr, 3),
        "refined_inner_rms": round(refined_inner, 4),
        "refined_before_rms": round(before, 4),
        "refined_after_rms": round(after, 4),
        "refined_ratio": round(ratio, 2),
        "start_cut_clean": s_clean,
        "end_cut_clean": e_clean,
        "boundary_warning": (not s_clean or not e_clean),
        "collapsed_fallback": collapsed,
        "original_start": round(region.start, 3),
        "original_end": round(region.end, 3),
    }


def _refine_region_bounds(audio: np.ndarray, sr: int, region: "CutRegion") -> tuple[int, int, dict]:
    """对单个删除区间做声学边界精确化(找能量谷),并记录诊断信息。

    设计原则(关键):本函数【永不否决删除】。
    删/不删完全由用户决策(EditItem.keep)决定。
    本函数只负责:把 Whisper 给的词边界微调到能量谷,让切点更准、拼接更平滑。
    切点是否"干净"(切在词中间)只记录到 diag 供 log,不改变行为。

    Returns:
        (start_sample, end_sample, diag)  diag 供 cut log
    """
    # 1) 精确化:找能量谷作为边界
    expected = _expected_token_duration_s(region)
    inner = _segment_energy(audio, sr, region.start, region.end)
    before = _segment_energy(audio, sr, max(0, region.start - 0.15), region.start)
    after = _segment_energy(audio, sr, region.end, region.end + 0.22)
    needs_voice_realign = (
        region.duration < expected * 0.75
        or (inner < 0.018 and max(before, after) > 0.04)
    )
    if needs_voice_realign:
        island = _find_voiced_island(audio, sr, region, "standard")
        if island is not None:
            s, e, extra = island
            diag = _diagnose_bounds(audio, sr, region, s, e, collapsed=False)
            diag.update(extra)
            return s, e, diag

    s = _find_energy_valley(audio, sr, int(region.start * sr), search_radius_ms=100)
    e = _find_energy_valley(audio, sr, int(region.end * sr), search_radius_ms=100)

    # 塌缩保护:精确化后区间不能比原始小太多,否则词没删干净。
    # 触发条件:区间 < 50ms,或 < 原始长度的 50%。
    orig_dur_samples = max(1, int(region.duration * sr))
    refined_dur_samples = e - s
    collapsed = (refined_dur_samples < int(0.05 * sr) or
                 refined_dur_samples < orig_dur_samples * 0.5)
    if collapsed:
        # 退回原始边界(仍删,不否决)。原始边界虽可能略偏,但至少删完整。
        s = max(0, int(region.start * sr))
        e = min(audio.size, int(region.end * sr))
        if e <= s + 2:
            e = s + max(int(0.02 * sr), orig_dur_samples)

    # 2) 诊断:切点是否干净(只记录,不否决)
    return s, e, _diagnose_bounds(audio, sr, region, s, e, collapsed=collapsed)


def _quality_from_diag(diag: dict) -> tuple[str, int]:
    """把切点诊断压成面向审核的质量标签。

    clean: 切点较干净,通常可放心
    review: 有一定黏连/切词风险,建议人工试听
    risky: 风险较高,应重点复核或换方案
    """
    score = 100
    if diag.get("boundary_warning"):
        score -= 35
    if diag.get("collapsed_fallback"):
        score -= 20
    ratio = float(diag.get("refined_ratio") or 0.0)
    if ratio > 0.8:
        score -= 25
    elif ratio > 0.6:
        score -= 15
    score = max(0, min(100, score))
    if score >= 80:
        return "clean", score
    if score >= 60:
        return "review", score
    return "risky", score


def _bounds_for_mode(
    audio: np.ndarray,
    sr: int,
    region: "CutRegion",
    mode: str = "standard",
) -> tuple[int, int, dict]:
    """为同一候选生成不同剪切边界。

    conservative: 少切一点,降低误删语义词的概率
    standard: 当前正式渲染策略,能量谷 + 零交叉
    clean: 多吃一点边界,优先降低黏连风险
    """
    if mode not in SUPPORTED_CUT_MODES:
        raise ValueError(f"未知剪切方案: {mode}")
    if mode == "standard":
        return _refine_region_bounds(audio, sr, region)

    audio_dur = audio.size / sr
    dur = max(0.0, region.duration)
    if mode == "clean":
        expected = _expected_token_duration_s(region)
        inner = _segment_energy(audio, sr, region.start, region.end)
        before = _segment_energy(audio, sr, max(0, region.start - 0.15), region.start)
        after = _segment_energy(audio, sr, region.end, region.end + 0.22)
        needs_voice_realign = (
            dur < expected * 0.9
            or (inner < 0.018 and max(before, after) > 0.04)
        )
        if needs_voice_realign:
            island = _find_voiced_island(audio, sr, region, "clean")
            if island is not None:
                s, e, extra = island
                diag = _diagnose_bounds(audio, sr, region, s, e, collapsed=False)
                diag.update(extra)
                return s, e, diag
    if mode == "conservative":
        inset = min(0.025, dur * 0.25)
        start_s = region.start + inset
        end_s = region.end - inset
    else:
        pad = min(0.05, max(0.015, dur * 0.2))
        start_s = max(0.0, region.start - pad)
        end_s = min(audio_dur, region.end + pad)

    if end_s <= start_s + 0.02:
        start_s, end_s = region.start, region.end

    s_anchor = int(start_s * sr)
    e_anchor = int(end_s * sr)
    if mode == "clean":
        s = _find_energy_valley(audio, sr, s_anchor, search_radius_ms=60)
        e = _find_energy_valley(audio, sr, e_anchor, search_radius_ms=60)
    else:
        s = _snap_to_zero_crossing(audio, s_anchor, sr, search_ms=3)
        e = _snap_to_zero_crossing(audio, e_anchor, sr, search_ms=3)

    if e <= s + 2:
        s = max(0, int(region.start * sr))
        e = min(audio.size, int(region.end * sr))
        if e <= s + 2:
            e = min(audio.size, s + max(2, int(0.02 * sr)))

    collapsed = (mode == "conservative" and (e - s) < int(region.duration * sr * 0.35))
    return s, e, _diagnose_bounds(audio, sr, region, s, e, collapsed=collapsed)


def _build_boundary_options(audio: np.ndarray, sr: int, region: "CutRegion") -> dict:
    """输出三种可试听/可展示的剪切边界方案。"""
    options = {}
    for mode in SUPPORTED_CUT_MODES:
        s, e, diag = _bounds_for_mode(audio, sr, region, mode)
        quality_label, quality_score = _quality_from_diag(diag)
        options[mode] = {
            "start": round(s / sr, 3),
            "end": round(e / sr, 3),
            "duration_ms": round((e - s) / sr * 1000),
            "quality_label": quality_label,
            "quality_score": quality_score,
            "boundary_warning": diag.get("boundary_warning", False),
            "collapsed_fallback": diag.get("collapsed_fallback", False),
            "alignment_note": diag.get("alignment_note"),
        }
    return options


# ============================================================
# 主渲染
# ============================================================

def apply_cuts(ctx: RenderContext, regions: list[CutRegion]) -> tuple[np.ndarray, list[dict]]:
    """应用所有删除区间,产出平滑后的 PCM。

    策略按 CutRegion.position 分流:
    - SENTENCE_INTERNAL : 删除 + 短静音填充(40ms,模拟自然停顿)
    - SENTENCE_BOUNDARY : 用底噪填充节奏保持(句间独立口癖)
    - UTTERANCE_EDGE    : 删除 + 轻微缩短停顿(句首/句末)

    Returns:
        (result_pcm, cut_log)  cut_log 是每个删除点的详细记录,供 debug
    """
    cut_log: list[dict] = []

    if not regions:
        return ctx.audio.copy(), cut_log

    audio = ctx.audio
    sr = ctx.sr

    # 1) 对每个 region 做声学边界精确化(能量谷吸附)。
    #    设计原则:用户决策为删的,全部执行,守卫无否决权。
    #    守卫只记录"切点是否干净"到 log(供用户参考),不改变删/不删。
    typed_regions = []
    warnings = 0
    for r in regions:
        orig_inner_rms = _segment_energy(audio, sr, r.start, r.end)
        orig_before_rms = _segment_energy(audio, sr, max(0, r.start - 0.15), r.start)
        orig_after_rms = _segment_energy(audio, sr, r.end, r.end + 0.15)

        mode = getattr(r, "mode", "standard")
        s, e, diag = _bounds_for_mode(audio, sr, r, mode)
        quality_label, quality_score = _quality_from_diag(diag)
        boundary_options = _build_boundary_options(audio, sr, r)
        typed_regions.append((s, e, r, quality_label, diag))
        if diag.get("boundary_warning"):
            warnings += 1
        cut_log.append({
            "original_text": r.original_text, "reason": r.reason,
            "original_start": round(r.start, 3), "original_end": round(r.end, 3),
            "original_duration_ms": round(r.duration * 1000),
            "refined_start": round(s / sr, 3), "refined_end": round(e / sr, 3),
            "refined_duration_ms": round((e - s) / sr * 1000),
            "boundary_shift_start_ms": round((s / sr - r.start) * 1000),
            "boundary_shift_end_ms": round((e / sr - r.end) * 1000),
            "applied": True, "skip_reason": None,
            "inner_rms": round(orig_inner_rms, 4), "before_rms": round(orig_before_rms, 4),
            "after_rms": round(orig_after_rms, 4),
            "refined_inner_rms": diag.get("refined_inner_rms"),
            "refined_ratio": diag.get("refined_ratio"),
            "start_cut_clean": diag.get("start_cut_clean"),
            "end_cut_clean": diag.get("end_cut_clean"),
            "boundary_warning": diag.get("boundary_warning", False),
            "quality_label": quality_label,
            "quality_score": quality_score,
            "alignment_note": diag.get("alignment_note"),
            "voiced_island_start": diag.get("voiced_island_start"),
            "voiced_island_end": diag.get("voiced_island_end"),
            "boundary_options": boundary_options,
            "selected_mode": mode,
            "position": r.position.value,
        })

    if warnings:
        logger.info(f"边界警告: {warnings} 处切点疑似切到词中间(已按用户决策执行删除,详见 log)")

    typed_regions.sort(key=lambda x: x[0])
    if not typed_regions:
        return ctx.audio.copy(), cut_log

    # 2) 顺序拼接
    out_chunks: list[tuple[str, np.ndarray]] = []
    cursor = 0

    for s, e, r, quality_label, diag in typed_regions:
        if s < cursor:
            continue
        keep_seg = _safe_slice(audio, cursor, s)
        if r.position == CutPosition.SENTENCE_INTERNAL:
            # 句内口癖:保留段 + 40ms 短静音填充(模拟自然停顿,避免两音黏连)
            base_gap_ms = settings.internal_gap_ms
            if getattr(r, "mode", "standard") == "clean":
                base_gap_ms = max(base_gap_ms, 70)
            if quality_label in ("review", "risky") or diag.get("boundary_warning"):
                base_gap_ms = max(base_gap_ms, 65)
            if r.duration >= 0.20:
                base_gap_ms = max(base_gap_ms, min(110, int(r.duration * 280)))
            gap_len = int(base_gap_ms * sr / 1000)
            gap = _make_floor_fill(gap_len, ctx.floor_noise)
            edge = min(int(0.015 * sr), gap.size // 2, keep_seg.size // 2)
            if edge > 4:
                fo, fi = _equal_power_fade(edge)
                keep_tail = keep_seg[-edge:] * fo
                gap_head = gap[:edge] * fi
                keep_seg = np.concatenate([keep_seg[:-edge], keep_tail + gap_head])
                gap = gap[edge:]
            out_chunks.append(("keep", keep_seg))
            out_chunks.append(("fill", gap))
        elif r.position == CutPosition.SENTENCE_BOUNDARY:
            # 句间:保留段 + 底噪填充(节奏保持)
            fill_len = int(ctx.speaker_pause_s * sr)
            fill = _make_floor_fill(fill_len, ctx.floor_noise)
            edge = min(int(0.02 * sr), fill.size // 2, keep_seg.size // 2)
            if edge > 4:
                fo, fi = _equal_power_fade(edge)
                keep_tail = keep_seg[-edge:] * fo
                fill_head = fill[:edge] * fi
                keep_seg = np.concatenate([keep_seg[:-edge], keep_tail + fill_head])
                fill = fill[edge:]
            out_chunks.append(("keep", keep_seg))
            out_chunks.append(("fill", fill))
        else:  # UTTERANCE_EDGE
            out_chunks.append(("keep", keep_seg))
        cursor = e

    if cursor < audio.size:
        out_chunks.append(("keep", _safe_slice(audio, cursor, audio.size)))

    # 3) 合并 chunk,对相邻 keep 做 crossfade
    result = _assemble_chunks(out_chunks, sr)
    # 4) 防爆音:软限幅
    result = _soft_clip(result)
    logger.info(f"smooth render: input {audio.size/sr:.1f}s -> output {result.size/sr:.1f}s")
    return result, cut_log


# ============================================================
# chunk 合并
# ============================================================

def _assemble_chunks(chunks: list[tuple[str, np.ndarray]], sr: int) -> np.ndarray:
    """合并 chunk 列表,相邻 keep 之间做交叉淡化。"""
    if not chunks:
        return np.zeros(0, dtype=np.float32)

    parts: list[np.ndarray] = []
    for i, (tag, data) in enumerate(chunks):
        if data.size == 0:
            continue
        if tag == "fill":
            parts.append(data)
        else:  # keep
            if not parts:
                parts.append(data)
            else:
                prev = parts.pop()
                merged = _crossfade(prev, data, sr)
                parts.append(merged)
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


# ============================================================
# 辅助
# ============================================================

def _safe_slice(audio: np.ndarray, s: int, e: int) -> np.ndarray:
    s = max(0, s)
    e = min(audio.size, e)
    if e <= s:
        return np.zeros(0, dtype=np.float32)
    return audio[s:e].copy()


def _soft_clip(x: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    """软限幅(tanh),防止拼接点瞬时过冲爆音。"""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= ceiling:
        return x
    scale = ceiling
    over = np.abs(x) > scale
    out = x.copy()
    out[over] = np.sign(x[over]) * scale * np.tanh(np.abs(x[over]) / scale)
    return out.astype(np.float32)


# ============================================================
# 单点预览缝合(供"试听删除后"用)
# ============================================================

def preview_single_cut(
    ctx: RenderContext,
    cut_start: float,
    cut_end: float,
    *,
    pad_before: float = 1.5,
    pad_after: float = 1.5,
    mode: str = "standard",
) -> tuple[np.ndarray, int]:
    """对单个剪辑点做局部缝合,返回预览片段(删除后的听感)。"""
    sr = ctx.sr
    audio = ctx.audio

    win_start = max(0.0, cut_start - pad_before)
    win_end = min(audio.size / sr, cut_end + pad_after)
    if win_end - win_start < 0.3:
        return np.zeros(0, dtype=np.float32), sr

    region = CutRegion(
        start=cut_start,
        end=cut_end,
        position=CutPosition.SENTENCE_INTERNAL,
        reason="preview",
        original_text="",
        source_ids=[],
    )
    cut_s, cut_e, _diag = _bounds_for_mode(audio, sr, region, mode)
    win_s = max(0, int(win_start * sr))
    win_e = min(audio.size, int(win_end * sr))
    if cut_e <= cut_s + 2:
        cut_e = cut_s + 2

    left = _safe_slice(audio, win_s, cut_s)
    right = _safe_slice(audio, cut_e, win_e)

    if left.size < int(0.03 * sr) or right.size < int(0.03 * sr):
        preview = _soft_clip(np.concatenate([left, right]).astype(np.float32))
        return preview, sr

    preview = _crossfade(left, right, sr)
    preview = _soft_clip(preview.astype(np.float32))
    return preview, sr
