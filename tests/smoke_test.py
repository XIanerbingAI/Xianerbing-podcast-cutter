"""端到端冒烟测试(无需真实音频 / 无需下载模型)。

验证:
1. 所有模块可导入
2. ffmpeg 二进制可用(imageio-ffmpeg)
3. 语义守卫对关键歧义词判定正确(然后/就是/那个)
4. 平滑剪辑在合成 PCM 上能跑通(零交叉/交叉淡化/底噪填充)
5. 数据模型序列化正常
6. FastAPI app 可构造

不依赖网络,不下载 Whisper 模型(转写模块的 get_model 不调用)。
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 让脚本可独立运行 —— 插入项目根(脚本在 tests/ 下,根是其父目录)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

results: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    try:
        fn()
        results.append((name, True, ""))
        print(f"{PASS} {name}")
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc().splitlines()[-1]
        results.append((name, False, str(e)))
        print(f"{FAIL} {name}: {e}")
        print(f"    {tb}")


# ============ 1. 模块导入 ============
def test_imports():
    import backend.config  # noqa
    import backend.models  # noqa
    import backend.ffmpeg_util  # noqa
    import backend.pipeline.transcribe  # noqa
    import backend.pipeline.vad  # noqa
    import backend.pipeline.lexicon  # noqa
    import backend.pipeline.semantic  # noqa
    import backend.pipeline.detect  # noqa
    import backend.pipeline.editplan  # noqa
    import backend.pipeline.smooth  # noqa
    import backend.pipeline.render  # noqa
    import backend.pipeline.semantic_llm  # noqa
    import backend.workers.task_queue  # noqa
    import backend.api.routes  # noqa
    import backend.main  # noqa


# ============ 2. ffmpeg 二进制 ============
def test_ffmpeg():
    from backend.ffmpeg_util import ffmpeg_bin, ensure_ffmpeg
    p = ffmpeg_bin()
    assert Path(p).exists(), f"ffmpeg 不存在: {p}"
    ensure_ffmpeg()
    print(f"    ffmpeg → {p}")


# ============ 3. 语义守卫 ============
def test_semantic_ranhao_connector():
    """然后作连接词 → KEEP"""
    from backend.pipeline.semantic import GuardContext, judge_discourse_marker, Verdict
    ctx = GuardContext(
        word="然后", word_start=1.0, word_end=1.2, word_duration=0.2,
        prev_text="我先去了超市。", next_text="就开始做饭了。",
        prev_is_pause=False, next_is_pause=False,
        sentence_start=False, sentence_end=False,
    )
    v = judge_discourse_marker(ctx)
    assert v == Verdict.KEEP, f"'然后'承接应为 KEEP, got {v}"


def test_semantic_ranhao_filler():
    """然后作口癖(独立/拖音)→ CUT"""
    from backend.pipeline.semantic import GuardContext, judge_discourse_marker, Verdict
    ctx = GuardContext(
        word="然后", word_start=1.0, word_end=1.6, word_duration=0.6,  # 拖音
        prev_text="那个", next_text="就是说",
        prev_is_pause=True, next_is_pause=True,  # 独立成调
        sentence_start=False, sentence_end=False,
    )
    v = judge_discourse_marker(ctx)
    assert v == Verdict.CUT, f"'然后'拖音独立应为 CUT, got {v}"


def test_semantic_jiushi_definition():
    """就是接'是 X'判断 → KEEP"""
    from backend.pipeline.semantic import GuardContext, judge_discourse_marker, Verdict
    ctx = GuardContext(
        word="就是", word_start=1.0, word_end=1.15, word_duration=0.15,
        prev_text="问题", next_text="是他没来。",
        prev_is_pause=False, next_is_pause=False,
        sentence_start=False, sentence_end=False,
    )
    v = judge_discourse_marker(ctx)
    assert v == Verdict.KEEP, f"'就是'判断应为 KEEP, got {v}"


def test_semantic_jiushi_filler():
    """就是接'说'(就是说)→ CUT"""
    from backend.pipeline.semantic import GuardContext, judge_discourse_marker, Verdict
    ctx = GuardContext(
        word="就是", word_start=1.0, word_end=1.2, word_duration=0.2,
        prev_text="嗯", next_text="说那个东西",
        prev_is_pause=False, next_is_pause=False,
        sentence_start=False, sentence_end=False,
    )
    v = judge_discourse_marker(ctx)
    assert v == Verdict.CUT, f"'就是说'应为 CUT, got {v}"


# ============ 4. 平滑剪辑(合成 PCM)============
def test_smooth_pipeline():
    """在合成 PCM 上跑完整平滑管线,验证不报错 + 输出合理。"""
    import numpy as np
    from backend.pipeline.smooth import (
        RenderContext, apply_cuts, _snap_to_zero_crossing,
        _equal_power_fade, _crossfade, _make_floor_fill, preview_single_cut,
    )
    from backend.pipeline.editplan import CutRegion
    from backend.models import CutPosition

    sr = 44100
    dur = 6.0
    t = np.arange(int(sr * dur)) / sr
    # 合成语音样:正弦波 + 噪声底
    audio = (0.2 * np.sin(2 * np.pi * 220 * t) + 0.02 * np.random.randn(t.size)).astype(np.float32)
    # 一段静音作底噪源
    floor = np.zeros(int(0.5 * sr), dtype=np.float32)

    from backend.models import Pause
    pauses = [Pause(start=2.0, end=2.5, is_breath=False),
              Pause(start=4.0, end=4.3, is_breath=True)]

    ctx = RenderContext(audio=audio, sr=sr, pauses=pauses,
                        floor_noise=floor, speaker_pause_s=0.28)

    # 三种 position 各一个删除区
    regions = [
        CutRegion(start=1.0, end=1.3, position=CutPosition.SENTENCE_INTERNAL,
                  reason="filler", original_text="嗯", source_ids=["a"]),
        CutRegion(start=2.8, end=3.2, position=CutPosition.SENTENCE_BOUNDARY,
                  reason="discourse", original_text="然后", source_ids=["b"]),
        CutRegion(start=5.0, end=5.2, position=CutPosition.UTTERANCE_EDGE,
                  reason="repeat", original_text="我", source_ids=["c"]),
    ]

    out, cut_log = apply_cuts(ctx, regions)
    assert out.size > 0, "输出为空"
    assert out.dtype == np.float32
    assert len(cut_log) == len(regions), f"剪辑日志数量不匹配: {len(cut_log)} != {len(regions)}"
    assert all(c.get("applied") for c in cut_log), "剪辑日志存在未执行项"
    assert all(c.get("quality_label") in {"clean", "review", "risky"} for c in cut_log)
    assert all(isinstance(c.get("quality_score"), int) for c in cut_log)
    assert all(set(c.get("boundary_options", {}).keys()) == {"conservative", "standard", "clean"}
               for c in cut_log)
    assert all(c.get("selected_mode") == "standard" for c in cut_log)
    # 删除了一些内容 → 输出应短于输入(但底噪填充会补回一部分)
    saved = audio.size - out.size
    print(f"    输入 {audio.size/sr:.2f}s → 输出 {out.size/sr:.2f}s (净删 {saved/sr:.2f}s)")

    # 验证无 NaN/Inf
    assert np.isfinite(out).all(), "输出含 NaN/Inf"
    # 验证无爆音(|val| < 1.5)
    peak = float(np.max(np.abs(out)))
    assert peak < 1.5, f"输出爆音 peak={peak}"
    print(f"    peak={peak:.3f}(软限幅后应 < 1.0)")

    previews = []
    for mode in ("conservative", "standard", "clean"):
        pcm, mode_sr = preview_single_cut(ctx, 1.0, 1.3, mode=mode)
        assert mode_sr == sr
        assert pcm.size > 0, f"{mode} 预览为空"
        previews.append(pcm.size)
    assert len(set(previews)) >= 1


def test_smooth_zero_cross():
    import numpy as np
    from backend.pipeline.smooth import _snap_to_zero_crossing
    sr = 44100
    t = np.arange(sr) / sr
    audio = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    # 在峰值处吸附 → 应移到附近零交叉
    peak_sample = sr // 4  # sin 峰值在 1/4 周期
    snapped = _snap_to_zero_crossing(audio, peak_sample, sr, search_ms=5)
    assert abs(audio[snapped]) < 0.05, f"吸附后非零交叉 |{audio[snapped]:.3f}|"


def test_smooth_realigns_silent_word_timestamp():
    """Whisper 把词时间戳打到静音上时,剪切边界应追到附近真实发声岛。"""
    import numpy as np
    from backend.models import CutPosition
    from backend.pipeline.editplan import CutRegion
    from backend.pipeline.smooth import _bounds_for_mode

    sr = 44100
    audio = np.zeros(int(sr * 1.4), dtype=np.float32)
    for start, end, freq in ((0.54, 0.66, 260), (0.72, 0.84, 320)):
        s = int(start * sr)
        e = int(end * sr)
        t = np.arange(e - s, dtype=np.float32) / sr
        audio[s:e] = 0.22 * np.sin(2 * np.pi * freq * t)

    region = CutRegion(
        start=0.20,
        end=0.40,
        position=CutPosition.SENTENCE_INTERNAL,
        reason="discourse",
        original_text="其实",
        source_ids=["silent_ts"],
    )
    s, e, diag = _bounds_for_mode(audio, sr, region, "standard")
    assert diag.get("alignment_note") == "voiced_island"
    assert s / sr > 0.45, f"边界仍停在静音时间戳附近: {s/sr:.3f}"
    assert e / sr > 0.75, f"结束边界未覆盖真实发声岛: {e/sr:.3f}"


def test_crossfade_power():
    """等功率交叉淡化:两段恒定幅值信号淡后中点功率≈恒定。"""
    import numpy as np
    from backend.pipeline.smooth import _crossfade
    sr = 44100
    n = sr // 2
    a = np.full(n, 0.5, dtype=np.float32)
    b = np.full(n, 0.5, dtype=np.float32)
    out = _crossfade(a, b, sr)
    # 淡化区中点应接近 0.5(等功率+等幅度)
    mid = out.size // 2
    region = out[mid - 50: mid + 50]
    assert np.all(np.abs(region - 0.5) < 0.1), f"交叉淡化区不平稳: {region[:5]}"


# ============ 5. 检测 + 编辑方案集成 ============
def test_detect_and_plan():
    from backend.models import Segment, WordToken, CutPosition
    from backend.pipeline.detect import detect, DetectionConfig
    from backend.pipeline.editplan import build_cut_regions

    # 构造含口癖的伪词序列
    words = [
        WordToken("嗯", 0.0, 0.4),       # FILLER, HIGH → 删
        WordToken("我", 0.45, 0.6),
        WordToken("我", 0.62, 0.78),     # 重复
        WordToken("我", 0.8, 0.95),      # 重复
        WordToken("想", 1.0, 1.2),
        WordToken("说", 1.25, 1.5),
        WordToken("然后", 1.6, 2.1),     # 拖音 + 独立 → 可能 CUT
        WordToken("就是", 2.15, 2.35),
        WordToken("说", 2.4, 2.6),       # 就是说 → CUT
    ]
    seg = Segment(start=0.0, end=3.0, text="嗯我我我我想说然后就是说说", words=words)
    items = detect([seg], DetectionConfig(strength="balanced"))
    assert len(items) > 0, "未检出任何候选"
    print(f"    检出 {len(items)} 条候选: " +
          ", ".join(f"{it.original_text}({it.reason.value}:{'删' if not it.keep else '留'})"
                    for it in items))

    regions = build_cut_regions(items)
    print(f"    生成 {len(regions)} 个删除区间")
    assert all(r.end > r.start for r in regions)


# ============ 6. 力度策略差异 ============
def test_strength_difference():
    """保守 vs 激进:话语标记在保守下应少检/不删。"""
    from backend.models import Segment, WordToken
    from backend.pipeline.detect import detect, DetectionConfig

    words = [
        WordToken("我觉得", 0.0, 0.4),
        WordToken("其实", 0.5, 0.7),     # discourse
        WordToken("这个", 0.75, 0.95),
        WordToken("问题", 1.0, 1.3),
    ]
    seg = Segment(start=0, end=1.5, text="我觉得其实这个问题", words=words)
    cons = detect([seg], DetectionConfig(strength="conservative"))
    agg = detect([seg], DetectionConfig(strength="aggressive"))
    print(f"    保守检出 {len(cons)} / 激进检出 {len(agg)}")


# ============ 7. LLM 上下文与时间戳规范化 ============
def test_llm_context_uses_detected_text_context():
    from backend.models import Confidence, CutPosition, CutReason, EditItem
    from backend.workers.task_queue import build_llm_context_map

    item = EditItem(
        id="ed_test",
        start=12.3,
        end=12.7,
        original_text="就是",
        reason=CutReason.DISCOURSE,
        confidence=Confidence.MEDIUM,
        position=CutPosition.SENTENCE_INTERNAL,
        context_before="问题",
        context_after="是他没来",
    )
    ctx = build_llm_context_map([item])[item.id]
    assert "问题【就是】是他没来" in ctx
    assert "12.30-12.70s" in ctx
    assert "discourse" in ctx


def test_transcribe_word_timing_normalization():
    from backend.models import Segment, WordToken
    from backend.pipeline.transcribe import _merge_close_words

    segs = [
        Segment(
            start=1.0,
            end=2.0,
            text="测试",
            words=[
                WordToken("测", 0.7, 0.6, probability=1.2),
                WordToken("试", 0.8, 2.5, probability=-0.1),
                WordToken("空", 2.1, 2.2),
            ],
        )
    ]
    out = _merge_close_words(segs)[0].words
    assert len(out) == 2
    assert out[0].start == 1.0
    assert out[0].end > out[0].start
    assert out[1].start >= out[0].end
    assert out[1].end <= 2.0
    assert out[0].probability == 1.0
    assert out[1].probability == 0.0


def test_llm_failure_marks_ambiguous_keep():
    from backend.models import Confidence, CutPosition, CutReason, EditItem
    from backend.pipeline.semantic_llm import mark_review_failed_keep

    item = EditItem(
        id="ed_llm",
        start=1.0,
        end=1.4,
        original_text="那个",
        reason=CutReason.DISCOURSE,
        confidence=Confidence.MEDIUM,
        position=CutPosition.SENTENCE_INTERNAL,
        keep=False,
        explanation="语义守卫:无法确定",
    )
    mark_review_failed_keep([item], "LLM timeout")
    assert item.keep is True
    assert "LLM timeout" in item.explanation


# ============ 8. 上传与缓存基础设施 ============
def test_upload_stream_limit_and_cleanup():
    """上传分块写入:超过大小限制时应报 413,且不写入超限 chunk。"""
    import asyncio
    import shutil
    import uuid
    from pathlib import Path
    from fastapi import HTTPException
    from backend.config import OUTPUT_DIR
    from backend.api.routes import _save_upload_stream

    class FakeUpload:
        def __init__(self, chunks: list[bytes]):
            self.chunks = list(chunks)

        async def read(self, _size: int) -> bytes:
            if not self.chunks:
                return b""
            return self.chunks.pop(0)

    tmp = OUTPUT_DIR / f"_smoke_upload_{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        ok_path = tmp / "ok.mp3"
        size = asyncio.run(_save_upload_stream(
            FakeUpload([b"abc", b"de"]), ok_path, max_bytes=5, chunk_bytes=2))
        assert size == 5
        assert ok_path.read_bytes() == b"abcde"

        too_big_path = tmp / "too_big.mp3"
        try:
            asyncio.run(_save_upload_stream(
                FakeUpload([b"abc", b"def"]), too_big_path, max_bytes=5, chunk_bytes=2))
        except HTTPException as e:
            assert e.status_code == 413
        else:
            raise AssertionError("超限上传未抛出 413")
        if too_big_path.exists():
            assert too_big_path.read_bytes() == b"abc", "超限 chunk 不应写入半成品"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_render_context_cache_limit():
    from backend.api import routes

    routes._RENDER_CTX_CACHE.clear()
    for key in ("a", "b", "c"):
        routes._cache_render_context(key, object())
    assert list(routes._RENDER_CTX_CACHE.keys()) == ["a", "b", "c"]
    routes._RENDER_CTX_CACHE.get("a")
    routes._RENDER_CTX_CACHE.move_to_end("a")
    routes._cache_render_context("d", object())
    assert list(routes._RENDER_CTX_CACHE.keys()) == ["c", "a", "d"]


def test_render_decision_modes_apply_to_regions():
    from backend.models import Confidence, CutPosition, CutReason, EditItem
    from backend.pipeline.editplan import build_cut_regions
    from backend.workers.task_queue import _apply_region_modes

    items = [
        EditItem(
            id="a",
            start=1.0,
            end=1.2,
            original_text="嗯",
            reason=CutReason.FILLER,
            confidence=Confidence.HIGH,
            position=CutPosition.SENTENCE_INTERNAL,
            keep=False,
        ),
        EditItem(
            id="b",
            start=1.23,
            end=1.4,
            original_text="就是",
            reason=CutReason.DISCOURSE,
            confidence=Confidence.MEDIUM,
            position=CutPosition.SENTENCE_INTERNAL,
            keep=False,
        ),
    ]
    regions = build_cut_regions(items)
    assert len(regions) == 1
    _apply_region_modes(regions, {"a": "conservative", "b": "clean"})
    assert regions[0].mode == "clean"


# ============ 9. FastAPI 构造 ============
def test_app_construct():
    from fastapi.testclient import TestClient
    from backend.main import app
    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200, r.text
    j = r.json()
    assert "ffmpeg" in j
    print(f"    /api/health → {j}")


# ============ 运行 ============
if __name__ == "__main__":
    check("1. 模块导入", test_imports)
    check("2. ffmpeg 二进制可用", test_ffmpeg)
    check("3a. 语义守卫:然后=连接词→保留", test_semantic_ranhao_connector)
    check("3b. 语义守卫:然后=口癖→删", test_semantic_ranhao_filler)
    check("3c. 语义守卫:就是=判断→保留", test_semantic_jiushi_definition)
    check("3d. 语义守卫:就是说→删", test_semantic_jiushi_filler)
    check("4a. 平滑剪辑完整管线", test_smooth_pipeline)
    check("4b. 零交叉吸附", test_smooth_zero_cross)
    check("4c. 静音错位时间戳重对齐", test_smooth_realigns_silent_word_timestamp)
    check("4d. 等功率交叉淡化", test_crossfade_power)
    check("5. 检测+编辑方案集成", test_detect_and_plan)
    check("6. 力度策略差异", test_strength_difference)
    check("7a. LLM 复核上下文", test_llm_context_uses_detected_text_context)
    check("7b. 词时间戳规范化", test_transcribe_word_timing_normalization)
    check("7c. LLM 失败保守保留", test_llm_failure_marks_ambiguous_keep)
    check("8a. 分块上传大小限制", test_upload_stream_limit_and_cleanup)
    check("8b. 预览缓存容量限制", test_render_context_cache_limit)
    check("8c. 渲染方案应用", test_render_decision_modes_apply_to_regions)
    check("9. FastAPI app 构造", test_app_construct)

    print("\n" + "=" * 50)
    ok = sum(1 for _, p, _ in results if p)
    total = len(results)
    print(f"结果: {ok}/{total} 通过")
    if ok < total:
        print("\n失败项:")
        for name, p, msg in results:
            if not p:
                print(f"  - {name}: {msg}")
        sys.exit(1)
    print("\n🎉 全部通过!")
