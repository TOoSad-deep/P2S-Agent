"""Tests for the VLM judge (all using injected fake clients — no network)."""
import json

from PIL import Image

import p2s_agent.core.llm.vlm_judge as vj
from p2s_agent.config import ModelConfig, use_active_model
from p2s_agent.core.llm.vlm_judge import judge_pairwise, judge_rubric


def _img(tmp_path, name, color):
    path = tmp_path / name
    Image.new("RGB", (32, 32), color).save(path)
    return path


def setup_function(_fn):
    vj._CACHE.clear()


def test_rubric_parses_valid_response(tmp_path):
    ref = _img(tmp_path, "ref.png", (255, 0, 0))
    rnd = _img(tmp_path, "rnd.png", (0, 0, 255))

    def fake(system_prompt, user_prompt, image_paths):
        assert image_paths and len(image_paths) == 1
        return json.dumps({
            "differences": ["color wrong"],
            "shape_fidelity": 0.9, "position_layout": 0.8,
            "color_fidelity": 0.2, "effects_fidelity": 0.5,
            "failure_type": "color",
            "revision_hints": ["change fill color to red"],
        })

    out = judge_rubric(ref, rnd, work_dir=tmp_path, judge_client=fake)
    assert out is not None
    assert out["semantic_scores"]["color_fidelity"] == 0.2
    assert out["failure_type"] == "color"
    assert out["revision_hints"] == ["change fill color to red"]


def test_rubric_malformed_response_degrades_to_none(tmp_path):
    ref = _img(tmp_path, "ref.png", (255, 0, 0))
    rnd = _img(tmp_path, "rnd.png", (0, 0, 255))
    out = judge_rubric(ref, rnd, work_dir=tmp_path, judge_client=lambda s, u, i: "not json at all")
    assert out is None


def test_pairwise_position_bias_yields_tie(tmp_path):
    ref = _img(tmp_path, "ref.png", (255, 0, 0))
    a = _img(tmp_path, "a.png", (250, 0, 0))
    b = _img(tmp_path, "b.png", (0, 0, 255))
    # client always answers "A" regardless of order -> contradictory -> tie
    out = judge_pairwise(ref, a, b, work_dir=tmp_path, judge_client=lambda s, u, i: '{"winner": "A"}')
    assert out == "tie"


def test_pairwise_consistent_winner(tmp_path):
    ref = _img(tmp_path, "ref.png", (255, 0, 0))
    a = _img(tmp_path, "a.png", (250, 0, 0))
    b = _img(tmp_path, "b.png", (0, 0, 255))
    calls = []

    def fake(system_prompt, user_prompt, image_paths):
        calls.append(1)
        # fwd panel order (A=a, B=b): a is closer -> "A"
        # rev panel order (A=b, B=a): a is closer -> "B"
        return '{"winner": "A"}' if len(calls) == 1 else '{"winner": "B"}'

    out = judge_pairwise(ref, a, b, work_dir=tmp_path, judge_client=fake)
    assert out == "A"
    assert len(calls) == 2


def test_pairwise_result_is_cached(tmp_path):
    ref = _img(tmp_path, "ref.png", (255, 0, 0))
    a = _img(tmp_path, "a.png", (250, 0, 0))
    b = _img(tmp_path, "b.png", (0, 0, 255))
    calls = []

    def fake(system_prompt, user_prompt, image_paths):
        calls.append(1)
        return '{"winner": "A"}' if len(calls) % 2 == 1 else '{"winner": "B"}'

    first = judge_pairwise(ref, a, b, work_dir=tmp_path, judge_client=fake)
    second = judge_pairwise(ref, a, b, work_dir=tmp_path, judge_client=fake)
    assert first == second == "A"
    assert len(calls) == 2, "second call must hit the cache"


def test_directed_pairwise_consistent_winner_b(tmp_path):
    ref = _img(tmp_path, "ref.png", (255, 0, 0))
    cur = _img(tmp_path, "cur.png", (0, 0, 255))
    cand = _img(tmp_path, "cand.png", (250, 0, 0))
    calls = []

    def fake(system_prompt, user_prompt, image_paths):
        calls.append(system_prompt)
        # fwd (A=cur, B=cand): candidate wins -> "B"
        # rev (A=cand, B=cur): candidate is now A -> "A"
        return '{"winner": "B"}' if len(calls) == 1 else '{"winner": "A"}'

    out = vj.judge_directed_pairwise(
        ref, cur, cand,
        user_feedback="make the water reflection brighter",
        work_dir=tmp_path,
        judge_client=fake,
    )
    assert out == "B"
    assert len(calls) == 2
    assert any("make the water reflection brighter" in s for s in calls)


def test_directed_pairwise_none_on_failure(tmp_path):
    ref = _img(tmp_path, "ref.png", (255, 0, 0))
    cur = _img(tmp_path, "cur.png", (0, 0, 255))
    cand = _img(tmp_path, "cand.png", (250, 0, 0))
    out = vj.judge_directed_pairwise(
        ref, cur, cand, user_feedback="x", work_dir=tmp_path,
        judge_client=lambda s, u, i: None,
    )
    assert out is None


# --- Bug 2: cache keyed by model identity + bounded -----------------------

def _model(model="gpt-4o"):
    return ModelConfig(api_key="k", base_url="https://api.openai.com/v1", model=model)


def test_cache_key_includes_model_identity():
    # Same prompt+images, two different models -> two DISTINCT cache entries.
    with use_active_model(_model("model-A")):
        key_a = vj._cache_key("rubric", "deadbeef")
    with use_active_model(_model("model-B")):
        key_b = vj._cache_key("rubric", "deadbeef")
    assert key_a != key_b


def test_no_cross_model_verdict_contamination(tmp_path):
    ref = _img(tmp_path, "ref.png", (255, 0, 0))
    a = _img(tmp_path, "a.png", (250, 0, 0))
    b = _img(tmp_path, "b.png", (0, 0, 255))

    with use_active_model(_model("model-A")):
        out_a = judge_pairwise(ref, a, b, work_dir=tmp_path, judge_client=lambda s, u, i: '{"winner": "A"}')
    assert out_a == "tie"

    # A different model must NOT read model-A's cached verdict.
    calls = []

    def fake(system_prompt, user_prompt, image_paths):
        calls.append(1)
        return '{"winner": "A"}' if len(calls) == 1 else '{"winner": "B"}'

    with use_active_model(_model("model-B")):
        out_b = judge_pairwise(ref, a, b, work_dir=tmp_path, judge_client=fake)
    assert out_b == "A"
    assert len(calls) == 2, "model-B must recompute, not reuse model-A's cache entry"


def test_cache_is_bounded(tmp_path):
    # Insert far more entries than the cap; the cache must not grow without limit.
    for i in range(vj._CACHE_MAX + 50):
        vj._cache_put(vj._cache_key("rubric", f"digest{i}"), {"i": i})
    assert len(vj._CACHE) <= vj._CACHE_MAX
