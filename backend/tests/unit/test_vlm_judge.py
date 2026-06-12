"""Tests for the VLM judge (all using injected fake clients — no network)."""
import json

from PIL import Image

import app.llm.vlm_judge as vj
from app.llm.vlm_judge import judge_pairwise, judge_rubric


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
