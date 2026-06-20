from pathlib import Path
from PIL import Image
from p2s_agent.core.pipeline.refinement import run_dsl_refinement_loop
from p2s_agent.core.pipeline.region_metrics import RegionVetoResult

_DSL_A = {"schema_version": 1, "canvas": {"width": 64, "height": 64, "background": "#000000"},
          "layers": [{"id": "c0", "type": "circle", "fill": {"type": "solid", "color": "#ff0000"},
                      "params": {"center": [0.5, 0.5], "radius": 0.2}, "opacity": 1.0}]}
_DSL_B = {"schema_version": 1, "canvas": {"width": 64, "height": 64, "background": "#000000"},
          "layers": [{"id": "c0", "type": "circle", "fill": {"type": "solid", "color": "#00ff00"},
                      "params": {"center": [0.5, 0.5], "radius": 0.3}, "opacity": 1.0}]}


def _veto_all(_render_path):
    return RegionVetoResult(True, 0.4, [{"id": "r1", "label": "sky", "ssim": 0.4,
                            "threshold": 0.9, "violated": True}],
                            "protected regions degraded: sky", True)


def _mk_eval(score_for_b):
    # _evaluate_dsl stand-in: renders to render_path (so it exists), scores by fill color.
    def _fake_eval(dsl, glsl, ref, render_path, **kw):
        Image.new("RGB", (64, 64), (0, 0, 0)).save(render_path)
        color = (dsl.get("layers", [{}])[0].get("fill", {}) or {}).get("color")
        score = score_for_b if color == "#00ff00" else 0.3
        return {}, {"final_score": score}, score, render_path
    return _fake_eval


def test_dsl_veto_rejects_globally_better_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr("p2s_agent.core.candidates.llm_scene.generate_llm_refinement",
                        lambda **k: dict(_DSL_B))
    monkeypatch.setattr("p2s_agent.core.pipeline.refinement._evaluate_dsl", _mk_eval(0.6))  # B scores HIGHER
    result = run_dsl_refinement_loop(
        preprocess={}, initial_dsl=dict(_DSL_A), initial_score=0.30,
        initial_metrics={}, initial_quality={"final_score": 0.30},
        reference_path=tmp_path / "ref.png",
        canvas_width=64, canvas_height=64, max_shader_chars=12000,
        max_iterations=1, threshold=0.80, high_score_stop=0.92,
        no_improvement_patience=2, loop_dir=tmp_path / "loop",
        protected_aspects=[],
        region_veto_fn=_veto_all,
    )
    assert result["best_dsl"] == _DSL_A
    assert result["history"][0].get("accepted") is not True
    assert result["history"][0].get("rejected_reason") == "protect_region_veto"


def test_dsl_veto_overrides_directed_acceptance(tmp_path, monkeypatch):
    # B scores LOWER (0.28 < 0.30): only directed acceptance could accept it; veto overrides.
    monkeypatch.setattr("p2s_agent.core.candidates.llm_scene.generate_llm_refinement",
                        lambda **k: dict(_DSL_B))
    monkeypatch.setattr("p2s_agent.core.pipeline.refinement._evaluate_dsl", _mk_eval(0.28))
    result = run_dsl_refinement_loop(
        preprocess={}, initial_dsl=dict(_DSL_A), initial_score=0.30,
        initial_metrics={}, initial_quality={"final_score": 0.30},
        reference_path=tmp_path / "ref.png",
        canvas_width=64, canvas_height=64, max_shader_chars=12000,
        max_iterations=1, threshold=0.80, high_score_stop=0.92,
        no_improvement_patience=2, loop_dir=tmp_path / "loop",
        protected_aspects=[],
        directed_acceptance={"score_drop_tolerance": 0.5},
        directed_pairwise_judge=lambda a, b: "B",
        region_veto_fn=_veto_all,
    )
    assert result["best_dsl"] == _DSL_A
    assert result["history"][0].get("accepted") is not True
    assert result["history"][0].get("human_goal_override") is None
    assert result["history"][0].get("rejected_reason") == "protect_region_veto"
