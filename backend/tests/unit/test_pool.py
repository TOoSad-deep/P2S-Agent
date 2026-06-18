"""Unit tests for run_candidate_pool observability: structured per-source errors.

A failing candidate source must NOT be silently dropped — the pool result must
still carry the other candidates AND a structured error record for the source
that raised (source name + exception type + short message), so the scoreboard
can surface it.
"""

from __future__ import annotations

from app.pipeline.pool import build_scoreboard, run_candidate_pool


def _make_preprocess(palette=None):
    return {
        "width": 64,
        "height": 64,
        "has_alpha": False,
        "alpha_coverage": 1.0,
        "palette": palette or ["#c86432", "#ffffff", "#000000", "#aaaaaa", "#555555"],
        "color_count_estimate": 12,
        "edge_sharpness": 0.1,
        "component_count_estimate": 1,
        "texture_score": 0.05,
        "photo_like_score": 0.1,
        "gradient_score": 0.2,
    }


def test_failing_source_surfaces_structured_error_and_keeps_others(monkeypatch):
    """When the rule source raises, the pool still returns the other candidates
    and records a structured error for 'rule' instead of silently dropping it."""

    def boom(*args, **kwargs):
        raise ValueError("rule blew up: bad geometry")

    monkeypatch.setattr("app.pipeline.pool.generate_rule_candidate", boom)

    candidates = run_candidate_pool(
        _make_preprocess(),
        {},
        cv_enabled=False,
        llm_enabled=False,
    )

    # Other sources still present (baseline + fallback at minimum).
    sources = {c.source for c in candidates}
    assert "baseline" in sources
    assert "fallback" in sources

    # The failing 'rule' source is surfaced as a structured error, not dropped.
    rule_records = [c for c in candidates if c.source == "rule"]
    assert rule_records, "rule source must remain visible as a failed candidate"
    err = rule_records[0].error
    assert err is not None
    assert err["source"] == "rule"
    assert err["error_type"] == "ValueError"
    assert "rule blew up" in err["message"]

    # Scoreboard aggregates a top-level source_errors list carrying the same record.
    scoreboard = build_scoreboard(candidates)
    source_errors = scoreboard["source_errors"]
    rule_errors = [e for e in source_errors if e["source"] == "rule"]
    assert rule_errors, source_errors
    assert rule_errors[0]["error_type"] == "ValueError"
    assert "rule blew up" in rule_errors[0]["message"]

    # The per-candidate scoreboard entry also carries the structured error.
    rule_entry = next(e for e in scoreboard["candidates"] if e["source"] == "rule")
    assert rule_entry["error"]["error_type"] == "ValueError"


def test_failing_decompose_source_adds_visible_placeholder(monkeypatch):
    """decompose previously appended NO placeholder on failure (silently dropped).
    It must now surface a failed-candidate record with a structured error."""

    def boom(*args, **kwargs):
        raise IndexError("decompose index out of range")

    monkeypatch.setattr("app.pipeline.pool.generate_decompose_candidate", boom)

    candidates = run_candidate_pool(
        _make_preprocess(),
        {},
        image_path="/tmp/does-not-need-to-exist.png",
        cv_enabled=False,
        llm_enabled=False,
    )

    decompose_records = [c for c in candidates if c.source == "decompose"]
    assert decompose_records, "decompose failure must be visible, not dropped"
    err = decompose_records[0].error
    assert err is not None
    assert err["error_type"] == "IndexError"
    assert "out of range" in err["message"]

    scoreboard = build_scoreboard(candidates)
    assert any(e["source"] == "decompose" for e in scoreboard["source_errors"])


def test_no_source_errors_when_all_succeed():
    """Happy path: no structured errors when nothing raises."""
    candidates = run_candidate_pool(
        _make_preprocess(),
        {},
        cv_enabled=False,
        llm_enabled=False,
    )
    assert all(c.error is None for c in candidates)
    scoreboard = build_scoreboard(candidates)
    assert scoreboard["source_errors"] == []
