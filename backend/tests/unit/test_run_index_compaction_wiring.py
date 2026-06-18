"""Wiring: the worker's run-index update chokepoint (_index_updated) triggers
opportunistic, threshold-gated compaction ONLY on a terminal status transition.

The compaction itself (run_index.compact_run_index/maybe_compact_run_index) is
unit-tested in test_run_index.py; here we only assert the trigger wiring.
"""
import app.routers.png_shader as P


def _spy(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "_RUN_INDEX_PATH", tmp_path / "idx.jsonl")
    calls: list[dict] = []
    monkeypatch.setattr(P, "maybe_compact_run_index", lambda **k: calls.append(k) or False)
    return calls


def test_index_updated_triggers_compaction_on_terminal_status(tmp_path, monkeypatch):
    calls = _spy(monkeypatch, tmp_path)
    P._index_updated("run_x", {"status": "completed"})
    assert len(calls) == 1
    assert calls[0].get("path") == tmp_path / "idx.jsonl"


def test_index_updated_no_compaction_on_nonterminal_status(tmp_path, monkeypatch):
    calls = _spy(monkeypatch, tmp_path)
    P._index_updated("run_x", {"status": "running"})
    assert calls == []


def test_index_updated_no_compaction_when_no_status_field(tmp_path, monkeypatch):
    calls = _spy(monkeypatch, tmp_path)
    P._index_updated("run_x", {"final_score": 0.9})
    assert calls == []
