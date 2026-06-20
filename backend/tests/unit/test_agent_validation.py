import pytest
from p2s_agent.core.errors import AgentInputError
from p2s_agent.core import validation as v

def test_coerce_int_ok():
    assert v.coerce_int(3, field="n", default=1, lo=1, hi=6) == 3

def test_coerce_int_rejects_non_int():
    with pytest.raises(AgentInputError):
        v.coerce_int("big", field="n", default=1, lo=1, hi=6)

def test_coerce_int_rejects_out_of_range():
    with pytest.raises(AgentInputError):
        v.coerce_int(99, field="n", default=1, lo=1, hi=6)

def test_validate_safe_id_blocks_traversal():
    with pytest.raises(AgentInputError):
        v.validate_safe_id("../etc", field="id")
    assert v.validate_safe_id("ok_id-1", field="id") == "ok_id-1"
