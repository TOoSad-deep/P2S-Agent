# backend/tests/unit/test_agent_web_boundary.py
"""p2s_agent must never import the web layer. This is THE enforcement of the L1 split."""
import ast
import pathlib

AGENT_ROOT = pathlib.Path(__file__).resolve().parents[2] / "p2s_agent"
FORBIDDEN_TOP = {"app", "fastapi", "starlette"}

def _imported_modules(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:   # absolute imports only
                yield node.module, node.lineno

def test_agent_package_never_imports_web():
    offenders = []
    for py in sorted(AGENT_ROOT.rglob("*.py")):
        for module, lineno in _imported_modules(py):
            if module.split(".")[0] in FORBIDDEN_TOP:
                offenders.append(f"{py.relative_to(AGENT_ROOT.parent)}:{lineno} imports {module}")
    assert not offenders, "p2s_agent must not import web layer:\n" + "\n".join(offenders)
