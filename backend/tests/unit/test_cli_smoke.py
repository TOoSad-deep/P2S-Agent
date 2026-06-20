import importlib, sys


def test_cli_imports_without_fastapi():
    # Importing the agent CLI must NOT pull in the web framework.
    for m in list(sys.modules):
        if m == "fastapi" or m.startswith("fastapi."):
            del sys.modules[m]
    mod = importlib.import_module("p2s_agent.cli")
    assert hasattr(mod, "main"), "cli must expose main()"
    assert "fastapi" not in sys.modules, "importing p2s_agent.cli must not import fastapi"
