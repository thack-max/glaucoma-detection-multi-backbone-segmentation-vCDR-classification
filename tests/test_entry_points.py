"""The console scripts declared in pyproject.toml must resolve to callables."""
from importlib.metadata import entry_points

import pytest


@pytest.mark.parametrize("name", ["glaucoma-screen", "glaucoma-specialist"])
def test_console_script_loads(name):
    (ep,) = [e for e in entry_points(group="console_scripts") if e.name == name]
    assert callable(ep.load())
