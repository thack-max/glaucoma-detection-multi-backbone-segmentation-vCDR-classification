"""The console scripts declared in pyproject.toml must work once installed.

Every check runs in a fresh interpreter whose cwd is a scratch directory and
whose PYTHONPATH is cleared, so neither the repo root nor the cwd can supply
``scripts/``, ``src/`` or ``configs/``: only what the installed distribution
provides is importable. An in-process ``ep.load()`` cannot catch packaging
bugs, because under pytest the repo root is on ``sys.path`` and ``scripts``
then imports as a namespace package even when it is not packaged.

Those subprocess tests can only vouch for this checkout if the installed
package IS this checkout (``installed_from_checkout``). Packaging of
``configs/`` is checked on a built wheel, because an editable install reads
the configs straight from the source tree.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
CONSOLE_SCRIPTS = ["glaucoma-screen", "glaucoma-specialist"]

# argv: <console-script name> [args for that script...]
_RUN_CONSOLE_SCRIPT = """
import sys
from importlib.metadata import entry_points

name = sys.argv[1]
(ep,) = [e for e in entry_points(group="console_scripts") if e.name == name]
sys.argv = [name, *sys.argv[2:]]
sys.exit(ep.load()())
"""

# argv: <console-script name>; prints the script's default --config path
_PRINT_DEFAULT_CONFIG = """
import sys
from importlib import import_module
from importlib.metadata import entry_points

name = sys.argv[1]
(ep,) = [e for e in entry_points(group="console_scripts") if e.name == name]
module = import_module(ep.module)
sys.argv = [name]
print(module.parse_args().config)
"""


# prints {console-script name: (entry-point module, its installed file)}
_LOCATE_ENTRY_POINTS = """
import json
from importlib.util import find_spec
from importlib.metadata import entry_points

print(json.dumps({
    e.name: (e.module, find_spec(e.module).origin)
    for e in entry_points(group="console_scripts")
    if e.name in ("glaucoma-screen", "glaucoma-specialist")
}))
"""


def _run_outside_repo(cwd: Path, code: str, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.fixture(scope="module")
def installed_from_checkout(tmp_path_factory):
    """Fail unless the installed console scripts run THIS checkout's code.

    The subprocess tests exercise whatever distribution `sys.executable` has
    installed. A venv shared with another worktree, or a stale install, would
    otherwise give false greens (or reds) for the tree under test.
    """
    result = _run_outside_repo(tmp_path_factory.mktemp("locate"), _LOCATE_ENTRY_POINTS)
    assert result.returncode == 0, result.stderr[-2000:]
    located = json.loads(result.stdout.strip().splitlines()[-1])
    missing = sorted(set(CONSOLE_SCRIPTS) - set(located))
    if missing:
        pytest.fail(f"console scripts {missing} not installed; run: pip install -e '.[dev]'")
    for name, (module, installed_file) in located.items():
        checkout_file = REPO_ROOT.joinpath(*module.split(".")).with_suffix(".py")
        if Path(installed_file).read_bytes() != checkout_file.read_bytes():
            pytest.fail(
                f"{name} runs {installed_file}, which differs from {checkout_file}: "
                f"the installed package is not this checkout. Reinstall from "
                f"{REPO_ROOT} (pip install -e '.[dev]') before running these tests."
            )


@pytest.mark.usefixtures("installed_from_checkout")
@pytest.mark.parametrize("name", CONSOLE_SCRIPTS)
def test_console_script_runs_outside_repo(name, tmp_path):
    result = _run_outside_repo(tmp_path, _RUN_CONSOLE_SCRIPT, name, "--help")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage:" in result.stdout


@pytest.mark.parametrize(
    ("name", "config_name"),
    [("glaucoma-screen", "screening.yaml"), ("glaucoma-specialist", "specialist.yaml")],
)
@pytest.mark.usefixtures("installed_from_checkout")
def test_default_config_resolves_outside_repo(name, config_name, tmp_path):
    result = _run_outside_repo(tmp_path, _PRINT_DEFAULT_CONFIG, name)
    assert result.returncode == 0, result.stderr[-2000:]
    config = Path(result.stdout.strip().splitlines()[-1])
    if not config.is_absolute():
        config = tmp_path / config
    assert config.name == config_name
    assert config.is_file(), f"default --config {config} does not exist"


@pytest.mark.usefixtures("installed_from_checkout")
@pytest.mark.parametrize(
    ("name", "config_name"),
    [("glaucoma-screen", "screening.yaml"), ("glaucoma-specialist", "specialist.yaml")],
)
def test_cwd_config_takes_precedence(name, config_name, tmp_path):
    # A project directory with its own configs/tier/<name>.yaml keeps using it
    # (the pre-packaging, cwd-relative default) instead of the packaged copy.
    local = tmp_path / "configs" / "tier" / config_name
    local.parent.mkdir(parents=True)
    shutil.copy(REPO_ROOT / "configs" / "tier" / config_name, local)

    result = _run_outside_repo(tmp_path, _PRINT_DEFAULT_CONFIG, name)
    assert result.returncode == 0, result.stderr[-2000:]
    config = Path(result.stdout.strip().splitlines()[-1])
    assert (tmp_path / config).resolve() == local.resolve()


def _write_fundus(path: Path) -> Path:
    Image.new("RGB", (64, 64), (180, 100, 60)).save(path)
    return path


@pytest.mark.usefixtures("installed_from_checkout")
def test_screen_with_default_config_outside_repo(tmp_path):
    image = _write_fundus(tmp_path / "fundus.png")
    output_dir = tmp_path / "out"

    # --baseline: segmenter only, so no DINOv2 weights are needed.
    result = _run_outside_repo(
        tmp_path, _RUN_CONSOLE_SCRIPT,
        "glaucoma-screen", "--image", str(image), "--baseline",
        "--output-dir", str(output_dir),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "Config: " in result.stdout and "screening.yaml" in result.stdout
    with open(output_dir / "summary.json") as f:
        assert json.load(f)["succeeded"] == 1


@pytest.mark.usefixtures("installed_from_checkout")
def test_screen_uses_cwd_config(tmp_path):
    with open(REPO_ROOT / "configs" / "tier" / "screening.yaml") as f:
        config = yaml.safe_load(f)
    config["output"]["output_dir"] = "clinic_out"
    local = tmp_path / "configs" / "tier" / "screening.yaml"
    local.parent.mkdir(parents=True)
    with open(local, "w") as f:
        yaml.safe_dump(config, f)
    image = _write_fundus(tmp_path / "fundus.png")

    # No --config / --output-dir: both must come from ./configs/tier/screening.yaml.
    result = _run_outside_repo(
        tmp_path, _RUN_CONSOLE_SCRIPT, "glaucoma-screen", "--image", str(image), "--baseline",
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert f"Config: {local.resolve()}" in result.stdout
    with open(tmp_path / "clinic_out" / "summary.json") as f:
        assert json.load(f)["succeeded"] == 1


def test_wheel_ships_scripts_and_configs(tmp_path):
    """A non-editable install must carry scripts/ and the default configs.

    Under an editable install (as in CI) the scripts' packaged config path
    points into the source checkout, so only the built wheel shows whether
    pyproject.toml actually packages configs/.
    """
    if importlib.util.find_spec("setuptools") is None:
        pytest.skip("setuptools not installed (it is in the [dev] extra)")
    # Build from a copy: setuptools writes build/ and *.egg-info into the source.
    source = tmp_path / "source"
    source.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy(REPO_ROOT / name, source / name)
    for name in ("src", "scripts", "configs"):
        shutil.copytree(
            REPO_ROOT / name, source / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    out = tmp_path / "dist"
    out.mkdir()
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import sys, setuptools.build_meta as b; print(b.build_wheel(sys.argv[1]))",
            str(out),
        ],
        cwd=source, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    wheel = out / result.stdout.strip().splitlines()[-1]
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
    for required in (
        "scripts/run_screening.py",
        "scripts/run_specialist.py",
        "configs/tier/screening.yaml",
        "configs/tier/specialist.yaml",
    ):
        assert required in names, f"{required} missing from {wheel.name}"
