"""pyproject.toml floors must match what the metadata and backends need."""

from __future__ import annotations

import re
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    tomllib = None

import pytest


PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    if tomllib is None:
        pytest.skip("tomllib needs Python 3.11+")
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)


def _floor(requirements: list[str], name: str) -> Version:
    for spec in requirements:
        req = Requirement(spec)
        if req.name == name:
            floors = [Version(s.version) for s in req.specifier if s.operator in (">=", "==")]
            assert floors, f"{spec!r} has no lower bound"
            return max(floors)
    raise AssertionError(f"{name} not in {requirements}")


def test_build_backend_reads_pep639_license_string(pyproject):
    # A `license = "MIT"` string (PEP 639) needs setuptools >= 77 at build
    # time; with an older (e.g. system, --no-build-isolation) setuptools,
    # metadata generation fails although the declared floor is met.
    assert isinstance(pyproject["project"]["license"], str)
    floor = _floor(pyproject["build-system"]["requires"], "setuptools")
    assert floor >= Version("77")


@pytest.mark.parametrize(
    ("package", "minimum"),
    # transformers 5.x (allowed by the unpinned transformers floor) treats
    # older versions as not installed: ACCELERATE_MIN_VERSION /
    # BITSANDBYTES_MIN_VERSION in transformers/utils/import_utils.py.
    [("accelerate", "1.1.0"), ("bitsandbytes", "0.46.1")],
)
def test_vlm_extra_floors_satisfy_transformers(pyproject, package, minimum):
    vlm = pyproject["project"]["optional-dependencies"]["vlm"]
    assert _floor(vlm, package) >= Version(minimum)


@pytest.mark.parametrize(
    ("sys_platform", "platform_machine", "installs_bitsandbytes"),
    [
        ("linux", "x86_64", True),
        ("linux", "aarch64", True),
        ("win32", "AMD64", True),
        ("darwin", "arm64", True),
        # bitsandbytes>=0.46.1 ships no Intel-macOS wheel and no sdist: an
        # unconditional requirement makes the whole extra unresolvable there.
        ("darwin", "x86_64", False),
    ],
)
def test_vlm_extra_resolves_without_bitsandbytes_wheels(
    pyproject, sys_platform, platform_machine, installs_bitsandbytes,
):
    env = {"sys_platform": sys_platform, "platform_machine": platform_machine}
    applies = {}
    for spec in pyproject["project"]["optional-dependencies"]["vlm"]:
        req = Requirement(spec)
        applies[req.name] = req.marker is None or req.marker.evaluate(env)
    assert applies["bitsandbytes"] is installs_bitsandbytes
    # load_in_4bit: false still needs these, on every platform.
    assert applies["accelerate"] and applies["peft"]


def test_readme_documents_vlm_extra():
    readme = (PYPROJECT.parent / "README.md").read_text()
    assert re.search(r"pip install -e \"?'?\.\[vlm\]", readme)
