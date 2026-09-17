"""Require an explicit compatibility audit when upstream transform semantics change."""

from __future__ import annotations

import ast
import json
import hashlib
from typing import cast
from pathlib import Path
from importlib.util import resolve_name
from typing_extensions import TypedDict

import pytest

ROOT = Path(__file__).resolve().parents[1]
STOCK_MODULE = "openai._utils._transform"
FALLBACK_MODULES = {"openai._utils._transform_plan", "openai._utils._transform_optimized"}


class Audit(TypedDict):
    upstream_revision: str
    sha256: dict[str, str]


def _changed_files(root: Path, fingerprints: dict[str, str]) -> list[str]:
    return [
        name
        for name, expected in fingerprints.items()
        if not (root / name).is_file() or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected
    ]


def _stock_imports(source: str, module: str) -> list[int]:
    violations: list[int] = []
    package = module.rsplit(".", 1)[0]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            if any(alias.name == STOCK_MODULE for alias in node.names):
                violations.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            imported = resolve_name("." * node.level + (node.module or ""), package) if node.level else node.module
            if imported == STOCK_MODULE:
                if any(alias.name != "PropertyInfo" for alias in node.names):
                    violations.append(node.lineno)
            elif imported == "openai._utils" and any(alias.name == "_transform" for alias in node.names):
                if module not in FALLBACK_MODULES:
                    violations.append(node.lineno)
    return violations


def test_audited_upstream_files_are_unchanged() -> None:
    audit = cast(Audit, json.loads(Path(__file__).with_name("transform_upstream.json").read_text()))
    assert len(audit["upstream_revision"]) == 40
    assert audit["sha256"]
    changed = _changed_files(ROOT, audit["sha256"])
    assert not changed, (
        f"Upstream transform dependencies changed: {changed}. Review their semantics and run the differential "
        "suite before intentionally updating tests/transform_upstream.json; see TRANSFORM_OPTIMIZATION.md."
    )


def test_sdk_imports_use_optimized_exports() -> None:
    violations: list[str] = []
    for path in (ROOT / "src/openai").rglob("*.py"):
        module = ".".join(path.relative_to(ROOT / "src").with_suffix("").parts)
        for line in _stock_imports(path.read_text(), module):
            violations.append(f"{path.relative_to(ROOT)}:{line}")
    assert not violations, f"SDK imports bypass the optimized transform exports: {violations}"


def test_fingerprint_detects_changed_and_missing_source(tmp_path: Path) -> None:
    original = (ROOT / "src/openai/_utils/_transform.py").read_bytes()
    copy = tmp_path / "_transform.py"
    copy.write_bytes(original)
    fingerprints = {copy.name: hashlib.sha256(original).hexdigest()}
    assert _changed_files(tmp_path, fingerprints) == []
    copy.write_bytes(original + b"\n# Synthetic upstream change.\n")
    assert _changed_files(tmp_path, fingerprints) == [copy.name]
    copy.unlink()
    assert _changed_files(tmp_path, fingerprints) == [copy.name]


@pytest.mark.parametrize(
    "source",
    [
        "from openai._utils._transform import transform",
        "from .._utils._transform import async_maybe_transform as run",
        "from openai._utils._transform import *",
        "import openai._utils._transform as stock",
        "from .._utils import _transform as stock",
    ],
)
def test_import_check_detects_bypasses(source: str) -> None:
    assert _stock_imports(source, "openai.resources.example") == [1]


@pytest.mark.parametrize(
    "source, module",
    [
        ("from .._utils import maybe_transform", "openai.resources.example"),
        ("from .._utils._transform import PropertyInfo", "openai.resources.example"),
        ("from . import _transform as stock", "openai._utils._transform_plan"),
        ("from . import _transform as stock", "openai._utils._transform_optimized"),
    ],
)
def test_import_check_allows_exports_metadata_and_fallback(source: str, module: str) -> None:
    assert _stock_imports(source, module) == []
