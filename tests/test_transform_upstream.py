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
FALLBACK_MODULES = {
    "openai._utils._transform_plan",
    "openai._utils._transform_optimized",
    "openai._utils._transform_fusion",
}


class ForkPatch(TypedDict):
    sha256: str
    reason: str


class Audit(TypedDict):
    upstream_revision: str
    sha256: dict[str, str]
    fork_patches: dict[str, ForkPatch]


def _reviewed_fingerprints(audit: Audit) -> dict[str, str]:
    fingerprints = audit["sha256"].copy()
    for name, patch in audit["fork_patches"].items():
        assert name in fingerprints, f"Missing upstream baseline for fork patch: {name}"
        assert patch["reason"] and patch["sha256"] != fingerprints[name]
        fingerprints[name] = patch["sha256"]
    return fingerprints


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
    changed = _changed_files(ROOT, _reviewed_fingerprints(audit))
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


@pytest.mark.parametrize(
    "source",
    [
        "from openai._utils._transform import transform",
        "from .._utils._transform import async_maybe_transform as run",
        "import openai._utils._transform as stock",
        "from .._utils import _transform as stock",
    ],
)
def test_import_check_detects_bypasses(source: str) -> None:
    assert _stock_imports(source, "openai.resources.example") == [1]
