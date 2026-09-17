from __future__ import annotations

import io
import sys
import tarfile
import zipfile
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NAME = "openai-python-optimized"


def make_project(
    root: Path, *, name: str = NAME, version: str = "3.14.1.post1", sdk_version: str | None = None
) -> None:
    (root / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "{version}"\n')
    source = root / "src/openai"
    source.mkdir(parents=True)
    (source / "_version.py").write_text(f'__version__ = "{sdk_version or version}"\n')


def make_artifacts(dist: Path, *, name: str = NAME, version: str = "3.14.1.post1") -> None:
    dist.mkdir()
    (dist / ".gitignore").write_text("*")  # uv build creates this alongside its distributions.
    basename = f"{name.replace('-', '_')}-{version}"
    metadata = f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n".encode()
    with zipfile.ZipFile(dist / f"{basename}-py3-none-any.whl", "w") as wheel:
        wheel.writestr(f"{basename}.dist-info/METADATA", metadata)
    with tarfile.open(dist / f"{basename}.tar.gz", "w:gz") as sdist:
        member = tarfile.TarInfo(f"{basename}/PKG-INFO")
        member.size = len(metadata)
        sdist.addfile(member, io.BytesIO(metadata))


def validate(root: Path, tag: str, *, artifacts: bool = False) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(ROOT / "scripts/validate-release.py"), "--root", str(root), "--tag", tag]
    if artifacts:
        command += ["--dist", str(root / "dist")]
    return subprocess.run(command, capture_output=True, text=True, check=False)


@pytest.mark.parametrize("version", ["3.14.1", "3.14.1.post1"])
def test_release_accepts_matching_source_and_artifacts(tmp_path: Path, version: str) -> None:
    make_project(tmp_path, version=version)
    make_artifacts(tmp_path / "dist", version=version)
    for artifacts in (False, True):
        result = validate(tmp_path, f"v{version}", artifacts=artifacts)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == NAME


@pytest.mark.parametrize(
    "name,sdk_version,tag,error",
    [
        ("openai", "3.14.1.post1", "v3.14.1.post1", "distribution name"),
        ("OpenAI", "3.14.1.post1", "v3.14.1.post1", "distribution name"),
        (NAME, "3.14.1", "v3.14.1.post1", "SDK version"),
        (NAME, "3.14.1.post1", "v3.14.1", "release tag"),
        (NAME, "3.14.1.post1", "3.14.1.post1", "release tag"),
    ],
)
def test_release_rejects_inconsistent_source(tmp_path: Path, name: str, sdk_version: str, tag: str, error: str) -> None:
    make_project(tmp_path, name=name, sdk_version=sdk_version)
    result = validate(tmp_path, tag)
    assert result.returncode != 0
    assert error in result.stderr


@pytest.mark.parametrize("name,version", [("openai", "3.14.1.post1"), (NAME, "3.14.0")])
@pytest.mark.parametrize("extension", ["*.whl", "*.tar.gz"])
def test_release_rejects_wrong_built_metadata(tmp_path: Path, name: str, version: str, extension: str) -> None:
    make_project(tmp_path)
    dist = tmp_path / "dist"
    make_artifacts(dist)
    bad = tmp_path / "bad"
    make_artifacts(bad, name=name, version=version)
    next(dist.glob(extension)).unlink()
    artifact = next(bad.glob(extension))
    artifact.rename(dist / artifact.name)
    result = validate(tmp_path, "v3.14.1.post1", artifacts=True)
    assert result.returncode != 0
    assert "metadata" in result.stderr


@pytest.mark.parametrize("change", ["missing-wheel", "missing-sdist", "extra-wheel", "extra-file"])
def test_release_rejects_incomplete_or_stale_artifacts(tmp_path: Path, change: str) -> None:
    make_project(tmp_path)
    dist = tmp_path / "dist"
    make_artifacts(dist)
    if change == "missing-wheel":
        next(dist.glob("*.whl")).unlink()
    elif change == "missing-sdist":
        next(dist.glob("*.tar.gz")).unlink()
    elif change == "extra-wheel":
        (dist / "stale.whl").write_bytes(next(dist.glob("*.whl")).read_bytes())
    else:
        (dist / "unexpected.txt").write_text("stale")
    result = validate(tmp_path, "v3.14.1.post1", artifacts=True)
    assert result.returncode != 0
    assert "one wheel and one sdist" in result.stderr


def test_publishing_requires_checks_and_isolates_oidc() -> None:
    path = ROOT / ".github/workflows/publish-pypi.yml"
    if not path.exists():
        pytest.skip("GitHub configuration is not included in source distributions")
    workflow = path.read_text()
    before_publish, publish = workflow.split("\n  publish:\n", 1)
    assert "id-token: write" not in before_publish
    assert "needs: [validate, checks]" in publish
    assert "uses: ./.github/workflows/ci.yml" in before_publish
    assert "git merge-base --is-ancestor HEAD origin/main" in before_publish
    assert "id-token: write" in publish
    assert "name: pypi" in publish
    assert "actions/download-artifact@" in publish
    assert "pypa/gh-action-pypi-publish@" in publish
    assert "actions/checkout@" not in publish
    assert "run:" not in publish
    assert "secrets." not in workflow
