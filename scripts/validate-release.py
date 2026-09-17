from __future__ import annotations

import re
import ast
import sys
import email
import tarfile
import zipfile
import argparse
from pathlib import Path
from email.message import Message

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "openai-python-optimized"


def normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def artifact_metadata(artifact: Path) -> Message:
    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as wheel:
            names = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
            if len(names) != 1:
                raise ValueError(f"Expected one metadata record in {artifact.name}")
            return email.message_from_bytes(wheel.read(names[0]))
    with tarfile.open(artifact) as sdist:
        members = [member for member in sdist.getmembers() if member.name.endswith("/PKG-INFO")]
        if len(members) != 1 or not members[0].isfile():
            raise ValueError(f"Expected one metadata record in {artifact.name}")
        contents = sdist.extractfile(members[0])
        assert contents is not None
        with contents:
            return email.message_from_bytes(contents.read())


def validate_release(root: Path, tag: str | None = None, dist: Path | None = None) -> str:
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    name, version = project["name"], project["version"]
    if not isinstance(name, str) or normalized_name(name) != PACKAGE:
        raise ValueError(f"The distribution name must be {PACKAGE!r}, found {name!r}")
    if tag is not None and tag != f"v{version}":
        raise ValueError(f"The release tag must be v{version}, found {tag!r}")

    module = ast.parse((root / "src/openai/_version.py").read_text())
    versions = [
        ast.literal_eval(statement.value)
        for statement in module.body
        if isinstance(statement, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__version__" for target in statement.targets)
    ]
    if versions != [version]:
        raise ValueError(f"SDK version must match project version {version!r}, found {versions!r}")

    if dist is not None:
        wheels = list(dist.glob("*.whl"))
        sdists = list(dist.glob("*.tar.gz"))
        artifacts = set(dist.iterdir()) - {dist / ".gitignore"}
        if len(wheels) != 1 or len(sdists) != 1 or artifacts != set(wheels + sdists):
            raise ValueError("Expected exactly one wheel and one sdist in the distribution directory")
        for artifact in wheels + sdists:
            metadata = artifact_metadata(artifact)
            if normalized_name(metadata.get("Name", "")) != PACKAGE or metadata.get_all("Version") != [version]:
                raise ValueError(f"Built metadata in {artifact.name} does not match {name} {version}")
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate fork package identity, versions, and built distributions.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--tag", help="Require this exact v<version> release tag")
    parser.add_argument("--dist", type=Path, help="Also validate the built wheel and source distribution")
    args = parser.parse_args()
    try:
        print(validate_release(args.root, args.tag, args.dist))
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
