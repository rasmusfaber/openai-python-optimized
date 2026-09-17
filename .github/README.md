# Fork automation

Rasmus Faber maintains `rasmusfaber/openai-python-optimized`. The distribution is
`openai-python-optimized`; application imports remain `openai`.

## CI and dependency updates

`ci.yml` runs lint and type checks, builds and validates both distributions, and
runs the full SDK tests with Pydantic v1 and v2 on Python 3.10 and 3.14. It runs on
pushes and pull requests, can be dispatched manually, and is reused by publishing.
Weekly and manual runs also test transforms on Python 3.10–3.14 and the
allowed-failure Python 3.15 prerelease, with both Pydantic versions.

Dependabot checks Python, Node tooling, and GitHub Actions weekly. Routine
updates retain the eight-day cooldown; security updates are exempt.

## One-time Trusted Publisher setup

In [PyPI publishing settings](https://pypi.org/manage/account/publishing/), add a
pending GitHub publisher for the first release, using these exact values:

| Field | Value |
| --- | --- |
| PyPI project name | `openai-python-optimized` |
| Owner | `rasmusfaber` |
| Repository | `openai-python-optimized` |
| Workflow filename | `publish-pypi.yml` |
| Environment | `pypi` |

If the project already exists under your account, add the publisher in that
project's Publishing settings instead. A pending publisher does not reserve the
project name. No PyPI API token or GitHub secret is needed.

In GitHub Settings → Environments, configure `pypi` with selected deployment
branches and tags: allow **tags** matching `v*`, with no branch rule.

See PyPI's guides to [adding a publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
and [using a publisher](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

## Releasing

1. Update the version in `pyproject.toml` and `src/openai/_version.py`, run
   `uv lock`, and review the lock diff. Use upstream's version plus `.postN`,
   starting with `3.14.1.post1` for this fork.
2. Merge the reviewed change into `main`. Run CI manually to include the full
   Python compatibility matrix before release.
3. Create a tag exactly matching `v<version>` on that commit and publish its
   GitHub release. For the first release this is `v3.14.1.post1`.
4. `publish-pypi.yml` checks the tag, package identity, SDK version, and ancestry
   from `main`, then runs CI again. Its upload job downloads only the wheel and
   sdist built and validated in that run and publishes them using OIDC.

The upload job has no checkout or build steps and is the only job with
`id-token: write`. All other jobs have read-only repository permissions.
Publishing a GitHub release is the action that initiates publication.

For a transient publishing failure, rerun the failed workflow. If one artifact
was uploaded before the failure, inspect PyPI before retrying: PyPI cannot replace
an existing file. Changes to released artifacts require a new version.
