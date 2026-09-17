<!-- Modified by Rasmus Faber: retain Python support while replacing automated policy reviews. -->
# Python Version Support Policy

This fork of the OpenAI Python SDK supports every fully released CPython version that has
not reached upstream end of life. The oldest supported version is declared by
`requires-python` in [`pyproject.toml`](pyproject.toml), documented in the
README, and tested on every pull request.

The fork maintainer may retain the most recently retired CPython version for up to six
months when the dependency graph, platform support, and security posture allow
it. This grace period is discretionary, is not an LTS commitment, and may end
early because of security, dependency, platform, or tooling requirements.
Active grace must be recorded below with an explicit end date and reason.

Minimum Python version increases:

- ship in an SDK minor release, not a normal patch release;
- are documented in the README and release notes;
- identify the final SDK release installable on the retired Python version;
- require approval from the fork CODEOWNER; and
- do not require a new SDK major version when documented APIs remain compatible
  on supported runtimes and `Requires-Python` prevents incompatible installs.

Removing a documented framework integration or public behavior is evaluated
separately. If supported users must change application code and no
compatibility layer preserves the contract, the change requires a major
release by default. Patch-level runtime removals are reserved for urgent
security exceptions and require unusually prominent communication.

The fork maintainer reviews this policy within 30 days of every October CPython
release and scheduled end of life. It does not normally raise the Python floor
more than once in a 12-month period. A scheduled upstream end of life may
require an earlier increase when the maximum six-month grace period would
expire before that cadence window ends; the compatibility history must record
that scheduled-EOL exception and its timing. Security and critical-dependency
exceptions must be recorded the same way. New stable CPython releases should
be added within 30 days of general availability when dependencies and CI
images are ready.

### Testing

- Pull requests run the high-value suite on the minimum and current stable
  CPython releases.
- A scheduled and manually dispatchable workflow tests every supported CPython
  release.
- The next CPython prerelease is allowed to fail until it becomes stable, with
  coverage beginning no later than its first release candidate.
- Built wheels and source distributions are checked for the authoritative
  `Requires-Python` value, and a resolver running on the retired interpreter
  must reject the new artifact.

### Maintainer review

The maintainer reviews CPython release and end-of-life dates when updating
upstream. Weekly and manual compatibility jobs exercise each supported runtime;
`scripts/check-python-version-policy.py` checks metadata, CI, and documentation
for agreement. Changes to support policy use the normal review and release process.

### Current compatibility

| SDK version | Python requirement |
| --- | --- |
| Fork 3.14.1.post1 | Python 3.10 or later |
| Upstream v2.48.0 | Final release installable on Python 3.9 |

The fully released upstream-supported matrix is currently Python 3.10 through
3.14. Python 3.15 is covered as an allowed-failure prerelease. There is no
active grace period.

Previously published SDK versions remain available. Unsupported Python
versions and older SDK releases receive no guaranteed fixes or security
backports. Users who need current SDK, dependency, and security fixes must use
a supported Python runtime.

For the upstream lifecycle and installer behavior, see the
[CPython version status](https://devguide.python.org/versions/),
[core metadata specification](https://packaging.python.org/en/latest/specifications/core-metadata/#requires-python),
and [PyPA guide to dropping old Python versions](https://packaging.python.org/en/latest/guides/dropping-older-python-versions/).
