# Compiled request transforms

This fork is maintained by Rasmus Faber. It reduces request serialization CPU by
compiling the SDK's current parameter annotations into reusable transform plans.
The package name and public SDK APIs remain unchanged.

## Integration and compatibility

The transform export block in `src/openai/_utils/__init__.py` is the single
integration point. Its four transform helpers use `_transform_optimized.py`,
backed by `_transform_plan.py` and `_transform_fusion.py`. A separate cache fix
in `_typing.py` uses `_identity_cache.py`. The upstream `_transform.py` stays
unchanged and still defines `PropertyInfo`.

Plans compose the copying required by every union member into one pass when
the actual data proves this safe. A matching discriminator does not skip other
union members. Unknown fields, container copying and sharing, numeric-list
identity, omitted values, model serialization, and alias/format metadata retain
upstream behavior.

The deliberate compatibility exception fixes the SDK's annotation-normalization
cache. Python considers reversed unions equal, but their transform order can
matter: if one member renames `a` to `b` and another renames `b` to `c`, the two
orders should produce `c` and `b`, respectively. Upstream's equality-keyed cache
could substitute the previously visited order. The fork uses retained identity
keys with the same 8,096-entry bound, so earlier requests cannot substitute a
different annotation. Python's own construction of typing aliases is unchanged;
the SDK honors the annotation object Python supplies.

Custom containers, iterators, file inputs, and other uncertain operations use
upstream fallback. Fallback is selected before consuming the
affected input; execution errors never trigger a retry of a partly consumed
transform. Ordinary sibling fields can still use compiled plans. The fusion
probe inspects only exact JSON containers and scalars, with bounded work and
depth. Shared containers, cycles, custom values, or probe limits select faithful
traversal; they do not reject requests. Unknown values retain their references.
Long lists beginning with a scalar also bypass fusion, preserving the cheap
numeric-array path without scanning the array.

Synchronous and asynchronous execution preserve their respective upstream
behavior, including existing differences in model dumping and dictionary-child
traversal. Annotation, field-plan, and fusion caches are bounded; fusion keys
contain shapes and dictionary keys, never scalar request values. TypedDict field
hints are resolved only when a dictionary is actually visited, including
recursive schemas. Cached proofs validate hint identities in first-visit order
and replay final schema visits to preserve upstream's cache eviction order.
Failed hint resolution restores only the completed prefix's cache order before
propagating the original exception, without retrying the failing resolver.
No resource or parameter modules are imported just to initialize the optimizer.

For diagnosis, direct imports reach the original walker with the cache fix:

```python
from openai._utils._transform import transform, async_transform
```

The unchanged upstream transform tests run through the optimized exports.
Additional tests compare values, types, identity/sharing, observable consumption,
exceptions, concurrent calls, and actual synchronous/asynchronous SDK requests
using mock HTTP. Benchmark inputs and regression fixtures are synthetic.

## Measuring performance

Run from this checkout after installing the locked development dependencies:

```sh
uv sync --locked --all-extras
uv run --locked --all-extras python scripts/benchmark-transform.py --items 1 50 200 --repeat 9
uv run --locked --all-extras python scripts/benchmark-transform.py --items 1 200 --repeat 9 --fallback
```

The benchmark compares the original walker with the cache fix against the
optimized implementation from the same SDK revision. It reports warm-cache
median/minimum transform time and complete
client request time over local mock HTTP, plus the first transform in a fresh
interpreter. First-transform timings exclude imports, process startup, and fixture
construction. Add `--json` for machine-readable results.

Responses inputs alternate text messages, function calls, and function outputs;
Chat inputs contain text messages. `--fallback` uses custom mappings to
exercise ordinary upstream traversal. Timings are workload- and machine-dependent;
mock-client results exclude real network and server latency. Historical prototype
multipliers are not performance guarantees for this implementation.

Measured on 2026-09-17 with Python 3.10.16, Pydantic 2.12.5, and SDK 3.14.1 on
Linux x86_64, using nine warm repetitions and item counts 1, 50, 200, and 2,000:

| API, 200 items | Reference transform | Optimized transform | Transform speedup | Mock-client speedup |
| --- | ---: | ---: | ---: | ---: |
| Responses, sync | 82.15 ms | 1.00 ms | 81.8× | 48.2× |
| Responses, async | 75.59 ms | 1.08 ms | 70.3× | 45.2× |
| Chat, sync | 36.41 ms | 1.50 ms | 24.2× | 16.6× |
| Chat, async | 36.24 ms | 1.45 ms | 25.0× | 17.4× |

At 2,000 items, Responses transforms improved 61–66× and mock clients 46–49×;
Chat transforms improved 19–25× and mock clients 15–20×. First transforms in
fresh interpreters at that size measured 692 → 28 ms for synchronous Responses
and 329 → 23 ms for synchronous Chat. Fallback-heavy warm transforms were
approximately unchanged against the reference walker with the cache fix
(0.99–1.00×); first-transform times differed by less than 1 ms at 200 items.

A separate control restored the audited upstream normalization helper for the
reference walker. Against that unmodified upstream baseline, 200-item transforms
improved 56–63× for Responses and about 20× for Chat. The identity-cache fix adds
overhead to each reference traversal: workloads using fallback throughout were
13–14% slower than unmodified upstream in this control. The table above isolates
the optimizer's gains from that cache-fix overhead.

## Updating from upstream

The reviewed baseline is SDK 3.14.1 at
`b77076d23b6f3e34453b0fadd8cd2a001627e365`. The manifest in
`tests/transform_upstream.json` records that revision and SHA-256 fingerprints of
the original transformer and the helper files whose semantics the plans use.
The original `_typing.py` fingerprint remains recorded alongside an explicit
reviewed fork-patch fingerprint and reason. Checks compare the patched file
against that reviewed version, so further upstream or fork drift still fails.
Complete files are fingerprinted deliberately: an unrelated helper edit can
require review even when it needs no optimizer change.

For each upstream update:

1. Merge the desired upstream revision, preserving the export change, optimizer
   modules, and normalization-cache fix (unless upstream has adopted it).
2. Review changes to every fingerprinted file. Check traversal precedence,
   annotation/metadata handling, model dumping, sentinels, and file conversion.
   Review dependency changes that affect typing or Pydantic behavior as well.
3. Review any SDK imports that bypass the `_utils` exports. The AST boundary test
   allows `PropertyInfo` imports and the three deliberate optimizer-module imports.
4. Adapt the optimizer or add conservative fallback for changed semantics. Add
   differential regressions, then intentionally update the reviewed revision and
   fingerprints. Tests never refresh the manifest automatically.
5. Run the upstream and added transform suites, lint, full SDK tests, and build.
   Exercise supported Python and Pydantic configurations and rerun benchmarks,
   including small and fallback-heavy inputs.

New schema types are resolved from their current annotations; there is no static
API dispatch table to refresh. The manifest signals a required review, rather
than proving compatibility by itself. Follow upstream's CODEOWNER and
custom-code-budget requirements before integrating serialization changes.

## License and attribution

Rasmus Faber wrote the fork's optimization modules and integration changes.
Upstream's Apache 2.0 license and applicable bundled license notices remain in
place. See `LICENSE` and the licenses shipped with the SDK distributions.
