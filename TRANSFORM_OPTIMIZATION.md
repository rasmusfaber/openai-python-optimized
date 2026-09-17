# Compiled request transforms

This fork is maintained by Rasmus Faber. It reduces request serialization CPU by
compiling the SDK's current parameter annotations into reusable transform plans.
The package name and public SDK APIs remain unchanged.

## Integration and compatibility

The only integration edit in existing SDK code is the transform export block in
`src/openai/_utils/__init__.py`. Its four transform helpers use
`_transform_optimized.py`, backed by `_transform_plan.py`. The upstream
`_transform.py` stays unchanged and still defines `PropertyInfo`.

Plans preserve upstream's cumulative union traversal and normalization-cache
updates. A matching discriminator does not skip other union members. Unknown
fields, container copying and sharing, numeric-list identity, omitted values,
model serialization, and alias/format metadata retain upstream behavior.

Custom containers, iterators, file inputs, and other uncertain operations use
upstream fallback. Fallback is selected before consuming the
affected input; execution errors never trigger a retry of a partly consumed
transform. Ordinary sibling fields can still use compiled plans. The optimizer
does not introduce payload limits or recursively inspect unknown fields.

Synchronous and asynchronous execution preserve their respective upstream
behavior, including existing differences in model dumping and dictionary-child
traversal. Annotation and field-plan caches are bounded. TypedDict field hints
are resolved only when a dictionary is actually visited, including recursive
schemas. No resource or parameter modules are imported just to initialize the
optimizer.

For diagnosis, direct imports still reach the original implementation:

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

The benchmark compares the original and optimized implementations from the same
SDK revision. It reports warm-cache median/minimum transform time and complete
client request time over local mock HTTP, plus the first transform in a fresh
interpreter. First-transform timings exclude imports, process startup, and fixture
construction. Add `--json` for machine-readable results.

Responses inputs alternate text messages, function calls, and function outputs;
Chat inputs contain text messages. `--fallback` uses custom mappings to
exercise ordinary upstream traversal. Timings are workload- and machine-dependent;
mock-client results exclude real network and server latency. Historical prototype
multipliers are not performance guarantees for this implementation.

The more aggressive prototype-style optimization combines repeated union walks.
It is not used here because upstream's normalization cache treats reversed unions
as equal, although their traversal order can change results for custom mappings,
iterators, and aliases. Omitting a cache update during one ordinary request can
therefore change a later fallback result. The compiled walker preserves these
updates and trades some potential speed for compatibility.

Measured on 2026-09-17 with Python 3.10.16, Pydantic 2.12.5, and SDK 3.14.1 on
Linux x86_64, using the commands above (median of nine warm repetitions):

| API, 200 items | Stock transform | Compiled transform | Transform speedup | Mock-client speedup |
| --- | ---: | ---: | ---: | ---: |
| Responses, sync | 54.44 ms | 22.28 ms | 2.44× | 2.26× |
| Responses, async | 51.02 ms | 24.23 ms | 2.11× | 2.15× |
| Chat, sync | 28.49 ms | 14.60 ms | 1.95× | 1.94× |
| Chat, async | 29.13 ms | 15.11 ms | 1.93× | 1.92× |

Single-item transforms improved by 1.67–1.94×. First-call synchronous transforms
at 200 items measured 52.95 → 26.20 ms for Responses and 34.37 → 13.04 ms for Chat.
Fallback-heavy warm transforms were approximately unchanged (0.99–1.00×);
first-call timings varied by about −1 to +4 ms relative to stock. These results
do not reproduce the historical prototype's larger multipliers.

## Updating from upstream

The reviewed baseline is SDK 3.14.1 at
`b77076d23b6f3e34453b0fadd8cd2a001627e365`. The manifest in
`tests/transform_upstream.json` records that revision and SHA-256 fingerprints of
the original transformer and the helper files whose semantics the plans use.
Complete files are fingerprinted deliberately: an unrelated helper edit can
require review even when it needs no optimizer change.

For each upstream update:

1. Merge the desired upstream revision, preserving the export change and the two
   optimizer modules.
2. Review changes to every fingerprinted file. Check traversal precedence,
   annotation/metadata handling, model dumping, sentinels, and file conversion.
   Review dependency changes that affect typing or Pydantic behavior as well.
3. Review any SDK imports that bypass the `_utils` exports. The AST boundary test
   allows `PropertyInfo` imports and the two deliberate fallback-module imports.
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
