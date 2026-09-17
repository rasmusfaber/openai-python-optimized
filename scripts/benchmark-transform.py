"""Compare upstream and optimized transforms using synthetic, offline requests."""

from __future__ import annotations

import sys
import json
import time
import asyncio
import argparse
import platform
import statistics
import subprocess
from copy import deepcopy
from typing import Any, Mapping, Callable, Awaitable, cast
from pathlib import Path
from functools import partial
from collections import UserDict
from unittest.mock import patch

import httpx2

import openai
from openai import OpenAI, AsyncOpenAI, _utils
from openai._utils import _transform as stock
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming
from openai.types.responses.response_create_params import ResponseCreateParamsNonStreaming


def request_body(api: str, count: int, fallback: bool = False) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for index in range(count):
        text = f"Synthetic message {index}: " + "benchmark text " * 20
        if api == "responses":
            if index % 3 == 0:
                items.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]})
            elif index % 3 == 1:
                items.append(
                    {"type": "function_call", "call_id": f"call_{index}", "name": "example", "arguments": "{}"}
                )
            else:
                items.append({"type": "function_call_output", "call_id": f"call_{index - 1}", "output": text})
        else:
            items.append({"role": "user", "content": [{"type": "text", "text": text}]})
    values = [UserDict(item) for item in items] if fallback else items
    return {"model": "gpt-test", "stream": False, "input" if api == "responses" else "messages": values}


def expected_type(api: str) -> object:
    return ResponseCreateParamsNonStreaming if api == "responses" else CompletionCreateParamsNonStreaming


def input_state(value: object) -> object:
    """Snapshot synthetic input values, container types and identities outside timings."""
    value_type, identity = type(value), id(value)
    if isinstance(value, (dict, UserDict)):
        return (
            value_type,
            identity,
            tuple((key, input_state(child)) for key, child in cast(Mapping[str, object], value).items()),
        )
    if isinstance(value, list):
        return value_type, identity, tuple(input_state(child) for child in cast(list[object], value))
    return value_type, value


def response(request: httpx2.Request) -> httpx2.Response:
    if request.url.path.endswith("/responses"):
        body: dict[str, Any] = {
            "id": "resp_test",
            "object": "response",
            "model": "gpt-test",
            "created_at": 0,
            "status": "completed",
            "output": [],
        }
    else:
        body = {"id": "chat_test", "object": "chat.completion", "model": "gpt-test", "created": 0, "choices": []}
    return httpx2.Response(200, json=body)


def measure(operation: Callable[[], object], repeat: int) -> dict[str, float]:
    operation()
    samples: list[float] = []
    for _ in range(repeat):
        started = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - started) * 1000)
    return {"median_ms": statistics.median(samples), "minimum_ms": min(samples)}


async def measure_async(operation: Callable[[], Awaitable[object]], repeat: int) -> dict[str, float]:
    await operation()
    samples: list[float] = []
    for _ in range(repeat):
        started = time.perf_counter()
        await operation()
        samples.append((time.perf_counter() - started) * 1000)
    return {"median_ms": statistics.median(samples), "minimum_ms": min(samples)}


def benchmark_sync(api: str, body: dict[str, Any], repeat: int, client: bool) -> dict[str, Any]:
    results: dict[str, Any] = {}
    if not client:
        expected = stock.transform(deepcopy(body), expected_type(api))
        assert _utils.transform(deepcopy(body), expected_type(api)) == expected
        for name, backend in (("stock", stock), ("optimized", _utils)):
            data = deepcopy(body)
            before = input_state(data)
            results[name] = measure(partial(backend.transform, data, expected_type(api)), repeat)
            assert input_state(data) == before, f"{name} mutated the benchmark input"
        return results
    module = (
        "openai.resources.responses.responses"
        if api == "responses"
        else "openai.resources.chat.completions.completions"
    )
    captured: list[bytes] = []
    capture = False

    def transport(request: httpx2.Request) -> httpx2.Response:
        if capture:
            captured.append(request.content)
        return response(request)

    fixture = body
    with OpenAI(
        api_key="test-key",
        base_url="https://example.test/v1",
        http_client=httpx2.Client(transport=httpx2.MockTransport(transport)),
    ) as sdk:

        def operation() -> object:
            if api == "responses":
                return sdk.responses.create(**cast(ResponseCreateParamsNonStreaming, body))
            return sdk.chat.completions.create(**cast(CompletionCreateParamsNonStreaming, body))

        for name, backend in (("stock", stock), ("optimized", _utils)):
            body = deepcopy(fixture)
            before = input_state(body)
            with patch(f"{module}.maybe_transform", backend.maybe_transform):
                capture = True
                operation()
                capture = False
                results[name] = measure(operation, repeat)
            assert input_state(body) == before, f"{name} mutated the benchmark input"
    assert len(captured) == 2 and captured[0] == captured[1], "Client request bodies differ"
    return results


async def benchmark_async(api: str, body: dict[str, Any], repeat: int, client: bool) -> dict[str, Any]:
    results: dict[str, Any] = {}
    if not client:
        expected = await stock.async_transform(deepcopy(body), expected_type(api))
        assert await _utils.async_transform(deepcopy(body), expected_type(api)) == expected
        for name, backend in (("stock", stock), ("optimized", _utils)):
            data = deepcopy(body)
            before = input_state(data)
            results[name] = await measure_async(partial(backend.async_transform, data, expected_type(api)), repeat)
            assert input_state(data) == before, f"{name} mutated the benchmark input"
        return results
    module = (
        "openai.resources.responses.responses"
        if api == "responses"
        else "openai.resources.chat.completions.completions"
    )
    captured: list[bytes] = []
    capture = False

    def transport(request: httpx2.Request) -> httpx2.Response:
        if capture:
            captured.append(request.content)
        return response(request)

    fixture = body
    async with AsyncOpenAI(
        api_key="test-key",
        base_url="https://example.test/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(transport)),
    ) as sdk:

        async def operation() -> object:
            if api == "responses":
                return await sdk.responses.create(**cast(ResponseCreateParamsNonStreaming, body))
            return await sdk.chat.completions.create(**cast(CompletionCreateParamsNonStreaming, body))

        for name, backend in (("stock", stock), ("optimized", _utils)):
            body = deepcopy(fixture)
            before = input_state(body)
            with patch(f"{module}.async_maybe_transform", backend.async_maybe_transform):
                capture = True
                await operation()
                capture = False
                results[name] = await measure_async(operation, repeat)
            assert input_state(body) == before, f"{name} mutated the benchmark input"
    assert len(captured) == 2 and captured[0] == captured[1], "Client request bodies differ"
    return results


def cold_worker(api: str, backend_name: str, mode: str, count: int, fixture: str) -> None:
    backend = stock if backend_name == "stock" else _utils
    body, annotation = request_body(api, count, fixture == "fallback"), expected_type(api)

    async def run_async() -> float:
        started = time.perf_counter()
        await backend.async_transform(body, annotation)
        return (time.perf_counter() - started) * 1000

    if mode == "async":
        elapsed = asyncio.run(run_async())
    else:
        started = time.perf_counter()
        backend.transform(body, annotation)
        elapsed = (time.perf_counter() - started) * 1000
    print(json.dumps({"first_transform_ms": elapsed}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, nargs="+", default=[1, 50, 200])
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable measurements.")
    parser.add_argument("--fallback", action="store_true", help="Use custom mapping items to measure fallback.")
    parser.add_argument(
        "--skip-cold", action="store_true", help="Skip first-transform measurements in fresh interpreters."
    )
    parser.add_argument("--cold-worker", nargs=5, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cold_worker:
        api, backend, mode, count, fixture = args.cold_worker
        cold_worker(api, backend, mode, int(count), fixture)
        return
    if args.repeat < 1 or any(count < 1 for count in args.items):
        parser.error("repeat and item counts must be positive")
    rows: list[dict[str, Any]] = []
    cold: list[dict[str, Any]] = []
    for api in ("responses", "chat"):
        for count in args.items:
            body = request_body(api, count, args.fallback)
            for mode in ("sync", "async"):
                for client in (False, True):
                    values = (
                        benchmark_sync(api, body, args.repeat, client)
                        if mode == "sync"
                        else asyncio.run(benchmark_async(api, body, args.repeat, client))
                    )
                    rows.append(
                        {
                            "api": api,
                            "items": count,
                            "mode": mode,
                            "layer": "client" if client else "transform",
                            **values,
                            "speedup": values["stock"]["median_ms"] / values["optimized"]["median_ms"],
                        }
                    )
        if not args.skip_cold:
            for mode in ("sync", "async"):
                row: dict[str, Any] = {"api": api, "items": max(args.items), "mode": mode}
                for backend in ("stock", "optimized"):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--cold-worker",
                            api,
                            backend,
                            mode,
                            str(max(args.items)),
                            "fallback" if args.fallback else "json",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    row[backend] = json.loads(result.stdout)
                cold.append(row)
    report = {
        "python": platform.python_version(),
        "sdk": openai.__version__,
        "platform": platform.platform(),
        "repeats": args.repeat,
        "fixture": "fallback" if args.fallback else "json",
        "warm": rows,
        "cold": cold,
        "cold_definition": "First transform in a fresh interpreter, excluding imports, fixture construction, and process startup.",
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(
        f"Python {report['python']} | OpenAI {report['sdk']} | {report['fixture']} fixture | "
        f"median of {args.repeat} warm repetitions"
    )
    print(f"{'API':10} {'items':>5} {'mode':5} {'layer':9} {'stock ms':>10} {'optimized ms':>13} {'speedup':>9}")
    for row in rows:
        print(
            f"{row['api']:10} {row['items']:5} {row['mode']:5} {row['layer']:9} "
            f"{row['stock']['median_ms']:10.3f} {row['optimized']['median_ms']:13.3f} {row['speedup']:8.1f}x"
        )
    if cold:
        print("\nFirst transform in a fresh interpreter (imports and startup excluded):")
        for row in cold:
            print(
                f"{row['api']:10} {row['mode']:5} {row['items']:5} items | "
                f"stock {row['stock']['first_transform_ms']:.3f} ms | optimized {row['optimized']['first_transform_ms']:.3f} ms"
            )


if __name__ == "__main__":
    main()
