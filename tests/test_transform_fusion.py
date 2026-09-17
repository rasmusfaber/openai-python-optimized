from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Union, cast
from typing_extensions import Literal, Annotated, TypedDict

import pytest

from openai import _utils
from openai._utils import _transform as stock, _transform_fusion as fusion, _transform_optimized as optimized
from openai._utils._transform_plan import Plan, compile_plan
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming
from openai.types.responses.response_create_params import ResponseCreateParamsNonStreaming


class Child(TypedDict):
    value: str


class First(TypedDict):
    left: Child
    children: list[Child]


class Second(TypedDict):
    right: Child
    numeric: list[int]


class Request(TypedDict):
    payload: Union[First, Second]


class Aliased(TypedDict):
    kind: Literal["first"]
    value: Annotated[str, stock.PropertyInfo(alias="renamed")]


class Other(TypedDict):
    kind: Literal["second"]


class LruValue:
    pass


class LruFirst(TypedDict):
    value: LruValue


class LruSecond(TypedDict):
    value: str


class LruRequest(TypedDict):
    first: LruFirst
    second: LruSecond
    last: LruFirst


class Numeric(TypedDict):
    values: list[int]


class InvalidationFirst(TypedDict):
    value: object


class InvalidationNew(TypedDict):
    value: dict[str, str]


class InvalidationLater(TypedDict):
    value: str


class InvalidationRoot(TypedDict):
    first: InvalidationFirst
    last: InvalidationLater


def payload(value: str = "a") -> dict[str, Any]:
    return {
        "payload": {
            "left": {"value": value},
            "right": {"value": value},
            "children": [{"value": value}],
            "numeric": [1, 2],
            "unknown": {"raw": [value]},
        }
    }


@pytest.mark.asyncio
async def test_fusion_skips_recursive_walk_and_preserves_copy_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    sync_calls, async_calls = 0, 0
    original_run, original_async_run = optimized._run, optimized._async_run

    def counted(data: object, plan: Plan) -> object:
        nonlocal sync_calls
        sync_calls += 1
        return original_run(data, plan)

    async def async_counted(data: object, plan: Plan) -> object:
        nonlocal async_calls
        async_calls += 1
        return await original_async_run(data, plan)

    monkeypatch.setattr(optimized, "_run", counted)
    monkeypatch.setattr(optimized, "_async_run", async_counted)
    data = payload()
    for result in [_utils.transform(data, Request), await _utils.async_transform(data, Request)]:
        assert result == stock.transform(data, Request)
        assert result is not data
        assert result["payload"] is not data["payload"]
        for key in ("left", "right", "children"):
            assert result["payload"][key] is not data["payload"][key]
        for key in ("numeric", "unknown"):
            assert result["payload"][key] is data["payload"][key]
        assert result["payload"]["children"][0] is not data["payload"]["children"][0]
    assert sync_calls == async_calls == 1


def test_cache_reuses_shape_without_reusing_scalar_values() -> None:
    assert _utils.transform(payload("before"), Request) == payload("before")
    assert _utils.transform(payload("after"), Request) == payload("after")


@pytest.mark.asyncio
async def test_nested_numeric_lists_keep_constant_probe_work(monkeypatch: pytest.MonkeyPatch) -> None:
    inspected_scalars = 0

    def counted_type(value: object) -> type:
        nonlocal inspected_scalars
        if type(value) is int:
            inspected_scalars += 1
        return type(value)

    monkeypatch.setattr(fusion, "type", counted_type, raising=False)
    data = {"values": [1] * 50_000}
    for result in [_utils.transform(data, Numeric), await _utils.async_transform(data, Numeric)]:
        assert result is not data
        assert result["values"] is data["values"]
    assert inspected_scalars <= 2


@pytest.mark.parametrize("warm", [False, True])
def test_fusion_preserves_hint_cache_eviction_order(warm: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    data = {name: {"value": "raw"} for name in ("first", "second", "last")}
    hint_cache = cast(Any, stock.get_type_hints)
    clear_hints = hint_cache.cache_clear
    capacity = hint_cache.cache_info().maxsize
    original_value = LruValue

    def after_eviction(transform: Any) -> object:
        clear_hints()
        fusion._CACHE.clear()
        monkeypatch.setitem(globals(), "LruValue", original_value)
        if warm:
            transform(data, LruRequest)
        transform(data, LruRequest)
        for index in range(capacity - 1):
            stock.get_type_hints(type(f"Filler{index}", (), {}), include_extras=True)
        monkeypatch.setitem(globals(), "LruValue", Annotated[str, stock.PropertyInfo(alias="renamed")])
        return transform({"value": "raw"}, LruFirst)

    try:
        expected = after_eviction(stock.transform)
        assert expected == {"value": "raw"}
        assert after_eviction(_utils.transform) == expected
    finally:
        clear_hints()
        fusion._CACHE.clear()


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("use_async", [False, True])
async def test_hint_failure_preserves_prefix_cache_order(
    warm: bool, use_async: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = """from __future__ import annotations
from typing_extensions import TypedDict
class Value: pass
class First(TypedDict): value: Value
class Second(TypedDict): value: str
class Broken(TypedDict): value: MissingName
class Root(TypedDict):
    first: First
    second: Second
    last: First
    error: Broken
"""
    hint_cache = cast(Any, stock.get_type_hints)
    capacity = hint_cache.cache_info().maxsize
    original = stock._get_type_hints
    data = {name: {"value": "raw"} for name in ("first", "second", "last")}
    data["error"] = {}

    async def after_failure(backend: ModuleType) -> object:
        hint_cache.cache_clear()
        fusion._CACHE.clear()
        module = ModuleType(f"fusion_hint_failure_{backend.__name__}")
        monkeypatch.setitem(sys.modules, module.__name__, module)
        exec(source, module.__dict__)

        async def transform(value: object, annotation: object) -> object:
            if use_async:
                return await backend.async_transform(value, annotation)
            return backend.transform(value, annotation)

        if warm:
            module.__dict__["MissingName"] = str
            await transform(data, module.Root)
            for annotation in (module.Root, module.First, module.Second):
                stock.get_type_hints(annotation, include_extras=True)
            for index in range(capacity - 3):
                stock.get_type_hints(type(f"WarmFiller{index}", (), {}), include_extras=True)
            del module.__dict__["MissingName"]

        calls: list[type] = []

        def counted(obj: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append(obj)
            return original(obj, *args, **kwargs)

        monkeypatch.setattr(stock, "_get_type_hints", counted)
        with pytest.raises(NameError, match="MissingName"):
            await transform(data, module.Root)
        assert calls == ([module.Broken] if warm else [module.Root, module.First, module.Second, module.Broken])
        for index in range(capacity - 1):
            stock.get_type_hints(type(f"Filler{index}", (), {}), include_extras=True)
        module.__dict__["Value"] = Annotated[str, stock.PropertyInfo(alias="renamed")]
        return await transform({"value": "raw"}, module.First)

    try:
        expected = await after_failure(stock)
        assert expected == {"value": "raw"}
        assert await after_failure(optimized) == expected
    finally:
        hint_cache.cache_clear()
        fusion._CACHE.clear()


def test_changed_hints_rebuild_before_resolving_later_old_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    data = {"first": {"value": {"key": "raw"}}, "last": {"value": "raw"}}
    clear_hints = cast(Any, stock.get_type_hints).cache_clear
    clear_hints()
    fusion._CACHE.clear()
    assert _utils.transform(data, InvalidationRoot)["first"]["value"] is data["first"]["value"]
    calls: list[type] = []
    original = stock._get_type_hints

    def counted(obj: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(obj)
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(stock, "_get_type_hints", counted)
    monkeypatch.setitem(InvalidationRoot.__annotations__, "first", InvalidationNew)
    clear_hints()
    try:
        result = _utils.transform(data, InvalidationRoot)
        assert calls == [InvalidationRoot, InvalidationNew, InvalidationLater]
        assert result["first"]["value"] is not data["first"]["value"]
        assert result == stock.transform(data, InvalidationRoot)
    finally:
        clear_hints()
        fusion._CACHE.clear()


@pytest.mark.parametrize("limit", ["_MAX_NODES", "_MAX_DEPTH", "_MAX_WORK", "_MAX_HINTS"])
def test_exhausted_fusion_budgets_use_faithful_fallback(limit: str, monkeypatch: pytest.MonkeyPatch) -> None:
    fusion._CACHE.clear()
    monkeypatch.setattr(fusion, limit, 0)
    data = payload()
    fused, unchanged = fusion.try_fuse(data, compile_plan(Request))
    assert not fused
    assert unchanged is data
    result = _utils.transform(data, Request)
    assert result == stock.transform(data, Request)
    assert result["payload"]["left"] is not data["payload"]["left"]
    assert result["payload"]["numeric"] is data["payload"]["numeric"]


def test_shape_cache_is_bounded() -> None:
    fusion._CACHE.clear()
    for index in range(40):
        data = {f"unknown_{index}": "raw"}
        assert _utils.transform(data, Request) == data
    assert len(fusion._CACHE) == 32


@pytest.mark.asyncio
async def test_metadata_union_is_not_pruned_by_discriminator() -> None:
    data = {"kind": "second", "value": "a"}
    annotation = Union[Aliased, Other]
    assert _utils.transform(data, annotation) == {"kind": "second", "renamed": "a"}
    assert await _utils.async_transform(data, annotation) == {"kind": "second", "renamed": "a"}


@pytest.mark.parametrize("api", ["responses", "chat"])
def test_repeated_generated_item_shapes_do_not_repeat_hint_analysis(api: str, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    original = stock.get_type_hints

    def counted(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    def body(count: int) -> dict[str, Any]:
        key, kind = ("input", "input_text") if api == "responses" else ("messages", "text")
        return {
            "model": "gpt-test",
            key: [{"role": "user", "content": [{"type": kind, "text": str(index)}]} for index in range(count)],
        }

    annotation = ResponseCreateParamsNonStreaming if api == "responses" else CompletionCreateParamsNonStreaming
    monkeypatch.setattr(stock, "get_type_hints", counted)
    _utils.transform(body(1), annotation)
    small_calls = calls
    calls = 0
    large = body(40)
    result = _utils.transform(large, annotation)
    assert calls <= small_calls * 2
    assert result == stock.transform(large, annotation)


@pytest.mark.asyncio
async def test_large_mixed_responses_requests_fuse_at_the_root(monkeypatch: pytest.MonkeyPatch) -> None:
    items: list[dict[str, Any]] = []
    for index in range(2_000):
        if index % 3 == 0:
            items.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": str(index)}]})
        elif index % 3 == 1:
            items.append({"type": "function_call", "call_id": f"call_{index}", "name": "example", "arguments": "{}"})
        else:
            items.append({"type": "function_call_output", "call_id": f"call_{index - 1}", "output": str(index)})
    data = {"model": "gpt-test", "input": items}
    sync_calls, async_calls = 0, 0
    original_run, original_async_run = optimized._run, optimized._async_run

    def counted(value: object, plan: Plan) -> object:
        nonlocal sync_calls
        sync_calls += 1
        return original_run(value, plan)

    async def async_counted(value: object, plan: Plan) -> object:
        nonlocal async_calls
        async_calls += 1
        return await original_async_run(value, plan)

    monkeypatch.setattr(optimized, "_run", counted)
    monkeypatch.setattr(optimized, "_async_run", async_counted)
    fusion._CACHE.clear()
    result = _utils.transform(data, ResponseCreateParamsNonStreaming)
    assert result == data
    assert sync_calls == 1
    fusion._CACHE.clear()
    assert await _utils.async_transform(data, ResponseCreateParamsNonStreaming) == result
    assert async_calls == 1
