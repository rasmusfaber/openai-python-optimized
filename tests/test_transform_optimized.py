from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Dict, List, Union, Iterable
from pathlib import Path
from datetime import datetime
from collections import UserDict
from typing_extensions import Literal, Annotated, TypedDict, get_args, override
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import Field

from openai import _utils, not_given
from openai._utils import _transform as stock, _transform_optimized as optimized
from openai._models import BaseModel
from openai._utils._typing import strip_annotated_type
from openai._utils._transform_plan import Plan, compile_plan, get_field_plans


class Child(TypedDict):
    value: str


class Nested(TypedDict):
    child: Child
    children: List[Child]
    numbers: List[int]


class Aliased(TypedDict):
    type: Literal["alias"]
    value: Annotated[str, stock.PropertyInfo(alias="renamed")]


class OtherArm(TypedDict):
    type: Literal["other"]
    renamed: Annotated[str, stock.PropertyInfo(alias="final")]


class WithIterator(TypedDict):
    items: Iterable[Aliased]
    sibling: Nested


class WithFile(TypedDict):
    file: Annotated[Path, stock.PropertyInfo(format="base64")]
    sibling: Nested


class Recursive(TypedDict, total=False):
    child: Recursive
    value: Annotated[str, stock.PropertyInfo(alias="renamed")]


class MapArm(TypedDict):
    child: Child


class ListArm(TypedDict):
    children: List[Child]


class RepeatedChild(TypedDict):
    first: Child
    second: Child


class ExcludedModel(BaseModel):
    value: str = Field(alias="alias")
    secret: str
    __api_exclude__ = {"secret"}


def nested_data() -> dict[str, object]:
    return {"child": {"value": "a"}, "children": [{"value": "b"}], "numbers": [1, 2], "unknown": {"raw": []}}


def test_package_uses_optimized_transform() -> None:
    assert _utils.transform is not stock.transform
    assert _utils.async_transform is not stock.async_transform
    assert _utils.PropertyInfo is stock.PropertyInfo


def test_repeated_calls_do_not_repeat_type_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    data = nested_data()
    expected = stock.transform(data, Nested)
    assert _utils.transform(data, Nested) == expected
    calls = 0
    original = stock._get_type_hints

    def counted(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(stock, "_get_type_hints", counted)
    assert _utils.transform(data, Nested) == expected
    assert calls == 0


@pytest.mark.asyncio
async def test_nested_copy_and_unknown_identity() -> None:
    data = nested_data()
    for result in [_utils.transform(data, Nested), await _utils.async_transform(data, Nested)]:
        assert result == stock.transform(data, Nested)
        assert result is not data
        assert result["child"] is not data["child"]
        assert result["children"] is not data["children"]
        assert result["numbers"] is data["numbers"]
        assert result["unknown"] is data["unknown"]


@pytest.mark.parametrize("annotation", [dict, Dict, list, List, List[int], Union[Aliased, OtherArm]])
@pytest.mark.asyncio
async def test_bare_containers_and_cumulative_union(annotation: object) -> None:
    for data in [{"type": "other", "value": "a", "omitted": not_given}, [1, 2], [{"value": "b"}]]:
        assert _utils.transform(data, annotation) == stock.transform(data, annotation)
        assert await _utils.async_transform(data, annotation) == await stock.async_transform(data, annotation)


@pytest.mark.asyncio
async def test_container_annotation_formats_children() -> None:
    annotation = Annotated[List[Union[int, datetime]], stock.PropertyInfo(format="iso8601")]
    data = [datetime(2026, 1, 1)]
    assert _utils.transform(data, annotation) == stock.transform(data, annotation)
    assert await _utils.async_transform(data, annotation) == await stock.async_transform(data, annotation)


@pytest.mark.parametrize("annotation", [object, Dict[str, object], List[object]])
@pytest.mark.asyncio
async def test_model_dump_preserves_sync_async_difference(annotation: object) -> None:
    model = ExcludedModel(alias="a", secret="b")
    for data in [model, {"model": model}, [model]]:
        assert _utils.transform(data, annotation) == stock.transform(data, annotation)
        assert await _utils.async_transform(data, annotation) == await stock.async_transform(data, annotation)


@pytest.mark.asyncio
async def test_one_shot_iterators() -> None:
    reads: list[str] = []

    def values() -> Iterable[dict[str, str]]:
        reads.append("read")
        yield {"value": "a"}

    expected = {"items": [{"renamed": "a"}], "sibling": nested_data()}
    assert _utils.transform({"items": values(), "sibling": nested_data()}, WithIterator) == expected
    assert reads == ["read"]
    reads.clear()
    assert await _utils.async_transform({"items": values(), "sibling": nested_data()}, WithIterator) == expected
    assert reads == ["read"]


@pytest.mark.asyncio
async def test_base64_path(tmp_path: Path) -> None:
    path = tmp_path / "file"
    path.write_bytes(b"hello")
    data = {"file": path, "sibling": nested_data()}
    assert _utils.transform(data, WithFile) == stock.transform(data, WithFile)
    assert await _utils.async_transform(data, WithFile) == await stock.async_transform(data, WithFile)


def test_recursive_and_concurrent_compilation() -> None:
    data = {"value": "a", "child": {"value": "b"}}
    expected = stock.transform(data, Recursive)

    def run(_: int) -> object:
        return _utils.transform(data, Recursive)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(20)))
    assert results == [expected] * 20


def test_deep_unknown_and_cycles_are_not_traversed() -> None:
    value: list[object] = []
    value.append(value)
    for _ in range(1500):
        value = [value]
    data = {"value": "a", "unknown": value}
    result = _utils.transform(data, Child)
    assert result["unknown"] is value


@pytest.mark.parametrize("annotation", [Union[MapArm, ListArm], Union[Dict[str, Child], ListArm]])
@pytest.mark.asyncio
async def test_cumulative_union_copy_behavior(annotation: object) -> None:
    data = nested_data()
    for result in [_utils.transform(data, annotation), await _utils.async_transform(data, annotation)]:
        assert result == stock.transform(data, annotation)
        assert result["child"] is not data["child"]
        assert result["children"] is not data["children"]


@pytest.mark.parametrize("annotation", [Union[List[int], List[Child]], Union[List[Child], List[int]]])
def test_union_numeric_list_copy_order(annotation: object) -> None:
    data = [{"value": "a"}]
    result = _utils.transform(data, annotation)
    assert result == stock.transform(data, annotation)
    assert result is not data
    assert result[0] is not data[0]


def test_shared_input_paths_create_separate_copies() -> None:
    shared = {"value": "a"}
    data = {"first": shared, "second": shared}
    result = _utils.transform(data, RepeatedChild)
    assert result == stock.transform(data, RepeatedChild)
    assert result["first"] is not result["second"]
    assert result["first"] is not shared


def test_mutable_metadata_remains_live() -> None:
    info = stock.PropertyInfo(discriminator="type")
    annotation = Annotated[List[datetime], info]
    data = [datetime(2026, 1, 1)]
    assert _utils.transform(data, annotation) == stock.transform(data, annotation)
    info.format = "iso8601"
    assert _utils.transform(data, annotation) == stock.transform(data, annotation)


def test_unknown_custom_types_are_not_compared() -> None:
    class UncomparableType(type):
        @override
        def __eq__(cls, other: object) -> bool:
            raise AssertionError("Unknown values must not invoke custom equality")

    class Uncomparable(metaclass=UncomparableType):
        pass

    value = Uncomparable()
    data = {"unknown": value}
    assert stock.transform(data, Child)["unknown"] is value
    assert _utils.transform(data, Child)["unknown"] is value


def test_custom_metadata_attribute_access_stays_stock() -> None:
    reads: list[str] = []

    class TrackedInfo(stock.PropertyInfo):
        @override
        def __getattribute__(self, name: str) -> Any:
            if name in {"alias", "format", "format_template"}:
                reads.append(name)
            return super().__getattribute__(name)

    annotation = Annotated[List[str], TrackedInfo()]
    data = ["a"]
    expected = stock.transform(data, annotation)
    expected_reads = reads[:]
    reads.clear()
    assert _utils.transform(data, annotation) == expected
    assert reads == expected_reads


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_equal_union_annotations_keep_distinct_cache_entries(use_async: bool) -> None:
    class First(TypedDict):
        a: Annotated[str, stock.PropertyInfo(alias="b")]

    class Second(TypedDict):
        b: Annotated[str, stock.PropertyInfo(alias="c")]

    forward, reverse = Union[First, Second], Union[Second, First]
    data = {"a": "x"}
    assert _utils.transform(data, forward) == {"c": "x"}
    for index in range(8097):
        strip_annotated_type(type(f"CacheEviction{index}", (), {}))

    if use_async:
        expected = await stock.async_transform(data, reverse)
        result = await _utils.async_transform(data, reverse)
    else:
        expected = stock.transform(data, reverse)
        result = _utils.transform(data, reverse)
    assert expected == {"b": "x"}
    assert result == expected
    assert compile_plan(forward) is not compile_plan(reverse)


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_absent_union_does_not_pollute_present_union(use_async: bool) -> None:
    class First(TypedDict):
        a: Annotated[str, stock.PropertyInfo(alias="b")]

    class Second(TypedDict):
        b: Annotated[str, stock.PropertyInfo(alias="c")]

    Wrapper = TypedDict("Wrapper", {"absent": Union[First, Second], "present": Union[Second, First]})
    data = {"present": {"a": "x"}}
    expected = stock.transform(data, Wrapper)
    assert expected == {"present": {"b": "x"}}
    for index in range(8097):
        strip_annotated_type(type(f"AbsentCacheEviction{index}", (), {}))
    result = await _utils.async_transform(data, Wrapper) if use_async else _utils.transform(data, Wrapper)
    assert result == expected
    fields = get_field_plans(Wrapper)
    assert fields["absent"] is not fields["present"]


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_warm_stock_union_cache_preserves_reverse_order(use_async: bool) -> None:
    class First(TypedDict):
        a: Annotated[str, stock.PropertyInfo(alias="b")]

    class Second(TypedDict):
        b: Annotated[str, stock.PropertyInfo(alias="c")]

    data = {"a": "x"}
    forward, reverse = Union[First, Second], Union[Second, First]
    assert _utils.transform(data, reverse) == {"b": "x"}
    for index in range(8097):
        strip_annotated_type(type(f"WarmCacheEviction{index}", (), {}))
    assert stock.transform(data, forward) == {"c": "x"}
    expected = stock.transform(data, reverse)
    assert expected == {"b": "x"}
    result = await _utils.async_transform(data, reverse) if use_async else _utils.transform(data, reverse)
    assert result == expected


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_python_container_annotation_order_stays_authoritative(use_async: bool) -> None:
    class First(TypedDict):
        a: Annotated[str, stock.PropertyInfo(alias="b")]

    class Second(TypedDict):
        b: Annotated[str, stock.PropertyInfo(alias="c")]

    data = [{"a": "x"}]
    forward, reverse = List[Union[First, Second]], List[Union[Second, First]]
    assert stock.transform([], forward) == []
    expected = stock.transform(data, reverse)
    # Python may canonicalize typing.List before the SDK sees the annotation.
    first_arm = get_args(get_args(reverse)[0])[0]
    assert expected == [{"c" if first_arm is First else "b": "x"}]
    result = await _utils.async_transform(data, reverse) if use_async else _utils.transform(data, reverse)
    assert result == expected


@pytest.mark.asyncio
async def test_unhashable_child_annotations_keep_stock_behavior() -> None:
    class UnhashableFields(TypedDict):
        value: Annotated[str, []]

    empty: dict[str, object] = {}
    assert _utils.transform(empty, UnhashableFields) == stock.transform(empty, UnhashableFields)
    with pytest.raises(TypeError):
        stock.transform({"value": "a"}, UnhashableFields)
    with pytest.raises(TypeError):
        _utils.transform({"value": "a"}, UnhashableFields)
    with pytest.raises(TypeError):
        await _utils.async_transform({"value": "a"}, UnhashableFields)


@pytest.mark.parametrize("annotation", [List[int], List[float]])
@pytest.mark.asyncio
async def test_numeric_lists_visit_only_the_root(annotation: object, monkeypatch: pytest.MonkeyPatch) -> None:
    sync_calls, async_calls = 0, 0
    original_run, original_async_run = optimized._run, optimized._async_run

    def counted_run(data: object, plan: Plan) -> object:
        nonlocal sync_calls
        sync_calls += 1
        return original_run(data, plan)

    async def counted_async_run(data: object, plan: Plan) -> object:
        nonlocal async_calls
        async_calls += 1
        return await original_async_run(data, plan)

    monkeypatch.setattr(optimized, "_run", counted_run)
    monkeypatch.setattr(optimized, "_async_run", counted_async_run)
    data = [1.0] * 10_000
    assert _utils.transform(data, annotation) is data
    assert await _utils.async_transform(data, annotation) is data
    assert sync_calls == async_calls == 1


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_metadata_mutation_preserves_reverse_union_order(use_async: bool) -> None:
    async def run(use_stock: bool) -> object:
        first_info, second_info = stock.PropertyInfo(), stock.PropertyInfo()
        First = TypedDict("First", {"a": Annotated[str, first_info]})
        Second = TypedDict("Second", {"b": Annotated[str, second_info]})
        forward, reverse = Union[First, Second], Union[Second, First]
        data = {"a": "x"}
        if use_async:
            async_operation = stock.async_transform if use_stock else _utils.async_transform
            assert await async_operation(data, forward) == data
            first_info.alias, second_info.alias = "b", "c"
            return await async_operation(data, reverse)
        operation = stock.transform if use_stock else _utils.transform
        assert operation(data, forward) == data
        first_info.alias, second_info.alias = "b", "c"
        return operation(data, reverse)

    expected = await run(True)
    assert expected == {"b": "x"}
    assert await run(False) == expected


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_json_warmup_preserves_custom_mapping_union_order(nested: bool, use_async: bool) -> None:
    async def run(use_stock: bool) -> object:
        class Item(TypedDict):
            a: str

        Wrapper = TypedDict("Wrapper", {"child": Union[Item, Iterable[str]]})
        data: object = {"child": {"a": "x"}} if nested else {"a": "x"}
        first_annotation = Wrapper if nested else Union[Item, Iterable[str]]
        second_annotation = Union[Iterable[str], Item]
        if use_async:
            async_operation = stock.async_transform if use_stock else _utils.async_transform
            assert await async_operation(data, first_annotation) == data
            return await async_operation(UserDict({"a": "x"}), second_annotation)
        operation = stock.transform if use_stock else _utils.transform
        assert operation(data, first_annotation) == data
        return operation(UserDict({"a": "x"}), second_annotation)

    expected = await run(True)
    assert expected == ["a"]
    assert await run(False) == expected


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_unvisited_future_hints_do_not_warm_generic_cache(
    use_async: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run(use_stock: bool) -> object:
        module = ModuleType(f"{__name__}_future_{use_stock}_{use_async}")
        monkeypatch.setitem(sys.modules, module.__name__, module)
        exec(
            """from __future__ import annotations
from typing import List, Union
from typing_extensions import Annotated, TypedDict
from openai._utils import PropertyInfo

class A(TypedDict):
    a: Annotated[str, PropertyInfo(alias="b")]
class B(TypedDict):
    b: Annotated[str, PropertyInfo(alias="c")]
class Left(TypedDict):
    values: List[Union[A, B]]
class Right(TypedDict):
    values: List[Union[B, A]]
class Wrapper(TypedDict):
    left: Left
    right: Right
""",
            module.__dict__,
        )
        annotation = module.__dict__["Wrapper"]
        data = {"right": {"values": [{"a": "x"}]}}
        if use_async:
            async_operation = stock.async_transform if use_stock else _utils.async_transform
            return await async_operation(data, annotation)
        operation = stock.transform if use_stock else _utils.transform
        return operation(data, annotation)

    expected = await run(True)
    assert expected == {"right": {"values": [{"b": "x"}]}}
    assert await run(False) == expected


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_annotation_changes_before_first_dictionary_visit(nested: bool, use_async: bool) -> None:
    async def run(use_stock: bool) -> object:
        Inner = TypedDict("Inner", {"value": str})
        Outer = TypedDict("Outer", {"child": Inner})
        first_data: object = {} if nested else "not a dictionary"
        first_annotation = Outer if nested else Inner
        if use_async:
            async_operation = stock.async_transform if use_stock else _utils.async_transform
            assert await async_operation(first_data, first_annotation) == first_data
            Inner.__annotations__["value"] = Annotated[str, stock.PropertyInfo(alias="renamed")]
            return await async_operation({"value": "x"}, Inner)
        operation = stock.transform if use_stock else _utils.transform
        assert operation(first_data, first_annotation) == first_data
        Inner.__annotations__["value"] = Annotated[str, stock.PropertyInfo(alias="renamed")]
        return operation({"value": "x"}, Inner)

    expected = await run(True)
    assert expected == {"renamed": "x"}
    assert await run(False) == expected


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_annotation_hash_failure_is_deferred_until_field_visit(use_async: bool) -> None:
    class BadMeta(type):
        @override
        def __hash__(cls) -> int:
            raise ValueError("synthetic annotation hash failure")

    class Bad(metaclass=BadMeta):
        pass

    Wrapper = TypedDict("Wrapper", {"absent": Bad})
    empty: dict[str, object] = {}
    present = {"absent": "value"}
    if use_async:
        assert await stock.async_transform(empty, Wrapper) == empty
        assert await _utils.async_transform(empty, Wrapper) == empty
        for async_operation in [stock.async_transform, _utils.async_transform]:
            with pytest.raises(ValueError, match="synthetic annotation hash failure"):
                await async_operation(present, Wrapper)
    else:
        assert stock.transform(empty, Wrapper) == empty
        assert _utils.transform(empty, Wrapper) == empty
        for operation in [stock.transform, _utils.transform]:
            with pytest.raises(ValueError, match="synthetic annotation hash failure"):
                operation(present, Wrapper)
