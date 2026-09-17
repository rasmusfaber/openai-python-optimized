from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Dict, List, Union, Iterable
from datetime import datetime
from collections import UserDict
from typing_extensions import Annotated, TypedDict, get_args, override

import pytest
from pydantic import Field

from openai import _utils
from openai._utils import _transform as stock
from openai._models import BaseModel

from .transform_helpers import transform


class Child(TypedDict):
    value: str


class Recursive(TypedDict, total=False):
    child: Recursive
    value: Annotated[str, stock.PropertyInfo(alias="renamed")]


class ExcludedModel(BaseModel):
    value: str = Field(alias="alias")
    secret: str
    __api_exclude__ = {"secret"}


@pytest.mark.parametrize("shape", ["root", "dict", "list"])
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_model_dump_preserves_sync_async_difference(shape: str, use_async: bool) -> None:
    model = ExcludedModel(alias="a", secret="b")
    data: object = model
    annotation: object = object
    if shape == "dict":
        data, annotation = {"model": model}, Dict[str, object]
    elif shape == "list":
        data, annotation = [model], List[object]
    assert await transform(data, annotation, use_async) == await transform(data, annotation, use_async, backend=stock)


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_recursive_schema(use_async: bool) -> None:
    data = {"value": "a", "child": {"value": "b"}}
    assert await transform(data, Recursive, use_async) == {"renamed": "a", "child": {"renamed": "b"}}


@pytest.mark.parametrize("annotation", [Union[List[int], List[Child]], Union[List[Child], List[int]]])
def test_union_numeric_list_copy_order(annotation: object) -> None:
    data = [{"value": "a"}]
    result = _utils.transform(data, annotation)
    assert result == stock.transform(data, annotation)
    assert result is not data
    assert result[0] is not data[0]


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_mutable_metadata_remains_live(use_async: bool) -> None:
    info = stock.PropertyInfo(discriminator="type")
    annotation = Annotated[List[Union[int, datetime]], info]
    data = [datetime(2026, 1, 1)]
    assert await transform(data, annotation, use_async) == data
    info.format = "iso8601"
    assert await transform(data, annotation, use_async) == await transform(data, annotation, use_async, backend=stock)


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
    result = await transform(data, reverse, use_async)
    assert result == expected


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_metadata_mutation_preserves_reverse_union_order(use_async: bool) -> None:
    for backend in (stock, _utils):
        first_info, second_info = stock.PropertyInfo(), stock.PropertyInfo()
        First = TypedDict("First", {"a": Annotated[str, first_info]})
        Second = TypedDict("Second", {"b": Annotated[str, second_info]})
        forward, reverse = Union[First, Second], Union[Second, First]
        data = {"a": "x"}
        assert await transform(data, forward, use_async, backend=backend) == data
        first_info.alias, second_info.alias = "b", "c"
        assert await transform(data, reverse, use_async, backend=backend) == {"b": "x"}


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_json_warmup_preserves_custom_mapping_union_order(use_async: bool) -> None:
    for backend in (stock, _utils):

        class Item(TypedDict):
            a: str

        Wrapper = TypedDict("Wrapper", {"child": Union[Item, Iterable[str]]})
        data = {"child": {"a": "x"}}
        assert await transform(data, Wrapper, use_async, backend=backend) == data
        result = await transform(UserDict({"a": "x"}), Union[Iterable[str], Item], use_async, backend=backend)
        assert result == ["a"]


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
        return await transform(data, annotation, use_async, backend=stock if use_stock else _utils)

    expected = await run(True)
    assert expected == {"right": {"values": [{"b": "x"}]}}
    assert await run(False) == expected


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_annotation_changes_before_first_dictionary_visit(nested: bool, use_async: bool) -> None:
    for backend in (stock, _utils):
        Inner = TypedDict("Inner", {"value": str})
        Outer = TypedDict("Outer", {"child": Inner})
        first_data: object = {} if nested else "not a dictionary"
        first_annotation = Outer if nested else Inner
        assert await transform(first_data, first_annotation, use_async, backend=backend) == first_data
        Inner.__annotations__["value"] = Annotated[str, stock.PropertyInfo(alias="renamed")]
        assert await transform({"value": "x"}, Inner, use_async, backend=backend) == {"renamed": "x"}


@pytest.mark.parametrize("error_type", [TypeError, ValueError])
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_annotation_hash_failure_is_deferred_until_field_visit(
    error_type: type[Exception], use_async: bool
) -> None:
    class BadMeta(type):
        @override
        def __hash__(cls) -> int:
            raise ValueError("synthetic annotation hash failure")

    class Bad(metaclass=BadMeta):
        pass

    UnhashableFields = TypedDict("UnhashableFields", {"absent": Annotated[str, []]})
    BadFields = TypedDict("BadFields", {"absent": Bad})
    annotation = UnhashableFields if error_type is TypeError else BadFields
    empty: dict[str, object] = {}
    for backend in (stock, _utils):
        assert await transform(empty, annotation, use_async, backend=backend) == empty
        with pytest.raises(error_type):
            await transform({"absent": "value"}, annotation, use_async, backend=backend)
