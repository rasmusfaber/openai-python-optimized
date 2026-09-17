from __future__ import annotations

from typing import Any, List, Union, Iterator, cast
from inspect import signature
from collections import abc
from typing_extensions import Required, Annotated, TypeAlias, TypedDict
from concurrent.futures import ThreadPoolExecutor

import pytest

from openai._utils import _transform as stock
from openai._utils._typing import strip_annotated_type


class First(TypedDict):
    a: Annotated[str, stock.PropertyInfo(alias="b")]


class Second(TypedDict):
    b: Annotated[str, stock.PropertyInfo(alias="c")]


Forward: TypeAlias = Union[First, Second]
Reverse: TypeAlias = Union[Second, First]


@pytest.fixture(autouse=True)
def clear_normalization_cache() -> Iterator[None]:
    cast(Any, strip_annotated_type).cache_clear()
    yield
    cast(Any, strip_annotated_type).cache_clear()


def annotations_and_data(shape: str) -> tuple[object, object, object, object, object]:
    data, forward_result, reverse_result = {"a": "x"}, {"c": "x"}, {"b": "x"}
    if shape == "union":
        return Forward, Reverse, data, forward_result, reverse_result
    if shape == "pep604":
        return First | Second, Second | First, data, forward_result, reverse_result
    if shape == "annotated":
        return Annotated[Forward, object()], Annotated[Reverse, object()], data, forward_result, reverse_result
    if shape == "required":
        return (
            Required[Annotated[Forward, object()]],
            Required[Annotated[Reverse, object()]],
            data,
            forward_result,
            reverse_result,
        )
    if shape == "dict":
        return (
            dict[str, Forward],
            dict[str, Reverse],
            {"item": data},
            {"item": forward_result},
            {"item": reverse_result},
        )
    if shape == "iterable":
        return abc.Iterable[Forward], abc.Iterable[Reverse], [data], [forward_result], [reverse_result]
    if shape == "sequence":
        return abc.Sequence[Forward], abc.Sequence[Reverse], [data], [forward_result], [reverse_result]
    if shape == "annotated_list":
        return (
            Annotated[list[Forward], object()],
            Annotated[list[Reverse], object()],
            [data],
            [forward_result],
            [reverse_result],
        )
    if shape == "required_list":
        return (
            Required[Annotated[list[Forward], object()]],
            Required[Annotated[list[Reverse], object()]],
            [data],
            [forward_result],
            [reverse_result],
        )
    assert shape == "list"
    return list[Forward], list[Reverse], [data], [forward_result], [reverse_result]


@pytest.mark.parametrize(
    "shape",
    [
        "union",
        "pep604",
        "annotated",
        "required",
        "list",
        "dict",
        "iterable",
        "sequence",
        "annotated_list",
        "required_list",
    ],
)
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_union_order_survives_cache_warmup(shape: str, use_async: bool) -> None:
    forward, reverse, data, forward_result, reverse_result = annotations_and_data(shape)
    for annotation, expected in [(reverse, reverse_result), (forward, forward_result), (reverse, reverse_result)]:
        result = await stock.async_transform(data, annotation) if use_async else stock.transform(data, annotation)
        assert result == expected


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_union_order_survives_cache_eviction(use_async: bool) -> None:
    data = {"a": "x"}
    assert stock.transform(data, Forward) == {"c": "x"}
    for index in range(8097):
        strip_annotated_type(type(f"NormalizationEviction{index}", (), {}))
    for annotation, expected in [(Reverse, {"b": "x"}), (Forward, {"c": "x"})]:
        result = await stock.async_transform(data, annotation) if use_async else stock.transform(data, annotation)
        assert result == expected


def test_cache_clear_bound_and_keyword_argument() -> None:
    cached = cast(Any, strip_annotated_type)
    assert tuple(signature(strip_annotated_type).parameters) == ("typ",)
    assert strip_annotated_type(typ=cast(type, Forward)) is Forward
    assert strip_annotated_type(typ=cast(type, Reverse)) is Reverse
    for index in range(8097):
        strip_annotated_type(type(f"BoundedNormalization{index}", (), {}))
    assert cached.cache_info().currsize == cached.cache_info().maxsize == 8096
    cached.cache_clear()
    assert cached.cache_info().currsize == cached.cache_info().hits == cached.cache_info().misses == 0


def test_concurrent_union_normalization_keeps_identity() -> None:
    annotations = [cast(type, Forward), cast(type, Reverse)] * 100
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(strip_annotated_type, annotations))
    assert all(result is annotation for result, annotation in zip(results, annotations, strict=True))


def test_existing_typing_alias_object_is_preserved() -> None:
    # Python may already have canonicalized these aliases during construction.
    # Normalization preserves the object it receives; it cannot undo that step.
    annotation = List[Reverse]
    assert strip_annotated_type(cast(type, annotation)) is annotation
