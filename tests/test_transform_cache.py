from __future__ import annotations

from typing import Any, Union, Iterator, cast
from typing_extensions import Required, Annotated, TypeAlias, TypedDict

import pytest

from openai import _utils
from openai._utils import _transform as stock
from openai._utils._typing import strip_annotated_type

from .transform_helpers import transform


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
    if shape == "pep604":
        forward, reverse = First | Second, Second | First
    elif shape == "required":
        forward, reverse = Required[Annotated[Forward, object()]], Required[Annotated[Reverse, object()]]
    elif shape == "list":
        return list[Forward], list[Reverse], [data], [forward_result], [reverse_result]
    elif shape == "dict":
        return (
            dict[str, Forward],
            dict[str, Reverse],
            {"item": data},
            {"item": forward_result},
            {"item": reverse_result},
        )
    else:
        forward, reverse = Forward, Reverse
    return forward, reverse, data, forward_result, reverse_result


@pytest.mark.parametrize("shape", ["union", "pep604", "required", "list", "dict"])
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_union_order_survives_cache_warmup(shape: str, use_async: bool) -> None:
    forward, reverse, data, forward_result, reverse_result = annotations_and_data(shape)
    for backend in (stock, _utils, stock, _utils):
        for annotation, expected in [(reverse, reverse_result), (forward, forward_result), (reverse, reverse_result)]:
            assert await transform(data, annotation, use_async, backend=backend) == expected
