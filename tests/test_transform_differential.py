from __future__ import annotations

import io
import json
import random
from typing import Any, Dict, List, Union, Callable, ClassVar, Iterable, Iterator, cast
from datetime import date, datetime, timezone
from collections import UserDict
from typing_extensions import Literal, Required, Annotated, TypedDict, override

import httpx2
import pytest
import pydantic

from openai import OpenAI, AsyncOpenAI, _utils
from openai._types import omit, not_given
from openai._utils import PropertyInfo, _transform as stock
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming
from openai.types.responses.response_create_params import ResponseCreateParamsNonStreaming

from .transform_helpers import transform, request_body


class First(TypedDict, total=False):
    type: Literal["first"]
    value: Annotated[str, PropertyInfo(alias="first_value")]
    first: List[Dict[str, str]]


class Second(TypedDict, total=False):
    type: Literal["second"]
    other: Annotated[str, PropertyInfo(alias="second_value")]
    second: Dict[str, List[str]]


class PlainFirst(TypedDict, total=False):
    first: List[Dict[str, str]]
    shared: Dict[str, List[str]]


class PlainSecond(TypedDict, total=False):
    second: Dict[str, List[str]]
    shared: Dict[str, List[int]]


class Nested(TypedDict, total=False):
    value: Union[First, Second, List[First], str, None]
    other: Union[PlainFirst, PlainSecond, Dict[str, List[str]]]
    items: Iterable[Union[First, Second]]
    time: Annotated[datetime, PropertyInfo(format="iso8601")]


class Model(pydantic.BaseModel):
    value: str = pydantic.Field(alias="wire_value")
    parsed_arguments: str = "private"
    __api_exclude__: ClassVar[set[str]] = {"parsed_arguments"}


def _input_ids(value: object, path: str = "$", seen: dict[int, str] | None = None) -> dict[int, str]:
    if seen is None:
        seen = {}
    if type(value) in (str, int, float, bool, type(None)) or id(value) in seen:
        return seen
    seen[id(value)] = path
    if isinstance(value, dict):
        for key, child in cast(dict[str, object], value).items():
            _input_ids(child, f"{path}.{key}", seen)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(cast(list[object], value)):
            _input_ids(child, f"{path}[{index}]", seen)
    return seen


def _shape(value: object, inputs: dict[int, str], seen: dict[int, int] | None = None) -> object:
    value_type = type(value)
    if seen is None:
        seen = {}
    if value_type in (str, int, float, bool, type(None)):
        return value_type.__name__, value
    if id(value) in seen:
        return "shared", seen[id(value)]
    seen[id(value)] = len(seen)
    identity = inputs.get(id(value))
    if isinstance(value, dict):
        return (
            value_type.__name__,
            identity,
            [(key, _shape(child, inputs, seen)) for key, child in cast(dict[str, object], value).items()],
        )
    if isinstance(value, (list, tuple)):
        return value_type.__name__, identity, [_shape(child, inputs, seen) for child in cast(list[object], value)]
    if identity is not None:
        return value_type.__name__, identity
    return value_type.__name__, value


async def _compare(factory: Callable[[], object], annotation: object, use_async: bool) -> None:
    left, right = factory(), factory()
    left_ids, right_ids = _input_ids(left), _input_ids(right)
    expected = await transform(left, annotation, use_async, backend=stock)
    actual = await transform(right, annotation, use_async)
    assert _shape(actual, right_ids) == _shape(expected, left_ids)


def _json_tree(rng: random.Random, depth: int) -> object:
    if depth == 0 or rng.random() < 0.45:
        return rng.choice([None, True, False, -1, 3, 0.5, "", "text"])
    if rng.random() < 0.4:
        return [_json_tree(rng, depth - 1) for _ in range(rng.randrange(4))]
    return {
        key: _json_tree(rng, depth - 1) for key in rng.sample(["first", "second", "shared", "extra"], rng.randrange(5))
    }


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("seed", [0, 1, 2, 5, 9, 10, 15, 17, 19])
async def test_generated_json_compositions(seed: int, use_async: bool) -> None:
    annotations: list[object] = [
        Any,
        dict,
        list,
        Dict[str, List[str]],
        Union[PlainFirst, PlainSecond],
        Union[PlainFirst, Dict[str, List[str]], PlainSecond],
        Union[List[PlainFirst], List[PlainSecond], Dict[str, PlainFirst], None],
        Required[Union[PlainFirst, PlainSecond]],
    ]
    for annotation in annotations:
        await _compare(lambda: _json_tree(random.Random(seed), 4), annotation, use_async)


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 7, 9, 12])
async def test_mixed_typed_values(seed: int, use_async: bool) -> None:
    def factory() -> dict[str, object]:
        rng = random.Random(seed)
        values: list[object] = [
            omit,
            not_given,
            Model(wire_value="example", parsed_arguments="parsed"),
            datetime(2026, 1, 2, tzinfo=timezone.utc),
            date(2026, 1, 2),
            UserDict({"value": "custom", "other": "custom"}),
            {"type": "first", "value": "one", "other": "two"},
            {"type": "future", "value": "one", "other": "two"},
            {"type": [], "value": "one", "other": "two"},
            [1, {"value": "x"}],
            ("a", "b"),
            b"bytes",
            None,
        ]
        return {key: rng.choice(values) for key in ("value", "other", "items", "time", "unknown")}

    await _compare(factory, Nested, use_async)


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_shared_inputs_keep_upstream_copy_boundaries(use_async: bool) -> None:
    def factory() -> dict[str, object]:
        numbers = [1, 2, 3]
        shared = {"numbers": numbers}
        return {"shared": shared, "second": shared, "unknown": shared, "first": [shared, shared]}

    await _compare(factory, Union[PlainFirst, PlainSecond, Dict[str, Any]], use_async)


class CountedIterator(Iterator[dict[str, str]]):
    def __init__(self) -> None:
        self.reads = 0

    @override
    def __next__(self) -> dict[str, str]:
        self.reads += 1
        if self.reads > 2:
            raise StopIteration
        return {"type": "first", "value": str(self.reads), "other": "also transformed"}


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_one_shot_inputs_are_not_replayed(use_async: bool) -> None:
    left, right = CountedIterator(), CountedIterator()
    left_body, right_body = {"items": left}, {"items": right}
    expected = await transform(left_body, Nested, use_async, backend=stock)
    actual = await transform(right_body, Nested, use_async)
    assert actual == expected
    assert right.reads == left.reads == 3


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_rejected_input_preserves_consumption(use_async: bool) -> None:
    class FailingIterator(Iterator[str]):
        def __init__(self) -> None:
            self.reads = 0

        @override
        def __next__(self) -> str:
            self.reads += 1
            if self.reads == 2:
                raise ValueError("synthetic iterator failure")
            return "value"

    observations: list[tuple[type[Exception], str, int]] = []
    for backend in (stock, _utils):
        values = FailingIterator()
        try:
            await transform(values, Iterable[str], use_async, backend=backend)
        except Exception as error:
            observations.append((type(error), str(error), values.reads))
    assert observations == [(ValueError, "synthetic iterator failure", 2)] * 2


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_unknown_deep_or_cyclic_data_is_not_traversed(use_async: bool) -> None:
    def factory() -> dict[str, object]:
        cycle: list[object] = []
        cycle.append(cycle)
        deep: object = "leaf"
        for _ in range(1500):
            deep = [deep]
        return {"unknown_cycle": cycle, "unknown_depth": deep}

    left, right = factory(), factory()
    expected = await transform(left, PlainFirst, use_async, backend=stock)
    actual = await transform(right, PlainFirst, use_async)
    assert set(actual) == set(expected)
    assert actual is not right
    for key in right:
        assert actual[key] is right[key]
        assert expected[key] is left[key]


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_file_objects_are_consumed_once(use_async: bool) -> None:
    annotation = Annotated[Union[str, io.IOBase], PropertyInfo(format="base64")]
    left, right = io.BytesIO(b"synthetic bytes"), io.BytesIO(b"synthetic bytes")
    expected = await transform(left, annotation, use_async, backend=stock)
    actual = await transform(right, annotation, use_async)
    assert actual == expected
    assert right.tell() == left.tell()


@pytest.mark.parametrize("api", ["responses", "chat"])
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_large_requests_match_on_wire(api: str, use_async: bool) -> None:
    body = request_body(api, 300)
    annotation = ResponseCreateParamsNonStreaming if api == "responses" else CompletionCreateParamsNonStreaming
    expected = await transform(body, annotation, use_async, backend=stock)
    requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        response: dict[str, Any] = {"id": "test", "model": "gpt-test", "created": 0, "choices": []}
        if api == "responses":
            response.update(object="response", created_at=0, status="completed", output=[])
        else:
            response["object"] = "chat.completion"
        return httpx2.Response(200, json=response)

    if use_async:
        async with AsyncOpenAI(
            api_key="test-key",
            base_url="https://example.test/v1",
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        ) as client:
            if api == "responses":
                await client.responses.create(**body)
            else:
                await client.chat.completions.create(**body)
    else:
        with OpenAI(
            api_key="test-key",
            base_url="https://example.test/v1",
            http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        ) as client:
            if api == "responses":
                client.responses.create(**body)
            else:
                client.chat.completions.create(**body)
    assert requests == [expected]
