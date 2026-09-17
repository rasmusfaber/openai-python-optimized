from __future__ import annotations

from types import ModuleType
from typing import Any, TypeVar, cast

from openai import _utils

_T = TypeVar("_T")


async def transform(data: _T, annotation: object, use_async: bool, *, backend: ModuleType = _utils) -> _T:
    if use_async:
        return cast(_T, await backend.async_transform(data, annotation))
    return cast(_T, backend.transform(data, annotation))


def request_body(api: str, count: int) -> dict[str, Any]:
    if api == "chat":
        return {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": [{"type": "text", "text": str(index)}]} for index in range(count)],
        }
    items: list[dict[str, Any]] = []
    for index in range(count):
        if index % 3 == 0:
            items.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": str(index)}]})
        elif index % 3 == 1:
            items.append({"type": "function_call", "call_id": f"call_{index}", "name": "example", "arguments": "{}"})
        else:
            items.append({"type": "function_call_output", "call_id": f"call_{index - 1}", "output": str(index)})
    return {"model": "gpt-test", "input": items}
