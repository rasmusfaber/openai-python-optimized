"""Compiled request transforms with upstream fallback."""

from __future__ import annotations

from typing import Any, TypeVar, cast
from typing_extensions import get_args

from . import _transform as stock
from ._utils import is_given
from ._transform_plan import Plan, compile_plan, get_field_plans
from ._transform_fusion import try_fuse

_T = TypeVar("_T")
_NONE_TYPE = type(None)


def transform(data: _T, expected_type: object) -> _T:
    return cast(_T, _run(data, compile_plan(expected_type)))


def maybe_transform(data: object, expected_type: object) -> Any | None:
    if data is None:
        return None
    return transform(data, expected_type)


async def async_transform(data: _T, expected_type: object) -> _T:
    return cast(_T, await _async_run(data, compile_plan(expected_type)))


async def async_maybe_transform(data: object, expected_type: object) -> Any | None:
    if data is None:
        return None
    return await async_transform(data, expected_type)


def _run(data: object, plan: Plan) -> object:
    data_type = type(data)
    if plan.kind == "fallback" or (
        data_type is not dict and data_type is not list and not _is_builtin_scalar(data_type)
    ):
        return stock._transform_recursive(data, annotation=plan.annotation, inner_type=plan.inner_type)

    if stock.strip_annotated_type(plan.inner_type) is not plan.stripped_type:
        return stock._transform_recursive(data, annotation=plan.annotation, inner_type=plan.inner_type)

    if data_type is list and plan.numeric:
        return data

    fused, fused_result = try_fuse(data, plan)
    if fused:
        return fused_result

    if data_type is dict:
        mapping = cast(dict[str, object], data)
        if plan.kind == "typed_dict":
            assert plan.stripped_type is not None
            fields = get_field_plans(plan.stripped_type)
            result: dict[str, object] = {}
            for key, value in mapping.items():
                if not is_given(value):
                    continue
                field = fields.get(key)
                if field is None:
                    result[key] = value
                else:
                    result[stock._maybe_transform_key(key, field.annotation)] = _run(value, field)
            return result
        if plan.kind == "dict":
            assert plan.child is not None
            return {key: _run(value, plan.child) for key, value in mapping.items()}
        if plan.kind == "iterable":
            return data

    if data_type is list and plan.kind in ("list", "iterable", "sequence"):
        assert plan.child is not None
        return [_run(value, plan.child) for value in cast(list[object], data)]

    if plan.kind == "union":
        for arm in plan.arms:
            data = _run(data, arm)
        return data

    annotated = stock._get_annotated_type(plan.annotation)
    if annotated is None:
        return data
    for info in get_args(annotated)[1:]:
        if isinstance(info, stock.PropertyInfo) and info.format is not None:
            return stock._format_data(data, info.format, info.format_template)
    return data


async def _async_run(data: object, plan: Plan) -> object:
    data_type = type(data)
    if plan.kind == "fallback" or (
        data_type is not dict and data_type is not list and not _is_builtin_scalar(data_type)
    ):
        return await stock._async_transform_recursive(data, annotation=plan.annotation, inner_type=plan.inner_type)

    if stock.strip_annotated_type(plan.inner_type) is not plan.stripped_type:
        return await stock._async_transform_recursive(data, annotation=plan.annotation, inner_type=plan.inner_type)

    if data_type is list and plan.numeric:
        return data

    fused, fused_result = try_fuse(data, plan)
    if fused:
        return fused_result

    if data_type is dict:
        mapping = cast(dict[str, object], data)
        if plan.kind == "typed_dict":
            assert plan.stripped_type is not None
            fields = get_field_plans(plan.stripped_type)
            result: dict[str, object] = {}
            for key, value in mapping.items():
                if not is_given(value):
                    continue
                field = fields.get(key)
                if field is None:
                    result[key] = value
                else:
                    result[stock._maybe_transform_key(key, field.annotation)] = await _async_run(value, field)
            return result
        if plan.kind == "dict":
            assert plan.child is not None
            # Stock async dict transforms deliberately use the sync walker.
            return {key: _run(value, plan.child) for key, value in mapping.items()}
        if plan.kind == "iterable":
            return data

    if data_type is list and plan.kind in ("list", "iterable", "sequence"):
        assert plan.child is not None
        return [await _async_run(value, plan.child) for value in cast(list[object], data)]

    if plan.kind == "union":
        for arm in plan.arms:
            data = await _async_run(data, arm)
        return data

    annotated = stock._get_annotated_type(plan.annotation)
    if annotated is None:
        return data
    for info in get_args(annotated)[1:]:
        if isinstance(info, stock.PropertyInfo) and info.format is not None:
            return await stock._async_format_data(data, info.format, info.format_template)
    return data


def _is_builtin_scalar(type_: type) -> bool:
    return type_ is str or type_ is int or type_ is float or type_ is bool or type_ is _NONE_TYPE
