"""Cached annotation analysis, written by Rasmus Faber for the optimized fork."""

from __future__ import annotations

from typing import Any, cast
from dataclasses import dataclass
from typing_extensions import Literal, get_args, override

from . import _transform as stock
from ._utils import lru_cache
from ._compat import get_origin, is_typeddict
from ._typing import (
    is_list_type,
    is_union_type,
    is_iterable_type,
    is_required_type,
    is_sequence_type,
    is_annotated_type,
)

Kind = Literal["fallback", "leaf", "typed_dict", "dict", "list", "iterable", "sequence", "union"]


@dataclass(frozen=True)
class Plan:
    annotation: type
    inner_type: type
    kind: Kind = "fallback"
    child: Plan | None = None
    arms: tuple[Plan, ...] = ()
    numeric: bool = False
    stripped_type: type | None = None


@dataclass(frozen=True, eq=False)
class _AnnotationKey:
    annotation: type

    @override
    def __hash__(self) -> int:
        return id(self.annotation)

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _AnnotationKey) and self.annotation is other.annotation


@dataclass(frozen=True, eq=False)
class _HintsKey:
    hints: dict[str, Any]

    @override
    def __hash__(self) -> int:
        return id(self.hints)

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _HintsKey) and self.hints is other.hints


def compile_plan(annotation: object) -> Plan:
    type_ = cast(type, annotation)
    try:
        hash(annotation)
    except Exception:
        return Plan(type_, type_)
    return _compile_cached(_AnnotationKey(type_))


@lru_cache(maxsize=512)
def _compile_cached(key: _AnnotationKey) -> Plan:
    # Only complete graphs enter the shared cache. A private build graph also
    # keeps concurrent and recursive construction from sharing partial plans.
    # Identity keys retain their annotations: equal unions can have different
    # arm orders, and a retained object cannot have its ID reused.
    return _build(key.annotation, key.annotation, {}, set())


def get_field_plans(expected_type: type) -> dict[str, Plan]:
    # Resolving a future annotation can populate Python's generic-type cache.
    # Match stock's visitation and cache eviction before reusing field plans.
    hints = stock.get_type_hints(expected_type, include_extras=True)
    return _compile_fields(_HintsKey(hints))


@lru_cache(maxsize=512)
def _compile_fields(key: _HintsKey) -> dict[str, Plan]:
    return {name: compile_plan(annotation) for name, annotation in key.hints.items() if annotation is not None}


def _strip_annotation(annotation: type) -> type:
    hash(annotation)
    while is_required_type(annotation) or is_annotated_type(annotation):
        annotation = cast(type, get_args(annotation)[0])
    return annotation


def _build(
    annotation: type,
    inner_type: type,
    memo: dict[tuple[int, int], Plan],
    active: set[tuple[int, int]],
) -> Plan:
    fallback = Plan(annotation, inner_type)
    # Active call frames and completed memoized plans retain both objects.
    key = (id(annotation), id(inner_type))
    if key in active:
        return fallback
    cached = memo.get(key)
    if cached is not None:
        return cached
    active.add(key)

    try:
        plan = _analyze(annotation, inner_type, memo, active)
    except Exception:
        # Eager analysis can reach unresolved forward references or unavailable
        # optional dependencies that the stock walk never visits for this data.
        # Execution errors are never caught or retried.
        plan = fallback
    finally:
        active.remove(key)
    memo[key] = plan
    return plan


def _analyze(
    annotation: type,
    inner_type: type,
    memo: dict[tuple[int, int], Plan],
    active: set[tuple[int, int]],
) -> Plan:
    stripped = _strip_annotation(inner_type)
    origin = get_origin(stripped) or stripped
    child: Plan | None = None
    arms: tuple[Plan, ...] = ()
    numeric = False
    if is_typeddict(stripped):
        kind: Kind = "typed_dict"
    elif origin == dict:
        kind = "dict"
        args = get_args(stripped)
        item_type = cast(type, args[1]) if len(args) > 1 else object
        child = _build(item_type, item_type, memo, active)
    elif is_list_type(stripped) or is_iterable_type(stripped) or is_sequence_type(stripped):
        kind = "list" if is_list_type(stripped) else "iterable" if is_iterable_type(stripped) else "sequence"
        args = get_args(stripped)
        item_type = cast(type, args[0]) if args else object
        numeric = stock._no_transform_needed(item_type)
        child = _build(annotation, item_type, memo, active)
    elif is_union_type(stripped):
        kind = "union"
        arms = tuple(_build(annotation, subtype, memo, active) for subtype in get_args(stripped))
    else:
        kind = "leaf"
    return Plan(annotation, inner_type, kind, child, arms, numeric, stripped_type=stripped)
