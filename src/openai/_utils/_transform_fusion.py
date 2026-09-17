"""Shape-directed copy fusion, written by Rasmus Faber for the optimized fork."""

from __future__ import annotations

from typing import Any, cast
from threading import Lock
from collections import OrderedDict
from dataclasses import field, dataclass
from typing_extensions import Literal, get_args, override

from . import _transform as stock
from ._typing import is_required_type, is_annotated_type
from ._transform_plan import Plan, compile_plan

_MAX_NODES = 100_000
_MAX_DEPTH = 64
_MAX_WORK = 250_000
_MAX_HINTS = 512  # Below the stock hint cache's 8096 entries.
_NONE_TYPE = type(None)


class _Decline(Exception):
    pass


@dataclass(frozen=True)
class _Shape:
    kind: Literal["dict", "list"]
    children: tuple[_Shape | None, ...]
    keys: tuple[str, ...] = ()
    fingerprint: int = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fingerprint", hash((self.kind, self.keys, self.children)))

    @override
    def __hash__(self) -> int:
        return self.fingerprint


@dataclass(frozen=True)
class _Effect:
    kind: Literal["dict", "list"]
    children: tuple[_Effect | None, ...] | None = None


@dataclass(frozen=True)
class _Dependency:
    annotation: type
    hints: dict[str, Any]
    prefix: tuple[type, ...]


@dataclass(frozen=True)
class _Compiled:
    plan: Plan
    effect: _Effect | None
    dependencies: tuple[_Dependency, ...]
    last_use: tuple[type, ...]


@dataclass(frozen=True)
class _Derived:
    effect: _Effect | None
    visits: tuple[int, ...]


_CACHE: OrderedDict[tuple[int, _Shape], _Compiled] = OrderedDict()
_LOCK = Lock()


def _probe(data: object) -> _Shape:
    pending: list[tuple[object, int, bool]] = [(data, 0, False)]
    seen: set[int] = set()
    completed: dict[int, _Shape] = {}
    canonical: dict[_Shape, _Shape] = {}
    remaining = _MAX_NODES
    while pending:
        value, depth, closing = pending.pop()
        value_type = type(value)
        if closing:
            if value_type is dict:
                mapping = cast(dict[str, object], value)
                shape = _Shape("dict", tuple(completed.get(id(item)) for item in mapping.values()), tuple(mapping))
            else:
                shape = _Shape("list", tuple(completed.get(id(item)) for item in cast(list[object], value)))
            completed[id(value)] = canonical.setdefault(shape, shape)
            continue
        remaining -= 1
        if remaining < 0:
            raise _Decline
        if (
            value_type is str
            or value_type is int
            or value_type is float
            or value_type is bool
            or value_type is _NONE_TYPE
        ):
            continue
        if value_type is not dict and value_type is not list:
            raise _Decline
        if depth >= _MAX_DEPTH or id(value) in seen:
            raise _Decline
        seen.add(id(value))
        pending.append((value, depth, True))
        if value_type is dict:
            mapping = cast(dict[str, object], value)
            if len(mapping) + len(pending) > remaining:
                raise _Decline
            for key, item in mapping.items():
                if type(key) is not str:
                    raise _Decline
                pending.append((item, depth + 1, False))
        else:
            items = cast(list[object], value)
            if len(items) + len(pending) > remaining:
                raise _Decline
            if len(items) > 64:
                first_type = type(items[0])
                # Avoid scanning nested vectors that the faithful numeric-list
                # shortcut can return unchanged. Mixed lists may decline too.
                if (
                    first_type is str
                    or first_type is int
                    or first_type is float
                    or first_type is bool
                    or first_type is _NONE_TYPE
                ):
                    raise _Decline
            pending.extend((item, depth + 1, False) for item in items)
    return completed[id(data)]


def _without_metadata(annotation: type) -> None:
    while is_required_type(annotation):
        annotation = cast(type, get_args(annotation)[0])
    if is_annotated_type(annotation):
        raise _Decline


class _Builder:
    def __init__(self) -> None:
        self.remaining = _MAX_WORK
        self.memo: dict[tuple[int, _Shape | None], _Derived] = {}
        self.plans: dict[int, Plan] = {}
        self.dependencies: dict[int, _Dependency] = {}
        self.last_use: dict[int, None] = {}
        self.previous: tuple[int, ...] = ()

    def step(self, count: int = 1) -> None:
        self.remaining -= count
        if self.remaining < 0:
            raise _Decline

    def record(self, visits: tuple[int, ...]) -> None:
        if visits is self.previous:
            return
        for identity in visits:
            self.last_use.pop(identity, None)
            self.last_use[identity] = None
        self.previous = visits
        self.step(len(visits))

    def prefix(self) -> tuple[type, ...]:
        return tuple(self.dependencies[identity].annotation for identity in self.last_use)

    def derive(self, plan: Plan, shape: _Shape | None) -> _Derived:
        self.step()
        key = (id(plan), shape)
        if key in self.memo:
            derived = self.memo[key]
            self.record(derived.visits)
            return derived
        self.plans[id(plan)] = plan  # Retain memo-key identities until the build completes.
        if plan.kind == "fallback":
            raise _Decline
        _without_metadata(plan.annotation)
        _without_metadata(plan.inner_type)
        touches: dict[int, None] = {}
        previous: tuple[int, ...] = ()

        def visit(child_plan: Plan, child_shape: _Shape | None) -> _Effect | None:
            nonlocal previous
            derived = self.derive(child_plan, child_shape)
            if derived.visits is not previous:
                self.step(len(derived.visits))
                for identity in derived.visits:
                    touches.pop(identity, None)
                    touches[identity] = None
                previous = derived.visits
            return derived.effect

        effect: _Effect | None = None
        if shape is not None and shape.kind == "dict" and plan.kind == "typed_dict":
            assert plan.stripped_type is not None
            identity = id(plan.stripped_type)
            dependency = self.dependencies.get(identity)
            if dependency is None:
                if len(self.dependencies) >= _MAX_HINTS:
                    raise _Decline
                self.step(len(self.last_use))
                prefix = self.prefix()
                hints = stock.get_type_hints(plan.stripped_type, include_extras=True)
                self.dependencies[identity] = _Dependency(plan.stripped_type, hints, prefix)
            else:
                hints = dependency.hints
            self.record((identity,))
            touches[identity] = None
            children: list[_Effect | None] = []
            for name, child_shape in zip(shape.keys, shape.children, strict=True):
                annotation = hints.get(name)
                children.append(None if annotation is None else visit(compile_plan(annotation), child_shape))
            effect = self.container(shape.kind, tuple(children))
        elif shape is not None and (
            (shape.kind == "dict" and plan.kind == "dict")
            or (shape.kind == "list" and plan.kind in ("list", "iterable", "sequence") and not plan.numeric)
        ):
            assert plan.child is not None
            effect = self.container(shape.kind, tuple(visit(plan.child, child) for child in shape.children))
        elif plan.kind == "union":
            if len(plan.arms) > self.remaining:
                raise _Decline
            for arm in plan.arms:
                effect = self.compose(effect, visit(arm, shape))
        derived = _Derived(effect, tuple(touches))
        self.memo[key] = derived
        return derived

    def container(self, kind: Literal["dict", "list"], children: tuple[_Effect | None, ...]) -> _Effect:
        self.step()
        return _Effect(kind, children if any(child is not None for child in children) else None)

    def compose(self, first: _Effect | None, second: _Effect | None) -> _Effect | None:
        self.step()
        if first is None:
            return second
        if second is None:
            return first
        assert first.kind == second.kind
        if first.children is None:
            return second
        if second.children is None:
            return first
        return self.container(
            first.kind, tuple(self.compose(a, b) for a, b in zip(first.children, second.children, strict=True))
        )


def try_fuse(data: object, plan: Plan) -> tuple[bool, object]:
    data_type = type(data)
    if not (
        plan.kind == "union"
        and (data_type is dict or data_type is list)
        or data_type is dict
        and plan.kind in ("typed_dict", "dict")
        or data_type is list
        and plan.kind in ("list", "iterable", "sequence")
        and not plan.numeric
    ):
        return False, data
    try:
        shape = _probe(data)
        key = (id(plan), shape)
        with _LOCK:
            entry = _CACHE.get(key)
            if entry is not None:
                _CACHE.move_to_end(key)
        if entry is not None:
            for dependency in entry.dependencies:
                try:
                    hints = stock.get_type_hints(dependency.annotation, include_extras=True)
                except BaseException:
                    _replay(dependency.prefix)
                    raise
                if hints is not dependency.hints:
                    # Rebuild immediately: later old dependencies may no longer
                    # precede descendants introduced by the changed hints.
                    entry = None
                    break
        if entry is None:
            builder = _Builder()
            try:
                derived = builder.derive(plan, shape)
            except BaseException:
                # Restore only successful visits before propagating a resolver
                # failure or declining fusion; never retry the failing lookup.
                _replay(builder.prefix())
                raise
            entry = _Compiled(
                plan,
                derived.effect,
                tuple(builder.dependencies.values()),
                tuple(builder.dependencies[identity].annotation for identity in derived.visits),
            )
            with _LOCK:
                _CACHE[key] = entry
                _CACHE.move_to_end(key)
                if len(_CACHE) > 32:
                    _CACHE.popitem(last=False)
    except _Decline:
        return False, data
    # First visits resolve hints in stock order. Fewer distinct schemas than
    # the hint cache can hold means none can be evicted during this traversal.
    # Replaying final visits restores stock's LRU order after memoization.
    _replay(entry.last_use)
    return True, _apply(data, entry.effect)


def _replay(visits: tuple[type, ...]) -> None:
    for type_ in visits:
        stock.get_type_hints(type_, include_extras=True)


def _apply(data: object, effect: _Effect | None) -> object:
    if effect is None:
        return data
    if effect.kind == "dict":
        mapping = cast(dict[str, object], data)
        if effect.children is None:
            return mapping.copy()
        return {key: _apply(value, child) for (key, value), child in zip(mapping.items(), effect.children, strict=True)}
    items = cast(list[object], data)
    if effect.children is None:
        return items.copy()
    return [_apply(value, child) for value, child in zip(items, effect.children, strict=True)]
