"""Annotation identity caching, written by Rasmus Faber for the optimized fork."""

from __future__ import annotations

from typing import Any, TypeVar, Callable, cast
from functools import wraps
from dataclasses import dataclass
from typing_extensions import override

from ._utils import lru_cache

_CallableT = TypeVar("_CallableT", bound=Callable[..., Any])


@dataclass(frozen=True, eq=False)
class _IdentityKey:
    """Retain the annotation so its ID cannot be reused before cache eviction."""

    value: object

    @override
    def __hash__(self) -> int:
        return id(self.value)

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _IdentityKey) and self.value is other.value


def identity_cache(*, maxsize: int) -> Callable[[_CallableT], _CallableT]:
    """Cache a unary annotation normalizer accepting a ``typ`` argument."""

    def decorate(function: _CallableT) -> _CallableT:
        @lru_cache(maxsize=maxsize)
        def cached(key: _IdentityKey) -> object:
            return cast(object, function(key.value))

        @wraps(function)
        def wrapper(typ: object) -> object:
            # Preserve the original normalizer's rejection of unhashable types.
            hash(typ)
            return cached(_IdentityKey(typ))

        for name in ("cache_clear", "cache_info", "cache_parameters"):
            setattr(wrapper, name, getattr(cached, name))
        return cast(_CallableT, wrapper)

    return decorate
