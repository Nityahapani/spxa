"""Freeze constructed mathematical objects without constraining subclass initialization."""

from __future__ import annotations

from abc import ABCMeta
from copy import deepcopy
from types import MappingProxyType
from typing import Any

import numpy as np


def _freeze_parameter(value: Any) -> Any:
    """Seal array data and nested parameter sequences without changing their values."""
    if isinstance(value, np.ndarray):
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    if isinstance(value, list | tuple):
        return tuple(_freeze_parameter(item) for item in value)
    return value


def _parameter_view(value: Any) -> Any:
    """Expose array metadata on fresh views, never on the stored canonical array."""
    if isinstance(value, np.ndarray):
        return value.view()
    if isinstance(value, tuple):
        return tuple(_parameter_view(item) for item in value)
    return value


class ImmutableMeta(ABCMeta):
    """Seal objects only after the complete most-derived initializer returns."""

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        instance = super().__call__(*args, **kwargs)
        for name, value in vars(instance).items():
            object.__setattr__(instance, name, _freeze_parameter(value))
        object.__setattr__(instance, "_sealed", True)
        return instance


class Immutable:
    """Prevent ordinary parameter mutation, including NumPy data/metadata changes."""

    def __getattribute__(self, name: str) -> Any:
        value = object.__getattribute__(self, name)
        if isinstance(value, np.ndarray | tuple) and vars(self).get("_sealed", False):
            return _parameter_view(value)
        return value

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError(f"{type(self).__name__} is immutable; construct a new instance")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError(f"{type(self).__name__} is immutable")
        object.__delattr__(self, name)

    def __deepcopy__(self, memo: dict[int, Any]) -> Immutable:
        """Copy a graph without reopening arrays or collapsing process identities."""
        copied = object.__new__(type(self))
        memo[id(self)] = copied
        for name, value in vars(self).items():
            item = value if isinstance(value, MappingProxyType) else deepcopy(value, memo)
            object.__setattr__(copied, name, _freeze_parameter(item))
        return copied

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore supported object states without making parameter arrays mutable."""
        for name, value in state.items():
            object.__setattr__(self, name, _freeze_parameter(value))
        object.__setattr__(self, "_sealed", True)
