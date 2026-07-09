"""Runtime compatibility shims for optional platform dependencies."""
from __future__ import annotations

import inspect
from collections import namedtuple


_ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")


def _getargspec(func):
    spec = inspect.getfullargspec(func)
    return _ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)


def apply_platform_compat() -> None:
    """Patch Python 3.11 removals used by older optional platform packages."""
    if not hasattr(inspect, "ArgSpec"):
        inspect.ArgSpec = _ArgSpec  # type: ignore[attr-defined]
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = _getargspec  # type: ignore[attr-defined]
