"""Runtime compatibility shims for optional platform dependencies."""
from __future__ import annotations

import inspect


def apply_platform_compat() -> None:
    """Patch Python 3.11 removals used by older optional platform packages."""
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
