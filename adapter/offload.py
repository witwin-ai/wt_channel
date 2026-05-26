"""Run the blocking GPU solve off the event loop.

The studio server awaits component button callbacks directly on its asyncio event loop,
so a multi-second synchronous solve blocks the loop. While blocked, the loop cannot flush
the ``_safe_broadcast`` messages the solve emits (notifications, object_created /
object_updated), and because ``server._safe_broadcast`` schedules them with an unreferenced
``asyncio.create_task`` (asyncio keeps only weak refs to tasks), the heavy solve's GC churn
can collect those tasks before they run -> the frontend silently receives nothing.

Running the solve in a worker thread keeps the loop free so the broadcasts deliver. All
solves share **one** worker thread because the platform's drjit / CUDA / OptiX runtime is
thread-affine -- reusing a single thread keeps that context consistent across solves.
"""
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wt-channel-solve")


async def run_off_loop(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Await ``func(*args, **kwargs)`` on the shared channel-solve worker thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_POOL, functools.partial(func, *args, **kwargs))
