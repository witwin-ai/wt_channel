"""Live radio map while the Tx is dragged: fast previews, then one refine.

``LiveTxScheduler`` owns the timing only; the RadioMap component supplies the three
actions. Every Tx move bumps a generation. One worker task previews at the newest
position (moves that arrive during a preview collapse into one more preview), waits
``settle_s`` of stillness, then runs the full-quality refine. A move during the refine
cancels it, and the refine is told through ``is_current()`` that its result is stale.
"""
import asyncio
import logging
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


class LiveTxScheduler:
    def __init__(
        self,
        preview: Callable[[], Awaitable[None]],
        refine: Callable[[Callable[[], bool]], Awaitable[None]],
        cancel_refine: Callable[[], None],
        *,
        settle_s: float = 1.5,
    ) -> None:
        self._preview = preview
        self._refine = refine
        self._cancel_refine = cancel_refine
        self.settle_s = float(settle_s)
        self._generation = 0
        self._refining = False
        self._task: Optional[asyncio.Task] = None
        self.last_error = ""

    def moved(self) -> None:
        """The Tx moved: preview soon, refine once it is still again."""
        self._generation += 1
        if self._refining:
            self._cancel_refine()
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._work())

    def reset(self) -> None:
        """Drop pending work (an explicit full solve is about to run)."""
        self._generation += 1
        if self._task is not None and not self._task.done() and not self._refining:
            self._task.cancel()
        self._task = None

    async def idle(self) -> None:
        """Wait until the worker has nothing left to do (tests)."""
        while self._task is not None and not self._task.done():
            await asyncio.wait({self._task})

    async def _work(self) -> None:
        try:
            while True:
                generation = self._generation
                await self._preview()
                if self._generation != generation:
                    continue
                await asyncio.sleep(self.settle_s)
                if self._generation != generation:
                    continue
                self._refining = True
                try:
                    await self._refine(lambda: self._generation == generation)
                finally:
                    self._refining = False
                if self._generation == generation:
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a broken solver must not spin the loop
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("live radio map update stopped: %s", self.last_error)
