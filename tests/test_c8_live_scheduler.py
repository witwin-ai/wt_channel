"""C8 — the live Tx scheduler (CPU, pure asyncio).

While the Tx is dragged the map follows with fast previews; moves that arrive during a
preview collapse into one more preview at the newest position. Once the Tx has been
still for ``settle_s`` one full-quality refine runs. A move during the refine cancels it
and its result must not be shown.
"""
import asyncio

from wt_channel.adapter.live import LiveTxScheduler


class _Recorder:
    def __init__(self):
        self.calls = []
        self.cancelled = 0
        self.preview_gate = None
        self.refine_gate = None
        self.refine_current = []

    async def preview(self):
        self.calls.append("preview")
        if self.preview_gate is not None:
            await self.preview_gate.wait()

    async def refine(self, is_current):
        self.calls.append("refine")
        if self.refine_gate is not None:
            await self.refine_gate.wait()
        self.refine_current.append(is_current())

    def cancel_refine(self):
        self.cancelled += 1


def _scheduler(rec, settle_s=0.01):
    return LiveTxScheduler(rec.preview, rec.refine, rec.cancel_refine, settle_s=settle_s)


def test_one_move_previews_then_refines_once():
    async def main():
        rec = _Recorder()
        sched = _scheduler(rec)
        sched.moved()
        await sched.idle()
        return rec

    rec = asyncio.run(main())
    assert rec.calls == ["preview", "refine"]
    assert rec.refine_current == [True]


def test_moves_during_a_preview_collapse_into_one_more_preview():
    async def main():
        rec = _Recorder()
        rec.preview_gate = asyncio.Event()
        sched = _scheduler(rec)
        sched.moved()
        await asyncio.sleep(0)
        for _ in range(5):
            sched.moved()
        rec.preview_gate.set()
        await sched.idle()
        return rec

    rec = asyncio.run(main())
    assert rec.calls == ["preview", "preview", "refine"]


def test_a_move_while_settling_skips_the_stale_refine():
    async def main():
        rec = _Recorder()
        sched = _scheduler(rec, settle_s=0.05)
        sched.moved()
        await asyncio.sleep(0.02)  # first preview done, settling
        sched.moved()
        await sched.idle()
        return rec

    rec = asyncio.run(main())
    assert rec.calls == ["preview", "preview", "refine"]


def test_a_move_during_the_refine_cancels_it_and_drops_its_result():
    async def main():
        rec = _Recorder()
        rec.refine_gate = asyncio.Event()
        sched = _scheduler(rec)
        sched.moved()
        while "refine" not in rec.calls:
            await asyncio.sleep(0.005)
        sched.moved()
        rec.refine_gate.set()
        await sched.idle()
        return rec

    rec = asyncio.run(main())
    assert rec.cancelled == 1
    assert rec.calls == ["preview", "refine", "preview", "refine"]
    assert rec.refine_current == [False, True]


def test_reset_drops_pending_work_before_an_explicit_solve():
    async def main():
        rec = _Recorder()
        sched = _scheduler(rec)
        sched.moved()
        sched.reset()
        await asyncio.sleep(0.05)
        await sched.idle()
        return rec

    assert asyncio.run(main()).calls == []


def test_a_failed_preview_stops_the_worker_instead_of_spinning():
    async def main():
        rec = _Recorder()

        async def broken():
            rec.calls.append("preview")
            raise RuntimeError("solver down")

        sched = LiveTxScheduler(broken, rec.refine, rec.cancel_refine, settle_s=0.01)
        sched.moved()
        await sched.idle()
        return rec, sched

    rec, sched = asyncio.run(main())
    assert rec.calls == ["preview"]
    assert "solver down" in sched.last_error
