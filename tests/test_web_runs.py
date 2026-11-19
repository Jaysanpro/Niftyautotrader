"""The run registry and its SSE stream.

Both bugs pinned here were found by watching a real screen, not by reading
the code, and both looked identical from the browser: a stream that stops
while the work carries on.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

from web.app import stream_run
from web.runs import Run, registry


def _drain(run: Run, limit: int = 60) -> list[dict]:
    """Read the real endpoint's stream to its end.

    Deliberately the handler itself rather than a copy of its logic: both
    bugs pinned here were in that handler, and a reimplementation here would
    have passed while the endpoint stayed broken.
    """

    async def read() -> list[dict]:
        out: list[dict] = []
        async for chunk in stream_run(run.run_id).body_iterator:
            if not chunk.startswith("data: "):
                continue          # a keep-alive comment
            out.append(json.loads(chunk[6:]))
            if len(out) >= limit:
                break
        return out

    return asyncio.run(read())


def test_events_are_not_replayed_twice():
    """Replaying ``events`` leaves the same events on the queue.

    Without a sequence number a subscriber sees everything in flight twice,
    which on a screen is a progress bar that jumps backwards.
    """
    run = registry.create("NIFTYIT", kind="screen")
    started = threading.Event()

    def work():
        started.set()
        for i in range(4):
            run.publish({"type": "progress", "completed": i + 1, "total": 4})
            time.sleep(0.05)
        return {"scanned": 4}

    thread = threading.Thread(
        target=lambda: registry.execute(run, work, result_type="screen"), daemon=True
    )
    thread.start()
    started.wait(5)
    time.sleep(0.12)          # let a couple of events land before subscribing

    events = _drain(run)
    thread.join(5)

    seqs = [e["seq"] for e in events]
    assert len(seqs) == len(set(seqs)), f"duplicate events: {seqs}"
    assert [e["completed"] for e in events if e["type"] == "progress"] == [1, 2, 3, 4]


def test_a_payload_carrying_done_does_not_end_the_stream():
    """``done`` is the terminal sentinel, so only ``True`` may end a stream.

    A screen's progress event once used ``done`` for its count of finished
    names. The first name set it to 1, the stream closed, and the console
    sat on one line while 63 more runs went by unseen.
    """
    run = registry.create("NIFTYIT", kind="screen")
    started = threading.Event()

    def work():
        started.set()
        for i in range(3):
            # Deliberately hostile: a count under the sentinel's name.
            run.publish({"type": "progress", "done": i + 1, "total": 3})
            time.sleep(0.08)
        return {"scanned": 3}

    # The events must arrive live rather than as replay. The check that
    # matters is in the follow loop, and a run that finished before the
    # subscriber connected never reaches it - this test passed against the
    # bug until it was made concurrent.
    thread = threading.Thread(
        target=lambda: registry.execute(run, work, result_type="screen"), daemon=True
    )
    thread.start()
    started.wait(5)

    events = _drain(run)
    thread.join(5)
    assert events[-1].get("done") is True, "only the terminal event ends the stream"
    assert sum(1 for e in events if e["type"] == "progress") == 3
    assert any(e["type"] == "screen" for e in events), "the result must still arrive"


def test_a_screen_keeps_only_warnings():
    """A screen is dozens of analyses; forwarding every INFO line buries it."""
    import logging

    screen_run = registry.create("NIFTY50", kind="screen")
    analyse_run = registry.create("TCS.NS")
    assert screen_run.min_level == logging.WARNING
    assert analyse_run.min_level == logging.INFO
