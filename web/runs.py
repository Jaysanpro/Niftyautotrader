"""Live progress for a run, so the console can show the desk thinking.

The pipeline is synchronous: ``analyse`` returns a decision and nothing in
between. But the agents already narrate themselves to the log - "fundamentals:
MILDLY_BULLISH (53%)", "research manager: BULLISH (conviction 85%)" - so the
trace exists already and only needs somewhere to go.

That is what this does: a run executes on its own thread, a logging handler
keyed by thread id puts that thread's records on that run's queue, and the
browser reads the queue over SSE. The framework is untouched; nothing in
``tradingagents_in`` knows a web page is watching.

Keyed by a context variable, because LangGraph fans the four analysts out
onto worker threads and copies the context into them - so the run id follows
the work, while a thread id does not. Thread id is kept as a fallback for any
executor that does not propagate context; without the context variable the
analysts, the most interesting part of the trace, were the exact lines that
went missing.
"""

from __future__ import annotations

import contextvars
import logging
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# A run is kept this long after finishing, so a browser that reconnects can
# still collect the result instead of being told the run never existed.
RETENTION_SECONDS = 900
MAX_RUNS = 64
# Bounded so a chatty run cannot grow without limit if nobody is reading.
QUEUE_MAX = 2048


@dataclass
class Run:
    run_id: str
    symbol: str
    status: str = "queued"          # queued | running | done | failed
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    decision: dict[str, Any] | None = None
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=QUEUE_MAX))
    thread_id: int | None = None

    def publish(self, event: dict[str, Any]) -> None:
        event.setdefault("at", time.time())
        self.events.append(event)
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            # A reader that cannot keep up loses lines, not the run.
            pass


# Set on the run's thread and carried into LangGraph's analyst workers.
_current_run: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "ita_run_id", default=None
)


class _RunRouter(logging.Handler):
    """Routes each log record to the run that produced it."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self._by_id: dict[str, Run] = {}
        self._by_thread: dict[int, Run] = {}
        self._lock = threading.Lock()

    def attach(self, run: Run, thread_id: int) -> None:
        with self._lock:
            self._by_id[run.run_id] = run
            self._by_thread[thread_id] = run

    def detach(self, run: Run, thread_id: int) -> None:
        with self._lock:
            self._by_id.pop(run.run_id, None)
            self._by_thread.pop(thread_id, None)

    def emit(self, record: logging.LogRecord) -> None:
        run_id = _current_run.get()
        with self._lock:
            run = self._by_id.get(run_id) if run_id else None
            if run is None:
                run = self._by_thread.get(record.thread)
        if run is None:
            return
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a bad format string must not kill a run
            return
        run.publish({
            "type": "log",
            "level": record.levelname.lower(),
            "logger": record.name.rsplit(".", 1)[-1],
            "message": message,
            "agent": _agent_of(message),
        })


# "fundamentals: MILDLY_BULLISH (53%)" and friends. The console highlights the
# agent that spoke; an unmatched line is still shown, just not attributed.
_AGENTS = (
    "fundamentals", "sentiment", "news_macro", "technical",
    "bull", "bear", "research manager", "trader",
    "risk manager", "portfolio manager",
)
_AGENT_RE = re.compile(
    r"^(" + "|".join(re.escape(a) for a in _AGENTS) + r")\b", re.IGNORECASE
)


def _agent_of(message: str) -> str | None:
    match = _AGENT_RE.match(message.strip())
    return match.group(1).lower().replace(" ", "_") if match else None


_router = _RunRouter()
_installed = False
_install_lock = threading.Lock()


def _install() -> None:
    global _installed
    with _install_lock:
        if _installed:
            return
        root = logging.getLogger("tradingagents_in")
        root.addHandler(_router)
        # Records must reach this handler even if the app's own level is higher.
        if root.level > logging.INFO or root.level == logging.NOTSET:
            root.setLevel(logging.INFO)
        _installed = True


class RunRegistry:
    """The runs this process knows about, newest first."""

    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()

    def create(self, symbol: str) -> Run:
        self._evict()
        run = Run(run_id=f"R-{uuid.uuid4().hex[:10]}", symbol=symbol)
        with self._lock:
            self._runs[run.run_id] = run
        return run

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            return self._runs.get(run_id)

    def recent(self, limit: int = 20) -> list[Run]:
        with self._lock:
            runs = sorted(self._runs.values(), key=lambda r: r.started_at, reverse=True)
        return runs[:limit]

    def _evict(self) -> None:
        cutoff = time.time() - RETENTION_SECONDS
        with self._lock:
            stale = [
                rid for rid, run in self._runs.items()
                if run.finished_at and run.finished_at < cutoff
            ]
            for rid in stale:
                self._runs.pop(rid, None)
            if len(self._runs) > MAX_RUNS:
                oldest = sorted(self._runs.values(), key=lambda r: r.started_at)
                for run in oldest[: len(self._runs) - MAX_RUNS]:
                    self._runs.pop(run.run_id, None)

    def execute(self, run: Run, work) -> None:
        """Run ``work()`` on this thread, narrating it onto the run's queue.

        ``work`` returns the decision as a dict. Any exception becomes a
        failed run with a message rather than a traceback nobody sees.
        """
        _install()
        run.thread_id = threading.get_ident()
        _router.attach(run, run.thread_id)
        token = _current_run.set(run.run_id)
        run.status = "running"
        run.publish({"type": "status", "status": "running", "symbol": run.symbol})
        try:
            decision = work()
            run.decision = decision
            run.status = "done"
            run.publish({"type": "decision", "decision": decision})
        except Exception as exc:  # noqa: BLE001 - surfaced to the browser
            run.status = "failed"
            run.error = f"{type(exc).__name__}: {exc}"
            run.publish({"type": "error", "error": run.error})
        finally:
            run.finished_at = time.time()
            run.publish({"type": "status", "status": run.status, "done": True})
            _current_run.reset(token)
            if run.thread_id is not None:
                _router.detach(run, run.thread_id)


registry = RunRegistry()
