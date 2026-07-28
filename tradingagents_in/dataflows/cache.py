"""On-disk TTL cache.

Market data endpoints are rate-limited and a single run touches each source
several times. Caching keeps a debate reproducible within a session and keeps
us off the wrong side of the data providers.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from tradingagents_in.utils.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")

_DEFAULT_DIR = Path(os.getenv("ITA_CACHE_DIR", ".cache/tradingagents_in"))
_DEFAULT_TTL = int(os.getenv("ITA_CACHE_TTL_SECONDS", "900"))


def _key(namespace: str, params: dict[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, default=str)
    digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
    safe_ns = "".join(c if c.isalnum() or c in "-_." else "_" for c in namespace)
    return f"{safe_ns}.{digest}"


class DiskCache:
    def __init__(self, directory: str | Path | None = None, ttl_seconds: int | None = None):
        self.dir = Path(directory or _DEFAULT_DIR)
        self.ttl = ttl_seconds if ttl_seconds is not None else _DEFAULT_TTL
        self.enabled = self.ttl > 0
        if self.enabled:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:  # read-only FS, container, etc.
                log.warning("cache disabled (%s)", exc)
                self.enabled = False

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.pkl"

    def get(self, namespace: str, params: dict[str, Any]) -> Any | None:
        if not self.enabled:
            return None
        path = self._path(_key(namespace, params))
        if not path.exists():
            return None
        if time.time() - path.stat().st_mtime > self.ttl:
            return None
        try:
            with path.open("rb") as fh:
                return pickle.load(fh)
        except Exception:
            # A corrupt entry is not worth failing a run over.
            path.unlink(missing_ok=True)
            return None

    def set(self, namespace: str, params: dict[str, Any], value: Any) -> None:
        if not self.enabled:
            return
        path = self._path(_key(namespace, params))
        tmp = path.with_suffix(".tmp")
        try:
            with tmp.open("wb") as fh:
                pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)
        except Exception as exc:  # pragma: no cover - best effort
            log.debug("cache write failed for %s: %s", namespace, exc)
            tmp.unlink(missing_ok=True)

    def memoize(
        self, namespace: str, params: dict[str, Any], producer: Callable[[], T]
    ) -> T:
        hit = self.get(namespace, params)
        if hit is not None:
            return hit
        value = producer()
        if value is not None:
            self.set(namespace, params, value)
        return value

    def clear(self) -> int:
        if not self.enabled:
            return 0
        removed = 0
        for path in self.dir.glob("*.pkl"):
            path.unlink(missing_ok=True)
            removed += 1
        return removed


_GLOBAL: DiskCache | None = None


def get_cache(config: dict[str, Any] | None = None) -> DiskCache:
    global _GLOBAL
    if config is not None:
        return DiskCache(config.get("cache_dir"), config.get("cache_ttl_seconds"))
    if _GLOBAL is None:
        _GLOBAL = DiskCache()
    return _GLOBAL
