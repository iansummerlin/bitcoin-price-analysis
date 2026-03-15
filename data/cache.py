"""Generic TTL-based file cache for external data sources.

All remote API calls must go through this cache. On cache hit within TTL,
the cached data is returned without any remote call. On miss or expiry,
the caller fetches fresh data and writes it back. On API failure, stale
cache (expired but present) is preferred over retries.

Cache directory: ``data/cache/`` (gitignored).
Atomic writes: temp file + ``os.replace`` to prevent corruption.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from config import ROOT_DIR

CACHE_DIR = ROOT_DIR / "data" / "cache"


def _ensure_cache_dir(namespace: str) -> Path:
    """Create and return the namespace directory inside the cache root."""
    ns_dir = CACHE_DIR / namespace
    ns_dir.mkdir(parents=True, exist_ok=True)
    return ns_dir


def _meta_path(data_path: Path) -> Path:
    return data_path.with_suffix(data_path.suffix + ".meta")


def _read_meta(meta_path: Path) -> dict[str, Any]:
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_atomic(path: Path, content: bytes) -> None:
    """Write bytes to *path* atomically via temp file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(fd, content)
        os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def cache_get(namespace: str, key: str, ttl_seconds: float) -> bytes | None:
    """Return cached data if present and within TTL, else ``None``.

    Parameters
    ----------
    namespace : str
        Data source family (e.g. ``"crossasset"``, ``"onchain"``).
    key : str
        Cache key within the namespace (e.g. ``"dxy_daily"``).
    ttl_seconds : float
        Maximum age in seconds before the entry is considered expired.

    Returns
    -------
    bytes or None
        Raw cached data, or ``None`` if missing or expired.
    """
    ns_dir = _ensure_cache_dir(namespace)
    data_path = ns_dir / key
    meta_path = _meta_path(data_path)

    if not data_path.exists():
        return None

    meta = _read_meta(meta_path)
    written_at = meta.get("written_at", 0.0)
    if (time.time() - written_at) > ttl_seconds:
        return None  # expired

    return data_path.read_bytes()


def cache_get_stale(namespace: str, key: str) -> bytes | None:
    """Return cached data regardless of TTL (stale fallback).

    Used when an API call fails — prefer stale data over no data.
    """
    ns_dir = _ensure_cache_dir(namespace)
    data_path = ns_dir / key
    if not data_path.exists():
        return None
    return data_path.read_bytes()


def cache_put(namespace: str, key: str, data: bytes) -> Path:
    """Write data to cache with current timestamp metadata.

    Returns the path where data was written.
    """
    ns_dir = _ensure_cache_dir(namespace)
    data_path = ns_dir / key
    meta_path = _meta_path(data_path)

    _write_atomic(data_path, data)
    meta = {"written_at": time.time(), "size": len(data)}
    _write_atomic(meta_path, json.dumps(meta).encode("utf-8"))
    return data_path


def cache_invalidate(namespace: str, key: str) -> bool:
    """Remove a cache entry. Returns True if it existed."""
    ns_dir = _ensure_cache_dir(namespace)
    data_path = ns_dir / key
    meta_path = _meta_path(data_path)
    existed = data_path.exists()
    for p in (data_path, meta_path):
        if p.exists():
            p.unlink()
    return existed
