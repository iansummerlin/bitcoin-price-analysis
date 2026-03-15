"""Tests for the shared data cache (Phase 12B)."""

from __future__ import annotations

import json
import time

import pytest

from data.cache import (
    CACHE_DIR,
    cache_get,
    cache_get_stale,
    cache_invalidate,
    cache_put,
    _meta_path,
)


@pytest.fixture(autouse=True)
def _clean_test_namespace(tmp_path, monkeypatch):
    """Redirect cache directory to a temp location for test isolation."""
    import data.cache as cache_mod

    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path / "cache")


class TestCachePutGet:
    def test_round_trip(self):
        data = b"hello world"
        cache_put("test_ns", "key1", data)
        result = cache_get("test_ns", "key1", ttl_seconds=3600)
        assert result == data

    def test_missing_key_returns_none(self):
        result = cache_get("test_ns", "nonexistent", ttl_seconds=3600)
        assert result is None

    def test_overwrite(self):
        cache_put("test_ns", "key1", b"first")
        cache_put("test_ns", "key1", b"second")
        result = cache_get("test_ns", "key1", ttl_seconds=3600)
        assert result == b"second"


class TestCacheTTL:
    def test_expired_returns_none(self):
        cache_put("test_ns", "key1", b"data")
        # Expire immediately by using TTL of 0
        result = cache_get("test_ns", "key1", ttl_seconds=0)
        assert result is None

    def test_not_expired_returns_data(self):
        cache_put("test_ns", "key1", b"data")
        result = cache_get("test_ns", "key1", ttl_seconds=9999)
        assert result == b"data"


class TestCacheStale:
    def test_stale_returns_expired_data(self):
        cache_put("test_ns", "key1", b"stale data")
        # Confirm it's expired via normal get
        assert cache_get("test_ns", "key1", ttl_seconds=0) is None
        # Stale get should still return it
        result = cache_get_stale("test_ns", "key1")
        assert result == b"stale data"

    def test_stale_returns_none_if_never_cached(self):
        result = cache_get_stale("test_ns", "never_existed")
        assert result is None


class TestCacheInvalidate:
    def test_invalidate_existing(self):
        cache_put("test_ns", "key1", b"data")
        assert cache_invalidate("test_ns", "key1") is True
        assert cache_get("test_ns", "key1", ttl_seconds=9999) is None
        assert cache_get_stale("test_ns", "key1") is None

    def test_invalidate_nonexistent(self):
        assert cache_invalidate("test_ns", "nonexistent") is False


class TestCacheMetadata:
    def test_meta_has_timestamp(self, tmp_path, monkeypatch):
        import data.cache as cache_mod

        before = time.time()
        path = cache_put("test_ns", "key1", b"data")
        after = time.time()

        meta_path = _meta_path(path)
        meta = json.loads(meta_path.read_text())
        assert before <= meta["written_at"] <= after
        assert meta["size"] == 4


class TestCacheNamespaces:
    def test_separate_namespaces(self):
        cache_put("ns_a", "key1", b"data_a")
        cache_put("ns_b", "key1", b"data_b")
        assert cache_get("ns_a", "key1", ttl_seconds=9999) == b"data_a"
        assert cache_get("ns_b", "key1", ttl_seconds=9999) == b"data_b"
