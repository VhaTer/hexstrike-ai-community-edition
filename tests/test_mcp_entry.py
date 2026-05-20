"""Tests for mcp_core/mcp_entry.py — lock, seed cache, prewarm singletons."""

import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from mcp_core import mcp_entry


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_scan_cache():
    """Patch mcp_entry._scan_cache with a dict-like that has .set()."""
    class DictCache(dict):
        def set(self, key, value, execution_time=None):
            self[key] = value
    cache = DictCache()
    with patch.object(mcp_entry, "_scan_cache", cache):
        yield cache


@pytest.fixture
def mock_op_metrics():
    """Patch _op_metrics in the seed function's import path."""
    fake = MagicMock()
    with patch("server_core.operational_metrics._op_metrics", fake):
        yield fake


@pytest.fixture
def tmp_lock_path(tmp_path):
    """Use a temp lock file so tests don't touch /tmp/hexstrike_mcp.lock."""
    lock = str(tmp_path / "hexstrike_mcp.lock")
    with patch.object(mcp_entry, "_LOCK_PATH", lock):
        yield lock


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset module-level state between tests."""
    mcp_entry._lock_fh = None
    yield


# ── _seed_scan_cache ──────────────────────────────────────────────────────────


class TestSeedScanCache:
    def test_seed_with_env_var(self, mock_scan_cache, mock_op_metrics):
        """When HEXSTRIKE_SEED_SCANS is set, seeds 4 cache entries."""
        with patch.dict(os.environ, {"HEXSTRIKE_SEED_SCANS": "scanme.nmap.org"}):
            logger = MagicMock()
            mcp_entry._seed_scan_cache(logger)
        seeds = {k: v for k, v in mock_scan_cache.items() if k.startswith("seed:")}
        assert len(seeds) == 4
        assert "seed:nmap:scanme.nmap.org" in seeds
        assert "seed:whatweb:scanme.nmap.org" in seeds
        assert "seed:nuclei:scanme.nmap.org" in seeds
        assert "seed:nikto:scanme.nmap.org" in seeds
        assert seeds["seed:nmap:scanme.nmap.org"]["result"]["success"] is True
        assert seeds["seed:nikto:scanme.nmap.org"]["result"]["success"] is False
        assert mock_op_metrics.record.call_count == 4

    def test_seed_without_env_var(self, mock_scan_cache):
        """Without env var, _seed_scan_cache does nothing."""
        logger = MagicMock()
        mcp_entry._seed_scan_cache(logger)
        assert len(mock_scan_cache) == 0

    def test_seed_empty_env_var(self, mock_scan_cache):
        """Empty string env var is treated as unset."""
        with patch.dict(os.environ, {"HEXSTRIKE_SEED_SCANS": ""}):
            logger = MagicMock()
            mcp_entry._seed_scan_cache(logger)
        assert len(mock_scan_cache) == 0


# ── _prewarm_singletons ──────────────────────────────────────────────────────


class TestPrewarmSingletons:
    def test_prewarm_spawns_thread(self):
        """_prewarm_singletons starts a daemon thread."""
        started = []

        def tracking_start(self_obj):
            started.append(True)
            assert self_obj.daemon is True
            assert self_obj.name == "prewarm"

        with patch.object(threading.Thread, "start", tracking_start):
            mcp_entry._prewarm_singletons(MagicMock())
        assert len(started) == 1

    def test_prewarm_warm_success(self):
        """Warm path calls get_decision_engine and get_tool_stats_store."""
        calls = []

        def fake_eng():
            eng = MagicMock()
            eng._get_parameter_optimizer = lambda: None
            calls.append("engine")
            return eng

        def fake_store():
            calls.append("store")
            return MagicMock()

        def sync_start(self_obj):
            self_obj._target()

        with patch("server_core.singletons.get_decision_engine", fake_eng):
            with patch("server_core.singletons.get_tool_stats_store", fake_store):
                with patch.object(threading.Thread, "start", sync_start):
                    mcp_entry._prewarm_singletons(MagicMock())

        assert "engine" in calls
        assert "store" in calls

    def test_prewarm_exception_handled(self):
        """Exception during prewarm is caught and logged (not re-raised)."""
        def fake_eng():
            raise RuntimeError("boom")

        logger = MagicMock()

        def sync_start(self_obj):
            self_obj._target()

        with patch("server_core.singletons.get_decision_engine", fake_eng):
            with patch.object(threading.Thread, "start", sync_start):
                mcp_entry._prewarm_singletons(logger)

        logger.debug.assert_any_call("🔍 Pre-warming lazy singletons...")
        # Also verifies the error was logged (no crash)
        assert any("Pre-warming skipped" in str(c) for c in logger.debug.call_args_list[-1])


# ── _acquire_lock ─────────────────────────────────────────────────────────────


class TestAcquireLock:
    def test_acquire_lock_success(self, tmp_lock_path):
        """Lock is acquired, _lock_fh is set, PID written."""
        mcp_entry._acquire_lock(MagicMock())
        assert mcp_entry._lock_fh is not None
        assert not mcp_entry._lock_fh.closed
        # Verify PID was written
        with open(tmp_lock_path, "r") as f:
            pid = f.read().strip()
        assert pid == str(os.getpid())
        # Clean up
        mcp_entry._lock_fh.close()
        os.unlink(tmp_lock_path)

    def test_acquire_lock_stale_removed(self, tmp_lock_path):
        """Old lock file (>TTL) is removed before acquiring."""
        with open(tmp_lock_path, "w") as f:
            f.write("stale")
        old_mtime = time.time() - (mcp_entry._LOCK_TTL + 5)
        os.utime(tmp_lock_path, (old_mtime, old_mtime))

        mcp_entry._acquire_lock(MagicMock())
        assert mcp_entry._lock_fh is not None
        assert not mcp_entry._lock_fh.closed
        mcp_entry._lock_fh.close()
        os.unlink(tmp_lock_path)

    def test_acquire_lock_pid_alive_exits(self, tmp_lock_path):
        """When lock file contains alive PID, sys.exit(1)."""
        with open(tmp_lock_path, "w") as f:
            f.write(str(os.getpid()))
        with pytest.raises(SystemExit) as exc:
            mcp_entry._acquire_lock(MagicMock())
        assert exc.value.code == 1

    def test_acquire_lock_pid_dead_proceeds(self, tmp_lock_path):
        """When lock file contains dead PID, proceeds (acquires new lock)."""
        with open(tmp_lock_path, "w") as f:
            f.write("999999999")  # unlikely to be alive
        mcp_entry._acquire_lock(MagicMock())
        assert mcp_entry._lock_fh is not None
        assert not mcp_entry._lock_fh.closed
        mcp_entry._lock_fh.close()
        os.unlink(tmp_lock_path)

    def test_acquire_lock_pid_corrupt_proceeds(self, tmp_lock_path):
        """When lock file has garbage content, proceeds."""
        with open(tmp_lock_path, "w") as f:
            f.write("not_a_pid\n")
        mcp_entry._acquire_lock(MagicMock())
        assert mcp_entry._lock_fh is not None
        assert not mcp_entry._lock_fh.closed
        mcp_entry._lock_fh.close()
        os.unlink(tmp_lock_path)

    def test_acquire_lock_empty_shutdown_proceeds(self, tmp_lock_path):
        """Empty lock file (clean shutdown marker) does not block."""
        with open(tmp_lock_path, "w") as f:
            pass  # empty file = clean shutdown signal
        mcp_entry._acquire_lock(MagicMock())
        assert mcp_entry._lock_fh is not None
        assert not mcp_entry._lock_fh.closed
        mcp_entry._lock_fh.close()
        os.unlink(tmp_lock_path)

    def test_acquire_lock_contention_exits(self, tmp_lock_path):
        """When fcntl lock is already held, sys.exit(1) is called."""
        lock_a = open(tmp_lock_path, "w")
        import fcntl
        fcntl.flock(lock_a, fcntl.LOCK_EX)

        with pytest.raises(SystemExit) as exc:
            mcp_entry._acquire_lock(MagicMock())
        assert exc.value.code == 1

        lock_a.close()
        os.unlink(tmp_lock_path)

    def test_acquire_lock_no_existing_file(self, tmp_lock_path):
        """Acquires lock when no lock file exists yet."""
        assert not os.path.exists(tmp_lock_path)
        mcp_entry._acquire_lock(MagicMock())
        assert mcp_entry._lock_fh is not None
        assert os.path.exists(tmp_lock_path)
        mcp_entry._lock_fh.close()
        os.unlink(tmp_lock_path)

    def test_release_cleans_up(self, tmp_lock_path):
        """_release_lock truncates PID, unlocks file, removes lock file."""
        mcp_entry._acquire_lock(MagicMock())
        assert os.path.exists(tmp_lock_path)
        assert mcp_entry._lock_fh is not None
        assert not mcp_entry._lock_fh.closed
        # PID should be present before release
        with open(tmp_lock_path, "r") as f:
            assert f.read().strip() == str(os.getpid())

        mcp_entry._release_lock()
        assert not os.path.exists(tmp_lock_path)
        assert mcp_entry._lock_fh is None or mcp_entry._lock_fh.closed

    def test_release_lock_already_unlinked(self, tmp_lock_path):
        """_release_lock handles already-deleted lock file."""
        mcp_entry._acquire_lock(MagicMock())
        assert os.path.exists(tmp_lock_path)
        os.unlink(tmp_lock_path)

        mcp_entry._release_lock()

    def test_release_lock_no_file_handle(self):
        """_release_lock handles None _lock_fh."""
        mcp_entry._lock_fh = None
        mcp_entry._release_lock()
