"""Coverage for tool_stats_store.py — 51% → 100%."""

import json
import os
import pytest
from unittest.mock import patch

from server_core.tool_stats_store import ToolStatsStore, STATS_FILE_NAME, MIN_RUNS_FOR_LIVE


class TestToolStatsStore:
    @pytest.fixture
    def store(self, tmp_path):
        return ToolStatsStore(data_dir=str(tmp_path))

    def test_record_success(self, store):
        store.record("nmap", True)
        stats = store.get_stats("nmap")
        assert stats["runs"] == 1
        assert stats["successes"] == 1

    def test_record_failure(self, store):
        store.record("nmap", False)
        stats = store.get_stats("nmap")
        assert stats["runs"] == 1
        assert stats["successes"] == 0

    def test_record_multiple(self, store):
        store.record("nmap", True)
        store.record("nmap", False)
        store.record("nmap", True)
        stats = store.get_stats("nmap")
        assert stats["runs"] == 3
        assert stats["successes"] == 2

    def test_get_stats_unseen_tool(self, store):
        stats = store.get_stats("nonexistent")
        assert stats["runs"] == 0
        assert stats["successes"] == 0

    def test_get_all_stats(self, store):
        store.record("nmap", True)
        store.record("sqlmap", False)
        all_stats = store.get_all_stats()
        assert "nmap" in all_stats
        assert "sqlmap" in all_stats
        assert all_stats["nmap"]["successes"] == 1

    def test_get_all_stats_isolation(self, store):
        store.record("nmap", True)
        all_stats = store.get_all_stats()
        all_stats["nmap"]["runs"] = 999
        assert store.get_stats("nmap")["runs"] == 1

    def test_live_effectiveness_none_below_min(self, store):
        for _ in range(MIN_RUNS_FOR_LIVE - 1):
            store.record("nmap", True)
        assert store.live_effectiveness("nmap") is None

    def test_live_effectiveness_value(self, store):
        for _ in range(MIN_RUNS_FOR_LIVE):
            store.record("nmap", True)
        assert store.live_effectiveness("nmap") == 1.0

    def test_blended_effectiveness_below_min(self, store):
        eff = store.blended_effectiveness("nmap", 0.85)
        assert eff == 0.85

    def test_blended_effectiveness_above_min(self, store):
        for _ in range(MIN_RUNS_FOR_LIVE):
            store.record("nmap", True)
        eff = store.blended_effectiveness("nmap", 0.5)
        assert eff == 1.0

    def test_reset(self, store):
        store.record("nmap", True)
        store.reset("nmap")
        assert store.get_stats("nmap")["runs"] == 0

    def test_reset_nonexistent(self, store):
        store.reset("never_seen")
        assert store.get_stats("never_seen")["runs"] == 0

    def test_persistence(self, tmp_path):
        store = ToolStatsStore(data_dir=str(tmp_path))
        store.record("nmap", True)
        store.record("nmap", False)
        store2 = ToolStatsStore(data_dir=str(tmp_path))
        assert store2.get_stats("nmap")["runs"] == 2
        assert store2.get_stats("nmap")["successes"] == 1

    def test_load_corrupt_json(self, tmp_path):
        stats_path = os.path.join(str(tmp_path), STATS_FILE_NAME)
        os.makedirs(str(tmp_path), exist_ok=True)
        with open(stats_path, "w") as f:
            f.write("not valid json")
        store = ToolStatsStore(data_dir=str(tmp_path))
        assert store.get_all_stats() == {}

    def test_load_invalid_shape(self, tmp_path):
        stats_path = os.path.join(str(tmp_path), STATS_FILE_NAME)
        os.makedirs(str(tmp_path), exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump({"nmap": "not_a_dict"}, f)
        store = ToolStatsStore(data_dir=str(tmp_path))
        assert "nmap" not in store.get_all_stats()

    def test_load_os_error(self, tmp_path):
        with patch("builtins.open", side_effect=OSError("permission denied")):
            store = ToolStatsStore(data_dir=str(tmp_path))
            assert store.get_all_stats() == {}

    def test_save_os_error(self, tmp_path, caplog):
        store = ToolStatsStore(data_dir=str(tmp_path))
        with patch("builtins.open", side_effect=OSError("disk full")):
            store.record("nmap", True)
        assert "failed to save" in caplog.text

    def test_default_data_dir_env(self):
        with patch.dict(os.environ, {"HEXSTRIKE_DATA_DIR": "/tmp/custom_stats_dir"}, clear=False):
            store = ToolStatsStore()
            assert "/tmp/custom_stats_dir" in store._stats_path
