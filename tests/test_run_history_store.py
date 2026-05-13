"""Coverage for run_history_store.py — 0% → 100%."""

from server_core.run_history_store import RunHistoryStore


class TestRunHistoryStore:
    def test_init_stores(self):
        s = RunHistoryStore()
        assert s._id_counter == 0
        assert len(s._entries) == 0
        assert s._lock is not None

    def test_record_creates_entry(self):
        s = RunHistoryStore()
        s.record("nmap", "/scan", {"target": "10.0.0.1"}, {"stdout": "ok", "return_code": 0, "success": True})
        entries = s.get_all()
        assert len(entries) == 1
        assert entries[0]["tool"] == "nmap"
        assert entries[0]["endpoint"] == "/scan"
        assert entries[0]["success"] is True
        assert entries[0]["return_code"] == 0

    def test_record_increments_id(self):
        s = RunHistoryStore()
        s.record("a", None, None, {})
        s.record("b", None, None, {})
        entries = s.get_all()
        assert entries[0]["id"] == 2
        assert entries[1]["id"] == 1

    def test_record_defaults_none_tool(self):
        s = RunHistoryStore()
        s.record(None, None, None, {})
        e = s.get_all()[0]
        assert e["tool"] == "unknown"
        assert e["endpoint"] == ""
        assert e["params"] == {}

    def test_record_defaults_result_fields(self):
        s = RunHistoryStore()
        s.record("test", "/", {}, {})
        e = s.get_all()[0]
        assert e["stdout"] == ""
        assert e["stderr"] == ""
        assert e["return_code"] == -1
        assert e["success"] is False
        assert e["timed_out"] is False
        assert e["partial_results"] is False
        assert e["execution_time"] == 0.0
        assert e["timestamp"] == ""

    def test_get_all_returns_copy(self):
        s = RunHistoryStore()
        s.record("a", None, None, {})
        entries = s.get_all()
        assert len(entries) == 1
        s.clear()
        assert len(s.get_all()) == 0

    def test_clear(self):
        s = RunHistoryStore()
        s.record("a", None, None, {})
        s.record("b", None, None, {})
        assert len(s.get_all()) == 2
        s.clear()
        assert len(s.get_all()) == 0

    def test_empty_get_all(self):
        s = RunHistoryStore()
        assert s.get_all() == []

    def test_record_timed_out(self):
        s = RunHistoryStore()
        s.record("x", "/y", {}, {"timed_out": True, "stderr": "timeout"})
        e = s.get_all()[0]
        assert e["timed_out"] is True
        assert e["stderr"] == "timeout"

    def test_record_partial_results(self):
        s = RunHistoryStore()
        s.record("x", "/y", {}, {"partial_results": True, "success": False})
        e = s.get_all()[0]
        assert e["partial_results"] is True
        assert e["success"] is False

    def test_max_entries(self):
        s = RunHistoryStore()
        for i in range(s.MAX_ENTRIES + 10):
            s.record(f"t{i}", None, None, {})
        assert len(s.get_all()) == s.MAX_ENTRIES
