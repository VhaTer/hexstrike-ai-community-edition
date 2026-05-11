import pytest
import json
import os
from unittest.mock import patch
from server_core.session_store import SessionStore


@pytest.fixture
def store(tmp_path):
    return SessionStore(data_dir=str(tmp_path / "data"))


class TestSessionStore:
    def test_save_and_load(self, store):
        store.save("session1", {"target": "10.0.0.1", "findings": []})
        loaded = store.load("session1")
        assert loaded is not None
        assert loaded["target"] == "10.0.0.1"

    def test_load_nonexistent(self, store):
        assert store.load("nonexistent") is None

    def test_save_idempotent(self, store):
        store.save("session1", {"target": "10.0.0.1"})
        store.save("session1", {"target": "10.0.0.2"})
        loaded = store.load("session1")
        assert loaded["target"] == "10.0.0.2"

    def test_archive_and_load_completed(self, store):
        session = {"session_id": "session1", "target": "10.0.0.1", "findings": []}
        store.save("session1", session)
        archived = store.archive("session1", session)
        assert archived is True
        assert store.load("session1") is None
        completed = store.load_completed("session1")
        assert completed is not None

    def test_archive_removes_active(self, store):
        session = {"session_id": "s1", "target": "10.0.0.1"}
        store.save("s1", session)
        store.archive("s1", session)
        assert store.load("s1") is None

    def test_list_active(self, store):
        store.save("s1", {"target": "10.0.0.1"})
        store.save("s2", {"target": "10.0.0.2"})
        active = store.list_active()
        assert "s1" in active
        assert "s2" in active

    def test_list_active_empty(self, store):
        assert store.list_active() == []

    def test_list_completed(self, store):
        s1 = {"session_id": "s1", "target": "10.0.0.1"}
        store.save("s1", s1)
        store.archive("s1", s1)
        s2 = {"session_id": "s2", "target": "10.0.0.2"}
        store.save("s2", s2)
        store.archive("s2", s2)
        completed = store.list_completed()
        assert len(completed) >= 2

    def test_list_completed_empty(self, store):
        assert store.list_completed() == []

    def test_delete(self, store):
        store.save("s1", {"target": "10.0.0.1"})
        assert store.delete("s1") is True
        assert store.load("s1") is None

    def test_delete_nonexistent(self, store):
        assert store.delete("nonexistent") is False

    def test_load_all_active(self, store):
        store.save("s1", {"target": "10.0.0.1", "id": 1})
        store.save("s2", {"target": "10.0.0.2", "id": 2})
        sessions = store.load_all_active()
        assert len(sessions) == 2

    def test_data_dir_property(self, store):
        assert store.data_dir.endswith("data")

    def test_prune_completed(self, store, tmp_path):
        store = SessionStore(data_dir=str(tmp_path / "data2"))
        for i in range(210):
            sid = f"s{i}"
            session = {"session_id": sid, "target": f"10.0.0.{i}"}
            store.save(sid, session)
            store.archive(sid, session)
        remaining = store.list_completed()
        assert len(remaining) <= 200

    def test_corrupt_session_file(self, store):
        path = store._session_path("corrupt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("not json")
        assert store.load("corrupt") is None

    def test_list_active_ignores_directories(self, store):
        os.makedirs(os.path.join(store._sessions_dir, "somedir"), exist_ok=True)
        active = store.list_active()
        assert "somedir" not in active

    # ── Save error paths ──

    def test_save_os_error(self, store):
        with patch("builtins.open", side_effect=OSError("permission denied")):
            result = store.save("s1", {"target": "10.0.0.1"})
            assert result is False

    def test_save_type_error(self, store):
        with patch("json.dump", side_effect=TypeError("not serializable")):
            result = store.save("s1", {"target": "10.0.0.1"})
            assert result is False

    # ── Archive error paths ──

    def test_archive_os_error(self, store):
        store.save("s1", {"target": "10.0.0.1"})
        with patch("builtins.open", side_effect=OSError("permission denied")):
            result = store.archive("s1", {"target": "10.0.0.1"})
            assert result is False

    def test_archive_type_error(self, store):
        store.save("s1", {"target": "10.0.0.1"})
        with patch("json.dump", side_effect=TypeError("not serializable")):
            result = store.archive("s1", {"target": "10.0.0.1"})
            assert result is False

    def test_archive_without_active_file(self, store):
        session = {"session_id": "orphan", "target": "10.0.0.1"}
        result = store.archive("orphan", session)
        assert result is True
        completed = store.load_completed("orphan")
        assert completed is not None

    # ── load_completed coverage ──

    def test_load_completed_nonexistent(self, store):
        assert store.load_completed("nonexistent") is None

    def test_load_completed_corrupt(self, store):
        path = store._completed_path("corrupt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("not json")
        assert store.load_completed("corrupt") is None

    # ── list_completed coverage ──

    def test_list_completed_skips_corrupt(self, store):
        store.save("good", {"session_id": "good", "target": "10.0.0.1"})
        store.archive("good", {"session_id": "good", "target": "10.0.0.1"})
        bad_path = os.path.join(store._completed_dir, "bad.json")
        with open(bad_path, "w") as f:
            f.write("not json")
        completed = store.list_completed()
        ids = [c["session_id"] for c in completed]
        assert "bad" not in ids
        assert "good" in ids

    def test_list_completed_skips_non_json(self, store):
        path = os.path.join(store._completed_dir, "readme.txt")
        with open(path, "w") as f:
            f.write("not json")
        completed = store.list_completed()
        assert isinstance(completed, list)

    def test_list_completed_no_dir(self, store):
        import shutil
        shutil.rmtree(store._completed_dir)
        result = store.list_completed()
        assert result == []

    # ── list_active missing dir ──

    def test_list_active_no_sessions_dir(self, store):
        import shutil
        shutil.rmtree(store._sessions_dir)
        assert store.list_active() == []

    # ── _prune_completed coverage ──

    def test_prune_completed_dir_missing(self, store):
        import shutil
        shutil.rmtree(store._completed_dir, ignore_errors=True)
        store._prune_completed()

    def test_prune_completed_os_error(self, store):
        for i in range(210):
            sid = f"s{i}"
            session = {"session_id": sid, "target": f"10.0.0.{i}"}
            store.save(sid, session)
            store.archive(sid, session)
        assert len(store.list_completed()) == 200
        for i in range(210, 230):
            sid = f"extra{i}"
            path = store._completed_path(sid)
            with open(path, "w") as f:
                json.dump({"session_id": sid, "target": f"10.0.0.{i}"}, f)
        with patch("os.remove", side_effect=OSError("permission denied")):
            store._prune_completed()
        assert len(store.list_completed()) == 220

    # ── load_all_active empty ──

    def test_load_all_active_empty(self, store):
        sessions = store.load_all_active()
        assert sessions == []

    # ── load_all_active skips corrupt files ──

    def test_load_all_active_skips_corrupt(self, store):
        store.save("good", {"target": "10.0.0.1", "id": 1})
        path = store._session_path("corrupt")
        with open(path, "w") as f:
            f.write("not json")
        sessions = store.load_all_active()
        assert len(sessions) == 1
        assert sessions[0]["target"] == "10.0.0.1"

    # ── _prune_completed skips non-JSON files ──

    def test_prune_completed_skips_non_json(self, store):
        non_json = os.path.join(store._completed_dir, "readme.txt")
        with open(non_json, "w") as f:
            f.write("not a session file")
        for i in range(210):
            sid = f"s{i}"
            session = {"session_id": sid, "target": f"10.0.0.{i}"}
            store.save(sid, session)
            store.archive(sid, session)
        assert os.path.exists(non_json)
        remaining = store.list_completed()
        assert len(remaining) <= 200

    # ── _default_data_dir coverage ──

    def test_default_data_dir(self, monkeypatch):
        monkeypatch.delenv("HEXSTRIKE_DATA_DIR", raising=False)
        store = SessionStore()
        assert isinstance(store.data_dir, str)
        assert os.path.isdir(store._sessions_dir)

    def test_default_data_dir_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HEXSTRIKE_DATA_DIR", str(tmp_path / "custom_dir"))
        store = SessionStore()
        assert store.data_dir == str(tmp_path / "custom_dir")
