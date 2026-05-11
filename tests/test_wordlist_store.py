import json
import os
import tempfile

import pytest

from server_core.wordlist_store import WordlistStore


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmp:
        yield WordlistStore(data_dir=tmp)


class TestWordlistStore:
    def test_init_creates_dir_and_file(self, store):
        assert os.path.exists(store.data_dir)
        assert os.path.exists(store._wordlists_file)

    def test_data_dir_property(self, store):
        assert store.data_dir == store._data_dir

    def test_save_and_load(self, store):
        info = {"path": "/usr/share/wordlists/rockyou.txt", "type": "password"}
        assert store.save("rockyou", info)
        loaded = store.load("rockyou")
        assert loaded == info

    def test_save_missing_path(self, store):
        assert store.save("bad", {"type": "password"}) is False

    def test_save_missing_type(self, store):
        assert store.save("bad", {"path": "/tmp/test"}) is False

    def test_save_not_dict(self, store):
        assert store.save("bad", "not_a_dict") is False

    def test_load_missing(self, store):
        assert store.load("nonexistent") is None

    def test_load_all_empty(self, store):
        assert store.load_all() == {}

    def test_save_then_load_all(self, store):
        store.save("wl1", {"path": "/tmp/a.txt", "type": "password"})
        store.save("wl2", {"path": "/tmp/b.txt", "type": "directory"})
        all_wl = store.load_all()
        assert len(all_wl) == 2
        assert all_wl["wl1"]["path"] == "/tmp/a.txt"

    def test_get_path(self, store):
        store.save("wl1", {"path": "/tmp/test.txt", "type": "password"})
        assert store.getPath("wl1") == "/tmp/test.txt"

    def test_get_path_missing(self, store):
        assert store.getPath("nonexistent") is None

    def test_delete_existing(self, store):
        store.save("wl1", {"path": "/tmp/a.txt", "type": "password"})
        assert store.delete("wl1") is True
        assert store.load("wl1") is None

    def test_delete_missing(self, store):
        assert store.delete("nonexistent") is False

    def test_delete_no_file(self, store):
        os.remove(store._wordlists_file)
        assert store.delete("wl1") is False

    def test_corrupted_json(self, store):
        with open(store._wordlists_file, "w") as f:
            f.write("not json")
        assert store.load("wl1") is None
        assert store.load_all() == {}

    def test_find_best_match_exact(self, store):
        store.save("wl1", {"path": "/tmp/a.txt", "type": "password", "recommended_for": ["bruteforce"]})
        match = store.find_best_match({"for_task": "bruteforce", "type": "password"})
        assert match is not None
        assert match["name"] == "wl1"

    def test_find_best_match_relaxed(self, store):
        store.save("wl1", {"path": "/tmp/a.txt", "type": "password", "recommended_for": ["bruteforce"]})
        store.save("wl2", {"path": "/tmp/b.txt", "type": "directory", "recommended_for": ["dirbust"]})
        match = store.find_best_match({"for_task": "bruteforce", "type": "does_not_exist"})
        assert match is not None
        assert match["name"] == "wl1"

    def test_find_best_match_fallback(self, store):
        store.save("wl1", {"path": "/tmp/a.txt", "type": "password", "recommended_for": ["bruteforce"]})
        match = store.find_best_match({"for_task": "nonexistent_task"})
        assert match is not None
        assert match["name"] == "wl1"

    def test_find_best_match_empty(self, store):
        assert store.find_best_match({"for_task": "anything"}) is None

    def test_find_best_match_tool(self, store):
        store.save("wl1", {"path": "/tmp/a.txt", "type": "password", "tool": ["hydra", "john"]})
        match = store.find_best_match({"tool": "hydra"})
        assert match is not None
        assert match["name"] == "wl1"

    def test_save_then_delete_then_load(self, store):
        store.save("wl1", {"path": "/tmp/a.txt", "type": "password"})
        store.delete("wl1")
        store.save("wl1", {"path": "/tmp/b.txt", "type": "password"})
        loaded = store.load("wl1")
        assert loaded["path"] == "/tmp/b.txt"

    def test_default_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = os.getcwd()
            os.chdir(tmp)
            try:
                s = WordlistStore(data_dir=None)
                assert s.data_dir is not None
            finally:
                os.chdir(orig)

    def test_load_no_file(self, store):
        os.remove(store._wordlists_file)
        assert store.load("anything") is None

    def test_load_all_no_file(self, store):
        os.remove(store._wordlists_file)
        assert store.load_all() == {}

    def test_delete_corrupt_json(self, store):
        store.save("wl1", {"path": "/tmp/a.txt", "type": "password"})
        with open(store._wordlists_file, "w") as f:
            f.write("{invalid")
        assert store.delete("wl1") is False

    def test_save_no_file_creates_dirs(self, store):
        import shutil
        shutil.rmtree(store.data_dir)
        assert store.save("wl1", {"path": "/tmp/a.txt", "type": "password"})
        assert store.load("wl1") is not None

    def test_save_no_wordlists_key(self, store):
        with open(store._wordlists_file, "w") as f:
            json.dump({"other_key": {}}, f)
        assert store.save("wl1", {"path": "/tmp/a.txt", "type": "password"})
        assert store.load("wl1") is not None

    def test_save_permission_error(self, store):
        os.chmod(store.data_dir, 0o444)
        try:
            assert store.save("wl1", {"path": "/tmp/a.txt", "type": "password"}) is False
        finally:
            os.chmod(store.data_dir, 0o755)

    def test_find_best_match_other_key(self, store):
        store.save("wl1", {"path": "/tmp/a.txt", "type": "password", "speed": "fast"})
        match = store.find_best_match({"speed": "fast"})
        assert match is not None
        assert match["name"] == "wl1"

    def test_find_best_match_relaxed_no_match(self, store):
        store.save("wl1", {"path": "/tmp/a.txt", "type": "password"})
        match = store.find_best_match({"for_task": "bruteforce"})
        assert match is not None
        assert match["name"] == "wl1"

    def test_find_best_match_tool_no_match(self, store):
        store.save("wl1", {"path": "/tmp/a.txt", "type": "password", "tool": ["hydra", "john"]})
        match = store.find_best_match({"tool": "nmap"})
        assert match is not None
        assert match["name"] == "wl1"

    def test_find_best_match_other_key_no_match(self, store):
        store.save("wl1", {"path": "/tmp/a.txt", "type": "password", "speed": "fast"})
        match = store.find_best_match({"speed": "slow"})
        assert match is not None
        assert match["name"] == "wl1"
