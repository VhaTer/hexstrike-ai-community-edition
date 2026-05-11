import pytest
from server_core.config_core import get_word_list, find_best_wordlist, get_word_list_path, get, set_value


class TestGetWordList:
    def test_exists(self):
        wl = get_word_list("rockyou")
        assert wl is not None
        assert wl["type"] == "password"

    def test_not_exists(self):
        assert get_word_list("nonexistent") is None


class TestFindBestWordlist:
    def test_exact_match(self):
        result = find_best_wordlist({"for_task": "password_cracking"})
        assert result is not None
        assert result["wordlist"]["type"] == "password"

    def test_tool_match(self):
        result = find_best_wordlist({"tool": "john"})
        assert result is not None
        assert "john" in result["wordlist"]["tool"]

    def test_type_match(self):
        result = find_best_wordlist({"type": "password"})
        assert result is not None
        assert result["wordlist"]["type"] == "password"

    def test_relaxed_for_task_fallback(self):
        result = find_best_wordlist({"for_task": "nonexistent_task", "type": "nonexistent"})
        assert result is not None

    def test_no_match_at_all(self):
        result = find_best_wordlist({"for_task": "nonexistent_task"})
        assert result is not None  # Falls back to first available

    def test_empty_criteria(self):
        result = find_best_wordlist({})
        assert result is not None

    def test_multiple_criteria(self):
        result = find_best_wordlist({"for_task": "dirbusting", "type": "directory"})
        assert result is not None
        assert result["wordlist"]["type"] == "directory"

    def test_empty_config(self, monkeypatch):
        monkeypatch.setattr("server_core.config_core._config", {"WORD_LISTS": {}})
        result = find_best_wordlist({"for_task": "anything"})
        assert result is None

    def test_other_key_match(self):
        """else branch in matches(): wl.get(key) == value (continue)."""
        result = find_best_wordlist({"speed": "fast"})
        assert result is not None

    def test_other_key_no_match(self):
        """else branch in matches(): wl.get(key) != value (return False), then fallback."""
        result = find_best_wordlist({"speed": "nonexistent"})
        assert result is not None  # falls back to first available

    def test_tool_not_matching_then_fallback(self):
        """tool branch: value not in wl.get('tool', []) (return False), then fallback."""
        result = find_best_wordlist({"tool": "nonexistent_tool"})
        assert result is not None  # falls back to first available

    def test_relaxed_match_after_failed_exact(self):
        """Exact match fails (type doesn't match), but relaxed for_task matches and returns."""
        result = find_best_wordlist({"for_task": "password_cracking", "speed": "nonexistent"})
        assert result is not None
        assert result["wordlist"]["type"] == "password"  # rockyou via relaxed for_task match


class TestGetWordListPath:
    def test_exists(self):
        path = get_word_list_path("rockyou")
        assert path is not None
        assert "/" in path

    def test_not_exists(self):
        assert get_word_list_path("nonexistent") is None

    def test_no_path_key(self, monkeypatch):
        monkeypatch.setattr("server_core.config_core._config", {
            "WORD_LISTS": {
                "nopath": {"type": "password"},
            }
        })
        assert get_word_list_path("nopath") is None


class TestGet:
    def test_existing_key(self):
        assert get("APP_NAME") == "HexStrike AI Pulse"

    def test_missing_key_default(self):
        assert get("NONEXISTENT", "fallback") == "fallback"

    def test_missing_key_no_default(self):
        assert get("NONEXISTENT") is None


class TestSetValue:
    def test_set_and_retrieve(self):
        set_value("TEST_KEY", "test_value")
        assert get("TEST_KEY") == "test_value"
