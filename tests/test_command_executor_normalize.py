"""
tests/test_command_executor_normalize.py

Regression tests for _normalize_result() in server_core/command_executor.py.

These tests validate that the canonical HexStrike result shape is always
produced regardless of what EnhancedCommandExecutor returns, and that
execute_command() applies normalization end-to-end.

Canonical shape keys:
    success, output, error, returncode, timed_out, partial_results,
    execution_time, timestamp
    + legacy aliases: stdout, stderr, return_code
"""

import pytest
from unittest.mock import MagicMock, patch
from server_core.command_executor import _normalize_result


# ---------------------------------------------------------------------------
# _normalize_result unit tests
# ---------------------------------------------------------------------------

class TestNormalizeResult:
    """Unit tests for the _normalize_result normalizer."""

    # --- already-normalized passthrough ---

    def test_passthrough_if_already_normalized(self):
        """Dicts that have 'output' but no 'stdout' are passed through unchanged."""
        already = {"success": True, "output": "hello", "error": "", "returncode": 0}
        result = _normalize_result(already)
        assert result is already  # same object, no copy

    def test_passthrough_error_dict_from_require(self):
        """_require() early-returns {success: False, error: ...} — must not blow up."""
        early = {"success": False, "error": "'target' is required"}
        result = _normalize_result(early)
        assert result is early
        assert result["success"] is False
        assert result["error"] == "'target' is required"

    # --- success path ---

    def test_success_stdout_becomes_output(self):
        raw = {
            "stdout": "nmap result\n",
            "stderr": "",
            "return_code": 0,
            "success": True,
            "timed_out": False,
            "partial_results": False,
            "execution_time": 1.5,
            "timestamp": "2026-01-01T00:00:00",
        }
        r = _normalize_result(raw)
        assert r["success"] is True
        assert r["output"] == "nmap result\n"
        assert r["error"] == ""
        assert r["returncode"] == 0
        assert r["timed_out"] is False
        assert r["execution_time"] == 1.5

    def test_success_empty_stdout_falls_back_to_stderr(self):
        """Some tools write results to stderr (e.g. hping3). Must not lose them."""
        raw = {
            "stdout": "",
            "stderr": "HPING result via stderr\n",
            "return_code": 0,
            "success": True,
            "timed_out": False,
            "partial_results": False,
            "execution_time": 2.0,
            "timestamp": "",
        }
        r = _normalize_result(raw)
        assert r["success"] is True
        assert r["output"] == "HPING result via stderr\n"
        assert r["error"] == ""

    # --- failure path ---

    def test_failure_stderr_becomes_error(self):
        raw = {
            "stdout": "",
            "stderr": "nmap: command not found\n",
            "return_code": 127,
            "success": False,
            "timed_out": False,
            "partial_results": False,
            "execution_time": 0.1,
            "timestamp": "",
        }
        r = _normalize_result(raw)
        assert r["success"] is False
        assert r["error"] == "nmap: command not found"
        assert r["returncode"] == 127

    def test_failure_no_stderr_error_is_empty(self):
        """Non-zero exit with no stderr -> error string is empty, not None."""
        raw = {
            "stdout": "partial output",
            "stderr": "",
            "return_code": 1,
            "success": False,
            "timed_out": False,
            "partial_results": False,
            "execution_time": 0.5,
            "timestamp": "",
        }
        r = _normalize_result(raw)
        assert r["success"] is False
        assert r["error"] == ""
        assert r["output"] == "partial output"

    # --- timeout path ---

    def test_timeout_success_is_false(self):
        """timed_out=True must always produce success=False."""
        raw = {
            "stdout": "partial scan\n",
            "stderr": "",
            "return_code": -1,
            "success": False,
            "timed_out": True,
            "partial_results": True,
            "execution_time": 300.0,
            "timestamp": "",
        }
        r = _normalize_result(raw)
        assert r["success"] is False
        assert r["timed_out"] is True
        assert r["partial_results"] is True
        assert "timed out" in r["error"].lower()
        assert "300" in r["error"]

    def test_timeout_with_stderr_appended_to_error(self):
        raw = {
            "stdout": "",
            "stderr": "Connection refused",
            "return_code": -1,
            "success": False,
            "timed_out": True,
            "partial_results": False,
            "execution_time": 60.0,
            "timestamp": "",
        }
        r = _normalize_result(raw)
        assert "timed out" in r["error"].lower()
        assert "Connection refused" in r["error"]

    def test_timeout_stderr_truncated_to_200_chars(self):
        long_stderr = "X" * 300
        raw = {
            "stdout": "",
            "stderr": long_stderr,
            "return_code": -1,
            "success": False,
            "timed_out": True,
            "partial_results": False,
            "execution_time": 300.0,
            "timestamp": "",
        }
        r = _normalize_result(raw)
        # The appended stderr portion must be at most 200 chars
        appended = r["error"].split(": ", 1)[1] if ": " in r["error"] else ""
        assert len(appended) <= 200

    # --- legacy key preservation ---

    def test_legacy_keys_preserved(self):
        """stdout, stderr, return_code must survive for legacy consumers."""
        raw = {
            "stdout": "output here",
            "stderr": "warn",
            "return_code": 0,
            "success": True,
            "timed_out": False,
            "partial_results": False,
            "execution_time": 1.0,
            "timestamp": "ts",
        }
        r = _normalize_result(raw)
        assert r["stdout"] == "output here"
        assert r["stderr"] == "warn"
        assert r["return_code"] == 0

    # --- canonical key completeness ---

    def test_all_canonical_keys_present(self):
        CANONICAL = {"success", "output", "error", "returncode", "timed_out",
                     "partial_results", "execution_time", "timestamp"}
        raw = {
            "stdout": "x", "stderr": "", "return_code": 0,
            "success": True, "timed_out": False, "partial_results": False,
            "execution_time": 1.0, "timestamp": "ts",
        }
        r = _normalize_result(raw)
        missing = CANONICAL - set(r.keys())
        assert not missing, f"Missing canonical keys: {missing}"

    def test_missing_optional_fields_default_gracefully(self):
        """Minimal raw dict (only stdout/stderr/return_code) must not raise."""
        raw = {"stdout": "hi", "stderr": "", "return_code": 0, "success": True}
        r = _normalize_result(raw)
        assert r["success"] is True
        assert r["output"] == "hi"
        assert r["returncode"] == 0
        assert r["timed_out"] is False
        assert r["execution_time"] == 0.0


# ---------------------------------------------------------------------------
# execute_command integration — normalization applied end-to-end
# ---------------------------------------------------------------------------

class TestExecuteCommandNormalization:
    """Verify execute_command() always returns the canonical shape."""

    def test_execute_command_returns_canonical_shape(self):
        """Mock EnhancedCommandExecutor so no real subprocess is spawned."""
        fake_raw = {
            "stdout": "scan done\n",
            "stderr": "",
            "return_code": 0,
            "success": True,
            "timed_out": False,
            "partial_results": False,
            "execution_time": 2.3,
            "timestamp": "2026-01-01T00:00:00",
        }
        mock_executor = MagicMock()
        mock_executor.execute.return_value = fake_raw

        with patch(
            "server_core.command_executor.EnhancedCommandExecutor",
            return_value=mock_executor,
        ):
            from server_core.command_executor import execute_command
            result = execute_command("nmap -sV 127.0.0.1", use_cache=False)

        assert result["success"] is True
        assert result["output"] == "scan done\n"
        assert result["error"] == ""
        assert result["returncode"] == 0
        assert "stdout" in result   # legacy key preserved

    def test_execute_command_cache_stores_result_not_empty_dict(self):
        """Cache must store the result dict, not an empty dict."""
        fake_raw = {
            "stdout": "output\n", "stderr": "", "return_code": 0,
            "success": True, "timed_out": False, "partial_results": False,
            "execution_time": 1.0, "timestamp": "",
        }
        mock_executor = MagicMock()
        mock_executor.execute.return_value = fake_raw

        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # cache miss

        with patch(
            "server_core.command_executor.EnhancedCommandExecutor",
            return_value=mock_executor,
        ):
            from server_core.command_executor import execute_command
            result = execute_command("nmap 1.2.3.4", use_cache=True, cache=mock_cache)

        # set() must be called with (command_str, result_dict) — not (command, {}, result)
        mock_cache.set.assert_called_once()
        call_args = mock_cache.set.call_args[0]
        assert call_args[0] == "nmap 1.2.3.4"
        stored = call_args[1]
        assert isinstance(stored, dict)
        assert stored.get("output") == "output\n"  # not empty dict
        assert "success" in stored

    def test_execute_command_timeout_normalized(self):
        fake_timeout = {
            "stdout": "partial\n",
            "stderr": "",
            "return_code": -1,
            "success": False,
            "timed_out": True,
            "partial_results": True,
            "execution_time": 300.0,
            "timestamp": "",
        }
        mock_executor = MagicMock()
        mock_executor.execute.return_value = fake_timeout

        with patch(
            "server_core.command_executor.EnhancedCommandExecutor",
            return_value=mock_executor,
        ):
            from server_core.command_executor import execute_command
            result = execute_command("masscan 10.0.0.0/8", use_cache=False)

        assert result["success"] is False
        assert result["timed_out"] is True
        assert "timed out" in result["error"].lower()
        assert result["partial_results"] is True
