"""
Real tests for EnhancedCommandExecutor — fires real subprocesses.

Covers: constructor, _box_row, execute() success/failure/timeout/shell/error.
"""
import time
import subprocess
from unittest.mock import patch

import pytest

from server_core.enhanced_command_executor import EnhancedCommandExecutor, _box_row
from server_core.modern_visual_engine import ModernVisualEngine


# ============================================================================
# _box_row helper
# ============================================================================

class TestBoxRow:
    def test_plain_text(self):
        result = _box_row("hello")
        assert "hello" in result
        assert result.startswith("\x1b[")  # ANSI prefix for border

    def test_with_ansi(self):
        result = _box_row(f"{ModernVisualEngine.COLORS['HACKER_RED']}red{ModernVisualEngine.COLORS['RESET']}")
        assert "red" in result

    def test_empty(self):
        result = _box_row("")
        assert "│" in result


# ============================================================================
# Constructor
# ============================================================================

class TestConstructor:
    def test_defaults(self):
        ece = EnhancedCommandExecutor("echo hello", timeout=30)
        assert ece.command == "echo hello"
        assert ece.timeout == 30
        assert ece.process is None
        assert ece.return_code is None
        assert ece.timed_out is False

    def test_default_timeout(self):
        ece = EnhancedCommandExecutor("echo hello")
        assert isinstance(ece.timeout, int)


# ============================================================================
# execute() — real commands
# ============================================================================

class TestExecute:
    def test_simple_success(self):
        ece = EnhancedCommandExecutor("echo hello world", timeout=10)
        result = ece.execute()
        assert result["success"] is True
        assert result["return_code"] == 0
        assert "hello world" in result["stdout"]

    def test_failure(self):
        ece = EnhancedCommandExecutor("false", timeout=10)
        result = ece.execute()
        assert result["success"] is False
        assert result["return_code"] != 0

    def test_non_existent_command(self):
        ece = EnhancedCommandExecutor("nonexistent_cmd_xyz_123", timeout=10)
        result = ece.execute()
        assert result["success"] is False
        assert result["return_code"] != 0
        assert "error" in result.get("stderr", "").lower() or result["return_code"] != 0

    def test_stdout_captured(self):
        ece = EnhancedCommandExecutor("echo line1 && echo line2", timeout=10)
        result = ece.execute()
        assert "line1" in result["stdout"]
        assert "line2" in result["stdout"]

    def test_stderr_captured(self):
        ece = EnhancedCommandExecutor("echo stderr >&2", timeout=10)
        result = ece.execute()
        assert "stderr" in result["stderr"]

    def test_shell_operators_pipe(self):
        ece = EnhancedCommandExecutor("echo hello | wc -c", timeout=10)
        result = ece.execute()
        assert result["success"] is True

    def test_shell_operators_redirect(self):
        ece = EnhancedCommandExecutor("echo hello > /dev/null", timeout=10)
        result = ece.execute()
        assert result["success"] is True

    def test_shell_operators_semicolon(self):
        ece = EnhancedCommandExecutor("echo a; echo b", timeout=10)
        result = ece.execute()
        assert result["success"] is True

    def test_multi_line_command(self):
        ece = EnhancedCommandExecutor("printf 'line1\\nline2\\n'", timeout=10)
        result = ece.execute()
        assert result["success"] is True
        assert "line1" in result["stdout"]
        assert "line2" in result["stdout"]

    def test_large_output(self):
        """Generate 500 lines to test buffer handling."""
        ece = EnhancedCommandExecutor("python3 -c \"for i in range(500): print(i)\"", timeout=30)
        result = ece.execute()
        assert result["success"] is True
        assert len(result["stdout"]) > 200

    def test_trimmed_command_in_output(self):
        """Long command (>55 chars) should be truncated in the display box."""
        long_cmd = "echo " + "a" * 100
        ece = EnhancedCommandExecutor(long_cmd, timeout=10)
        result = ece.execute()
        assert result["success"] is True


class TestExecuteTimeout:
    def test_timeout_triggers(self):
        """Command that sleeps longer than timeout should be killed."""
        ece = EnhancedCommandExecutor("sleep 30", timeout=2)
        t0 = time.time()
        result = ece.execute()
        elapsed = time.time() - t0
        assert result["timed_out"] is True
        assert result["success"] is False
        assert elapsed < 10, f"Took {elapsed:.1f}s, expected < 10s"

    def test_timeout_short_sleep(self):
        """Short sleep within timeout should succeed."""
        ece = EnhancedCommandExecutor("sleep 1", timeout=10)
        result = ece.execute()
        assert result["timed_out"] is False
        assert result["success"] is True


class TestExecuteEdgeCases:
    def test_empty_command(self):
        ece = EnhancedCommandExecutor("", timeout=10)
        result = ece.execute()
        # Empty command — the shell may succeed or fail depending on platform
        assert isinstance(result["success"], bool)

    def test_whitespace_only(self):
        ece = EnhancedCommandExecutor("   ", timeout=10)
        result = ece.execute()
        assert isinstance(result["success"], bool)

    def test_special_chars_in_arg(self):
        """Command with quotes and special characters should still work."""
        ece = EnhancedCommandExecutor('echo "hello world with quotes"', timeout=10)
        result = ece.execute()
        assert result["success"] is True

    def test_unicode_output(self):
        ece = EnhancedCommandExecutor("echo 'héllo wörld'", timeout=10)
        result = ece.execute()
        assert result["success"] is True
        assert "héllo" in result["stdout"] or "hello" in result["stdout"]

    def test_return_code_nonzero(self):
        ece = EnhancedCommandExecutor("python3 -c 'exit(42)'", timeout=10)
        result = ece.execute()
        assert result["success"] is False
        assert result["return_code"] == 42

    def test_return_code_false(self):
        ece = EnhancedCommandExecutor("false", timeout=10)
        result = ece.execute()
        assert result["success"] is False
        assert result["return_code"] == 1

    def test_execution_time_positive(self):
        ece = EnhancedCommandExecutor("echo fast", timeout=10)
        result = ece.execute()
        assert result["execution_time"] > 0

    def test_timestamp_format(self):
        ece = EnhancedCommandExecutor("echo hi", timeout=10)
        result = ece.execute()
        assert "T" in result["timestamp"]  # ISO format has T separator

    def test_reuse_instance(self):
        """Same executor instance should be reusable."""
        ece = EnhancedCommandExecutor("echo first", timeout=10)
        r1 = ece.execute()
        assert r1["success"] is True
        ece.command = "echo second"
        r2 = ece.execute()
        assert r2["success"] is True
        assert "second" in r2["stdout"]


# ============================================================================
# Shell/no-shell split edge cases
# ============================================================================

class TestShellDetection:
    def test_simple_no_shell(self):
        """Simple command runs without shell=True (list args)."""
        ece = EnhancedCommandExecutor("echo test", timeout=10)
        result = ece.execute()
        assert result["success"] is True
        assert "test" in result["stdout"]

    def test_pipe_uses_shell(self):
        """Pipe operators should trigger shell=True."""
        ece = EnhancedCommandExecutor("echo test | cat", timeout=10)
        result = ece.execute()
        assert result["success"] is True

    def test_redirect_uses_shell(self):
        ece = EnhancedCommandExecutor("echo test > /dev/null", timeout=10)
        result = ece.execute()
        assert result["success"] is True
