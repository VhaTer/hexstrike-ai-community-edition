"""Tests for exec_direct — arbitrary code execution primitives."""

import pytest

from mcp_core.exec_direct import exec_direct, _execute_code, _HANDLERS


def test_unknown_tool():
    result = exec_direct("nonexistent", {})
    assert result["success"] is False
    assert "Unknown exec tool" in result.get("error", "")


def test_execute_code_unknown_language():
    result = _execute_code({"code": "print(1)", "language": "brainfuck"})
    assert result["success"] is False
    assert "unsupported language" in result.get("error", "")


def test_execute_code_empty_code():
    result = _execute_code({"code": "", "language": "python"})
    assert result["success"] is False
    assert "code is required" in result.get("error", "")


def test_python_success():
    result = _execute_code({
        "code": "print('hello from hexstrike')",
        "language": "python",
        "timeout": 10,
    })
    assert result["success"] is True
    assert result["exit_code"] == 0
    assert "hello from hexstrike" in result["stdout"]
    assert result["language"] == "python"
    assert result["timeout"] is False


def test_python_expression():
    result = _execute_code({
        "code": "print(sum(range(100)))",
        "language": "python",
        "timeout": 10,
    })
    assert result["success"] is True
    assert result["exit_code"] == 0
    assert "4950" in result["stdout"]


def test_python_stderr():
    result = _execute_code({
        "code": "import sys; print('stdout'); print('stderr', file=sys.stderr)",
        "language": "python",
    })
    assert result["success"] is True
    assert "stdout" in result["stdout"]
    assert "stderr" in result["stderr"]


def test_python_non_zero_exit():
    result = _execute_code({
        "code": "import sys; sys.exit(42)",
        "language": "python",
        "timeout": 5,
    })
    assert result["success"] is True
    assert result["exit_code"] == 42


def test_python_syntax_error():
    result = _execute_code({
        "code": "if True print('invalid')",
        "language": "python",
        "timeout": 5,
    })
    assert result["success"] is True
    assert result["exit_code"] != 0
    assert "SyntaxError" in result["stderr"] or "invalid syntax" in result["stderr"]


def test_bash_success():
    result = _execute_code({
        "code": "echo 'hello from bash'",
        "language": "bash",
        "timeout": 5,
    })
    assert result["success"] is True
    assert result["exit_code"] == 0
    assert "hello from bash" in result["stdout"]


def test_bash_pipeline():
    result = _execute_code({
        "code": "echo 'foo\nbar\nbaz' | wc -l",
        "language": "bash",
        "timeout": 5,
    })
    assert result["success"] is True
    assert "3" in result["stdout"]


def test_sandbox_param_reserved():
    """sandbox=False is accepted but ignored (reserved for future)."""
    result = _execute_code({
        "code": "print('sandbox test')",
        "language": "python",
        "sandbox": True,
    })
    assert result["success"] is True
    assert "sandbox test" in result["stdout"]


def test_timeout():
    result = _execute_code({
        "code": "import time; time.sleep(10)",
        "language": "python",
        "timeout": 1,
    })
    assert result["success"] is False
    assert result["timeout"] is True
    assert "timed out" in result["stderr"].lower()


def test_language_shorthand():
    result = _execute_code({"code": "echo sh", "language": "sh", "timeout": 5})
    assert result["success"] is True


def test_dispatch_integration():
    result = exec_direct("execute_code", {
        "code": "print('dispatched')",
        "language": "python",
        "timeout": 5,
    })
    assert result["success"] is True
    assert "dispatched" in result["stdout"]


def test_default_language_is_python():
    result = _execute_code({"code": "print('default lang')", "timeout": 5})
    assert result["success"] is True
    assert "default lang" in result["stdout"]


def test_node_skip_if_not_installed():
    """Node tests are best-effort — skip if node not installed."""
    import shutil
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    result = _execute_code({
        "code": "console.log('hello from node')",
        "language": "node",
        "timeout": 5,
    })
    assert result["success"] is True
    assert "hello from node" in result["stdout"]
