import pytest
from unittest.mock import patch
from mcp_core.osint_direct import osint_exec
from mcp_core._helpers import require


@pytest.fixture
def mock_exec():
    with patch("mcp_core.osint_direct.execute_command") as m:
        m.return_value = {"success": True, "output": "ok", "returncode": 0}
        yield m


class TestOsintRequire:
    def test_require_all_present(self):
        assert require({"a": "x", "b": "y"}, "a", "b") == {}

    def test_require_missing(self):
        result = require({"a": "x"}, "a", "b")
        assert result == {"success": False, "error": "'b' is required"}


class TestOsintExecRouting:
    def test_unknown_tool(self):
        result = osint_exec("nonexistent", {})
        assert result["success"] is False
        assert "Unknown" in result["error"]

    def test_sherlock_routing(self, mock_exec):
        result = osint_exec("sherlock", {"username": "testuser"})
        assert result["success"] is True

    def test_spiderfoot_routing(self, mock_exec):
        result = osint_exec("spiderfoot", {"target": "example.com"})
        assert result["success"] is True

    def test_sublist3r_routing(self, mock_exec):
        result = osint_exec("sublist3r", {"domain": "example.com"})
        assert result["success"] is True

    def test_parsero_routing(self, mock_exec):
        result = osint_exec("parsero", {"target": "https://example.com"})
        assert result["success"] is True

    def test_all_tools_covered(self):
        from mcp_core.osint_direct import _HANDLERS
        expected = {"sherlock", "spiderfoot", "sublist3r", "parsero"}
        assert set(_HANDLERS.keys()) == expected


class TestSherlock:
    def test_missing_username(self, mock_exec):
        result = osint_exec("sherlock", {})
        assert result["success"] is False
        assert "username" in result["error"]

    def test_command_format(self, mock_exec):
        osint_exec("sherlock", {"username": "testuser"})
        cmd = mock_exec.call_args[0][0]
        assert "sherlock testuser" in cmd
        assert "--json" in cmd


class TestSublist3r:
    def test_default_threads(self, mock_exec):
        osint_exec("sublist3r", {"domain": "example.com"})
        cmd = mock_exec.call_args[0][0]
        assert "sublist3r -d example.com -t 3" in cmd

    def test_custom_threads(self, mock_exec):
        osint_exec("sublist3r", {"domain": "example.com", "threads": 10})
        cmd = mock_exec.call_args[0][0]
        assert "-t 10" in cmd

    def test_with_engine(self, mock_exec):
        osint_exec("sublist3r", {"domain": "example.com", "engine": "baidu"})
        cmd = mock_exec.call_args[0][0]
        assert "-e baidu" in cmd

    def test_uses_cache(self, mock_exec):
        osint_exec("sublist3r", {"domain": "example.com"})
        assert mock_exec.call_args[1].get("use_cache") is True


class TestParsero:
    def test_additional_args(self, mock_exec):
        osint_exec("parsero", {"target": "https://example.com", "additional_args": "--follow"})
        cmd = mock_exec.call_args[0][0]
        assert "parsero -u https://example.com --follow" in cmd

    def test_uses_cache(self, mock_exec):
        osint_exec("parsero", {"target": "https://example.com"})
        assert mock_exec.call_args[1].get("use_cache") is True
