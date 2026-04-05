"""
tests/test_testssl_direct.py

Unit tests for mcp_core/testssl_direct.py

Strategy:
- Mock execute_command to avoid needing testssl.sh installed
- Verify command construction for each parameter combination
- Verify error handling for missing required fields
"""

import pytest
from unittest.mock import patch, MagicMock

import mcp_core.testssl_direct as _mod

MOCK_OK = {"success": True, "output": "testssl output", "returncode": 0}


def run_testssl(data: dict) -> dict:
    with patch("mcp_core.testssl_direct.execute_command", return_value=MOCK_OK) as mock:
        result = _mod.testssl_exec("testssl", data)
        return result, mock


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestTestsslValidation:
    def test_missing_target_returns_error(self):
        result, _ = run_testssl({})
        assert result["success"] is False
        assert "target" in result["error"]

    def test_unknown_tool_returns_error(self):
        result = _mod.testssl_exec("unknown_tool", {"target": "example.com"})
        assert result["success"] is False
        assert "Unknown" in result["error"]


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------

class TestTestsslCommandConstruction:
    def test_basic_command(self):
        _, mock = run_testssl({"target": "example.com"})
        cmd = mock.call_args[0][0]
        assert "testssl.sh" in cmd
        assert "example.com" in cmd

    def test_quiet_by_default(self):
        _, mock = run_testssl({"target": "example.com"})
        cmd = mock.call_args[0][0]
        assert "--quiet" in cmd

    def test_quiet_disabled(self):
        _, mock = run_testssl({"target": "example.com", "quiet": False})
        cmd = mock.call_args[0][0]
        assert "--quiet" not in cmd

    def test_protocols_by_default(self):
        _, mock = run_testssl({"target": "example.com"})
        cmd = mock.call_args[0][0]
        assert "--protocols" in cmd

    def test_protocols_disabled(self):
        _, mock = run_testssl({"target": "example.com", "protocols": False})
        cmd = mock.call_args[0][0]
        assert "--protocols" not in cmd

    def test_server_defaults_by_default(self):
        _, mock = run_testssl({"target": "example.com"})
        cmd = mock.call_args[0][0]
        assert "--server-defaults" in cmd

    def test_vulnerable_flag(self):
        _, mock = run_testssl({"target": "example.com", "vulnerable": True})
        cmd = mock.call_args[0][0]
        assert "--vulnerable" in cmd

    def test_full_flag(self):
        _, mock = run_testssl({"target": "example.com", "full": True})
        cmd = mock.call_args[0][0]
        assert "--full" in cmd

    def test_headers_flag(self):
        _, mock = run_testssl({"target": "example.com", "headers": True})
        cmd = mock.call_args[0][0]
        assert "--headers" in cmd

    def test_starttls(self):
        _, mock = run_testssl({"target": "mail.example.com", "starttls": "smtp"})
        cmd = mock.call_args[0][0]
        assert "--starttls smtp" in cmd

    def test_severity_filter(self):
        _, mock = run_testssl({"target": "example.com", "severity": "high"})
        cmd = mock.call_args[0][0]
        assert "--severity HIGH" in cmd

    def test_json_output(self):
        _, mock = run_testssl({"target": "example.com", "json_output": True})
        cmd = mock.call_args[0][0]
        assert "--jsonfile" in cmd

    def test_custom_jsonfile(self):
        _, mock = run_testssl({
            "target": "example.com",
            "json_output": True,
            "jsonfile": "/tmp/custom.json"
        })
        cmd = mock.call_args[0][0]
        assert "/tmp/custom.json" in cmd

    def test_ipv4_only(self):
        _, mock = run_testssl({"target": "example.com", "ipv4_only": True})
        cmd = mock.call_args[0][0]
        assert " -4" in cmd

    def test_ipv6_only(self):
        _, mock = run_testssl({"target": "example.com", "ipv6_only": True})
        cmd = mock.call_args[0][0]
        assert " -6" in cmd

    def test_proxy(self):
        _, mock = run_testssl({"target": "example.com", "proxy": "http://proxy:8080"})
        cmd = mock.call_args[0][0]
        assert "--proxy http://proxy:8080" in cmd

    def test_specific_ip(self):
        _, mock = run_testssl({"target": "example.com", "ip": "1.2.3.4"})
        cmd = mock.call_args[0][0]
        assert "--ip 1.2.3.4" in cmd

    def test_target_with_port(self):
        _, mock = run_testssl({"target": "example.com:8443"})
        cmd = mock.call_args[0][0]
        assert "example.com:8443" in cmd

    def test_additional_args(self):
        _, mock = run_testssl({"target": "example.com", "additional_args": "--sneaky"})
        cmd = mock.call_args[0][0]
        assert "--sneaky" in cmd

    def test_target_last(self):
        """Target must be the last argument in the command."""
        _, mock = run_testssl({"target": "example.com", "vulnerable": True})
        cmd = mock.call_args[0][0]
        assert cmd.endswith("example.com")

    def test_forward_secrecy(self):
        _, mock = run_testssl({"target": "example.com", "forward_secrecy": True})
        cmd = mock.call_args[0][0]
        assert "--fs" in cmd

    def test_rating_only(self):
        _, mock = run_testssl({"target": "example.com", "rating_only": True})
        cmd = mock.call_args[0][0]
        assert "--rating" in cmd


# ---------------------------------------------------------------------------
# Return value
# ---------------------------------------------------------------------------

class TestTestsslReturnValue:
    def test_returns_success(self):
        result, _ = run_testssl({"target": "example.com"})
        assert result["success"] is True

    def test_returns_output(self):
        result, _ = run_testssl({"target": "example.com"})
        assert "output" in result
