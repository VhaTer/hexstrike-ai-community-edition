"""
tests/test_vuln_intel_direct.py

Unit tests for mcp_core/vuln_intel_direct.py
Covers: vulnx
"""

import pytest
from unittest.mock import patch

import mcp_core.vuln_intel_direct as _mod

MOCK_OK = {"success": True, "output": "vulnx output", "returncode": 0}


def run_vulnx(data: dict):
    with patch("mcp_core.vuln_intel_direct.execute_command", return_value=MOCK_OK) as mock:
        result = _mod.vuln_intel_exec("vulnx", data)
        return result, mock


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestVulnxValidation:
    def test_unknown_tool(self):
        result = _mod.vuln_intel_exec("unknown", {"cve_id": "CVE-2021-44228"})
        assert result["success"] is False
        assert "Unknown" in result["error"]

    def test_missing_cve_and_search(self):
        result, _ = run_vulnx({})
        assert result["success"] is False
        assert "required" in result["error"]

    def test_cve_id_alone_is_valid(self):
        result, _ = run_vulnx({"cve_id": "CVE-2021-44228"})
        assert result["success"] is True

    def test_search_alone_is_valid(self):
        result, _ = run_vulnx({"search": "apache log4j"})
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------

class TestVulnxCommandConstruction:
    def test_cve_id_in_command(self):
        _, mock = run_vulnx({"cve_id": "CVE-2021-44228"})
        cmd = mock.call_args[0][0]
        assert "vulnx" in cmd
        assert "CVE-2021-44228" in cmd

    def test_search_in_command(self):
        _, mock = run_vulnx({"search": "log4j"})
        cmd = mock.call_args[0][0]
        assert "--search log4j" in cmd

    def test_cve_flag(self):
        _, mock = run_vulnx({"cve_id": "CVE-2021-44228"})
        cmd = mock.call_args[0][0]
        assert "--cve-id CVE-2021-44228" in cmd

    def test_auth_key(self):
        _, mock = run_vulnx({"cve_id": "CVE-2021-44228", "auth_key": "mykey123"})
        cmd = mock.call_args[0][0]
        assert "--auth-key mykey123" in cmd

    def test_output_format(self):
        _, mock = run_vulnx({"cve_id": "CVE-2021-44228", "output": "json"})
        cmd = mock.call_args[0][0]
        assert "--output json" in cmd

    def test_additional_args(self):
        _, mock = run_vulnx({"cve_id": "CVE-2021-44228", "additional_args": "--verbose"})
        cmd = mock.call_args[0][0]
        assert "--verbose" in cmd

    def test_both_cve_and_search(self):
        _, mock = run_vulnx({"cve_id": "CVE-2021-44228", "search": "log4j"})
        cmd = mock.call_args[0][0]
        assert "--cve-id" in cmd
        assert "--search" in cmd

    def test_returns_success(self):
        result, _ = run_vulnx({"cve_id": "CVE-2021-44228"})
        assert result["success"] is True
