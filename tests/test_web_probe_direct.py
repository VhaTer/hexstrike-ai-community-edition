"""
tests/test_web_probe_direct.py

Unit tests for mcp_core/web_probe_direct.py
Covers: whatweb, commix, joomscan
"""

import pytest
from unittest.mock import patch

import mcp_core.web_probe_direct as _mod

MOCK_OK = {"success": True, "output": "tool output", "returncode": 0}


def run_probe(tool: str, data: dict):
    with patch("mcp_core.web_probe_direct.execute_command", return_value=MOCK_OK) as mock:
        result = _mod.web_probe_exec(tool, data)
        return result, mock


# ---------------------------------------------------------------------------
# Validation — common
# ---------------------------------------------------------------------------

class TestWebProbeValidation:
    def test_unknown_tool(self):
        result = _mod.web_probe_exec("unknown", {"url": "http://example.com"})
        assert result["success"] is False
        assert "Unknown" in result["error"]

    def test_whatweb_missing_url(self):
        result, _ = run_probe("whatweb", {})
        assert result["success"] is False
        assert "url" in result["error"]

    def test_commix_missing_url(self):
        result, _ = run_probe("commix", {})
        assert result["success"] is False
        assert "url" in result["error"]

    def test_joomscan_missing_url(self):
        result, _ = run_probe("joomscan", {})
        assert result["success"] is False
        assert "url" in result["error"]


# ---------------------------------------------------------------------------
# WhatWeb
# ---------------------------------------------------------------------------

class TestWhatWeb:
    def test_basic_command(self):
        _, mock = run_probe("whatweb", {"url": "http://example.com"})
        cmd = mock.call_args[0][0]
        assert "whatweb" in cmd
        assert "example.com" in cmd

    def test_default_aggression(self):
        _, mock = run_probe("whatweb", {"url": "http://example.com"})
        cmd = mock.call_args[0][0]
        assert "--aggression=1" in cmd

    def test_custom_aggression(self):
        _, mock = run_probe("whatweb", {"url": "http://example.com", "aggression": 3})
        cmd = mock.call_args[0][0]
        assert "--aggression=3" in cmd

    def test_no_errors_flag(self):
        _, mock = run_probe("whatweb", {"url": "http://example.com", "no_errors": True})
        cmd = mock.call_args[0][0]
        assert "--no-errors" in cmd

    def test_additional_args(self):
        _, mock = run_probe("whatweb", {"url": "http://example.com", "additional_args": "--color=never"})
        cmd = mock.call_args[0][0]
        assert "--color=never" in cmd

    def test_returns_success(self):
        result, _ = run_probe("whatweb", {"url": "http://example.com"})
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Commix
# ---------------------------------------------------------------------------

class TestCommix:
    def test_basic_command(self):
        _, mock = run_probe("commix", {"url": "http://example.com/vuln?id=1"})
        cmd = mock.call_args[0][0]
        assert "commix" in cmd
        assert "example.com" in cmd

    def test_batch_by_default(self):
        _, mock = run_probe("commix", {"url": "http://example.com/vuln?id=1"})
        cmd = mock.call_args[0][0]
        assert "--batch" in cmd

    def test_batch_disabled(self):
        _, mock = run_probe("commix", {"url": "http://example.com/vuln?id=1", "batch": False})
        cmd = mock.call_args[0][0]
        assert "--batch" not in cmd

    def test_level(self):
        _, mock = run_probe("commix", {"url": "http://example.com/vuln?id=1", "level": "2"})
        cmd = mock.call_args[0][0]
        assert "--level=2" in cmd

    def test_technique(self):
        _, mock = run_probe("commix", {"url": "http://example.com/vuln?id=1", "technique": "classic"})
        cmd = mock.call_args[0][0]
        assert "--technique=classic" in cmd

    def test_os_target(self):
        _, mock = run_probe("commix", {"url": "http://example.com/vuln?id=1", "os": "unix"})
        cmd = mock.call_args[0][0]
        assert "--os=unix" in cmd

    def test_proxy(self):
        _, mock = run_probe("commix", {"url": "http://example.com/vuln?id=1", "proxy": "http://127.0.0.1:8080"})
        cmd = mock.call_args[0][0]
        assert "--proxy=http://127.0.0.1:8080" in cmd

    def test_url_flag(self):
        _, mock = run_probe("commix", {"url": "http://example.com/vuln?id=1"})
        cmd = mock.call_args[0][0]
        assert "--url=" in cmd

    def test_additional_args(self):
        _, mock = run_probe("commix", {"url": "http://example.com/vuln?id=1", "additional_args": "--tor"})
        cmd = mock.call_args[0][0]
        assert "--tor" in cmd

    def test_returns_success(self):
        result, _ = run_probe("commix", {"url": "http://example.com/vuln?id=1"})
        assert result["success"] is True


# ---------------------------------------------------------------------------
# JoomScan
# ---------------------------------------------------------------------------

class TestJoomScan:
    def test_basic_command(self):
        _, mock = run_probe("joomscan", {"url": "http://joomla.example.com"})
        cmd = mock.call_args[0][0]
        assert "joomscan" in cmd
        assert "joomla.example.com" in cmd

    def test_url_flag(self):
        _, mock = run_probe("joomscan", {"url": "http://joomla.example.com"})
        cmd = mock.call_args[0][0]
        assert "--url" in cmd

    def test_enum_components_by_default(self):
        _, mock = run_probe("joomscan", {"url": "http://joomla.example.com"})
        cmd = mock.call_args[0][0]
        assert "--enumerate-components" in cmd

    def test_enum_components_disabled(self):
        _, mock = run_probe("joomscan", {"url": "http://joomla.example.com", "enum_components": False})
        cmd = mock.call_args[0][0]
        assert "--enumerate-components" not in cmd

    def test_random_agent(self):
        _, mock = run_probe("joomscan", {"url": "http://joomla.example.com", "random_agent": True})
        cmd = mock.call_args[0][0]
        assert "--random-agent" in cmd

    def test_cookie(self):
        _, mock = run_probe("joomscan", {"url": "http://joomla.example.com", "cookie": "session=abc123"})
        cmd = mock.call_args[0][0]
        assert "session=abc123" in cmd

    def test_additional_args(self):
        _, mock = run_probe("joomscan", {"url": "http://joomla.example.com", "additional_args": "--ec"})
        cmd = mock.call_args[0][0]
        assert "--ec" in cmd

    def test_returns_success(self):
        result, _ = run_probe("joomscan", {"url": "http://joomla.example.com"})
        assert result["success"] is True
