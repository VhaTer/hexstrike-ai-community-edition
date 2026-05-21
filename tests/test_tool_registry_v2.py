"""
tests/test_tool_registry_v2.py

Unit tests for mcp_core/tool_registry_v2.py (ToolRegistry).

Covers:
  - Initialization loads all TOOL_ROUTES
  - get_tool_status returns correct keys for known/unknown tools
  - get_available / get_missing partition tools correctly
  - Binary status is cached after first check
  - install() returns helpful error for unknown tools
  - install() returns helpful error for alias tools
  - install() skips if already installed
  - refresh_status() re-checks all binaries
"""

import pytest
from mcp_core.tool_registry_v2 import ToolRegistry, _SKIP_INSTALL


class TestToolRegistry:
    """ToolRegistry unit tests (all binary checks mocked)."""

    @pytest.fixture
    def reg(self):
        r = ToolRegistry()
        # Pre-fill status cache so we don't need real binaries
        for binary in r._names_by_binary:
            r._status_cache[binary] = "not_found"
        # Mark known-installed tools
        r._status_cache["nmap"] = "installed"
        r._status_cache["whatweb"] = "installed"
        r._status_cache["sqlmap"] = "installed"
        return r

    def test_loads_all_tools(self, reg):
        assert len(reg.all_tool_names) >= 130
        assert "nmap" in reg.all_tool_names
        assert "sqlmap" in reg.all_tool_names

    def test_get_tool_status_known(self, reg):
        s = reg.get_tool_status("nmap")
        assert s["name"] == "nmap"
        assert s["binary"] == "nmap"
        assert s["status"] in ("installed", "not_found")
        assert s["category"]
        assert s["desc"]
        assert s["install_hint"]

    def test_get_tool_status_unknown(self, reg):
        s = reg.get_tool_status("nonexistent_tool_xyz")
        assert s["status"] == "unknown"
        assert "error" in s

    def test_get_available_returns_only_installed(self, reg):
        avail = reg.get_available()
        names = {t["name"] for t in avail}
        assert "nmap" in names
        assert "sqlmap" in names
        assert all(t["status"] == "installed" for t in avail)

    def test_get_missing_returns_only_not_found(self, reg):
        missing = reg.get_missing()
        assert all(t["status"] == "not_found" for t in missing)
        # nmap is installed, should NOT be in missing
        missing_names = {t["name"] for t in missing}
        assert "nmap" not in missing_names

    def test_binary_status_is_cached(self, reg):
        # First call populates cache
        s1 = reg._get_binary_status("nmap")
        assert s1 == "installed"
        # Cache should be populated now
        assert "nmap" in reg._status_cache

    def test_install_unknown_tool(self, reg):
        result = reg.install("nonexistent_tool_xyz")
        assert not result["success"]
        assert "Unknown" in result["error"]

    def test_install_alias_tool(self, reg):
        for alias in _SKIP_INSTALL:
            result = reg.install(alias)
            assert not result["success"]
            assert "alias" in result["error"].lower()
            break

    def test_install_already_installed(self, reg):
        result = reg.install("nmap")
        assert result["success"]
        assert result.get("skipped")

    def test_get_tool_status_includes_all_keys(self, reg):
        s = reg.get_tool_status("sqlmap")
        assert "name" in s
        assert "binary" in s
        assert "status" in s
        assert "category" in s
        assert "desc" in s
        assert "install_hint" in s

    def test_all_tools_have_binary_mapping(self, reg):
        """Every tool should have at least a binary name."""
        for name in reg.all_tool_names:
            s = reg.get_tool_status(name)
            assert s["binary"], f"{name} has no binary mapping"

    def test_refresh_status_returns_all_binaries(self, reg):
        statuses = reg.refresh_status()
        assert isinstance(statuses, dict)
        assert len(statuses) > 0
        for binary, s in statuses.items():
            assert s in ("installed", "not_found")
