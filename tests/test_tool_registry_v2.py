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
  - _get_binary_status cache miss with shutil.which
  - _get_auto_install_cmd for apt/pip/gem/unknown tools
  - install() with auto-install success, failure, timeout, missing manager
  - _get_install_hint gem/go branches
"""

import subprocess
import pytest
from unittest.mock import patch, MagicMock
from mcp_core.tool_registry_v2 import ToolRegistry, _SKIP_INSTALL, _INSTALL_HINTS


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
        """Every tool should have at least a binary name (primitives with no binary are allowed)."""
        _PRIMITIVE_TOOLS = {"execute_code", "browser_fetch", "browser_screenshot", "browser_eval"}
        for name in reg.all_tool_names:
            if name in _PRIMITIVE_TOOLS:
                continue
            s = reg.get_tool_status(name)
            assert s["binary"], f"{name} has no binary mapping"

    def test_refresh_status_returns_all_binaries(self, reg):
        statuses = reg.refresh_status()
        assert isinstance(statuses, dict)
        assert len(statuses) > 0
        for binary, s in statuses.items():
            assert s in ("installed", "not_found")

    def test_get_binary_status_cache_miss_calls_which(self):
        """When cache is empty, _get_binary_status calls shutil.which."""
        reg = ToolRegistry()
        reg._status_cache.clear()
        with patch("mcp_core.tool_registry_v2.shutil.which", return_value=None):
            s = reg._get_binary_status("nmap")
            assert s == "not_found"
            assert reg._status_cache.get("nmap") == "not_found"

    def test_get_install_hint_gem_branch(self):
        """_get_install_hint returns gem hint for gem-based tools."""
        reg = ToolRegistry()
        hint = reg._get_install_hint("whatweb")
        assert hint.startswith("gem install")

    def test_get_install_hint_go_branch(self):
        """_get_install_hint returns go hint for go-based tools."""
        reg = ToolRegistry()
        hint = reg._get_install_hint("dalfox")
        assert hint.startswith("go install")

    def test_get_install_hint_unknown(self):
        """_get_install_hint returns manual instruction for unknown tool."""
        reg = ToolRegistry()
        hint = reg._get_install_hint("imaginary-tool-v2")
        assert "manually" in hint

    def test_get_auto_install_cmd_apt(self):
        """_get_auto_install_cmd returns apt command for apt tools."""
        reg = ToolRegistry()
        cmd = reg._get_auto_install_cmd("nmap")
        assert cmd == ["sudo", "apt", "install", "-y", "nmap"]

    def test_get_auto_install_cmd_pip(self):
        """_get_auto_install_cmd returns pip command for pip tools."""
        reg = ToolRegistry()
        cmd = reg._get_auto_install_cmd("sqlmap")
        assert cmd == ["pip", "install", "sqlmap"]

    def test_get_auto_install_cmd_gem(self):
        """_get_auto_install_cmd returns gem command for gem tools."""
        reg = ToolRegistry()
        cmd = reg._get_auto_install_cmd("whatweb")
        assert cmd == ["gem", "install", "whatweb"]

    def test_get_auto_install_cmd_none(self):
        """_get_auto_install_cmd returns None for unsupported method."""
        reg = ToolRegistry()
        cmd = reg._get_auto_install_cmd("dalfox")  # go method
        assert cmd is None

    def test_get_auto_install_cmd_unknown(self):
        """_get_auto_install_cmd returns None for unregistered tool."""
        reg = ToolRegistry()
        cmd = reg._get_auto_install_cmd("imaginary-tool-v2")
        assert cmd is None

    def test_install_no_auto_method(self):
        """install() returns hint when no auto-install method exists."""
        reg = ToolRegistry()
        # dalfox has go method -> no auto cmd
        with patch.object(reg, "_get_binary_status", return_value="not_found"):
            with patch.object(reg, "_get_install_hint", return_value="go install dalfox"):
                reg._routes["dalfox"] = ("recon_exec", "dalfox")
                result = reg.install("dalfox")
        assert not result["success"]
        assert "install_command" in result

    def test_install_subprocess_success(self):
        """install() succeeds when subprocess returns 0."""
        reg = ToolRegistry()
        with patch.object(reg, "_get_binary_status", return_value="not_found"):
            with patch.object(reg, "_get_auto_install_cmd", return_value=["echo", "ok"]):
                with patch("subprocess.run", return_value=MagicMock(returncode=0)):
                    reg._routes["whatweb"] = ("recon_exec", "whatweb")
                    result = reg.install("whatweb")
        assert result["success"]

    def test_install_subprocess_failure(self):
        """install() returns error when subprocess fails."""
        reg = ToolRegistry()
        with patch.object(reg, "_get_binary_status", return_value="not_found"):
            with patch.object(reg, "_get_auto_install_cmd", return_value=["false"]):
                with patch("subprocess.run", return_value=MagicMock(returncode=1, stderr="error")):
                    reg._routes["whatweb"] = ("recon_exec", "whatweb")
                    result = reg.install("whatweb")
        assert not result["success"]
        assert "Install failed" in result["error"]

    def test_install_subprocess_timeout(self):
        """install() returns error on subprocess timeout."""
        reg = ToolRegistry()
        with patch.object(reg, "_get_binary_status", return_value="not_found"):
            with patch.object(reg, "_get_auto_install_cmd", return_value=["sleep", "10"]):
                with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="sleep", timeout=1)):
                    reg._routes["whatweb"] = ("recon_exec", "whatweb")
                    result = reg.install("whatweb")
        assert not result["success"]
        assert "timed out" in result["error"]

    def test_install_subprocess_file_not_found(self):
        """install() returns hint when package manager is missing."""
        reg = ToolRegistry()
        with patch.object(reg, "_get_binary_status", return_value="not_found"):
            with patch.object(reg, "_get_auto_install_cmd", return_value=["apt", "install", "-y", "whatweb"]):
                with patch("subprocess.run", side_effect=FileNotFoundError):
                    reg._routes["whatweb"] = ("recon_exec", "whatweb")
                    result = reg.install("whatweb")
        assert not result["success"]
        assert "install_command" in result

    def test_install_subprocess_exception(self):
        """install() handles generic exceptions gracefully."""
        reg = ToolRegistry()
        with patch.object(reg, "_get_binary_status", return_value="not_found"):
            with patch.object(reg, "_get_auto_install_cmd", return_value=["apt", "install", "-y", "whatweb"]):
                with patch("subprocess.run", side_effect=RuntimeError("boom")):
                    reg._routes["whatweb"] = ("recon_exec", "whatweb")
                    result = reg.install("whatweb")
        assert not result["success"]
        assert "boom" in result["error"]

    def test_get_auto_install_cmd_unknown_binary(self):
        """_get_auto_install_cmd returns None for binary not in install hints."""
        reg = ToolRegistry()
        cmd = reg._get_auto_install_cmd("completely_fake_binary_xyz")
        assert cmd is None
