"""
tests/test_osint_mcp_tools.py

Unit tests for mcp_tools/osint/* — no network, no Flask server needed.

Strategy:
- Patch mcp_core.osint_direct.osint_exec to intercept direct calls
- Register tools against a real FastMCP instance
- Use FastMCP 3.x public API: await mcp.get_tool()
- Inject a mock Context for ctx: Context param
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from fastmcp import FastMCP


# ---------------------------------------------------------------------------
# Helpers — identical pattern to test_wifi_mcp_tools.py
# ---------------------------------------------------------------------------

def make_mock_context():
    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.debug = AsyncMock()
    ctx.report_progress = AsyncMock()
    return ctx


def make_mock_osint_exec(success=True, extra=None):
    response = {"success": success, "output": "mocked", "returncode": 0}
    if extra:
        response.update(extra)
    return MagicMock(return_value=response)


def make_mcp_with(register_fn, logger=None):
    mcp = FastMCP("test-hexstrike-osint")
    if logger is None:
        logger = MagicMock()
    register_fn(mcp, MagicMock(), logger)
    return mcp, logger


async def call_tool(mcp, name, **kwargs):
    tool = await mcp.get_tool(name)
    assert tool is not None, f"Tool '{name}' not registered."
    ctx = make_mock_context()
    return await tool.fn(ctx, **kwargs)


async def assert_tool_registered(mcp, name):
    tool = await mcp.get_tool(name)
    assert tool is not None, f"Tool '{name}' not registered."


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ===========================================================================
# sherlock
# ===========================================================================

class TestSherlock:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mcp_tools.osint.sherlock import register_osint_sherlock_tool
        with patch("mcp_core.osint_direct.osint_exec", make_mock_osint_exec()) as mock:
            self.mock_exec = mock
            self.mcp, self.logger = make_mcp_with(register_osint_sherlock_tool)
            yield

    def test_tool_registered(self):
        run(assert_tool_registered(self.mcp, "sherlock"))

    def test_calls_correct_tool(self):
        result = run(call_tool(self.mcp, "sherlock", username="testuser"))
        self.mock_exec.assert_called_once_with("sherlock", {"username": "testuser"})
        assert result["success"] is True

    def test_returns_failure_on_error(self):
        with patch("mcp_core.osint_direct.osint_exec", make_mock_osint_exec(success=False)):
            mcp, _ = make_mcp_with(
                __import__("mcp_tools.osint.sherlock", fromlist=["register_osint_sherlock_tool"]).register_osint_sherlock_tool
            )
            result = run(call_tool(mcp, "sherlock", username="testuser"))
            assert result["success"] is False


# ===========================================================================
# spiderfoot
# ===========================================================================

class TestSpiderfoot:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mcp_tools.osint.spiderfoot import register_osint_spiderfoot_tool
        with patch("mcp_core.osint_direct.osint_exec", make_mock_osint_exec()) as mock:
            self.mock_exec = mock
            self.mcp, self.logger = make_mcp_with(register_osint_spiderfoot_tool)
            yield

    def test_tool_registered(self):
        run(assert_tool_registered(self.mcp, "spiderfoot"))

    def test_calls_correct_tool(self):
        result = run(call_tool(self.mcp, "spiderfoot", target="example.com"))
        self.mock_exec.assert_called_once_with("spiderfoot", {"target": "example.com"})
        assert result["success"] is True

    def test_returns_failure_on_error(self):
        with patch("mcp_core.osint_direct.osint_exec", make_mock_osint_exec(success=False)):
            mcp, _ = make_mcp_with(
                __import__("mcp_tools.osint.spiderfoot", fromlist=["register_osint_spiderfoot_tool"]).register_osint_spiderfoot_tool
            )
            result = run(call_tool(mcp, "spiderfoot", target="example.com"))
            assert result["success"] is False


# ===========================================================================
# sublist3r
# ===========================================================================

class TestSublist3r:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mcp_tools.osint.sublist3r import register_osint_sublist3r_tool
        with patch("mcp_core.osint_direct.osint_exec", make_mock_osint_exec()) as mock:
            self.mock_exec = mock
            self.mcp, self.logger = make_mcp_with(register_osint_sublist3r_tool)
            yield

    def test_tool_registered(self):
        run(assert_tool_registered(self.mcp, "sublist3r"))

    def test_calls_correct_tool_defaults(self):
        result = run(call_tool(self.mcp, "sublist3r", domain="example.com"))
        self.mock_exec.assert_called_once_with(
            "sublist3r", {"domain": "example.com", "threads": 3, "engine": ""}
        )
        assert result["success"] is True

    def test_calls_correct_tool_with_engine(self):
        result = run(call_tool(self.mcp, "sublist3r", domain="example.com", threads=5, engine="google"))
        self.mock_exec.assert_called_once_with(
            "sublist3r", {"domain": "example.com", "threads": 5, "engine": "google"}
        )
        assert result["success"] is True

    def test_returns_failure_on_error(self):
        with patch("mcp_core.osint_direct.osint_exec", make_mock_osint_exec(success=False)):
            mcp, _ = make_mcp_with(
                __import__("mcp_tools.osint.sublist3r", fromlist=["register_osint_sublist3r_tool"]).register_osint_sublist3r_tool
            )
            result = run(call_tool(mcp, "sublist3r", domain="example.com"))
            assert result["success"] is False


# ===========================================================================
# parsero
# ===========================================================================

class TestParsero:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mcp_tools.osint.parsero import register_osint_parsero_tool
        with patch("mcp_core.osint_direct.osint_exec", make_mock_osint_exec()) as mock:
            self.mock_exec = mock
            self.mcp, self.logger = make_mcp_with(register_osint_parsero_tool)
            yield

    def test_tool_registered(self):
        run(assert_tool_registered(self.mcp, "parsero"))

    def test_calls_correct_tool_no_args(self):
        result = run(call_tool(self.mcp, "parsero", target="https://example.com"))
        self.mock_exec.assert_called_once_with(
            "parsero", {"target": "https://example.com", "additional_args": ""}
        )
        assert result["success"] is True

    def test_calls_correct_tool_with_args(self):
        result = run(call_tool(self.mcp, "parsero", target="https://example.com", additional_args="-o"))
        self.mock_exec.assert_called_once_with(
            "parsero", {"target": "https://example.com", "additional_args": "-o"}
        )
        assert result["success"] is True

    def test_returns_failure_on_error(self):
        with patch("mcp_core.osint_direct.osint_exec", make_mock_osint_exec(success=False)):
            mcp, _ = make_mcp_with(
                __import__("mcp_tools.osint.parsero", fromlist=["register_osint_parsero_tool"]).register_osint_parsero_tool
            )
            result = run(call_tool(mcp, "parsero", target="https://example.com"))
            assert result["success"] is False


# ===========================================================================
# osint_direct.py unit tests — test handlers directly (no MCP overhead)
# ===========================================================================

class TestOsintDirectHandlers:
    """Test the _direct.py handlers directly — faster, no FastMCP needed."""

    def test_sherlock_requires_username(self):
        with patch("mcp_core.osint_direct.execute_command") as mock_exec:
            from mcp_core.osint_direct import osint_exec
            result = osint_exec("sherlock", {})
            mock_exec.assert_not_called()
            assert result["success"] is False
            assert "username" in result["error"]

    def test_spiderfoot_requires_target(self):
        with patch("mcp_core.osint_direct.execute_command") as mock_exec:
            from mcp_core.osint_direct import osint_exec
            result = osint_exec("spiderfoot", {})
            mock_exec.assert_not_called()
            assert result["success"] is False
            assert "target" in result["error"]

    def test_sublist3r_requires_domain(self):
        with patch("mcp_core.osint_direct.execute_command") as mock_exec:
            from mcp_core.osint_direct import osint_exec
            result = osint_exec("sublist3r", {})
            mock_exec.assert_not_called()
            assert result["success"] is False
            assert "domain" in result["error"]

    def test_parsero_requires_target(self):
        with patch("mcp_core.osint_direct.execute_command") as mock_exec:
            from mcp_core.osint_direct import osint_exec
            result = osint_exec("parsero", {})
            mock_exec.assert_not_called()
            assert result["success"] is False
            assert "target" in result["error"]

    def test_unknown_tool_returns_error(self):
        from mcp_core.osint_direct import osint_exec
        result = osint_exec("nonexistent_tool", {"target": "x"})
        assert result["success"] is False
        assert "Unknown" in result["error"]

    def test_sherlock_builds_correct_command(self):
        with patch("mcp_core.osint_direct.execute_command", return_value={"success": True, "output": ""}) as mock:
            from mcp_core.osint_direct import osint_exec
            osint_exec("sherlock", {"username": "vhater"})
            cmd = mock.call_args[0][0]
            assert "sherlock" in cmd
            assert "vhater" in cmd
            assert "--json" in cmd

    def test_sublist3r_builds_correct_command_with_engine(self):
        with patch("mcp_core.osint_direct.execute_command", return_value={"success": True, "output": ""}) as mock:
            from mcp_core.osint_direct import osint_exec
            osint_exec("sublist3r", {"domain": "example.com", "threads": 5, "engine": "google"})
            cmd = mock.call_args[0][0]
            assert "-d example.com" in cmd
            assert "-t 5" in cmd
            assert "-e google" in cmd

    def test_parsero_builds_correct_command_with_args(self):
        with patch("mcp_core.osint_direct.execute_command", return_value={"success": True, "output": ""}) as mock:
            from mcp_core.osint_direct import osint_exec
            osint_exec("parsero", {"target": "https://example.com", "additional_args": "-o"})
            cmd = mock.call_args[0][0]
            assert "parsero" in cmd
            assert "https://example.com" in cmd
            assert "-o" in cmd
