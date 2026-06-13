"""
tests/test_bugbounty_engine.py

Unit tests for mcp_core/bugbounty_engine.py
Covers: _execute_bb_phase(), register_bugbounty_tools(), all 5 MCP tools
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from fastmcp import FastMCP


@pytest.fixture
def mock_ctx():
    ctx = AsyncMock()
    ctx.info = AsyncMock()
    ctx.report_progress = AsyncMock()
    return ctx


@pytest.fixture
def mock_bb_mgr():
    mgr = MagicMock()
    mgr.create_reconnaissance_workflow.return_value = {
        "phases": [
            {"name": "subdomain_enum", "description": "Find subdomains",
             "tools": [{"tool": "nmap", "params": {}}]},
        ],
        "estimated_time": 600,
        "tools_count": 1,
    }
    mgr.create_vulnerability_hunting_workflow.return_value = {
        "vulnerability_tests": [
            {"priority": 1, "vulnerability_type": "rce", "tools": ["nmap"],
             "test_scenarios": [{"payloads": ["test"]}]},
        ],
        "estimated_time": 1800,
        "priority_score": 85,
    }
    mgr.create_business_logic_testing_workflow.return_value = {
        "business_logic_tests": [
            {"category": "auth", "tests": [
                {"name": "jwt_bypass", "method": "automated"},
            ]},
        ],
        "estimated_time": 480,
        "manual_testing_required": True,
    }
    mgr.create_osint_workflow.return_value = {
        "osint_phases": [
            {"name": "domain_intel", "tools": [{"tool": "whois", "params": {}}]},
        ],
        "estimated_time": 240,
        "intelligence_types": ["whois"],
    }
    return mgr


from contextlib import contextmanager


@contextmanager
def _patch_bb_tools(results_map=None):
    """Context manager: patches run_security_tool with canned per-tool results."""
    if results_map is None:
        results_map = {}

    async def _mock(ctx, tool_name, params):
        data = results_map.get(tool_name, {})
        if not data:
            return {"success": False, "error": f"Unknown tool: {tool_name}", "returncode": 1}
        return {
            "success": data.get("success", True),
            "output": data.get("output", ""),
            "stdout": data.get("output", ""),
            "error": data.get("error", ""),
            "returncode": 0 if data.get("success", True) else 1,
            "duration": 0.1,
            **data,
        }

    with patch("mcp_core.server_setup.run_security_tool", new=_mock):
        yield


# ---------------------------------------------------------------------------
# Test _execute_bb_phase()
# ---------------------------------------------------------------------------

class TestExecuteBbPhase:

    @pytest.mark.asyncio
    async def test_dry_run_returns_skipped(self, mock_ctx):
        from mcp_core.bugbounty_engine import _execute_bb_phase
        phase = {"name": "recon", "description": "test",
                 "tools": [{"tool": "nmap", "params": {}}]}
        result = await _execute_bb_phase(phase, "example.com", mock_ctx, dry_run=True)
        assert result["success"] is True
        assert "nmap" in result["tools_skipped"]

    @pytest.mark.asyncio
    async def test_manual_tools_are_skipped(self, mock_ctx):
        from mcp_core.bugbounty_engine import _execute_bb_phase
        phase = {"name": "osint", "description": "OSINT phase",
                 "tools": [{"tool": "shodan", "params": {}}]}
        result = await _execute_bb_phase(phase, "example.com", mock_ctx)
        assert "shodan" in result["tools_skipped"]

    @pytest.mark.asyncio
    async def test_executable_tools_run(self, mock_ctx):
        from mcp_core.bugbounty_engine import _execute_bb_phase
        phase = {"name": "recon", "description": "Recon",
                 "tools": [{"tool": "nmap", "params": {}}]}

        with _patch_bb_tools({"nmap": {"success": True, "output": "scan results"}}):
            result = await _execute_bb_phase(phase, "example.com", mock_ctx)

        assert "nmap" in result["tools_executed"]
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_tool_failure_skipped(self, mock_ctx):
        from mcp_core.bugbounty_engine import _execute_bb_phase
        phase = {"name": "recon", "description": "Recon",
                 "tools": [{"tool": "nmap", "params": {}}]}

        with _patch_bb_tools({"nmap": {"success": False, "error": "timeout"}}):
            result = await _execute_bb_phase(phase, "example.com", mock_ctx)

        assert "nmap" in result["tools_skipped"]
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_exception_during_tool_caught(self, mock_ctx):
        from mcp_core.bugbounty_engine import _execute_bb_phase

        async def _crash_mock(ctx, tool_name, params):
            raise RuntimeError("crashed")

        phase = {"name": "recon", "description": "Recon",
                 "tools": [{"tool": "nmap", "params": {}}]}

        with patch("mcp_core.server_setup.run_security_tool", new=_crash_mock):
            result = await _execute_bb_phase(phase, "example.com", mock_ctx)

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_mixed_tools(self, mock_ctx):
        from mcp_core.bugbounty_engine import _execute_bb_phase
        phase = {"name": "mixed", "description": "Mixed",
                 "tools": [
                     {"tool": "nmap", "params": {}},
                     {"tool": "shodan", "params": {}},
                     {"tool": "gobuster", "params": {}},
                 ]}

        with _patch_bb_tools({"nmap": {"success": True, "output": "ok"},
                               "gobuster": {"success": True, "output": "ok"}}):
            result = await _execute_bb_phase(phase, "example.com", mock_ctx)

        assert "nmap" in result["tools_executed"]
        assert "gobuster" in result["tools_executed"]
        assert "shodan" in result["tools_skipped"]

    @pytest.mark.asyncio
    async def test_gather_exception_skipped(self, mock_ctx):
        from mcp_core.bugbounty_engine import _execute_bb_phase
        async def fake_gather(*a, **kw):
            return [RuntimeError("tool crashed")]
        phase = {"name": "recon", "description": "Recon",
                 "tools": [{"tool": "nmap", "params": {}}]}
        with patch("mcp_core.bugbounty_engine.asyncio.gather", fake_gather):
            result = await _execute_bb_phase(phase, "example.com", mock_ctx)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_empty_phase_no_tools(self, mock_ctx):
        from mcp_core.bugbounty_engine import _execute_bb_phase
        phase = {"name": "empty", "description": "No tools", "tools": []}
        result = await _execute_bb_phase(phase, "example.com", mock_ctx)
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Test register_bugbounty_tools()
# ---------------------------------------------------------------------------

class TestRegisterBbTools:

    def test_registers_five_tools(self):
        mcp = FastMCP("test")
        from mcp_core.bugbounty_engine import register_bugbounty_tools
        register_bugbounty_tools(mcp)
        names = [t.name for t in __import__('asyncio').run(mcp.list_tools())]
        assert "bb_recon" in names
        assert "bb_hunt" in names
        assert "bb_business" in names
        assert "bb_osint" in names
        assert "bb_full" in names


# ---------------------------------------------------------------------------
# Test bb_recon tool
# ---------------------------------------------------------------------------

class TestBbRecon:

    @pytest.mark.asyncio
    async def test_dry_run_no_execution(self, mock_ctx, mock_bb_mgr):
        mcp = FastMCP("test")
        from mcp_core.bugbounty_engine import register_bugbounty_tools
        register_bugbounty_tools(mcp)

        with patch("mcp_core.bugbounty_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.bugbounty_engine.get_bugbounty_manager", return_value=mock_bb_mgr):
            tool = await mcp.get_tool("bb_recon")
            result = await tool.fn(domain="example.com", dry_run=True)

        assert result["success"] is True
        assert result["mode"] == "DRY RUN"
        assert "phase_results" in result

    @pytest.mark.asyncio
    async def test_live_with_execution(self, mock_ctx, mock_bb_mgr):
        mcp = FastMCP("test")
        from mcp_core.bugbounty_engine import register_bugbounty_tools
        register_bugbounty_tools(mcp)

        with patch("mcp_core.bugbounty_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.bugbounty_engine.get_bugbounty_manager", return_value=mock_bb_mgr), \
             _patch_bb_tools({"nmap": {"success": True, "output": "ok"}}):
            tool = await mcp.get_tool("bb_recon")
            result = await tool.fn(domain="example.com", dry_run=False)

        assert result["success"] is True
        assert result["mode"] == "LIVE"

    @pytest.mark.asyncio
    async def test_scope_parsing(self, mock_ctx, mock_bb_mgr):
        mcp = FastMCP("test")
        from mcp_core.bugbounty_engine import register_bugbounty_tools
        register_bugbounty_tools(mcp)

        with patch("mcp_core.bugbounty_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.bugbounty_engine.get_bugbounty_manager", return_value=mock_bb_mgr):
            tool = await mcp.get_tool("bb_recon")
            result = await tool.fn(
                domain="example.com", dry_run=True,
                scope=["example.com", "api.example.com"],
                out_of_scope=["dev.example.com"],
            )

        assert result["success"] is True


# ---------------------------------------------------------------------------
# Test bb_hunt tool
# ---------------------------------------------------------------------------

class TestBbHunt:

    @pytest.mark.asyncio
    async def test_hunt_success(self, mock_ctx, mock_bb_mgr):
        mcp = FastMCP("test")
        from mcp_core.bugbounty_engine import register_bugbounty_tools
        register_bugbounty_tools(mcp)

        with patch("mcp_core.bugbounty_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.bugbounty_engine.get_bugbounty_manager", return_value=mock_bb_mgr):
            tool = await mcp.get_tool("bb_hunt")
            result = await tool.fn(domain="example.com")

        assert result["success"] is True
        assert result["summary"]["vulns_prioritized"] == 1
        assert result["summary"]["top_priority"] == "rce"

    @pytest.mark.asyncio
    async def test_hunt_custom_priority(self, mock_ctx, mock_bb_mgr):
        mcp = FastMCP("test")
        from mcp_core.bugbounty_engine import register_bugbounty_tools
        register_bugbounty_tools(mcp)

        with patch("mcp_core.bugbounty_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.bugbounty_engine.get_bugbounty_manager", return_value=mock_bb_mgr):
            tool = await mcp.get_tool("bb_hunt")
            result = await tool.fn(
                domain="example.com",
                priority_vulns=["rce", "sqli"],
                program_type="api",
            )

        assert result["success"] is True


# ---------------------------------------------------------------------------
# Test bb_business tool
# ---------------------------------------------------------------------------

class TestBbBusiness:

    @pytest.mark.asyncio
    async def test_business_success(self, mock_ctx, mock_bb_mgr):
        mcp = FastMCP("test")
        from mcp_core.bugbounty_engine import register_bugbounty_tools
        register_bugbounty_tools(mcp)

        with patch("mcp_core.bugbounty_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.bugbounty_engine.get_bugbounty_manager", return_value=mock_bb_mgr):
            tool = await mcp.get_tool("bb_business")
            result = await tool.fn(domain="example.com")

        assert result["success"] is True
        assert result["summary"]["categories"] == 1
        assert result["summary"]["total_tests"] == 1


# ---------------------------------------------------------------------------
# Test bb_osint tool
# ---------------------------------------------------------------------------

class TestBbOsint:

    @pytest.mark.asyncio
    async def test_osint_success(self, mock_ctx, mock_bb_mgr):
        mcp = FastMCP("test")
        from mcp_core.bugbounty_engine import register_bugbounty_tools
        register_bugbounty_tools(mcp)

        with patch("mcp_core.bugbounty_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.bugbounty_engine.get_bugbounty_manager", return_value=mock_bb_mgr):
            tool = await mcp.get_tool("bb_osint")
            result = await tool.fn(domain="example.com")

        assert result["success"] is True
        assert "whois" in result["intel_types"]


# ---------------------------------------------------------------------------
# Test bb_full tool
# ---------------------------------------------------------------------------

class TestBbFull:

    @pytest.mark.asyncio
    async def test_full_dry_run(self, mock_ctx, mock_bb_mgr):
        mcp = FastMCP("test")
        from mcp_core.bugbounty_engine import register_bugbounty_tools
        register_bugbounty_tools(mcp)

        with patch("mcp_core.bugbounty_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.bugbounty_engine.get_bugbounty_manager", return_value=mock_bb_mgr):
            tool = await mcp.get_tool("bb_full")
            result = await tool.fn(domain="example.com", dry_run=True)

        assert result["success"] is True
        assert result["mode"] == "DRY RUN"
        assert "engagement" in result
        assert "summary" in result
        assert "reconnaissance" in result["engagement"]

    @pytest.mark.asyncio
    async def test_full_live(self, mock_ctx, mock_bb_mgr):
        mcp = FastMCP("test")
        from mcp_core.bugbounty_engine import register_bugbounty_tools
        register_bugbounty_tools(mcp)

        with patch("mcp_core.bugbounty_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.bugbounty_engine.get_bugbounty_manager", return_value=mock_bb_mgr), \
             _patch_bb_tools({"nmap": {"success": True, "output": "ok"}}):
            tool = await mcp.get_tool("bb_full")
            result = await tool.fn(domain="example.com", dry_run=False)

        assert result["success"] is True
        assert result["mode"] == "LIVE"

    @pytest.mark.asyncio
    async def test_full_custom_params(self, mock_ctx, mock_bb_mgr):
        mcp = FastMCP("test")
        from mcp_core.bugbounty_engine import register_bugbounty_tools
        register_bugbounty_tools(mcp)

        with patch("mcp_core.bugbounty_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.bugbounty_engine.get_bugbounty_manager", return_value=mock_bb_mgr):
            tool = await mcp.get_tool("bb_full")
            result = await tool.fn(
                domain="example.com", dry_run=True,
                scope=["example.com"], out_of_scope=["old.example.com"],
                priority_vulns=["rce"], program_type="api",
            )

        assert result["success"] is True
