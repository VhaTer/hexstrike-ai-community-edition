"""
tests/test_ctf_engine.py

Unit tests for mcp_core/ctf_engine.py
Covers: _execute_ctf_step_real(), register_ctf_tools(), and all 4 MCP tools
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call

from fastmcp import FastMCP
from server_core.workflows.ctf.CTFChallenge import CTFChallenge


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ctx():
    ctx = AsyncMock()
    ctx.info = AsyncMock()
    ctx.report_progress = AsyncMock()
    ctx.warning = MagicMock()
    return ctx


@pytest.fixture
def mock_automator():
    automator = MagicMock()
    automator._validate_flag_format.return_value = False
    automator._generate_manual_guidance.return_value = ["Try harder"]
    return automator


@pytest.fixture
def challenge():
    return CTFChallenge(
        name="Test Challenge",
        category="web",
        description="A test web challenge",
        difficulty="easy",
        target="http://example.com",
        points=100,
    )


@pytest.fixture
def sample_step():
    return {
        "step": 1,
        "action": "recon",
        "description": "Initial reconnaissance",
        "tools": ["nmap", "gobuster"],
        "parallel": False,
        "estimated_time": 300,
    }


@pytest.fixture
def manual_step():
    return {
        "step": 2,
        "action": "analyze",
        "description": "Manual code analysis",
        "tools": ["manual", "ida"],
        "parallel": False,
        "estimated_time": 600,
    }


@pytest.fixture
def parallel_step():
    return {
        "step": 3,
        "action": "scan",
        "description": "Parallel scanning",
        "tools": ["nmap", "ffuf", "gobuster"],
        "parallel": True,
        "estimated_time": 120,
    }


def _mock_direct_tools(results_map=None):
    """Build mock DIRECT_TOOLS — _run_tool calls exec_func(tool_key, params) via run_in_executor.

    Each entry: (sync_callable, tool_key). The callable receives (tool_key, params)
    and returns a dict compatible with _normalize_tool_result.
    """
    if results_map is None:
        results_map = {}
    default = {"success": True, "output": "mock output"}
    def _make_exec(result):
        def _exec(tool, params):
            return {**result, "tool": tool}
        return _exec
    all_tools = list(results_map.keys()) or ["nmap", "gobuster", "ffuf"]
    return {t: (_make_exec(results_map.get(t, default)), t) for t in all_tools}


# ---------------------------------------------------------------------------
# Test _execute_ctf_step_real()
# ---------------------------------------------------------------------------

class TestExecuteCtfStep:

    @pytest.mark.asyncio
    async def test_manual_step_returns_guidance(self, mock_ctx, challenge, manual_step):
        from mcp_core.ctf_engine import _execute_ctf_step_real
        result = await _execute_ctf_step_real(manual_step, challenge, mock_ctx)
        assert result["success"] is True
        assert "[MANUAL]" in result["output"]
        assert "ida" in result["tools_skipped"]

    @pytest.mark.asyncio
    async def test_executable_step_runs_tools(self, mock_ctx, challenge, sample_step):
        from mcp_core.ctf_engine import _execute_ctf_step_real
        with patch("mcp_core.server_setup.get_direct_tools",
                   return_value=_mock_direct_tools({"nmap": {"success": True, "output": "nmap result"},
                                                     "gobuster": {"success": True, "output": "gobuster result"}})):
            result = await _execute_ctf_step_real(sample_step, challenge, mock_ctx)
        assert result["success"] is True
        assert "nmap" in result["tools_executed"]
        assert "gobuster" in result["tools_executed"]
        assert len(result["tools_skipped"]) == 0

    @pytest.mark.asyncio
    async def test_tool_failure_sets_no_success(self, mock_ctx, challenge, sample_step):
        from mcp_core.ctf_engine import _execute_ctf_step_real
        with patch("mcp_core.server_setup.get_direct_tools",
                   return_value=_mock_direct_tools({"nmap": {"success": False, "error": "connection refused"},
                                                     "gobuster": {"success": False, "error": "connection refused"}})):
            result = await _execute_ctf_step_real(sample_step, challenge, mock_ctx)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_exception_during_tool_is_caught(self, mock_ctx, challenge, sample_step):
        from mcp_core.ctf_engine import _execute_ctf_step_real
        def _crash_exec(tool, params):
            raise RuntimeError("Tool crashed")
        with patch("mcp_core.server_setup.get_direct_tools",
                   return_value={"nmap": (_crash_exec, "nmap"),
                                  "gobuster": (_crash_exec, "gobuster")}):
            result = await _execute_ctf_step_real(sample_step, challenge, mock_ctx)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_parallel_execution(self, mock_ctx, challenge, parallel_step):
        from mcp_core.ctf_engine import _execute_ctf_step_real
        with patch("mcp_core.server_setup.get_direct_tools",
                   return_value=_mock_direct_tools({"nmap": {"success": True, "output": "nmap scan done"},
                                                     "ffuf": {"success": True, "output": "ffuf scan done"},
                                                     "gobuster": {"success": False, "error": "timeout"}})):
            result = await _execute_ctf_step_real(parallel_step, challenge, mock_ctx)
        assert "nmap" in result["tools_executed"]
        assert "ffuf" in result["tools_executed"]
        assert "gobuster" not in result["tools_executed"]

    @pytest.mark.asyncio
    async def test_gather_exception_returns_error(self, mock_ctx, challenge, parallel_step):
        from mcp_core.ctf_engine import _execute_ctf_step_real
        async def fake_gather(*a, **kw):
            return [RuntimeError("tool crashed")]
        with patch("mcp_core.ctf_engine.asyncio.gather", fake_gather):
            result = await _execute_ctf_step_real(parallel_step, challenge, mock_ctx)
        assert "[ERROR]" in result["output"]

    @pytest.mark.asyncio
    async def test_flag_extraction_from_output(self, mock_ctx, challenge, sample_step):
        from mcp_core.ctf_engine import _execute_ctf_step_real
        with patch("mcp_core.server_setup.get_direct_tools",
                   return_value=_mock_direct_tools({"nmap": {"success": True, "output": "Found flag{hidden_flag_123} in response"},
                                                     "gobuster": {"success": True, "output": "no flags"}})):
            result = await _execute_ctf_step_real(sample_step, challenge, mock_ctx)
        assert any("flag{hidden_flag_123}" in c for c in result["flag_candidates"])

    @pytest.mark.asyncio
    async def test_hex_string_flag_extraction(self, mock_ctx, challenge, sample_step):
        from mcp_core.ctf_engine import _execute_ctf_step_real
        hex_hash = "5d41402abc4b2a76b9719d911017c592"
        with patch("mcp_core.server_setup.get_direct_tools",
                   return_value=_mock_direct_tools({"nmap": {"success": True, "output": f"Hash found: {hex_hash}"},
                                                     "gobuster": {"success": True, "output": ""}})):
            result = await _execute_ctf_step_real(sample_step, challenge, mock_ctx)
        assert any(hex_hash in c for c in result["flag_candidates"])

    @pytest.mark.asyncio
    async def test_output_truncated(self, mock_ctx, challenge, sample_step):
        from mcp_core.ctf_engine import _execute_ctf_step_real
        with patch("mcp_core.server_setup.get_direct_tools",
                   return_value=_mock_direct_tools({"nmap": {"success": True, "output": "A" * 5000},
                                                     "gobuster": {"success": True, "output": ""}})):
            result = await _execute_ctf_step_real(sample_step, challenge, mock_ctx)
        assert len(result["output"]) < 10000

    @pytest.mark.asyncio
    async def test_empty_tools_returns_success(self, mock_ctx, challenge):
        from mcp_core.ctf_engine import _execute_ctf_step_real
        step = {"step": 1, "action": "noop", "description": "No tools", "tools": [], "parallel": False}
        result = await _execute_ctf_step_real(step, challenge, mock_ctx)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_null_output_handled(self, mock_ctx, challenge, sample_step):
        from mcp_core.ctf_engine import _execute_ctf_step_real
        with patch("mcp_core.server_setup.get_direct_tools",
                   return_value=_mock_direct_tools({"nmap": {"success": True, "output": ""},
                                                     "gobuster": {"success": True, "output": ""}})):
            result = await _execute_ctf_step_real(sample_step, challenge, mock_ctx)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_ctf_format_flag(self, mock_ctx, challenge, sample_step):
        from mcp_core.ctf_engine import _execute_ctf_step_real
        with patch("mcp_core.server_setup.get_direct_tools",
                   return_value=_mock_direct_tools({"nmap": {"success": True, "output": "CTF{found_it}"},
                                                     "gobuster": {"success": True, "output": ""}})):
            result = await _execute_ctf_step_real(sample_step, challenge, mock_ctx)
        assert any("CTF{found_it}" in c for c in result["flag_candidates"])

    @pytest.mark.asyncio
    async def test_mixed_manual_and_executable(self, mock_ctx, challenge):
        from mcp_core.ctf_engine import _execute_ctf_step_real
        step = {
            "step": 1, "action": "mixed", "description": "Mixed tools",
            "tools": ["nmap", "manual", "gobuster", "wireshark"], "parallel": False,
        }
        with patch("mcp_core.server_setup.get_direct_tools",
                   return_value=_mock_direct_tools({"nmap": {"success": True, "output": "OK"},
                                                     "gobuster": {"success": True, "output": "OK"}})):
            result = await _execute_ctf_step_real(step, challenge, mock_ctx)
        assert "nmap" in result["tools_executed"]
        assert "gobuster" in result["tools_executed"]
        assert "manual" in result["tools_skipped"]
        assert "wireshark" in result["tools_skipped"]


# ---------------------------------------------------------------------------
# Test register_ctf_tools()
# ---------------------------------------------------------------------------

class TestRegisterCtfTools:

    def test_registers_four_tools(self):
        mcp = FastMCP("test")
        from mcp_core.ctf_engine import register_ctf_tools
        register_ctf_tools(mcp)
        assert asyncio.run(mcp.list_tools()) is not None

    def test_tool_names_present(self):
        mcp = FastMCP("test")
        from mcp_core.ctf_engine import register_ctf_tools
        register_ctf_tools(mcp)
        names = asyncio.run(mcp.list_tools())
        tool_names = [t.name for t in names]
        assert "ctf_analyze" in tool_names
        assert "ctf_tools" in tool_names
        assert "ctf_solve" in tool_names
        assert "ctf_team" in tool_names


import asyncio


# ---------------------------------------------------------------------------
# Test ctf_analyze tool
# ---------------------------------------------------------------------------

class TestCtfAnalyze:

    @pytest.mark.asyncio
    async def test_analyze_returns_workflow(self, mock_ctx):
        workflow = {
            "tools": ["nmap", "gobuster"],
            "estimated_time": 1800,
            "success_probability": 0.85,
            "workflow_steps": [{"step": 1, "action": "recon", "tools": ["nmap"]}],
        }
        mock_manager = MagicMock()
        mock_manager.create_ctf_challenge_workflow.return_value = workflow

        mcp = FastMCP("test")
        from mcp_core.ctf_engine import register_ctf_tools
        register_ctf_tools(mcp)

        with patch("mcp_core.ctf_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.ctf_engine.get_ctf_manager", return_value=mock_manager):
            tool = await mcp.get_tool("ctf_analyze")
            result = await tool.fn(
                name="Test Challenge",
                category="web",
                description="SQL injection test",
                difficulty="medium",
                target="http://example.com",
                points=200,
            )

        assert result["success"] is True
        assert result["challenge"] == "Test Challenge"
        assert result["category"] == "web"
        assert result["workflow"]["estimated_time"] == 1800

    @pytest.mark.asyncio
    async def test_analyze_default_difficulty(self, mock_ctx):
        workflow = {"tools": [], "estimated_time": 600, "success_probability": 0.5, "workflow_steps": []}
        mock_manager = MagicMock()
        mock_manager.create_ctf_challenge_workflow.return_value = workflow

        mcp = FastMCP("test")
        from mcp_core.ctf_engine import register_ctf_tools
        register_ctf_tools(mcp)

        with patch("mcp_core.ctf_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.ctf_engine.get_ctf_manager", return_value=mock_manager):
            tool = await mcp.get_tool("ctf_analyze")
            result = await tool.fn(
                name="Crypto1",
                category="crypto",
                description="RSA attack",
            )

        assert result["success"] is True
        assert result["difficulty"] == "unknown"


# ---------------------------------------------------------------------------
# Test ctf_tools tool
# ---------------------------------------------------------------------------

class TestCtfTools:

    @pytest.mark.asyncio
    async def test_tools_with_description(self, mock_ctx):
        mock_tool_mgr = MagicMock()
        mock_tool_mgr.suggest_tools_for_challenge.return_value = ["sqlmap", "nmap"]
        mock_tool_mgr.get_tool_command.side_effect = lambda t, target: f"{t} {target}"
        mock_tool_mgr.tool_categories = {"web_recon": ["nmap", "gobuster"]}

        mcp = FastMCP("test")
        from mcp_core.ctf_engine import register_ctf_tools
        register_ctf_tools(mcp)

        with patch("mcp_core.ctf_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.ctf_engine.get_ctf_tools", return_value=mock_tool_mgr):
            tool = await mcp.get_tool("ctf_tools")
            result = await tool.fn(
                category="web",
                description="Find SQL injection",
                target="http://example.com",
            )

        assert result["success"] is True
        assert "sqlmap" in result["suggested_tools"]
        assert "web_recon" in result["category_arsenal"]

    @pytest.mark.asyncio
    async def test_tools_no_description(self, mock_ctx):
        mock_tool_mgr = MagicMock()
        mock_tool_mgr.suggest_tools_for_challenge.return_value = []
        mock_tool_mgr.tool_categories = {"misc_encoding": ["base64"]}

        mcp = FastMCP("test")
        from mcp_core.ctf_engine import register_ctf_tools
        register_ctf_tools(mcp)

        with patch("mcp_core.ctf_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.ctf_engine.get_ctf_tools", return_value=mock_tool_mgr):
            tool = await mcp.get_tool("ctf_tools")
            result = await tool.fn(category="misc")

        assert result["success"] is True
        assert result["suggested_tools"] == []
        assert "misc_encoding" in result["category_arsenal"]

    @pytest.mark.asyncio
    async def test_tool_command_fallback_on_exception(self, mock_ctx):
        mock_tool_mgr = MagicMock()
        mock_tool_mgr.suggest_tools_for_challenge.return_value = ["sqlmap"]
        mock_tool_mgr.get_tool_command.side_effect = Exception("tool not found")
        mock_tool_mgr.tool_categories = {}

        mcp = FastMCP("test")
        from mcp_core.ctf_engine import register_ctf_tools
        register_ctf_tools(mcp)

        with patch("mcp_core.ctf_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.ctf_engine.get_ctf_tools", return_value=mock_tool_mgr):
            tool = await mcp.get_tool("ctf_tools")
            result = await tool.fn(category="web", description="test", target="example.com")

        assert result["success"] is True
        assert "sqlmap example.com" in result["commands"].get("sqlmap", "")


# ---------------------------------------------------------------------------
# Test ctf_solve tool
# ---------------------------------------------------------------------------

class TestCtfSolve:

    @pytest.mark.asyncio
    async def test_dry_run_returns_planned(self, mock_ctx, mock_automator):
        workflow = {
            "tools": ["nmap"],
            "estimated_time": 3600,
            "success_probability": 0.75,
            "workflow_steps": [
                {"step": 1, "action": "recon", "description": "Scan", "tools": ["nmap"],
                 "parallel": False, "estimated_time": 300},
            ],
        }
        mock_manager = MagicMock()
        mock_manager.create_ctf_challenge_workflow.return_value = workflow

        mcp = FastMCP("test")
        from mcp_core.ctf_engine import register_ctf_tools
        register_ctf_tools(mcp)

        with patch("mcp_core.ctf_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.ctf_engine.get_ctf_manager", return_value=mock_manager), \
             patch("mcp_core.ctf_engine.get_ctf_automator", return_value=mock_automator):
            tool = await mcp.get_tool("ctf_solve")
            result = await tool.fn(
                name="Test", category="web", description="test", dry_run=True,
            )

        assert result["status"] == "planned"
        assert "steps_planned" in result

    @pytest.mark.asyncio
    async def test_flag_found_solved(self, mock_ctx, mock_automator):
        mock_automator._validate_flag_format.return_value = True
        workflow = {
            "tools": ["nmap"],
            "estimated_time": 3600,
            "success_probability": 0.75,
            "workflow_steps": [
                {"step": 1, "action": "recon", "description": "Scan", "tools": ["nmap"],
                 "parallel": False, "estimated_time": 300},
            ],
        }
        mock_manager = MagicMock()
        mock_manager.create_ctf_challenge_workflow.return_value = workflow

        mcp = FastMCP("test")
        from mcp_core.ctf_engine import register_ctf_tools
        register_ctf_tools(mcp)

        with patch("mcp_core.ctf_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.ctf_engine.get_ctf_manager", return_value=mock_manager), \
             patch("mcp_core.ctf_engine.get_ctf_automator", return_value=mock_automator), \
             patch("mcp_core.server_setup.get_direct_tools",
                   return_value=_mock_direct_tools({"nmap": {"success": True, "output": "Found flag{solved_it}"}})):
            tool = await mcp.get_tool("ctf_solve")
            result = await tool.fn(
                name="Test", category="web", description="test", dry_run=False,
            )

        assert result["status"] == "solved"
        assert result["flag"] == "flag{solved_it}"

    @pytest.mark.asyncio
    async def test_no_flag_needs_manual(self, mock_ctx, mock_automator):
        workflow = {
            "tools": ["nmap"],
            "estimated_time": 3600,
            "success_probability": 0.75,
            "workflow_steps": [
                {"step": 1, "action": "recon", "description": "Scan", "tools": ["nmap"],
                 "parallel": False, "estimated_time": 300},
            ],
        }
        mock_manager = MagicMock()
        mock_manager.create_ctf_challenge_workflow.return_value = workflow

        mcp = FastMCP("test")
        from mcp_core.ctf_engine import register_ctf_tools
        register_ctf_tools(mcp)

        with patch("mcp_core.ctf_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.ctf_engine.get_ctf_manager", return_value=mock_manager), \
             patch("mcp_core.ctf_engine.get_ctf_automator", return_value=mock_automator), \
             patch("mcp_core.server_setup.get_direct_tools",
                   return_value=_mock_direct_tools({"nmap": {"success": True, "output": "No flags here"}})):
            tool = await mcp.get_tool("ctf_solve")
            result = await tool.fn(
                name="Test", category="web", description="test", dry_run=False,
            )

        assert result["status"] == "needs_manual_intervention"
        assert len(result["manual_guidance"]) > 0

    @pytest.mark.asyncio
    async def test_max_steps_respected(self, mock_ctx, mock_automator):
        workflow = {
            "tools": ["nmap"],
            "estimated_time": 7200,
            "success_probability": 0.5,
            "workflow_steps": [
                {"step": i, "action": f"step_{i}", "description": "test",
                 "tools": ["nmap"], "parallel": False, "estimated_time": 300}
                for i in range(1, 12)
            ],
        }
        mock_manager = MagicMock()
        mock_manager.create_ctf_challenge_workflow.return_value = workflow

        mcp = FastMCP("test")
        from mcp_core.ctf_engine import register_ctf_tools
        register_ctf_tools(mcp)

        with patch("mcp_core.ctf_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.ctf_engine.get_ctf_manager", return_value=mock_manager), \
             patch("mcp_core.ctf_engine.get_ctf_automator", return_value=mock_automator), \
             patch("mcp_core.server_setup.get_direct_tools",
                   return_value=_mock_direct_tools({"nmap": {"success": False, "error": "failed"}})):
            tool = await mcp.get_tool("ctf_solve")
            result = await tool.fn(
                name="Test", category="web", description="test",
                dry_run=False, max_steps=3,
            )

        assert len(result["steps_executed"]) <= 3

    @pytest.mark.asyncio
    async def test_confidence_increments(self, mock_ctx, mock_automator):
        workflow = {
            "tools": ["nmap"],
            "estimated_time": 3600,
            "success_probability": 0.75,
            "workflow_steps": [
                {"step": 1, "action": "recon", "description": "Scan", "tools": ["nmap"],
                 "parallel": False, "estimated_time": 300},
                {"step": 2, "action": "exploit", "description": "Exploit", "tools": ["nmap"],
                 "parallel": False, "estimated_time": 300},
            ],
        }
        mock_manager = MagicMock()
        mock_manager.create_ctf_challenge_workflow.return_value = workflow

        mcp = FastMCP("test")
        from mcp_core.ctf_engine import register_ctf_tools
        register_ctf_tools(mcp)

        with patch("mcp_core.ctf_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.ctf_engine.get_ctf_manager", return_value=mock_manager), \
             patch("mcp_core.ctf_engine.get_ctf_automator", return_value=mock_automator), \
             patch("mcp_core.server_setup.get_direct_tools",
                   return_value=_mock_direct_tools({"nmap": {"success": True, "output": "OK"}})):
            tool = await mcp.get_tool("ctf_solve")
            result = await tool.fn(
                name="Test", category="web", description="test", dry_run=False,
            )

        assert result["confidence"] >= 0.2


# ---------------------------------------------------------------------------
# Test ctf_team tool
# ---------------------------------------------------------------------------

class TestCtfTeam:

    @pytest.mark.asyncio
    async def test_team_strategy_returned(self, mock_ctx):
        strategy = {
            "assignments": {"alice": []},
            "collaboration_opportunities": [],
            "expected_score": 500,
        }
        mock_coordinator = MagicMock()
        mock_coordinator.optimize_team_strategy.return_value = strategy

        mcp = FastMCP("test")
        from mcp_core.ctf_engine import register_ctf_tools
        register_ctf_tools(mcp)

        with patch("mcp_core.ctf_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.ctf_engine.get_ctf_coordinator", return_value=mock_coordinator):
            tool = await mcp.get_tool("ctf_team")
            result = await tool.fn(
                team_skills={"alice": ["web"], "bob": ["pwn"]},
                challenges=[{"name": "Web1", "category": "web", "description": "test", "points": 100}],
            )

        assert result["success"] is True
        assert "strategy" in result
        assert result["strategy"] == strategy

    @pytest.mark.asyncio
    async def test_team_empty_challenges(self, mock_ctx):
        strategy = {"assignments": {}, "collaboration_opportunities": [], "expected_score": 0}
        mock_coordinator = MagicMock()
        mock_coordinator.optimize_team_strategy.return_value = strategy

        mcp = FastMCP("test")
        from mcp_core.ctf_engine import register_ctf_tools
        register_ctf_tools(mcp)

        with patch("mcp_core.ctf_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.ctf_engine.get_ctf_coordinator", return_value=mock_coordinator):
            tool = await mcp.get_tool("ctf_team")
            result = await tool.fn(
                team_skills={"alice": ["web"]},
                challenges=[],
            )

        assert result["success"] is True
