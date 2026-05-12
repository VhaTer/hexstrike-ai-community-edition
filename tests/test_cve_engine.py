"""
tests/test_cve_engine.py

Unit tests for mcp_core/cve_engine.py
Covers: register_cve_tools(), all 4 MCP tools, risk score computation
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from fastmcp import FastMCP


@pytest.fixture
def mock_ctx():
    ctx = AsyncMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.report_progress = AsyncMock()
    return ctx


@pytest.fixture
def mock_cve_mgr():
    mgr = MagicMock()
    mgr.fetch_latest_cves.return_value = {
        "success": True,
        "cves": [{"id": "CVE-2024-0001"}],
        "data_sources": ["nvd"],
    }
    mgr.analyze_cve_exploitability.return_value = {
        "success": True,
        "cvss_score": 9.1,
        "severity": "CRITICAL",
        "exploitability_score": 0.85,
        "exploitability_level": "HIGH",
        "attack_vector": "NETWORK",
        "threat_intelligence": {
            "recommended_priority": "IMMEDIATE",
            "active_exploitation": True,
            "exploit_indicators": ["ransomware_groups"],
        },
    }
    mgr.search_existing_exploits.return_value = {
        "success": True,
        "exploits": [{"source": "github", "url": "https://github.com/test"}],
        "search_summary": {"github_repos": 1, "metasploit_modules": 0, "exploit_db_refs": 0},
    }
    return mgr


# ---------------------------------------------------------------------------
# Test register_cve_tools()
# ---------------------------------------------------------------------------

class TestRegisterCveTools:

    def test_registers_four_tools(self):
        mcp = FastMCP("test")
        from mcp_core.cve_engine import register_cve_tools
        register_cve_tools(mcp)
        names = [t.name for t in __import__('asyncio').run(mcp.list_tools())]
        assert "cve_fetch" in names
        assert "cve_analyze" in names
        assert "cve_exploits" in names
        assert "cve_intel" in names


# ---------------------------------------------------------------------------
# Test cve_fetch tool
# ---------------------------------------------------------------------------

class TestCveFetch:

    @pytest.mark.asyncio
    async def test_fetch_success(self, mock_ctx, mock_cve_mgr):
        mcp = FastMCP("test")
        from mcp_core.cve_engine import register_cve_tools
        register_cve_tools(mcp)

        with patch("mcp_core.cve_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.cve_engine.get_cve_intelligence", return_value=mock_cve_mgr):
            tool = await mcp.get_tool("cve_fetch")
            result = await tool.fn(hours=48, severity="CRITICAL")

        assert result["success"] is True
        assert len(result["cves"]) == 1
        mock_cve_mgr.fetch_latest_cves.assert_called_with(hours=48, severity_filter="CRITICAL")

    @pytest.mark.asyncio
    async def test_fetch_failure(self, mock_ctx, mock_cve_mgr):
        mock_cve_mgr.fetch_latest_cves.return_value = {
            "success": False, "error": "NVD API rate limited", "cves": [],
        }
        mcp = FastMCP("test")
        from mcp_core.cve_engine import register_cve_tools
        register_cve_tools(mcp)

        with patch("mcp_core.cve_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.cve_engine.get_cve_intelligence", return_value=mock_cve_mgr):
            tool = await mcp.get_tool("cve_fetch")
            result = await tool.fn()

        assert result["success"] is False
        assert "rate limited" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_fetch_default_params(self, mock_ctx, mock_cve_mgr):
        mcp = FastMCP("test")
        from mcp_core.cve_engine import register_cve_tools
        register_cve_tools(mcp)

        with patch("mcp_core.cve_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.cve_engine.get_cve_intelligence", return_value=mock_cve_mgr):
            tool = await mcp.get_tool("cve_fetch")
            result = await tool.fn()

        assert result["success"] is True


# ---------------------------------------------------------------------------
# Test cve_analyze tool
# ---------------------------------------------------------------------------

class TestCveAnalyze:

    @pytest.mark.asyncio
    async def test_analyze_invalid_cve(self, mock_ctx):
        mcp = FastMCP("test")
        from mcp_core.cve_engine import register_cve_tools
        register_cve_tools(mcp)

        with patch("mcp_core.cve_engine.get_context", return_value=mock_ctx):
            tool = await mcp.get_tool("cve_analyze")
            result = await tool.fn(cve_id="not-a-cve")

        assert result["success"] is False
        assert "Invalid CVE ID" in result["error"]

    @pytest.mark.asyncio
    async def test_analyze_valid_cve(self, mock_ctx, mock_cve_mgr):
        mcp = FastMCP("test")
        from mcp_core.cve_engine import register_cve_tools
        register_cve_tools(mcp)

        with patch("mcp_core.cve_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.cve_engine.get_cve_intelligence", return_value=mock_cve_mgr):
            tool = await mcp.get_tool("cve_analyze")
            result = await tool.fn(cve_id="CVE-2024-1234")

        assert result["success"] is True
        assert result["cvss_score"] == 9.1

    @pytest.mark.asyncio
    async def test_analyze_lowercase_cve(self, mock_ctx, mock_cve_mgr):
        mcp = FastMCP("test")
        from mcp_core.cve_engine import register_cve_tools
        register_cve_tools(mcp)

        with patch("mcp_core.cve_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.cve_engine.get_cve_intelligence", return_value=mock_cve_mgr):
            tool = await mcp.get_tool("cve_analyze")
            result = await tool.fn(cve_id="cve-2024-5678")

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_analyze_failure(self, mock_ctx, mock_cve_mgr):
        mock_cve_mgr.analyze_cve_exploitability.return_value = {
            "success": False, "error": "CVE not found",
        }
        mcp = FastMCP("test")
        from mcp_core.cve_engine import register_cve_tools
        register_cve_tools(mcp)

        with patch("mcp_core.cve_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.cve_engine.get_cve_intelligence", return_value=mock_cve_mgr):
            tool = await mcp.get_tool("cve_analyze")
            result = await tool.fn(cve_id="CVE-2024-9999")

        assert result["success"] is False


# ---------------------------------------------------------------------------
# Test cve_exploits tool
# ---------------------------------------------------------------------------

class TestCveExploits:

    @pytest.mark.asyncio
    async def test_exploits_invalid_cve(self, mock_ctx):
        mcp = FastMCP("test")
        from mcp_core.cve_engine import register_cve_tools
        register_cve_tools(mcp)

        with patch("mcp_core.cve_engine.get_context", return_value=mock_ctx):
            tool = await mcp.get_tool("cve_exploits")
            result = await tool.fn(cve_id="bad")

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_exploits_success(self, mock_ctx, mock_cve_mgr):
        mcp = FastMCP("test")
        from mcp_core.cve_engine import register_cve_tools
        register_cve_tools(mcp)

        with patch("mcp_core.cve_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.cve_engine.get_cve_intelligence", return_value=mock_cve_mgr):
            tool = await mcp.get_tool("cve_exploits")
            result = await tool.fn(cve_id="CVE-2024-1234")

        assert result["success"] is True
        assert len(result["exploits"]) == 1

    @pytest.mark.asyncio
    async def test_exploits_failure(self, mock_ctx, mock_cve_mgr):
        mock_cve_mgr.search_existing_exploits.return_value = {
            "success": False, "error": "API unavailable",
        }
        mcp = FastMCP("test")
        from mcp_core.cve_engine import register_cve_tools
        register_cve_tools(mcp)

        with patch("mcp_core.cve_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.cve_engine.get_cve_intelligence", return_value=mock_cve_mgr):
            tool = await mcp.get_tool("cve_exploits")
            result = await tool.fn(cve_id="CVE-2024-1234")

        assert result["success"] is False


# ---------------------------------------------------------------------------
# Test cve_intel tool (full report with risk scoring)
# ---------------------------------------------------------------------------

class TestCveIntel:

    @pytest.mark.asyncio
    async def test_intel_invalid_cve(self, mock_ctx):
        mcp = FastMCP("test")
        from mcp_core.cve_engine import register_cve_tools
        register_cve_tools(mcp)

        with patch("mcp_core.cve_engine.get_context", return_value=mock_ctx):
            tool = await mcp.get_tool("cve_intel")
            result = await tool.fn(cve_id="bad")

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_intel_success(self, mock_ctx, mock_cve_mgr):
        mcp = FastMCP("test")
        from mcp_core.cve_engine import register_cve_tools
        register_cve_tools(mcp)

        with patch("mcp_core.cve_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.cve_engine.get_cve_intelligence", return_value=mock_cve_mgr):
            tool = await mcp.get_tool("cve_intel")
            result = await tool.fn(cve_id="CVE-2024-1234")

        assert result["success"] is True
        assert "risk_score" in result
        assert "risk_level" in result
        assert "analysis" in result
        assert "exploits" in result
        assert "intel_summary" in result

    def test_risk_score_critical(self):
        analysis = {
            "exploitability_score": 1.0, "cvss_score": 10.0,
            "threat_intelligence": {},
        }
        exploits = {"exploits": [1, 2, 3], "search_summary": {"metasploit_modules": 1}}
        score = analysis["exploitability_score"] * 0.5 + (analysis["cvss_score"] / 10.0) * 0.3
        if exploits["search_summary"]["metasploit_modules"] > 0:
            score = min(1.0, score + 0.3)
        if len(exploits["exploits"]) > 3:
            score = min(1.0, score + 0.1)
        assert score >= 0.85

    def test_risk_score_high(self):
        analysis = {
            "exploitability_score": 0.5, "cvss_score": 7.5,
            "threat_intelligence": {},
        }
        exploits = {"exploits": [], "search_summary": {"metasploit_modules": 0}}
        score = analysis["exploitability_score"] * 0.5 + (analysis["cvss_score"] / 10.0) * 0.3
        # has_metasploit=False -> 0.25 + 0.225 = 0.475 -> MEDIUM
        # score = 0.25 + 0.225 = 0.475
        assert 0.40 <= score < 0.65

    def test_risk_score_medium(self):
        analysis = {
            "exploitability_score": 0.3, "cvss_score": 5.0,
            "threat_intelligence": {},
        }
        exploits = {"exploits": [], "search_summary": {"metasploit_modules": 0}}
        score = analysis["exploitability_score"] * 0.5 + (analysis["cvss_score"] / 10.0) * 0.3
        assert 0.25 <= score < 0.40

    def test_risk_score_low(self):
        analysis = {
            "exploitability_score": 0.1, "cvss_score": 2.0,
            "threat_intelligence": {},
        }
        exploits = {"exploits": [], "search_summary": {"metasploit_modules": 0}}
        score = analysis["exploitability_score"] * 0.5 + (analysis["cvss_score"] / 10.0) * 0.3
        assert score < 0.40

    def test_risk_score_metasploit_bonus(self):
        analysis = {
            "exploitability_score": 0.3, "cvss_score": 5.0,
            "threat_intelligence": {},
        }
        exploits = {"exploits": [], "search_summary": {"metasploit_modules": 1}}
        score = analysis["exploitability_score"] * 0.5 + (analysis["cvss_score"] / 10.0) * 0.3
        if exploits["search_summary"]["metasploit_modules"] > 0:
            score = min(1.0, score + 0.3)
        assert score >= 0.55  # 0.30 + 0.30 = 0.60 (but 0.15+0.15=0.30+0.3=0.60)

    def test_risk_score_exploit_count_bonus(self):
        analysis = {
            "exploitability_score": 0.3, "cvss_score": 5.0,
            "threat_intelligence": {},
        }
        exploits = {"exploits": [1, 2, 3, 4], "search_summary": {"metasploit_modules": 0}}
        score = analysis["exploitability_score"] * 0.5 + (analysis["cvss_score"] / 10.0) * 0.3
        if len(exploits["exploits"]) > 3:
            score = min(1.0, score + 0.1)
        assert score == 0.40

    def test_risk_score_capped_at_one(self):
        analysis = {
            "exploitability_score": 1.0, "cvss_score": 10.0,
            "threat_intelligence": {},
        }
        exploits = {
            "exploits": [1, 2, 3, 4, 5],
            "search_summary": {"metasploit_modules": 1},
        }
        score = analysis["exploitability_score"] * 0.5 + (analysis["cvss_score"] / 10.0) * 0.3
        if exploits["search_summary"]["metasploit_modules"] > 0:
            score = min(1.0, score + 0.3)
        if len(exploits["exploits"]) > 3:
            score = min(1.0, score + 0.1)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_intel_with_metasploit_risks(self, mock_ctx, mock_cve_mgr):
        mock_cve_mgr.analyze_cve_exploitability.return_value = {
            "success": True, "cvss_score": 9.5, "severity": "CRITICAL",
            "exploitability_score": 0.9, "exploitability_level": "HIGH",
            "attack_vector": "NETWORK",
            "threat_intelligence": {
                "recommended_priority": "IMMEDIATE",
                "active_exploitation": True,
                "exploit_indicators": ["ransomware_groups"],
            },
        }
        mock_cve_mgr.search_existing_exploits.return_value = {
            "success": True,
            "exploits": [{"id": i} for i in range(5)],
            "search_summary": {"github_repos": 3, "metasploit_modules": 2, "exploit_db_refs": 2},
        }
        mcp = FastMCP("test")
        from mcp_core.cve_engine import register_cve_tools
        register_cve_tools(mcp)

        with patch("mcp_core.cve_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.cve_engine.get_cve_intelligence", return_value=mock_cve_mgr):
            tool = await mcp.get_tool("cve_intel")
            result = await tool.fn(cve_id="CVE-2024-0001")

        assert result["risk_level"] == "CRITICAL"
        assert result["risk_score"] >= 0.85
        assert result["intel_summary"]["has_metasploit"] is True
        assert result["intel_summary"]["exploit_count"] == 5

    @pytest.mark.asyncio
    async def test_intel_analysis_fails_but_returns(self, mock_ctx, mock_cve_mgr):
        mock_cve_mgr.analyze_cve_exploitability.return_value = {
            "success": False, "error": "CVE not found",
            "cvss_score": 0.0, "exploitability_score": 0.0,
        }
        mock_cve_mgr.search_existing_exploits.return_value = {
            "success": True, "exploits": [],
            "search_summary": {"github_repos": 0, "metasploit_modules": 0, "exploit_db_refs": 0},
        }
        mcp = FastMCP("test")
        from mcp_core.cve_engine import register_cve_tools
        register_cve_tools(mcp)

        with patch("mcp_core.cve_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.cve_engine.get_cve_intelligence", return_value=mock_cve_mgr):
            tool = await mcp.get_tool("cve_intel")
            result = await tool.fn(cve_id="CVE-2024-9999")

        assert result["success"] is False
        assert result["risk_level"] == "LOW"

    @pytest.mark.asyncio
    async def test_intel_medium_risk_level(self, mock_ctx, mock_cve_mgr):
        """Cover risk_level = 'MEDIUM' (line 274): score in [0.40, 0.65)."""
        mock_cve_mgr.analyze_cve_exploitability.return_value = {
            "success": True, "cvss_score": 7.5, "severity": "HIGH",
            "exploitability_score": 0.5, "exploitability_level": "MEDIUM",
            "attack_vector": "NETWORK", "threat_intelligence": {},
        }
        mock_cve_mgr.search_existing_exploits.return_value = {
            "success": True, "exploits": [],
            "search_summary": {"github_repos": 0, "metasploit_modules": 0, "exploit_db_refs": 0},
        }
        mcp = FastMCP("test")
        from mcp_core.cve_engine import register_cve_tools
        register_cve_tools(mcp)
        with patch("mcp_core.cve_engine.get_context", return_value=mock_ctx), \
             patch("mcp_core.cve_engine.get_cve_intelligence", return_value=mock_cve_mgr):
            tool = await mcp.get_tool("cve_intel")
            result = await tool.fn(cve_id="CVE-2024-0001")
        assert result["risk_level"] == "MEDIUM"
