"""
Test suite for bug bounty reconnaissance workflow module.
Tests: mcp_tools/bugbounty_workflow/bug_bounty_recon.py
Coverage target: 50%+
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from mcp_tools.bugbounty_workflow.bug_bounty_recon import register_bug_bounty_recon_tools


class TestBugBountyRecon:
    """Test bug bounty reconnaissance workflows."""

    @pytest.fixture
    def mock_mcp(self):
        """Create mock MCP instance."""
        mcp = Mock()
        mcp._tools = {}
        
        def tool_decorator():
            def decorator(fn):
                mcp._tools[fn.__name__] = fn
                return fn
            return decorator
        
        mcp.tool = tool_decorator
        return mcp

    @pytest.fixture
    def mock_hexstrike_client(self):
        """Create mock hexstrike client."""
        return Mock()

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return Mock()

    @pytest.fixture
    def mock_context(self):
        """Create mock FastMCP context."""
        ctx = Mock()
        ctx.info = AsyncMock()
        ctx.error = AsyncMock()
        ctx.report_progress = AsyncMock()
        return ctx

    @pytest.fixture
    def setup_tools(self, mock_mcp, mock_hexstrike_client, mock_logger):
        """Register tools and return them for testing."""
        register_bug_bounty_recon_tools(mock_mcp, mock_hexstrike_client, mock_logger)
        return mock_mcp._tools, mock_hexstrike_client, mock_logger

    @pytest.mark.asyncio
    async def test_reconnaissance_workflow_web(self, setup_tools, mock_context):
        """Test reconnaissance workflow for web program."""
        tools, mock_client, mock_logger = setup_tools
        workflow_fn = tools['bugbounty_reconnaissance_workflow']

        mock_client.safe_post.return_value = {
            "success": True,
            "workflow": {
                "domain": "example.com",
                "program_type": "web",
                "tools_count": 25,
                "estimated_time": 3600,
                "phases": [
                    {"name": "Subdomain Enumeration", "tools": 5},
                    {"name": "Port Scanning", "tools": 3},
                    {"name": "Web Enumeration", "tools": 8},
                ]
            }
        }

        result = await workflow_fn(mock_context, "example.com", "*.example.com", "test.example.com", "web")

        assert result["success"] is True
        assert result["workflow"]["tools_count"] == 25
        mock_context.report_progress.assert_called()

    @pytest.mark.asyncio
    async def test_reconnaissance_workflow_api(self, setup_tools, mock_context):
        """Test reconnaissance workflow for API program."""
        tools, mock_client, mock_logger = setup_tools
        workflow_fn = tools['bugbounty_reconnaissance_workflow']

        mock_client.safe_post.return_value = {
            "success": True,
            "workflow": {
                "domain": "api.example.com",
                "program_type": "api",
                "tools_count": 18,
                "estimated_time": 2400,
                "phases": [
                    {"name": "API Discovery", "tools": 4},
                    {"name": "Endpoint Enumeration", "tools": 6},
                    {"name": "Authentication Testing", "tools": 8},
                ]
            }
        }

        result = await workflow_fn(mock_context, "api.example.com", "", "", "api")

        assert result["success"] is True
        assert result["workflow"]["program_type"] == "api"
        call_args = mock_client.safe_post.call_args
        assert call_args[0][1]["program_type"] == "api"

    @pytest.mark.asyncio
    async def test_reconnaissance_workflow_mobile(self, setup_tools, mock_context):
        """Test reconnaissance workflow for mobile program."""
        tools, mock_client, mock_logger = setup_tools
        workflow_fn = tools['bugbounty_reconnaissance_workflow']

        mock_client.safe_post.return_value = {
            "success": True,
            "workflow": {
                "program_type": "mobile",
                "tools_count": 22,
                "estimated_time": 4200
            }
        }

        result = await workflow_fn(mock_context, "example.com", "", "", "mobile")

        assert result["success"] is True
        assert result["workflow"]["program_type"] == "mobile"

    @pytest.mark.asyncio
    async def test_reconnaissance_workflow_with_scope(self, setup_tools, mock_context):
        """Test reconnaissance workflow with scope specification."""
        tools, mock_client, mock_logger = setup_tools
        workflow_fn = tools['bugbounty_reconnaissance_workflow']

        mock_client.safe_post.return_value = {
            "success": True,
            "workflow": {
                "domain": "example.com",
                "scope": ["*.example.com", "api.example.com"],
                "out_of_scope": ["test.example.com", "staging.example.com"],
                "tools_count": 20
            }
        }

        result = await workflow_fn(
            mock_context, 
            "example.com", 
            "*.example.com, api.example.com",
            "test.example.com, staging.example.com"
        )

        assert result["success"] is True
        call_args = mock_client.safe_post.call_args
        assert len(call_args[0][1]["scope"]) == 2
        assert len(call_args[0][1]["out_of_scope"]) == 2

    @pytest.mark.asyncio
    async def test_reconnaissance_workflow_failure(self, setup_tools, mock_context):
        """Test reconnaissance workflow failure handling."""
        tools, mock_client, mock_logger = setup_tools
        workflow_fn = tools['bugbounty_reconnaissance_workflow']

        mock_client.safe_post.return_value = {
            "success": False,
            "error": "Domain not supported"
        }

        result = await workflow_fn(mock_context, "invalid.domain", "", "", "web")

        assert result["success"] is False
        mock_context.error.assert_called()

    @pytest.mark.asyncio
    async def test_vulnerability_hunting_basic(self, setup_tools, mock_context):
        """Test vulnerability hunting workflow with default priorities."""
        tools, mock_client, mock_logger = setup_tools
        hunting_fn = tools['bugbounty_vulnerability_hunting']

        mock_client.safe_post.return_value = {
            "success": True,
            "workflow": {
                "domain": "example.com",
                "priority_vulns": ["rce", "sqli", "xss", "idor", "ssrf"],
                "bounty_range": "unknown",
                "vulnerabilities": [
                    {"type": "XSS", "severity": "HIGH"},
                    {"type": "SQLI", "severity": "CRITICAL"},
                ]
            }
        }

        result = await hunting_fn(mock_context, "example.com")

        assert result["success"] is True
        assert len(result["workflow"]["vulnerabilities"]) == 2

    @pytest.mark.asyncio
    async def test_vulnerability_hunting_custom_priorities(self, setup_tools, mock_context):
        """Test vulnerability hunting with custom priority vulnerabilities."""
        tools, mock_client, mock_logger = setup_tools
        hunting_fn = tools['bugbounty_vulnerability_hunting']

        mock_client.safe_post.return_value = {
            "success": True,
            "workflow": {
                "priority_vulns": ["rce", "authentication", "logic-flaw"],
                "bounty_range": "high"
            }
        }

        result = await hunting_fn(
            mock_context,
            "example.com",
            "rce,authentication,logic-flaw",
            "high"
        )

        assert result["success"] is True
        call_args = mock_client.safe_post.call_args
        assert len(call_args[0][1]["priority_vulns"]) == 3
        assert call_args[0][1]["bounty_range"] == "high"

    @pytest.mark.asyncio
    async def test_vulnerability_hunting_bounty_ranges(self, setup_tools, mock_context):
        """Test vulnerability hunting with different bounty ranges."""
        tools, mock_client, mock_logger = setup_tools
        hunting_fn = tools['bugbounty_vulnerability_hunting']

        bounty_ranges = ["low", "medium", "high", "critical"]
        mock_client.safe_post.return_value = {
            "success": True,
            "workflow": {"vulnerabilities": []}
        }

        for bounty_range in bounty_ranges:
            result = await hunting_fn(mock_context, "example.com", "", bounty_range)
            assert result["success"] is True
            call_args = mock_client.safe_post.call_args
            assert call_args[0][1]["bounty_range"] == bounty_range

    @pytest.mark.asyncio
    async def test_vulnerability_hunting_failure(self, setup_tools, mock_context):
        """Test vulnerability hunting workflow failure."""
        tools, mock_client, mock_logger = setup_tools
        hunting_fn = tools['bugbounty_vulnerability_hunting']

        mock_client.safe_post.return_value = {
            "success": False,
            "error": "Analysis timed out"
        }

        result = await hunting_fn(mock_context, "example.com")

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_reconnaissance_empty_scope(self, setup_tools, mock_context):
        """Test reconnaissance with empty scope parameters."""
        tools, mock_client, mock_logger = setup_tools
        workflow_fn = tools['bugbounty_reconnaissance_workflow']

        mock_client.safe_post.return_value = {
            "success": True,
            "workflow": {"tools_count": 15}
        }

        result = await workflow_fn(mock_context, "example.com", "", "")

        assert result["success"] is True
        call_args = mock_client.safe_post.call_args
        assert call_args[0][1]["scope"] == []
        assert call_args[0][1]["out_of_scope"] == []

    @pytest.mark.asyncio
    async def test_vulnerability_hunting_empty_keywords(self, setup_tools, mock_context):
        """Test vulnerability hunting with empty keyword filter."""
        tools, mock_client, mock_logger = setup_tools
        hunting_fn = tools['bugbounty_vulnerability_hunting']

        mock_client.safe_post.return_value = {
            "success": True,
            "workflow": {"vulnerabilities": []}
        }

        result = await hunting_fn(mock_context, "example.com", "", "")

        assert result["success"] is True
        call_args = mock_client.safe_post.call_args
        assert call_args[0][1]["bounty_range"] == ""
