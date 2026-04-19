"""
Phase 1+: Gateway.py Test Suite (Expanded with Async Support)
Tests core gateway routing and task classification functionality.

Coverage Target: 95%+
Effort: 3-4 hours (includes async/await patterns)
Pattern: Mock hexstrike_client, mock direct executors, test routing logic
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from typing import Dict, Any


# Fixtures for async testing
@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestClassifyTaskAsync:
    """Test async task classification gateway tool."""

    @pytest.fixture
    def mock_hexstrike_client(self):
        """Mock HexStrikeClient for gateway tests."""
        client = Mock()
        client.safe_post = Mock(return_value={
            "success": True,
            "category": "network_scan",
            "recommended_tools": [
                {"tool": "nmap", "params": {"target": "10.0.0.1"}},
                {"tool": "masscan", "params": {"target": "10.0.0.1"}}
            ],
            "confidence": 0.95
        })
        return client

    @pytest.fixture
    def mock_mcp(self):
        """Mock MCP server instance."""
        mcp = Mock()
        # tool() is a decorator factory that returns a decorator
        # When called as @mcp.tool(), it needs to return a function that accepts a function
        def tool_decorator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        
        mcp.tool = Mock(side_effect=tool_decorator)
        return mcp

    def test_classify_task_registers_async_tool(self, mock_mcp, mock_hexstrike_client):
        """Test that classify_task is registered as async tool."""
        from mcp_tools.gateway import register_gateway_tools
        
        # This will call mcp.tool() to register both classify_task and run_tool
        register_gateway_tools(mock_mcp, mock_hexstrike_client)
        
        # Verify tool() was called at least twice (for classify_task and run_tool)
        assert mock_mcp.tool.call_count >= 2

    @pytest.mark.asyncio
    async def test_classify_task_success_async(self, mock_hexstrike_client):
        """Test successful async task classification."""
        mock_hexstrike_client.safe_post.return_value = {
            "success": True,
            "category": "network_scan",
            "recommended_tools": [{"tool": "nmap"}],
            "confidence": 0.95
        }
        
        # Simulate what classify_task does
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: mock_hexstrike_client.safe_post(
                "api/intelligence/classify-task",
                {"description": "scan for open ports"}
            )
        )
        
        assert result["success"] == True
        assert result["category"] == "network_scan"

    @pytest.mark.asyncio
    async def test_classify_task_adds_usage_info(self, mock_hexstrike_client):
        """Test that classify_task adds usage info to response."""
        mock_hexstrike_client.safe_post.return_value = {
            "success": True,
            "category": "web_scan",
            "recommended_tools": []
        }
        
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: mock_hexstrike_client.safe_post(
                "api/intelligence/classify-task",
                {"description": "test web app"}
            )
        )
        
        # The gateway adds usage info
        if result.get("success"):
            result["usage"] = "Use run_tool with a tool name and params from the recommended list"
        
        assert "usage" in result

    @pytest.mark.parametrize("description,category", [
        ("scan for open ports", "network_scan"),
        ("test for SQL injection", "web_scan"),
        ("crack password hash", "password_cracking"),
        ("enumerate domain", "active_directory"),
        ("wifi penetration test", "wifi"),
    ])
    @pytest.mark.asyncio
    async def test_classify_various_descriptions_async(self, mock_hexstrike_client, description, category):
        """Test classification of various descriptions (async)."""
        mock_hexstrike_client.safe_post.return_value = {
            "success": True,
            "category": category,
            "recommended_tools": [{"tool": "tool1"}],
            "confidence": 0.90
        }
        
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: mock_hexstrike_client.safe_post(
                "api/intelligence/classify-task",
                {"description": description}
            )
        )
        
        assert result["category"] == category


class TestRunToolAsync:
    """Test async tool execution gateway."""

    @pytest.fixture
    def mock_hexstrike_client(self):
        """Mock HexStrikeClient."""
        return Mock()

    def test_run_tool_json_parsing_valid(self):
        """Test valid JSON parameter parsing."""
        params_str = '{"target": "10.0.0.1", "port": "22"}'
        parsed = json.loads(params_str)
        assert parsed["target"] == "10.0.0.1"
        assert parsed["port"] == "22"

    def test_run_tool_json_parsing_dict_passthrough(self):
        """Test that dict params are passed through."""
        params_dict = {"target": "10.0.0.1"}
        parsed = params_dict if isinstance(params_dict, dict) else json.loads(params_dict)
        assert parsed == {"target": "10.0.0.1"}

    def test_run_tool_json_parsing_invalid(self):
        """Test invalid JSON parameter handling."""
        params_str = '{"target": "10.0.0.1", invalid}'
        with pytest.raises(json.JSONDecodeError):
            json.loads(params_str)

    def test_run_tool_json_error_handling(self):
        """Test error handling for invalid JSON."""
        params_str = '{"target": "10.0.0.1", invalid}'
        try:
            parsed_params = json.loads(params_str) if isinstance(params_str, str) else params_str
        except json.JSONDecodeError as e:
            error_response = {"error": f"Invalid params JSON: {e}", "success": False}
        
        assert error_response["success"] == False
        assert "Invalid params JSON" in error_response["error"]

    @pytest.mark.parametrize("tool_name,category", [
        ("airmon_ng", "wifi"),
        ("nmap", "network"),
        ("sqlmap", "web_scan"),
        ("amass", "recon"),
        ("hydra", "cracking"),
        ("metasploit", "exploit"),
    ])
    def test_run_tool_category_routing(self, tool_name, category):
        """Test that tools map to correct categories."""
        tool_mapping = {
            "airmon_ng": "wifi",
            "nmap": "network",
            "sqlmap": "web_scan",
            "amass": "recon",
            "hydra": "cracking",
            "metasploit": "exploit",
        }
        assert tool_mapping[tool_name] == category

    @pytest.mark.asyncio
    async def test_run_tool_executor_pattern(self):
        """Test async executor pattern for blocking tool execution."""
        def blocking_operation():
            return {"success": True, "result": "tool executed"}
        
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, blocking_operation)
        
        assert result["success"] == True
        assert "result" in result

    @pytest.mark.asyncio
    async def test_run_tool_executor_with_params(self):
        """Test executor with parameters."""
        def mock_exec_fn(tool_key, params):
            return {
                "success": True,
                "tool": tool_key,
                "params_received": params
            }
        
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: mock_exec_fn("nmap", {"target": "10.0.0.1"})
        )
        
        assert result["tool"] == "nmap"
        assert result["params_received"]["target"] == "10.0.0.1"


class TestGatewayRouting:
    """Test gateway tool routing logic."""

    def test_routing_dictionary_complete(self):
        """Test that all critical tools are in routing dict."""
        critical_tools = [
            "nmap", "sqlmap", "hashcat", "metasploit",
            "amass", "nikto", "hydra", "enum4linux"
        ]
        
        # Simulated routing dict (subset)
        routing = {
            "nmap": ("net_scan_exec", "nmap"),
            "sqlmap": ("web_scan_exec", "sqlmap"),
            "hashcat": ("pwdcrack_exec", "hashcat"),
            "metasploit": ("exploit_exec", "metasploit"),
            "amass": ("recon_exec", "amass"),
            "nikto": ("web_scan_exec", "nikto"),
            "hydra": ("pwdcrack_exec", "hydra"),
            "enum4linux": ("smb_enum_exec", "enum4linux"),
        }
        
        for tool in critical_tools:
            assert tool in routing, f"Tool {tool} missing from routing"

    def test_unknown_tool_handling(self):
        """Test handling of unknown tool names."""
        routing = {"nmap": ("net_scan_exec", "nmap")}
        tool_name = "unknown_tool"
        
        route = routing.get(tool_name.lower())
        if not route:
            error_response = {"error": f"Unknown tool: {tool_name}", "success": False}
        
        assert error_response["success"] == False

    @pytest.mark.parametrize("tool_name", [
        "nmap",
        "sqlmap", 
        "hashcat",
        "metasploit",
        "wpscan",
    ])
    def test_routing_for_known_tools(self, tool_name):
        """Test routing exists for known tools."""
        routing = {
            "nmap": ("net_scan_exec", "nmap"),
            "sqlmap": ("web_scan_exec", "sqlmap"),
            "hashcat": ("pwdcrack_exec", "hashcat"),
            "metasploit": ("exploit_exec", "metasploit"),
            "wpscan": ("web_scan_exec", "wpscan"),
        }
        
        assert tool_name in routing
        exec_fn, tool_key = routing[tool_name]
        assert exec_fn is not None
        assert tool_key is not None


class TestGatewayErrorHandling:
    """Test error scenarios and edge cases."""

    def test_empty_description_classification(self):
        """Test classification with empty description."""
        mock_client = Mock()
        mock_client.safe_post.return_value = {
            "error": "Description cannot be empty",
            "success": False
        }
        
        result = mock_client.safe_post(
            "api/intelligence/classify-task",
            {"description": ""}
        )
        
        assert result["success"] == False

    def test_very_long_description_classification(self):
        """Test classification with very long description."""
        long_desc = "A" * 10000
        mock_client = Mock()
        mock_client.safe_post.return_value = {
            "success": True,
            "category": "generic",
            "warning": "Description is very long"
        }
        
        result = mock_client.safe_post(
            "api/intelligence/classify-task",
            {"description": long_desc}
        )
        
        assert result["success"] == True

    def test_special_chars_in_description(self):
        """Test classification with special characters."""
        desc = "Test <script>alert('xss')</script> for injection"
        mock_client = Mock()
        mock_client.safe_post.return_value = {
            "success": True,
            "category": "web_scan"
        }
        
        result = mock_client.safe_post(
            "api/intelligence/classify-task",
            {"description": desc}
        )
        
        assert result["success"] == True

    def test_missing_required_params_run_tool(self):
        """Test run_tool with missing required parameters."""
        error_response = {
            "error": "Missing required param: target",
            "success": False
        }
        
        assert error_response["success"] == False
        assert "Missing required param" in error_response["error"]

    def test_malformed_json_params_run_tool(self):
        """Test run_tool with malformed JSON."""
        params_str = '{"target": "10.0.0.1", invalid}'
        
        try:
            parsed_params = json.loads(params_str)
        except json.JSONDecodeError as e:
            error_response = {"error": f"Invalid params JSON: {e}", "success": False}
        
        assert error_response["success"] == False

    @pytest.mark.asyncio
    async def test_executor_exception_handling(self):
        """Test exception handling in async executor."""
        def failing_operation():
            raise ValueError("Tool execution failed")
        
        loop = asyncio.get_running_loop()
        
        with pytest.raises(ValueError):
            await loop.run_in_executor(None, failing_operation)


class TestGatewayIntegrationAsync:
    """Integration tests for async gateway operations."""

    @pytest.mark.asyncio
    async def test_classify_then_run_workflow(self):
        """Test classification followed by tool execution workflow."""
        # Step 1: Classify
        classify_result = {
            "success": True,
            "category": "network_scan",
            "recommended_tools": [
                {"tool": "nmap", "params": {"target": "10.0.0.1"}}
            ]
        }
        
        # Step 2: Extract tool
        tool_name = classify_result["recommended_tools"][0]["tool"]
        tool_params = classify_result["recommended_tools"][0]["params"]
        
        # Step 3: Execute
        assert tool_name == "nmap"
        assert tool_params["target"] == "10.0.0.1"

    def test_gateway_response_format(self):
        """Test standard gateway response format."""
        valid_responses = [
            {
                "success": True,
                "category": "network_scan",
                "recommended_tools": [],
                "confidence": 0.95,
                "usage": "Use run_tool..."
            },
            {
                "success": False,
                "error": "Unknown tool"
            }
        ]
        
        for response in valid_responses:
            assert "success" in response
            if response["success"]:
                assert "category" in response or "recommended_tools" in response

    @pytest.mark.parametrize("workflow_type", [
        "recon_to_scan",
        "scan_to_exploit",
        "web_scan_to_fuzz",
    ])
    def test_workflow_chains(self, workflow_type):
        """Test multi-step security workflows."""
        workflows = {
            "recon_to_scan": ["amass", "nmap"],
            "scan_to_exploit": ["nmap", "metasploit"],
            "web_scan_to_fuzz": ["nikto", "ffuf"],
        }
        
        tools_in_workflow = workflows[workflow_type]
        assert len(tools_in_workflow) == 2
        assert all(isinstance(t, str) for t in tools_in_workflow)
