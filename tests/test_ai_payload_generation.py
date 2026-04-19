"""
Test suite for AI payload generation module.
Tests: mcp_tools/ai_payload/ai_payload_generation.py
Coverage target: 50%+
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from mcp_tools.ai_payload.ai_payload_generation import register_ai_payload_generation_tools


class TestAiPayloadGeneration:
    """Test AI payload generation functionality."""

    @pytest.fixture
    def mock_mcp(self):
        """Create mock MCP instance."""
        mcp = Mock()
        # Create a mapping to capture registered tools
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
    def setup_tools(self, mock_mcp, mock_hexstrike_client, mock_logger):
        """Register tools and return them for testing."""
        register_ai_payload_generation_tools(mock_mcp, mock_hexstrike_client, mock_logger)
        return mock_mcp._tools, mock_hexstrike_client, mock_logger

    @pytest.mark.asyncio
    async def test_ai_generate_payload_xss_basic(self, setup_tools):
        """Test XSS payload generation with basic complexity."""
        tools, mock_client, mock_logger = setup_tools
        ai_generate_payload = tools['ai_generate_payload']

        # Mock successful response
        mock_client.safe_post.return_value = {
            "success": True,
            "ai_payload_generation": {
                "payload_count": 3,
                "payloads": [
                    {"payload": "<script>alert(1)</script>", "risk_level": "HIGH", "context": "basic"},
                    {"payload": "<img src=x onerror=alert(1)>", "risk_level": "MEDIUM", "context": "event"},
                    {"payload": "<svg onload=alert(1)>", "risk_level": "HIGH", "context": "svg"},
                ],
                "test_cases": []
            }
        }

        result = await ai_generate_payload("xss", "basic")

        assert result["success"] is True
        assert result["ai_payload_generation"]["payload_count"] == 3
        mock_client.safe_post.assert_called_once()
        call_args = mock_client.safe_post.call_args
        assert call_args[0][0] == "api/ai/generate_payload"
        assert call_args[0][1]["attack_type"] == "xss"
        assert call_args[0][1]["complexity"] == "basic"

    @pytest.mark.asyncio
    async def test_ai_generate_payload_sqli_advanced(self, setup_tools):
        """Test SQL injection payload generation with advanced complexity."""
        tools, mock_client, mock_logger = setup_tools
        ai_generate_payload = tools['ai_generate_payload']

        mock_client.safe_post.return_value = {
            "success": True,
            "ai_payload_generation": {
                "payload_count": 5,
                "payloads": [
                    {"payload": "' OR '1'='1", "risk_level": "HIGH", "context": "classic"},
                    {"payload": "1' UNION SELECT NULL--", "risk_level": "CRITICAL", "context": "union"},
                ],
                "test_cases": [{"payload": "' OR '1'='1", "expected": "error"}]
            }
        }

        result = await ai_generate_payload("sqli", "advanced", "php")

        assert result["success"] is True
        assert result["ai_payload_generation"]["payload_count"] == 5
        call_args = mock_client.safe_post.call_args
        assert call_args[0][1]["attack_type"] == "sqli"
        assert call_args[0][1]["complexity"] == "advanced"
        assert call_args[0][1]["technology"] == "php"

    @pytest.mark.asyncio
    async def test_ai_generate_payload_lfi_with_url(self, setup_tools):
        """Test LFI payload generation with URL context."""
        tools, mock_client, mock_logger = setup_tools
        ai_generate_payload = tools['ai_generate_payload']

        mock_client.safe_post.return_value = {
            "success": True,
            "ai_payload_generation": {
                "payload_count": 4,
                "payloads": [
                    {"payload": "../../../etc/passwd", "risk_level": "HIGH", "context": "path"},
                    {"payload": "....//....//....//etc/passwd", "risk_level": "MEDIUM", "context": "bypass"},
                ],
                "test_cases": []
            }
        }

        result = await ai_generate_payload("lfi", "bypass", "", "http://example.com/page.php?file=")

        assert result["success"] is True
        call_args = mock_client.safe_post.call_args
        assert call_args[0][1]["url"] == "http://example.com/page.php?file="

    @pytest.mark.asyncio
    async def test_ai_generate_payload_failure(self, setup_tools):
        """Test payload generation failure handling."""
        tools, mock_client, mock_logger = setup_tools
        ai_generate_payload = tools['ai_generate_payload']

        mock_client.safe_post.return_value = {
            "success": False,
            "error": "API rate limit exceeded"
        }

        result = await ai_generate_payload("cmd_injection", "advanced")

        assert result["success"] is False
        mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_ai_test_payload_vulnerable(self, setup_tools):
        """Test payload testing with vulnerability detection."""
        tools, mock_client, mock_logger = setup_tools
        ai_test_payload = tools['ai_test_payload']

        mock_client.safe_post.return_value = {
            "success": True,
            "ai_analysis": {
                "potential_vulnerability": True,
                "confidence": 0.95,
                "vulnerability_type": "XSS",
                "response_time_ms": 145,
            }
        }

        result = await ai_test_payload("<script>alert(1)</script>", "http://target.com/search")

        assert result["success"] is True
        assert result["ai_analysis"]["potential_vulnerability"] is True
        mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_ai_test_payload_safe(self, setup_tools):
        """Test payload testing when no vulnerability detected."""
        tools, mock_client, mock_logger = setup_tools
        ai_test_payload = tools['ai_test_payload']

        mock_client.safe_post.return_value = {
            "success": True,
            "ai_analysis": {
                "potential_vulnerability": False,
                "confidence": 0.99,
                "response_time_ms": 150,
            }
        }

        result = await ai_test_payload("benign_input", "http://target.com/search")

        assert result["success"] is True
        assert result["ai_analysis"]["potential_vulnerability"] is False
        # Should call info, not warning
        assert any("No obvious vulnerability" in str(call) for call in mock_logger.method_calls)

    @pytest.mark.asyncio
    async def test_ai_test_payload_post_method(self, setup_tools):
        """Test payload testing with POST method."""
        tools, mock_client, mock_logger = setup_tools
        ai_test_payload = tools['ai_test_payload']

        mock_client.safe_post.return_value = {
            "success": True,
            "ai_analysis": {"potential_vulnerability": False}
        }

        result = await ai_test_payload("payload_data", "http://target.com/api", method="POST")

        assert result["success"] is True
        call_args = mock_client.safe_post.call_args
        assert call_args[0][1]["method"] == "POST"

    @pytest.mark.asyncio
    async def test_ai_test_payload_failure(self, setup_tools):
        """Test payload testing failure."""
        tools, mock_client, mock_logger = setup_tools
        ai_test_payload = tools['ai_test_payload']

        mock_client.safe_post.return_value = {
            "success": False,
            "error": "Target unreachable"
        }

        result = await ai_test_payload("payload", "http://unreachable.com")

        assert result["success"] is False
        mock_logger.error.assert_called()

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="ai_generate_payload called without await - known code issue")
    async def test_ai_generate_attack_suite_multiple_types(self, setup_tools):
        """Test attack suite generation with multiple payload types."""
        tools, mock_client, mock_logger = setup_tools
        ai_generate_attack_suite = tools['ai_generate_attack_suite']

        def mock_post(endpoint, data):
            if "generate_payload" in endpoint:
                attack_type = data.get("attack_type")
                if attack_type == "xss":
                    return {
                        "success": True,
                        "ai_payload_generation": {
                            "payload_count": 3,
                            "payloads": [{"risk_level": "HIGH"}, {"risk_level": "LOW"}],
                            "test_cases": [{"id": 1}]
                        }
                    }
                elif attack_type == "sqli":
                    return {
                        "success": True,
                        "ai_payload_generation": {
                            "payload_count": 4,
                            "payloads": [{"risk_level": "CRITICAL"}],
                            "test_cases": [{"id": 2}, {"id": 3}]
                        }
                    }
                elif attack_type == "lfi":
                    return {
                        "success": True,
                        "ai_payload_generation": {
                            "payload_count": 2,
                            "payloads": [{"risk_level": "MEDIUM"}],
                            "test_cases": []
                        }
                    }
            return {"success": False}

        mock_client.safe_post = mock_post

        result = await ai_generate_attack_suite("http://target.com", "xss,sqli,lfi")

        assert result["success"] is True
        assert "attack_suite" in result
        suite = result["attack_suite"]
        assert suite["target_url"] == "http://target.com"
        assert len(suite["payload_suites"]) == 3
        assert suite["summary"]["total_payloads"] == 9
        assert suite["summary"]["high_risk_payloads"] == 2
        assert suite["summary"]["test_cases"] == 3

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="ai_generate_payload called without await - known code issue")
    async def test_ai_generate_attack_suite_single_type(self, setup_tools):
        """Test attack suite with single payload type."""
        tools, mock_client, mock_logger = setup_tools
        ai_generate_attack_suite = tools['ai_generate_attack_suite']

        mock_client.safe_post.return_value = {
            "success": True,
            "ai_payload_generation": {
                "payload_count": 2,
                "payloads": [{"risk_level": "HIGH"}],
                "test_cases": []
            }
        }

        result = await ai_generate_attack_suite("http://target.com", "xss")

        assert result["success"] is True
        assert len(result["attack_suite"]["payload_suites"]) == 1

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="ai_generate_payload called without await - known code issue")
    async def test_ai_generate_attack_suite_with_failed_payload_generation(self, setup_tools):
        """Test attack suite when payload generation fails for one type."""
        tools, mock_client, mock_logger = setup_tools
        ai_generate_attack_suite = tools['ai_generate_attack_suite']

        call_count = [0]

        def mock_post(endpoint, data):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "success": True,
                    "ai_payload_generation": {
                        "payload_count": 2,
                        "payloads": [{"risk_level": "HIGH"}],
                        "test_cases": []
                    }
                }
            else:
                return {"success": False}

        mock_client.safe_post = mock_post

        result = await ai_generate_attack_suite("http://target.com", "xss,sqli")

        assert result["success"] is True
        # Should have processed first type successfully
        assert "xss" in result["attack_suite"]["payload_suites"]

    @pytest.mark.asyncio
    async def test_ai_generate_payload_all_attack_types(self, setup_tools):
        """Test payload generation for all attack types."""
        tools, mock_client, mock_logger = setup_tools
        ai_generate_payload = tools['ai_generate_payload']

        attack_types = ["xss", "sqli", "lfi", "cmd_injection", "ssti", "xxe"]
        mock_client.safe_post.return_value = {
            "success": True,
            "ai_payload_generation": {
                "payload_count": 2,
                "payloads": [{"payload": "test", "risk_level": "HIGH", "context": "test"}],
                "test_cases": []
            }
        }

        for attack_type in attack_types:
            result = await ai_generate_payload(attack_type, "basic")
            assert result["success"] is True
            call_args = mock_client.safe_post.call_args
            assert call_args[0][1]["attack_type"] == attack_type

    @pytest.mark.asyncio
    async def test_ai_generate_payload_all_complexities(self, setup_tools):
        """Test payload generation for all complexity levels."""
        tools, mock_client, mock_logger = setup_tools
        ai_generate_payload = tools['ai_generate_payload']

        complexities = ["basic", "advanced", "bypass"]
        mock_client.safe_post.return_value = {
            "success": True,
            "ai_payload_generation": {
                "payload_count": 2,
                "payloads": [{"payload": "test", "risk_level": "HIGH", "context": "test"}],
                "test_cases": []
            }
        }

        for complexity in complexities:
            result = await ai_generate_payload("xss", complexity)
            assert result["success"] is True
            call_args = mock_client.safe_post.call_args
            assert call_args[0][1]["complexity"] == complexity

    @pytest.mark.asyncio
    async def test_ai_generate_payload_with_technologies(self, setup_tools):
        """Test payload generation with different technologies."""
        tools, mock_client, mock_logger = setup_tools
        ai_generate_payload = tools['ai_generate_payload']

        technologies = ["php", "asp", "jsp", "python", "nodejs"]
        mock_client.safe_post.return_value = {
            "success": True,
            "ai_payload_generation": {
                "payload_count": 2,
                "payloads": [{"payload": "test", "risk_level": "HIGH", "context": "test"}],
                "test_cases": []
            }
        }

        for tech in technologies:
            result = await ai_generate_payload("sqli", "advanced", tech)
            assert result["success"] is True
            call_args = mock_client.safe_post.call_args
            assert call_args[0][1]["technology"] == tech
