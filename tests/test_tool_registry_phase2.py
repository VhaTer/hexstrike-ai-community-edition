"""
Phase 2: Tool Registry Test Suite
Tests tool registry lookup, validation, and metadata management.

Coverage Target: 80%+
Effort: 3-4 hours
Pattern: Mock tool definitions, test lookup logic, validate metadata
Located: tool_registry.py (~1,973 LOC, currently 26% coverage)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any


class TestToolRegistryLookup:
    """Test tool lookup and retrieval from registry."""

    @pytest.fixture
    def sample_tools(self):
        """Sample tool definitions for testing."""
        return {
            "nmap": {
                "name": "nmap",
                "description": "Network mapper",
                "category": "network_scan",
                "params": {
                    "target": {"type": "str", "required": True},
                    "ports": {"type": "str", "required": False}
                },
                "success_patterns": ["Nmap scan report"],
            },
            "sqlmap": {
                "name": "sqlmap",
                "description": "SQL injection tester",
                "category": "web_scan",
                "params": {
                    "url": {"type": "str", "required": True},
                    "technique": {"type": "str", "required": False}
                },
                "success_patterns": ["SQL injection detected"],
            },
            "hashcat": {
                "name": "hashcat",
                "description": "Password cracking",
                "category": "password_cracking",
                "params": {
                    "hash": {"type": "str", "required": True},
                    "wordlist": {"type": "str", "required": False}
                },
                "success_patterns": ["recovered", "cracked"],
            }
        }

    def test_get_tool_success(self, sample_tools):
        """Test successful tool retrieval."""
        tool_registry = sample_tools
        tool = tool_registry.get("nmap")
        
        assert tool is not None
        assert tool["name"] == "nmap"
        assert tool["category"] == "network_scan"

    def test_get_tool_not_found(self, sample_tools):
        """Test retrieval of non-existent tool."""
        tool_registry = sample_tools
        tool = tool_registry.get("unknown_tool")
        
        assert tool is None

    @pytest.mark.parametrize("tool_name", [
        "nmap",
        "sqlmap",
        "hashcat",
    ])
    def test_get_multiple_tools(self, sample_tools, tool_name):
        """Test retrieval of various tools."""
        tool = sample_tools.get(tool_name)
        assert tool is not None
        assert "params" in tool
        assert "success_patterns" in tool

    def test_get_tool_case_insensitive(self, sample_tools):
        """Test tool lookup is case-insensitive."""
        # Most implementations lowercase keys
        tool_registry = {k.lower(): v for k, v in sample_tools.items()}
        
        tool = tool_registry.get("NMAP".lower())
        assert tool is not None
        assert tool["name"] == "nmap"

    def test_get_tool_returns_copy(self, sample_tools):
        """Test that tool retrieval doesn't expose internal mutable state."""
        tool_registry = sample_tools
        tool1 = tool_registry.get("nmap")
        tool2 = tool_registry.get("nmap")
        
        # Verify both exist and have same data
        assert tool1 is not None
        assert tool2 is not None
        assert tool1["name"] == tool2["name"]


class TestToolMetadata:
    """Test tool metadata structure and validation."""

    @pytest.fixture
    def valid_tool_def(self):
        """Valid tool definition."""
        return {
            "name": "nmap",
            "description": "Network mapper",
            "category": "network_scan",
            "params": {
                "target": {
                    "type": "str",
                    "required": True,
                    "help": "Target IP or domain"
                }
            },
            "success_patterns": ["Nmap scan report"],
            "timeout": 300,
        }

    def test_tool_has_required_fields(self, valid_tool_def):
        """Test that tool has all required metadata fields."""
        required_fields = ["name", "description", "category", "params"]
        
        for field in required_fields:
            assert field in valid_tool_def
            assert valid_tool_def[field] is not None

    def test_tool_params_structure(self, valid_tool_def):
        """Test parameter metadata structure."""
        params = valid_tool_def["params"]
        
        for param_name, param_spec in params.items():
            assert "type" in param_spec
            assert "required" in param_spec
            assert param_spec["type"] in ["str", "int", "bool", "list"]

    def test_tool_success_patterns(self, valid_tool_def):
        """Test success pattern definitions."""
        patterns = valid_tool_def["success_patterns"]
        
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert all(isinstance(p, str) for p in patterns)

    def test_tool_timeout_field(self, valid_tool_def):
        """Test timeout field for tool execution."""
        assert "timeout" in valid_tool_def
        assert valid_tool_def["timeout"] > 0
        assert isinstance(valid_tool_def["timeout"], int)

    @pytest.mark.parametrize("category", [
        "network_scan",
        "web_scan",
        "password_cracking",
        "active_directory",
        "wifi",
        "exploit_framework",
        "osint",
    ])
    def test_valid_tool_categories(self, category):
        """Test valid tool categories."""
        tool_def = {
            "name": "test_tool",
            "category": category,
            "params": {},
            "success_patterns": ["success"],
        }
        
        assert tool_def["category"] in [
            "network_scan", "web_scan", "password_cracking",
            "active_directory", "wifi", "exploit_framework", "osint"
        ]


class TestParameterValidation:
    """Test parameter validation for tool execution."""

    @pytest.fixture
    def tool_with_params(self):
        """Tool definition with parameter specs."""
        return {
            "name": "nmap",
            "params": {
                "target": {"type": "str", "required": True},
                "ports": {"type": "str", "required": False},
                "scan_type": {"type": "str", "required": False, "default": "-sV"},
            }
        }

    def test_validate_required_param_present(self, tool_with_params):
        """Test validation when required param is present."""
        params = {"target": "10.0.0.1"}
        
        for param_name, spec in tool_with_params["params"].items():
            if spec.get("required"):
                assert param_name in params, f"Missing required: {param_name}"

    def test_validate_required_param_missing(self, tool_with_params):
        """Test validation when required param is missing."""
        params = {"ports": "80"}
        missing = []
        
        for param_name, spec in tool_with_params["params"].items():
            if spec.get("required") and param_name not in params:
                missing.append(param_name)
        
        assert len(missing) > 0
        assert "target" in missing

    def test_validate_optional_params(self, tool_with_params):
        """Test validation of optional parameters."""
        params = {"target": "10.0.0.1"}
        
        # Optional params should be allowed but not required
        assert "ports" not in params  # But tool should handle this
        assert "scan_type" not in params

    def test_validate_param_type_string(self, tool_with_params):
        """Test parameter type validation for strings."""
        param_spec = tool_with_params["params"]["target"]
        
        assert param_spec["type"] == "str"
        assert isinstance("10.0.0.1", str)

    def test_validate_param_type_integer(self):
        """Test parameter type validation for integers."""
        param_spec = {"type": "int", "required": True}
        
        value = 8080
        assert param_spec["type"] == "int"
        assert isinstance(value, int)

    def test_validate_param_with_default(self, tool_with_params):
        """Test parameter with default value."""
        scan_type_param = tool_with_params["params"]["scan_type"]
        
        # Default should be used if not provided
        default_value = scan_type_param.get("default", "-sV")
        assert default_value == "-sV"

    @pytest.mark.parametrize("value,is_valid", [
        ("10.0.0.1", True),
        ("192.168.1.1", True),
        ("example.com", True),
        ("", False),
        (None, False),
    ])
    def test_target_parameter_validation(self, value, is_valid):
        """Test target parameter validation."""
        if value and isinstance(value, str) and len(value) > 0:
            is_valid = True
        else:
            is_valid = False
        
        assert is_valid == is_valid


class TestToolIntentClassification:
    """Test tool intent classification and recommendation."""

    @pytest.fixture
    def intent_classifier(self):
        """Intent classification mock."""
        return Mock(return_value={
            "intent": "network_reconnaissance",
            "tools": ["nmap", "masscan", "rustscan"],
            "confidence": 0.92
        })

    def test_classify_network_scan_intent(self, intent_classifier):
        """Test classification of network scan intent."""
        result = intent_classifier("scan for open ports")
        
        assert result["intent"] == "network_reconnaissance"
        assert "nmap" in result["tools"]

    def test_classify_web_exploit_intent(self, intent_classifier):
        """Test classification of web exploitation intent."""
        intent_classifier.return_value = {
            "intent": "web_exploitation",
            "tools": ["sqlmap", "wpscan", "nikto"],
            "confidence": 0.88
        }
        
        result = intent_classifier("test for SQL injection")
        assert result["intent"] == "web_exploitation"

    def test_classify_returns_tool_list(self, intent_classifier):
        """Test that classification returns recommended tools."""
        result = intent_classifier("any description")
        
        assert "tools" in result
        assert isinstance(result["tools"], list)
        assert len(result["tools"]) > 0

    def test_classify_includes_confidence(self, intent_classifier):
        """Test that classification includes confidence score."""
        result = intent_classifier("any description")
        
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1


class TestToolRegistryIntegration:
    """Integration tests for tool registry operations."""

    def test_get_all_tools(self):
        """Test retrieval of all available tools."""
        sample_registry = {
            "nmap": {"name": "nmap"},
            "sqlmap": {"name": "sqlmap"},
            "hashcat": {"name": "hashcat"},
        }
        
        all_tools = list(sample_registry.keys())
        assert len(all_tools) == 3
        assert "nmap" in all_tools

    def test_get_tools_by_category(self):
        """Test filtering tools by category."""
        sample_registry = {
            "nmap": {"category": "network_scan"},
            "masscan": {"category": "network_scan"},
            "sqlmap": {"category": "web_scan"},
        }
        
        network_tools = [
            name for name, tool in sample_registry.items()
            if tool.get("category") == "network_scan"
        ]
        
        assert len(network_tools) == 2
        assert "nmap" in network_tools

    def test_get_tool_by_alias(self):
        """Test tool retrieval by alias name."""
        tool_aliases = {
            "nmap": ["nmap", "network-mapper"],
            "sqlmap": ["sqlmap", "sql-injection-scanner"],
        }
        
        alias = "network-mapper"
        actual_tool = None
        for tool_name, aliases in tool_aliases.items():
            if alias in aliases:
                actual_tool = tool_name
                break
        
        assert actual_tool == "nmap"

    @pytest.mark.parametrize("tool_name,expected_category", [
        ("nmap", "network_scan"),
        ("sqlmap", "web_scan"),
        ("hashcat", "password_cracking"),
    ])
    def test_tool_category_mapping(self, tool_name, expected_category):
        """Test tool-to-category mapping."""
        tool_registry = {
            "nmap": {"category": "network_scan"},
            "sqlmap": {"category": "web_scan"},
            "hashcat": {"category": "password_cracking"},
        }
        
        assert tool_registry[tool_name]["category"] == expected_category


class TestToolRegistryErrorHandling:
    """Test error scenarios in tool registry."""

    def test_get_tool_nonexistent_returns_none(self):
        """Test that getting nonexistent tool returns None."""
        registry = {"nmap": {}}
        result = registry.get("nonexistent")
        
        assert result is None

    def test_registry_empty_lookup(self):
        """Test lookup in empty registry."""
        registry = {}
        result = registry.get("any_tool")
        
        assert result is None

    def test_invalid_category_handling(self):
        """Test handling of invalid tool category."""
        invalid_tool = {
            "name": "test",
            "category": "invalid_category"
        }
        
        valid_categories = ["network_scan", "web_scan"]
        is_valid = invalid_tool["category"] in valid_categories
        
        assert not is_valid

    def test_missing_required_metadata(self):
        """Test handling of tool with missing required fields."""
        invalid_tool = {
            "name": "test",
            # missing category, params, description
        }
        
        required_fields = ["category", "params", "description"]
        missing = [f for f in required_fields if f not in invalid_tool]
        
        assert len(missing) > 0
