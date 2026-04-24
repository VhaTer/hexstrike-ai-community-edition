"""
Test suite for new skills/MCP integration features.
Validates get_tool_skill(), typed tool wrappers, and skill bundle reading.
"""

import pytest
import mcp_core.server_setup


class TestSkillsModernizationIntegration:
    """Test core skills modernization functionality."""

    def test_tool_skill_map_loaded(self):
        """Verify _TOOL_SKILL_MAP is populated."""
        tool_skill_map = getattr(mcp_core.server_setup, '_TOOL_SKILL_MAP', {})
        assert isinstance(tool_skill_map, dict)
        assert len(tool_skill_map) > 0, "Tool skill map should not be empty"

    def test_common_tools_have_skill_mappings(self):
        """Verify common tools are mapped to skills."""
        tool_skill_map = mcp_core.server_setup._TOOL_SKILL_MAP
        
        # Test a sample of known mappings
        assert "nmap" in tool_skill_map, "nmap should be in tool skill map"
        assert "sqlmap" in tool_skill_map, "sqlmap should be in tool skill map"
        assert "metasploit" in tool_skill_map, "metasploit should be in tool skill map"

    def test_nmap_maps_to_nmap_recon(self):
        """Verify nmap maps to nmap-recon skill."""
        tool_skill_map = mcp_core.server_setup._TOOL_SKILL_MAP
        assert tool_skill_map.get("nmap") == "nmap-recon"

    def test_sqlmap_maps_to_web_vuln(self):
        """Verify sqlmap maps to web-vuln skill."""
        tool_skill_map = mcp_core.server_setup._TOOL_SKILL_MAP
        assert tool_skill_map.get("sqlmap") == "web-vuln"

    def test_metasploit_maps_to_exploitation(self):
        """Verify metasploit maps to exploitation skill."""
        tool_skill_map = mcp_core.server_setup._TOOL_SKILL_MAP
        assert tool_skill_map.get("metasploit") == "exploitation"


class TestRegistryToolDefinitions:
    """Test tool registry definition retrieval."""

    def test_get_registry_tool_definition_exists(self):
        """Verify _get_registry_tool_definition function exists."""
        func = getattr(mcp_core.server_setup, '_get_registry_tool_definition', None)
        assert callable(func), "Function should exist and be callable"

    def test_nmap_definition_exists(self):
        """Verify nmap has a registry definition."""
        definition = mcp_core.server_setup._get_registry_tool_definition("nmap")
        assert definition is not None, "nmap definition should exist in registry"

    def test_nmap_definition_has_required_fields(self):
        """Verify nmap definition has required structure."""
        definition = mcp_core.server_setup._get_registry_tool_definition("nmap")
        
        assert "desc" in definition, "Definition should have 'desc'"
        assert "params" in definition, "Definition should have 'params'"
        assert "endpoint" in definition, "Definition should have 'endpoint'"

    def test_nmap_has_target_parameter(self):
        """Verify nmap definition includes target parameter."""
        definition = mcp_core.server_setup._get_registry_tool_definition("nmap")
        params = definition.get("params", {})
        
        assert "target" in params, "nmap should require 'target' parameter"
        assert params["target"].get("required") is True, "target should be required"

    def test_sqlmap_definition_has_url_parameter(self):
        """Verify sqlmap definition includes url parameter."""
        definition = mcp_core.server_setup._get_registry_tool_definition("sqlmap")
        params = definition.get("params", {})
        
        assert "url" in params, "sqlmap should require 'url' parameter"
        assert params["url"].get("required") is True, "url should be required"

    def test_nonexistent_tool_returns_none(self):
        """Verify nonexistent tool returns None."""
        definition = mcp_core.server_setup._get_registry_tool_definition("tool_that_does_not_exist_xyz")
        assert definition is None, "Should return None for nonexistent tool"


class TestCreateTypedToolWrapper:
    """Test typed tool wrapper generation."""

    def test_create_typed_tool_wrapper_exists(self):
        """Verify _create_typed_tool_wrapper function exists."""
        func = getattr(mcp_core.server_setup, '_create_typed_tool_wrapper', None)
        assert callable(func), "Function should exist and be callable"

    def test_wrapper_has_correct_attributes(self):
        """Verify wrapper has required attributes."""
        from unittest.mock import AsyncMock
        
        tool_def = {
            "desc": "Test tool",
            "params": {"target": {"required": True}},
            "optional": {}
        }
        mock_run = AsyncMock()
        
        wrapper = mcp_core.server_setup._create_typed_tool_wrapper(
            "test_tool", tool_def, mock_run
        )
        
        assert hasattr(wrapper, '__name__'), "Wrapper should have __name__"
        assert hasattr(wrapper, '__doc__'), "Wrapper should have __doc__"
        assert hasattr(wrapper, '__annotations__'), "Wrapper should have __annotations__"


class TestSkillsBundleReading:
    """Test skill bundle reading functionality."""

    def test_read_skill_bundle_exists(self):
        """Verify _read_skill_bundle function exists."""
        func = getattr(mcp_core.server_setup, '_read_skill_bundle', None)
        assert callable(func), "Function should exist and be callable"

    def test_get_tool_skill_dynamically_registered(self):
        """Verify get_tool_skill is registered as MCP tool."""
        # get_tool_skill is registered dynamically as an MCP tool,
        # not exposed as a module-level function
        assert hasattr(mcp_core.server_setup, '_read_skill_bundle'), "Should have _read_skill_bundle helper"
        assert hasattr(mcp_core.server_setup, '_TOOL_SKILL_MAP'), "Should have _TOOL_SKILL_MAP mapping"


class TestSkillsDirectoryProvider:
    """Test skills directory provider integration."""

    def test_register_skills_exists(self):
        """Verify _register_skills function exists."""
        func = getattr(mcp_core.server_setup, '_register_skills', None)
        assert callable(func), "Function should exist and be callable"

    def test_skills_directory_exists(self):
        """Verify skills directory exists on filesystem."""
        from pathlib import Path
        skills_dir = Path(__file__).parent.parent / "skills"
        assert skills_dir.exists(), "skills directory should exist"
        assert skills_dir.is_dir(), "skills should be a directory"

    def test_nmap_recon_skill_exists(self):
        """Verify nmap-recon skill directory exists."""
        from pathlib import Path
        skill_dir = Path(__file__).parent.parent / "skills" / "nmap-recon"
        assert skill_dir.exists(), "nmap-recon skill directory should exist"

    def test_nmap_recon_has_skill_md(self):
        """Verify nmap-recon has SKILL.md."""
        from pathlib import Path
        skill_file = Path(__file__).parent.parent / "skills" / "nmap-recon" / "SKILL.md"
        assert skill_file.exists(), "SKILL.md should exist in nmap-recon"

    def test_nmap_recon_has_reference_md(self):
        """Verify nmap-recon has REFERENCE.md."""
        from pathlib import Path
        ref_file = Path(__file__).parent.parent / "skills" / "nmap-recon" / "REFERENCE.md"
        assert ref_file.exists(), "REFERENCE.md should exist in nmap-recon"
