"""Test suite for CTF Tool Manager"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from server_core.workflows.ctf.toolManager import CTFToolManager


class TestCTFToolManager:
    """Comprehensive tests for CTFToolManager"""

    @pytest.fixture
    def tool_manager(self):
        """Create CTFToolManager instance"""
        return CTFToolManager()

    def test_initialization(self, tool_manager):
        """Test CTFToolManager initialization"""
        assert tool_manager is not None
        assert hasattr(tool_manager, 'tool_commands')
        assert isinstance(tool_manager.tool_commands, dict)
        assert len(tool_manager.tool_commands) > 0

    def test_tool_commands_structure(self, tool_manager):
        """Test that tool_commands has expected entries"""
        expected_tools = ['httpx', 'katana', 'sqlmap', 'hashcat', 'john', 'checksec', 'pwntools']
        for tool in expected_tools:
            assert tool in tool_manager.tool_commands
            assert isinstance(tool_manager.tool_commands[tool], str)
            assert len(tool_manager.tool_commands[tool]) > 0

    def test_get_tool_command_simple(self, tool_manager):
        """Test getting simple tool command with target"""
        command = tool_manager.get_tool_command("hashcat", "target.txt")
        assert command is not None
        assert "hashcat" in command
        assert "target.txt" in command

    def test_get_tool_command_with_target(self, tool_manager):
        """Test getting tool command with target variable"""
        command = tool_manager.get_tool_command("nikto", "example.com")
        assert command is not None
        assert "nikto" in command

    def test_get_tool_command_with_additional_args(self, tool_manager):
        """Test getting tool command with additional arguments"""
        command = tool_manager.get_tool_command("sqlmap", "example.com", "--technique=B")
        assert command is not None
        assert "sqlmap" in command

    def test_get_tool_command_nonexistent_tool(self, tool_manager):
        """Test getting command for nonexistent tool"""
        command = tool_manager.get_tool_command("nonexistent_tool", "example.com")
        assert command is not None  # Should handle gracefully

    def test_get_category_tools_web(self, tool_manager):
        """Test getting web category tools"""
        tools = tool_manager.get_category_tools("web_recon")
        assert isinstance(tools, list)
        assert len(tools) > 0
        assert "httpx" in tools or any("web" in str(tool).lower() for tool in tools)

    def test_get_category_tools_crypto(self, tool_manager):
        """Test getting crypto category tools"""
        tools = tool_manager.get_category_tools("crypto_hash")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_category_tools_pwn(self, tool_manager):
        """Test getting pwn category tools"""
        tools = tool_manager.get_category_tools("pwn_analysis")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_category_tools_forensics(self, tool_manager):
        """Test getting forensics category tools"""
        tools = tool_manager.get_category_tools("forensics_file")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_category_tools_rev(self, tool_manager):
        """Test getting reverse engineering category tools"""
        tools = tool_manager.get_category_tools("rev_static")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_category_tools_misc(self, tool_manager):
        """Test getting misc category tools"""
        tools = tool_manager.get_category_tools("misc_encoding")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_category_tools_osint(self, tool_manager):
        """Test getting OSINT category tools"""
        tools = tool_manager.get_category_tools("osint_domain")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_category_tools_unknown(self, tool_manager):
        """Test getting tools for unknown category"""
        tools = tool_manager.get_category_tools("unknown_category")
        assert isinstance(tools, list)

    def test_suggest_tools_for_challenge_web(self, tool_manager):
        """Test suggesting tools for web challenge"""
        description = "SQL injection vulnerability in login form"
        tools = tool_manager.suggest_tools_for_challenge(description, "web")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_suggest_tools_for_challenge_crypto(self, tool_manager):
        """Test suggesting tools for crypto challenge"""
        description = "Break MD5 hash with dictionary attack"
        tools = tool_manager.suggest_tools_for_challenge(description, "crypto")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_suggest_tools_for_challenge_pwn(self, tool_manager):
        """Test suggesting tools for pwn challenge"""
        description = "Exploit buffer overflow in binary"
        tools = tool_manager.suggest_tools_for_challenge(description, "pwn")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_suggest_tools_for_challenge_forensics(self, tool_manager):
        """Test suggesting tools for forensics challenge"""
        description = "Extract hidden data from image using steganography"
        tools = tool_manager.suggest_tools_for_challenge(description, "forensics")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_suggest_tools_for_challenge_rev(self, tool_manager):
        """Test suggesting tools for reverse engineering challenge"""
        description = "Reverse engineer encrypted binary to find flag"
        tools = tool_manager.suggest_tools_for_challenge(description, "rev")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_suggest_tools_for_challenge_misc(self, tool_manager):
        """Test suggesting tools for misc challenge"""
        description = "Decode base64 encoded string"
        tools = tool_manager.suggest_tools_for_challenge(description, "misc")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_suggest_tools_for_challenge_osint(self, tool_manager):
        """Test suggesting tools for OSINT challenge"""
        description = "Find all subdomains for target domain"
        tools = tool_manager.suggest_tools_for_challenge(description, "osint")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_suggest_tools_for_challenge_empty_description(self, tool_manager):
        """Test suggesting tools with empty description"""
        tools = tool_manager.suggest_tools_for_challenge("", "web")
        assert isinstance(tools, list)

    def test_suggest_tools_for_challenge_long_description(self, tool_manager):
        """Test suggesting tools with very long description"""
        description = "This is a very long description " * 50
        tools = tool_manager.suggest_tools_for_challenge(description, "web")
        assert isinstance(tools, list)

    def test_suggest_tools_returns_unique_tools(self, tool_manager):
        """Test that suggested tools are unique"""
        description = "Complex web challenge with multiple vectors"
        tools = tool_manager.suggest_tools_for_challenge(description, "web")
        assert len(tools) == len(set(tools))  # All tools should be unique

    def test_get_tool_command_with_empty_target(self, tool_manager):
        """Test get_tool_command with empty target"""
        command = tool_manager.get_tool_command("nikto", "")
        assert command is not None

    def test_get_tool_command_with_special_chars_target(self, tool_manager):
        """Test get_tool_command with special characters in target"""
        command = tool_manager.get_tool_command("sqlmap", "http://example.com:8080/test?id=1&name=value")
        assert command is not None

    def test_get_tool_command_preserves_base_command(self, tool_manager):
        """Test that base command is preserved"""
        tool = "sqlmap"
        original_command = tool_manager.tool_commands[tool]
        command = tool_manager.get_tool_command(tool, "example.com")
        assert tool in command or original_command in command

    def test_multiple_category_lookups(self, tool_manager):
        """Test multiple category lookups return consistent results"""
        categories = ["web_recon", "crypto_hash", "pwn_exploit", "forensics_image", "rev_static", "misc_encoding", "osint_domain"]
        results = {}
        for category in categories:
            tools = tool_manager.get_category_tools(category)
            results[category] = tools
            assert isinstance(tools, list)
            assert len(tools) > 0

        # Verify results are consistent
        for category in categories:
            tools1 = tool_manager.get_category_tools(category)
            tools2 = tool_manager.get_category_tools(category)
            assert tools1 == tools2

    def test_tool_commands_have_valid_strings(self, tool_manager):
        """Test all tool commands are valid strings"""
        for tool_name, command in tool_manager.tool_commands.items():
            assert isinstance(tool_name, str)
            assert isinstance(command, str)
            assert len(tool_name) > 0
            assert len(command) > 0
            assert not command.isspace()
