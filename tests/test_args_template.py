"""
PHASE 1 TEST TEMPLATE: args.py (25% → 95%+)
============================================

Goal: Achieve 95%+ coverage for mcp_core/args.py

This module handles command-line argument parsing. Key areas to test:
1. Parsing all argument types (string, int, bool, list)
2. Default values are correctly set
3. Help messages work
4. Argument validation
5. Custom argument combinations
"""

import pytest
import sys
from unittest.mock import Mock, patch
from io import StringIO
import argparse


# ========== FIXTURE SETUP ==========

@pytest.fixture
def original_argv():
    """Save original sys.argv to restore it after tests"""
    original = sys.argv.copy()
    yield original
    sys.argv = original


# ========== TEST SECTION 1: Basic Argument Parsing ==========

class TestBasicArgumentParsing:
    """Test basic CLI argument parsing"""

    def test_parse_args_defaults(self, original_argv):
        """Test parse_args returns default values when no args provided"""
        sys.argv = ["hexstrike_mcp.py"]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        assert args.server == "http://127.0.0.1:8888"
        assert args.timeout == 600
        assert args.debug is False
        assert args.compact is False
        assert args.profile == []
        assert args.auth_token == ""
        assert args.disable_ssl_verify is False

    def test_parse_args_server_url(self, original_argv):
        """Test parsing custom server URL"""
        sys.argv = ["hexstrike_mcp.py", "--server", "http://custom.com:9999"]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        assert args.server == "http://custom.com:9999"

    def test_parse_args_timeout(self, original_argv):
        """Test parsing timeout value"""
        sys.argv = ["hexstrike_mcp.py", "--timeout", "600"]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        assert args.timeout == 600
        assert isinstance(args.timeout, int)

    def test_parse_args_debug_flag(self, original_argv):
        """Test parsing debug flag"""
        sys.argv = ["hexstrike_mcp.py", "--debug"]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        assert args.debug is True

    def test_parse_args_compact_flag(self, original_argv):
        """Test parsing compact mode flag"""
        sys.argv = ["hexstrike_mcp.py", "--compact"]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        assert args.compact is True

    def test_parse_args_auth_token(self, original_argv):
        """Test parsing authentication token"""
        sys.argv = ["hexstrike_mcp.py", "--auth-token", "my_secret_token"]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        assert args.auth_token == "my_secret_token"

    def test_parse_args_disable_ssl_verify(self, original_argv):
        """Test parsing SSL verification disable flag"""
        sys.argv = ["hexstrike_mcp.py", "--disable-ssl-verify"]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        assert args.disable_ssl_verify is True


# ========== TEST SECTION 2: Profile Argument Parsing ==========

class TestProfileArgumentParsing:
    """Test parsing of tool profiles"""

    def test_parse_args_single_profile(self, original_argv):
        """Test parsing single tool profile"""
        sys.argv = ["hexstrike_mcp.py", "--profile", "recon"]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        assert args.profile == ["recon"]
        assert len(args.profile) == 1

    def test_parse_args_multiple_profiles(self, original_argv):
        """Test parsing multiple tool profiles"""
        sys.argv = ["hexstrike_mcp.py", "--profile", "recon", "web_crawl", "exploit_framework"]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        assert args.profile == ["recon", "web_crawl", "exploit_framework"]
        assert len(args.profile) == 3

    def test_parse_args_full_profile(self, original_argv):
        """Test parsing 'full' profile"""
        sys.argv = ["hexstrike_mcp.py", "--profile", "full"]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        assert "full" in args.profile

    def test_parse_args_default_profile(self, original_argv):
        """Test parsing 'default' profile"""
        sys.argv = ["hexstrike_mcp.py", "--profile", "default"]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        assert "default" in args.profile

    def test_parse_args_no_profile(self, original_argv):
        """Test default when no profile specified"""
        sys.argv = ["hexstrike_mcp.py"]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        assert args.profile == []

    @pytest.mark.parametrize("profiles", [
        ["recon"],
        ["web_crawl"],
        ["exploit_framework"],
        ["recon", "web_crawl"],
        ["web_crawl", "exploit_framework", "binary_analysis"],
    ])
    def test_parse_args_various_profiles(self, original_argv, profiles):
        """Parametrized test for various profile combinations"""
        sys.argv = ["hexstrike_mcp.py", "--profile"] + profiles
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        assert args.profile == profiles


# ========== TEST SECTION 3: Argument Validation ==========

class TestArgumentValidation:
    """Test argument validation and type checking"""

    def test_timeout_is_integer(self, original_argv):
        """Test that timeout is parsed as integer"""
        sys.argv = ["hexstrike_mcp.py", "--timeout", "500"]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        assert isinstance(args.timeout, int)
        assert args.timeout == 500

    def test_invalid_timeout_raises_error(self, original_argv):
        """Test that non-integer timeout raises error"""
        sys.argv = ["hexstrike_mcp.py", "--timeout", "not_a_number"]
        
        from mcp_core.args import parse_args
        
        with pytest.raises(SystemExit):  # argparse exits on error
            parse_args()

    def test_negative_timeout_allowed(self, original_argv):
        """Test that negative timeout is allowed (parsed but may be invalid semantically)"""
        sys.argv = ["hexstrike_mcp.py", "--timeout", "-100"]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        # argparse allows it, but business logic may reject it
        assert args.timeout == -100

    def test_server_url_string(self, original_argv):
        """Test that server URL is parsed as string"""
        sys.argv = ["hexstrike_mcp.py", "--server", "http://example.com"]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        assert isinstance(args.server, str)
        assert args.server == "http://example.com"

    def test_boolean_flags_are_boolean(self, original_argv):
        """Test that boolean flags are correctly parsed as bool"""
        sys.argv = ["hexstrike_mcp.py", "--debug", "--compact"]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        assert isinstance(args.debug, bool)
        assert isinstance(args.compact, bool)
        assert args.debug is True
        assert args.compact is True

    def test_boolean_flags_default_false(self, original_argv):
        """Test that boolean flags default to False"""
        sys.argv = ["hexstrike_mcp.py"]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        assert args.debug is False
        assert args.compact is False
        assert args.disable_ssl_verify is False


# ========== TEST SECTION 4: Combined Arguments ==========

class TestCombinedArguments:
    """Test parsing combinations of arguments"""

    def test_all_arguments_combined(self, original_argv):
        """Test parsing all arguments together"""
        sys.argv = [
            "hexstrike_mcp.py",
            "--server", "http://custom.com:8888",
            "--timeout", "600",
            "--debug",
            "--compact",
            "--profile", "recon", "web_crawl",
            "--auth-token", "token123",
            "--disable-ssl-verify"
        ]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        assert args.server == "http://custom.com:8888"
        assert args.timeout == 600
        assert args.debug is True
        assert args.compact is True
        assert args.profile == ["recon", "web_crawl"]
        assert args.auth_token == "token123"
        assert args.disable_ssl_verify is True

    def test_server_and_timeout_only(self, original_argv):
        """Test parsing only server and timeout"""
        sys.argv = ["hexstrike_mcp.py", "--server", "http://localhost", "--timeout", "500"]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        assert args.server == "http://localhost"
        assert args.timeout == 500
        assert args.debug is False  # Other flags remain default
        assert args.compact is False

    def test_debug_and_profile_combo(self, original_argv):
        """Test debug with multiple profiles"""
        sys.argv = [
            "hexstrike_mcp.py",
            "--debug",
            "--profile", "recon", "exploit_framework"
        ]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        assert args.debug is True
        assert "recon" in args.profile
        assert "exploit_framework" in args.profile

    @pytest.mark.parametrize("server,timeout", [
        ("http://localhost:8888", 300),
        ("http://192.168.1.1:9999", 600),
        ("http://remote.server.com:8080", 120),
    ])
    def test_various_server_timeout_combos(self, original_argv, server, timeout):
        """Parametrized test for various server/timeout combinations"""
        sys.argv = ["hexstrike_mcp.py", "--server", server, "--timeout", str(timeout)]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        assert args.server == server
        assert args.timeout == timeout


# ========== TEST SECTION 5: Default Values ==========

class TestDefaultValues:
    """Test that default values are correctly set"""

    def test_server_default(self, original_argv):
        """Test server URL default"""
        sys.argv = ["hexstrike_mcp.py"]
        
        from mcp_core.args import parse_args
        from mcp_core.hexstrike_client import DEFAULT_HEXSTRIKE_SERVER
        
        args = parse_args()
        
        assert args.server == DEFAULT_HEXSTRIKE_SERVER
        assert args.server == "http://127.0.0.1:8888"

    def test_timeout_default(self, original_argv):
        """Test timeout default"""
        sys.argv = ["hexstrike_mcp.py"]
        
        from mcp_core.args import parse_args
        from mcp_core.hexstrike_client import DEFAULT_REQUEST_TIMEOUT
        
        args = parse_args()
        
        assert args.timeout == DEFAULT_REQUEST_TIMEOUT
        assert args.timeout == 600

    def test_all_defaults(self, original_argv):
        """Test all arguments have sensible defaults"""
        sys.argv = ["hexstrike_mcp.py"]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        # All defaults should be set
        assert hasattr(args, 'server')
        assert hasattr(args, 'timeout')
        assert hasattr(args, 'debug')
        assert hasattr(args, 'compact')
        assert hasattr(args, 'profile')
        assert hasattr(args, 'auth_token')
        assert hasattr(args, 'disable_ssl_verify')


# ========== TEST SECTION 6: Help Messages and Usage ==========

class TestHelpMessages:
    """Test help messages and usage text"""

    def test_help_flag_exists(self, original_argv):
        """Test that --help flag is available"""
        sys.argv = ["hexstrike_mcp.py", "--help"]
        
        from mcp_core.args import parse_args
        
        # --help causes SystemExit, which is expected
        with pytest.raises(SystemExit) as exc_info:
            parse_args()
        
        # Exit code should be 0 for successful help
        assert exc_info.value.code == 0

    def test_help_short_flag(self, original_argv):
        """Test that -h flag is available"""
        sys.argv = ["hexstrike_mcp.py", "-h"]
        
        from mcp_core.args import parse_args
        
        with pytest.raises(SystemExit) as exc_info:
            parse_args()
        
        assert exc_info.value.code == 0


# ========== TEST SECTION 7: Edge Cases ==========

class TestEdgeCases:
    """Test edge cases and unusual inputs"""

    def test_empty_profile_list(self, original_argv):
        """Test empty profile list"""
        sys.argv = ["hexstrike_mcp.py", "--profile"]  # --profile without values
        
        from mcp_core.args import parse_args
        
        # This should cause an error because --profile requires at least one value
        with pytest.raises(SystemExit):
            parse_args()

    def test_duplicate_arguments(self, original_argv):
        """Test handling of duplicate arguments (last one wins)"""
        sys.argv = [
            "hexstrike_mcp.py",
            "--server", "http://first.com",
            "--server", "http://second.com"
        ]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        # argparse uses the last value
        assert args.server == "http://second.com"

    def test_long_timeout_value(self, original_argv):
        """Test parsing very large timeout value"""
        sys.argv = ["hexstrike_mcp.py", "--timeout", "999999"]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        assert args.timeout == 999999

    def test_special_chars_in_auth_token(self, original_argv):
        """Test auth token with special characters"""
        sys.argv = ["hexstrike_mcp.py", "--auth-token", "token_with_$pec!@l_ch@rs"]
        
        from mcp_core.args import parse_args
        args = parse_args()
        
        assert args.auth_token == "token_with_$pec!@l_ch@rs"

    def test_localhost_variations(self, original_argv):
        """Test various localhost address variations"""
        localhost_urls = [
            "http://127.0.0.1:8888",
            "http://localhost:8888",
            "http://0.0.0.0:8888",
            "https://127.0.0.1:8888"
        ]
        
        for url in localhost_urls:
            sys.argv = ["hexstrike_mcp.py", "--server", url]
            from mcp_core.args import parse_args
            args = parse_args()
            assert args.server == url


# ========== Conftest Helper: Run All Tests ==========

if __name__ == "__main__":
    # Run with: pytest tests/test_args_template.py -v
    pytest.main([__file__, "-v", "--tb=short"])
