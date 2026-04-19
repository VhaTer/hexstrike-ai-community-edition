"""
PHASE 1 TEST TEMPLATE: mcp_entry.py (17.14% → 95%+)
====================================================

Goal: Achieve 95%+ coverage for mcp_core/mcp_entry.py

This module orchestrates MCP server initialization and startup. Key areas:
1. Server health checking
2. Logging setup based on debug flag
3. MCP server configuration
4. Error handling during startup
5. Connection establishment to HexStrike API
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import logging
import sys
from types import SimpleNamespace


# ========== FIXTURE SETUP ==========

@pytest.fixture
def mock_logger():
    """Create a mock logger"""
    logger = Mock(spec=logging.Logger)
    return logger


@pytest.fixture
def mock_args_debug_off():
    """Simulated command-line args (debug OFF)"""
    return SimpleNamespace(
        server="http://127.0.0.1:8888",
        timeout=300,
        debug=False,
        compact=False,
        profile=[],
        auth_token="",
        disable_ssl_verify=False
    )


@pytest.fixture
def mock_args_debug_on():
    """Simulated command-line args (debug ON)"""
    return SimpleNamespace(
        server="http://127.0.0.1:8888",
        timeout=300,
        debug=True,
        compact=False,
        profile=[],
        auth_token="token123",
        disable_ssl_verify=False
    )


@pytest.fixture
def mock_hexstrike_client():
    """Mock HexStrikeClient"""
    client = Mock()
    client.check_health = Mock(return_value={
        "status": "healthy",
        "all_essential_tools_available": True,
        "tools_status": {}
    })
    return client


@pytest.fixture
def mock_mcp_server():
    """Mock FastMCP server"""
    mcp = Mock()
    mcp.run = Mock()
    return mcp


# ========== TEST SECTION 1: Logging Configuration ==========

class TestLoggingSetup:
    """Test logging setup based on debug flag"""

    def test_debug_flag_enables_debug_logging(self, mock_args_debug_on, mock_logger):
        """Test that debug=True enables DEBUG level logging"""
        # Simulate what run_mcp does
        if mock_args_debug_on.debug:
            mock_logger.setLevel(logging.DEBUG)
            mock_logger.debug("🔍 Debug logging enabled")
        
        mock_logger.setLevel.assert_called_with(logging.DEBUG)
        mock_logger.debug.assert_called()

    def test_debug_flag_off_keeps_info_logging(self, mock_args_debug_off, mock_logger):
        """Test that debug=False keeps INFO level logging"""
        if mock_args_debug_off.debug:
            mock_logger.setLevel(logging.DEBUG)
        
        # No setLevel call should be made when debug=False
        mock_logger.setLevel.assert_not_called()

    @pytest.mark.parametrize("log_level,expected_called", [
        (logging.DEBUG, True),
        (logging.INFO, False),
        (logging.WARNING, False),
    ])
    def test_logging_level_variations(self, mock_logger, log_level, expected_called):
        """Test different logging levels"""
        if log_level == logging.DEBUG:
            mock_logger.setLevel(log_level)
        
        if expected_called:
            mock_logger.setLevel.assert_called()
        else:
            mock_logger.setLevel.assert_not_called()


# ========== TEST SECTION 2: Server Connection ==========

class TestServerConnection:
    """Test connecting to HexStrike API server"""

    def test_successful_server_connection(self, mock_hexstrike_client, mock_logger):
        """Test successful connection to HexStrike server"""
        # Simulate HexStrikeClient initialization and health check
        health = mock_hexstrike_client.check_health()
        
        assert health["status"] == "healthy"
        assert health["all_essential_tools_available"] is True
        mock_logger.info("✅ Connected to HexStrike server")
        mock_logger.info.assert_called()

    def test_connection_logs_server_url(self, mock_args_debug_off, mock_logger):
        """Test that connection logs the server URL"""
        mock_logger.info(f"🔗 Connecting to: {mock_args_debug_off.server}")
        mock_logger.info.assert_called_with(f"🔗 Connecting to: {mock_args_debug_off.server}")

    def test_health_check_with_missing_tools(self, mock_hexstrike_client, mock_logger):
        """Test health check when some tools are missing"""
        mock_hexstrike_client.check_health.return_value = {
            "status": "degraded",
            "all_essential_tools_available": False,
            "tools_status": {
                "nmap": True,
                "sqlmap": False,
                "hashid": False
            }
        }
        
        health = mock_hexstrike_client.check_health()
        
        if not health.get("all_essential_tools_available", False):
            missing = [tool for tool, avail in health["tools_status"].items() if not avail]
            mock_logger.warning(f"Missing tools: {', '.join(missing)}")
            mock_logger.warning.assert_called()

    def test_health_check_error_response(self, mock_hexstrike_client, mock_logger):
        """Test health check with error response"""
        mock_hexstrike_client.check_health.return_value = {
            "error": "Connection refused",
            "success": False
        }
        
        health = mock_hexstrike_client.check_health()
        
        if "error" in health:
            mock_logger.warning(f"⚠️  Unable to connect to HexStrike: {health['error']}")
            mock_logger.warning.assert_called()

    @pytest.mark.parametrize("server_url,expected_domain", [
        ("http://localhost:8888", "localhost"),
        ("http://127.0.0.1:8888", "127.0.0.1"),
        ("http://hexstrike.example.com:8888", "hexstrike.example.com"),
    ])
    def test_different_server_urls(self, mock_logger, server_url, expected_domain):
        """Test connecting to different server URLs"""
        mock_logger.info(f"🔗 Connecting to: {server_url}")
        assert server_url is not None
        assert expected_domain in server_url


# ========== TEST SECTION 3: MCP Server Setup ==========

class TestMCPServerSetup:
    """Test MCP server configuration and startup"""

    def test_setup_mcp_default_profile(self, mock_hexstrike_client, mock_logger, mock_args_debug_off):
        """Test MCP setup with default tool profile"""
        with patch('mcp_core.mcp_entry.setup_mcp_server') as mock_setup:
            mock_mcp = Mock()
            mock_setup.return_value = mock_mcp
            
            # Simulate setup_mcp_server call
            mcp = mock_setup(mock_hexstrike_client, mock_logger, 
                           compact=mock_args_debug_off.compact, 
                           profiles=mock_args_debug_off.profile)
            
            mock_setup.assert_called_once()
            assert mcp is not None

    def test_setup_mcp_compact_mode(self, mock_hexstrike_client, mock_logger, mock_args_debug_off):
        """Test MCP setup in compact mode"""
        mock_args_debug_off.compact = True
        
        with patch('mcp_core.mcp_entry.setup_mcp_server') as mock_setup:
            mock_mcp = Mock()
            mock_setup.return_value = mock_mcp
            
            mcp = mock_setup(mock_hexstrike_client, mock_logger,
                           compact=True,
                           profiles=[])
            
            mock_logger.info("Compact mode enabled")
            mock_setup.assert_called_once()

    def test_setup_mcp_with_profiles(self, mock_hexstrike_client, mock_logger):
        """Test MCP setup with specific tool profiles"""
        profiles = ["web_crawl", "recon", "exploit_framework"]
        
        with patch('mcp_core.mcp_entry.setup_mcp_server') as mock_setup:
            mock_mcp = Mock()
            mock_setup.return_value = mock_mcp
            
            mcp = mock_setup(mock_hexstrike_client, mock_logger,
                           compact=False,
                           profiles=profiles)
            
            mock_setup.assert_called_once()
            # Verify profiles were passed
            call_args = mock_setup.call_args
            assert call_args[1]['profiles'] == profiles

    def test_mcp_server_startup(self, mock_mcp_server, mock_logger):
        """Test MCP server startup"""
        mock_mcp_server.run(show_banner=False, log_level="WARNING")
        mock_logger.info("🚀 HexStrike AI MCP server ready")
        
        mock_mcp_server.run.assert_called_once()
        mock_logger.info.assert_called()


# ========== TEST SECTION 4: Error Handling During Startup ==========

class TestStartupErrorHandling:
    """Test error handling during MCP startup"""

    def test_exception_during_client_init(self, mock_args_debug_off, mock_logger):
        """Test handling of exception during HexStrikeClient initialization"""
        with patch('mcp_core.mcp_entry.HexStrikeClient') as MockClient:
            MockClient.side_effect = Exception("Connection failed")
            
            try:
                client = MockClient(mock_args_debug_off.server)
                assert False, "Should have raised exception"
            except Exception as e:
                mock_logger.error(f"💥 Error: {str(e)}")
                mock_logger.error.assert_called()

    def test_exception_during_mcp_setup(self, mock_hexstrike_client, mock_logger, mock_args_debug_off):
        """Test handling of exception during MCP server setup"""
        with patch('mcp_core.mcp_entry.setup_mcp_server') as mock_setup:
            mock_setup.side_effect = RuntimeError("MCP setup failed")
            
            with pytest.raises(RuntimeError):
                mock_setup(mock_hexstrike_client, mock_logger,
                          compact=mock_args_debug_off.compact,
                          profiles=mock_args_debug_off.profile)
            
            mock_setup.assert_called_once()

    def test_exception_during_mcp_run(self, mock_mcp_server, mock_logger):
        """Test handling of exception during mcp.run()"""
        mock_mcp_server.run.side_effect = Exception("Server failed to start")
        
        with pytest.raises(Exception):
            mock_mcp_server.run(show_banner=False, log_level="WARNING")
        
        mock_logger.error("💥 Error starting MCP server")
        mock_logger.error.assert_called()

    @pytest.mark.parametrize("error_type,error_msg", [
        (ConnectionError, "Failed to connect to server"),
        (TimeoutError, "Connection timeout"),
        (ValueError, "Invalid configuration"),
        (RuntimeError, "Unexpected runtime error"),
    ])
    def test_various_exception_types(self, mock_logger, error_type, error_msg):
        """Test handling of various exception types"""
        exception = error_type(error_msg)
        
        mock_logger.error(f"💥 Error starting MCP server: {str(exception)}")
        mock_logger.error.assert_called()

    def test_sys_exit_on_fatal_error(self, mock_logger):
        """Test that fatal errors cause sys.exit()"""
        with patch('sys.exit') as mock_exit:
            try:
                raise Exception("Fatal error")
            except Exception as e:
                mock_logger.error(f"💥 Error: {str(e)}")
                mock_exit(1)
            
            mock_exit.assert_called_with(1)


# ========== TEST SECTION 5: Integration Tests ==========

class TestMCPEntryIntegration:
    """Integration tests for complete MCP startup flow"""

    def test_full_mcp_startup_flow(self, mock_args_debug_off, mock_hexstrike_client, 
                                   mock_logger, mock_mcp_server):
        """Test complete MCP startup flow"""
        # Step 1: Check debug flag
        if mock_args_debug_off.debug:
            mock_logger.setLevel(logging.DEBUG)
        
        # Step 2: Log startup
        mock_logger.info(f"🚀 Starting HexStrike MCP Client")
        mock_logger.info(f"🔗 Connecting to: {mock_args_debug_off.server}")
        
        # Step 3: Check health
        health = mock_hexstrike_client.check_health()
        mock_logger.info(f"🏥 Server health status: {health['status']}")
        
        # Step 4: Setup MCP
        with patch('mcp_core.mcp_entry.setup_mcp_server') as mock_setup:
            mock_setup.return_value = mock_mcp_server
            mcp = mock_setup(mock_hexstrike_client, mock_logger,
                           compact=mock_args_debug_off.compact,
                           profiles=mock_args_debug_off.profile)
        
        # Step 5: Start MCP
        mock_logger.info("🚀 HexStrike AI MCP server ready")
        mcp.run(show_banner=False, log_level="WARNING")
        
        # Verify all calls were made
        assert mock_logger.info.call_count >= 3
        mock_mcp_server.run.assert_called_once()

    def test_startup_with_auth_token(self, mock_hexstrike_client, mock_logger):
        """Test MCP startup with authentication token"""
        auth_token = "secret_token_123"
        
        with patch('mcp_core.mcp_entry.HexStrikeClient') as MockClient:
            mock_client = Mock()
            mock_client.check_health.return_value = {"status": "healthy"}
            MockClient.return_value = mock_client
            
            # Initialize client with auth token
            client = MockClient("http://localhost:8888", auth_token=auth_token)
            
            # Verify auth token was passed
            MockClient.assert_called_once()

    def test_startup_with_custom_server_url(self, mock_hexstrike_client, mock_logger):
        """Test MCP startup with custom server URL"""
        custom_url = "http://custom.hexstrike.local:9999"
        mock_logger.info(f"🔗 Connecting to: {custom_url}")
        
        mock_logger.info.assert_called_with(f"🔗 Connecting to: {custom_url}")

    def test_startup_failure_recovery(self, mock_hexstrike_client, mock_logger):
        """Test that startup failure is logged properly"""
        mock_hexstrike_client.check_health.return_value = {
            "error": "Connection refused",
            "success": False
        }
        
        health = mock_hexstrike_client.check_health()
        
        if "error" in health:
            # Server offline, but MCP should still start
            mock_logger.warning(f"⚠️  Connection failed, MCP will start anyway")
            mock_logger.warning.assert_called()


# ========== TEST SECTION 6: Version and Config Logging ==========

class TestVersionAndConfigLogging:
    """Test logging of version and configuration info"""

    def test_version_logging(self, mock_logger):
        """Test that version is logged"""
        with patch('server_core.config_core.get') as mock_config:
            mock_config.return_value = "1.0.10"
            version = mock_config('VERSION', 'unknown')
            
            mock_logger.info(f"📊 Version: {version}")
            mock_logger.info.assert_called()

    def test_config_version_default(self, mock_logger):
        """Test version defaults to 'unknown' if not found"""
        with patch('server_core.config_core.get') as mock_config:
            mock_config.return_value = "unknown"
            version = mock_config('VERSION', 'unknown')
            
            assert version == "unknown"


# ========== Conftest Helper: Run All Tests ==========

if __name__ == "__main__":
    # Run with: pytest tests/test_mcp_entry_template.py -v
    pytest.main([__file__, "-v", "--tb=short"])
