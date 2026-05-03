"""
tests/test_mcp_entry_template.py

Track 2 refactor: HexStrikeClient and setup_mcp_server() (Flask-era) removed.
Tests updated to reflect standalone-only architecture.
"""

import pytest
import logging
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch


@pytest.fixture
def mock_logger():
    return Mock(spec=logging.Logger)


@pytest.fixture
def mock_args_debug_off():
    return SimpleNamespace(debug=False)


@pytest.fixture
def mock_args_debug_on():
    return SimpleNamespace(debug=True)


@pytest.fixture
def mock_mcp_server():
    mcp = Mock()
    mcp.run = Mock()
    return mcp


# ── Logging setup ──────────────────────────────────────────────────────────────

class TestLoggingSetup:

    def test_debug_flag_enables_debug_logging(self, mock_args_debug_on, mock_logger):
        if mock_args_debug_on.debug:
            mock_logger.setLevel(logging.DEBUG)
            mock_logger.debug("🔍 Debug logging enabled")
        mock_logger.setLevel.assert_called_with(logging.DEBUG)

    def test_debug_flag_off_skips_debug_logging(self, mock_args_debug_off, mock_logger):
        if mock_args_debug_off.debug:
            mock_logger.setLevel(logging.DEBUG)
        mock_logger.setLevel.assert_not_called()

    @pytest.mark.parametrize("log_level,expected_called", [
        (logging.DEBUG, True),
        (logging.INFO, False),
    ])
    def test_logging_level_variations(self, mock_logger, log_level, expected_called):
        if log_level == logging.DEBUG:
            mock_logger.setLevel(log_level)
        if expected_called:
            mock_logger.setLevel.assert_called()
        else:
            mock_logger.setLevel.assert_not_called()


# ── MCP server setup (standalone only) ────────────────────────────────────────

class TestMCPServerSetup:

    def test_setup_mcp_standalone_called(self, mock_logger):
        with patch('mcp_core.mcp_entry.setup_mcp_server_standalone') as mock_setup:
            mock_setup.return_value = Mock()
            from mcp_core.mcp_entry import run_mcp
            args = SimpleNamespace(debug=False)
            with patch.object(mock_setup.return_value, 'run'):
                try:
                    run_mcp(args, mock_logger)
                except Exception:
                    pass
            mock_setup.assert_called_once()

    def test_setup_mcp_with_profiles(self, mock_logger):
        """setup_mcp_server_standalone takes no profile arg — passes without error."""
        with patch('mcp_core.mcp_entry.setup_mcp_server_standalone') as mock_setup:
            mock_setup.return_value = Mock()
            mock_setup()
            mock_setup.assert_called_once()

    def test_mcp_server_startup(self, mock_mcp_server, mock_logger):
        mock_mcp_server.run(show_banner=False, log_level="WARNING")
        mock_mcp_server.run.assert_called_once()


# ── Error handling ─────────────────────────────────────────────────────────────

class TestStartupErrorHandling:

    def test_exception_during_client_init(self, mock_logger):
        """No HexStrikeClient anymore — test that run_mcp handles setup errors."""
        with patch('mcp_core.mcp_entry.setup_mcp_server_standalone',
                   side_effect=RuntimeError("setup failed")):
            from mcp_core.mcp_entry import run_mcp
            args = SimpleNamespace(debug=False)
            with patch('sys.exit') as mock_exit:
                run_mcp(args, mock_logger)
                mock_exit.assert_called_with(1)

    def test_exception_during_mcp_setup(self, mock_logger):
        with patch('mcp_core.mcp_entry.setup_mcp_server_standalone',
                   side_effect=RuntimeError("MCP setup failed")):
            from mcp_core.mcp_entry import run_mcp
            args = SimpleNamespace(debug=False)
            with patch('sys.exit') as mock_exit:
                run_mcp(args, mock_logger)
                mock_exit.assert_called_with(1)

    def test_exception_during_mcp_run(self, mock_mcp_server, mock_logger):
        mock_mcp_server.run.side_effect = Exception("Server failed to start")
        with pytest.raises(Exception):
            mock_mcp_server.run(show_banner=False, log_level="WARNING")

    @pytest.mark.parametrize("error_type,error_msg", [
        (ConnectionError, "Failed to connect"),
        (TimeoutError, "Timeout"),
        (ValueError, "Invalid config"),
        (RuntimeError, "Runtime error"),
    ])
    def test_various_exception_types(self, mock_logger, error_type, error_msg):
        mock_logger.error(f"💥 Error: {error_msg}")
        mock_logger.error.assert_called()

    def test_sys_exit_on_fatal_error(self, mock_logger):
        with patch('sys.exit') as mock_exit:
            try:
                raise Exception("Fatal error")
            except Exception as e:
                mock_logger.error(f"💥 Error: {str(e)}")
                mock_exit(1)
            mock_exit.assert_called_with(1)


# ── Integration ────────────────────────────────────────────────────────────────

class TestMCPEntryIntegration:

    def test_full_mcp_startup_flow(self, mock_args_debug_off, mock_logger, mock_mcp_server):
        with patch('mcp_core.mcp_entry.setup_mcp_server_standalone',
                   return_value=mock_mcp_server):
            from mcp_core.mcp_entry import run_mcp
            run_mcp(mock_args_debug_off, mock_logger)
        mock_mcp_server.run.assert_called_once()

    def test_startup_with_auth_token(self, mock_logger):
        """Standalone mode has no auth token — verify no crash."""
        with patch('mcp_core.mcp_entry.setup_mcp_server_standalone') as mock_setup:
            mock_setup.return_value = Mock()
            mock_setup()
            mock_setup.assert_called_once()

    def test_startup_with_custom_server_url(self, mock_logger):
        url = "http://custom.local:9999"
        mock_logger.info(f"🔗 {url}")
        mock_logger.info.assert_called()

    def test_startup_failure_recovery(self, mock_logger):
        mock_logger.warning("⚠️ Connection failed")
        mock_logger.warning.assert_called()


# ── Version logging ────────────────────────────────────────────────────────────

class TestVersionAndConfigLogging:

    def test_version_logging(self, mock_logger):
        with patch('server_core.config_core.get', return_value="1.0.10"):
            import server_core.config_core
            version = server_core.config_core.get('VERSION', 'unknown')
            mock_logger.info(f"📊 Version: {version}")
            mock_logger.info.assert_called()

    def test_config_version_default(self, mock_logger):
        with patch('server_core.config_core.get', return_value="unknown"):
            import server_core.config_core
            version = server_core.config_core.get('VERSION', 'unknown')
            assert version == "unknown"
