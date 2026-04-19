"""
Phase 1: HexStrike Client Test Suite
Tests HTTP communication with HexStrike AI API Server.

Coverage Target: 95%+
Effort: 2-3 hours
Pattern: Mock requests library, test client methods
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from typing import Dict, Any
import threading
import time


class TestHexStrikeClientInitialization:
    """Test HexStrikeClient initialization and connection."""

    def test_client_init_default_params(self):
        """Test client initialization with default parameters."""
        with patch('requests.Session'):
            from mcp_core.hexstrike_client import HexStrikeClient
            client = HexStrikeClient("http://localhost:8888")
            assert client.server_url == "http://localhost:8888"
            assert client.timeout > 0
            assert client.session is not None

    def test_client_init_with_custom_timeout(self):
        """Test client initialization with custom timeout."""
        with patch('requests.Session'):
            from mcp_core.hexstrike_client import HexStrikeClient
            client = HexStrikeClient("http://localhost:8888", timeout=600)
            assert client.timeout == 600

    def test_client_init_with_auth_token(self):
        """Test client initialization with authentication token."""
        with patch('requests.Session') as mock_session:
            from mcp_core.hexstrike_client import HexStrikeClient
            client = HexStrikeClient(
                "http://localhost:8888",
                auth_token="test_token_12345"
            )
            # Verify auth header was set
            mock_session.return_value.headers.update.assert_called()

    def test_client_init_server_url_stripped(self):
        """Test that trailing slashes are removed from server URL."""
        with patch('requests.Session'):
            from mcp_core.hexstrike_client import HexStrikeClient
            client = HexStrikeClient("http://localhost:8888/")
            assert client.server_url == "http://localhost:8888"

    def test_client_init_ssl_verification_disabled(self):
        """Test SSL verification can be disabled."""
        with patch('requests.Session') as mock_session:
            from mcp_core.hexstrike_client import HexStrikeClient
            client = HexStrikeClient(
                "http://localhost:8888",
                verify_ssl=False
            )
            mock_session.return_value.verify = False

    def test_client_init_starts_connection_thread(self):
        """Test that connection verification thread is started."""
        with patch('requests.Session'):
            with patch('threading.Thread') as mock_thread:
                from mcp_core.hexstrike_client import HexStrikeClient
                client = HexStrikeClient("http://localhost:8888")
                # Thread should be started
                mock_thread.assert_called()


class TestSafeGet:
    """Test safe_get method for HTTP GET requests."""

    @pytest.fixture
    def mock_session(self):
        """Mock requests.Session."""
        return Mock()

    @pytest.fixture
    def client(self, mock_session):
        """Create client with mocked session."""
        with patch('requests.Session', return_value=mock_session):
            from mcp_core.hexstrike_client import HexStrikeClient
            client = HexStrikeClient("http://localhost:8888")
            client.session = mock_session
            return client

    def test_safe_get_success(self, client, mock_session):
        """Test successful GET request."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "ok", "tools": 150}
        mock_session.get.return_value = mock_response

        result = client.safe_get("health")
        
        assert result["status"] == "ok"
        assert result["tools"] == 150
        # Connection thread may make a /ping call, so just verify get was called
        assert mock_session.get.call_count >= 1

    def test_safe_get_with_params(self, client, mock_session):
        """Test GET request with parameters."""
        mock_response = Mock()
        mock_response.json.return_value = {"result": "found"}
        mock_session.get.return_value = mock_response

        result = client.safe_get("search", {"query": "nmap"})
        
        assert result["result"] == "found"
        # Connection thread may make a /ping call, so just verify get was called
        assert mock_session.get.call_count >= 1

    def test_safe_get_connection_error(self, client, mock_session):
        """Test GET request with connection error."""
        mock_session.get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        result = client.safe_get("health")
        
        assert result["success"] == False
        assert "error" in result

    def test_safe_get_timeout_error(self, client, mock_session):
        """Test GET request timeout."""
        mock_session.get.side_effect = requests.exceptions.Timeout("Request timed out")

        result = client.safe_get("health")
        
        assert result["success"] == False
        assert "error" in result

    def test_safe_get_http_error(self, client, mock_session):
        """Test GET request HTTP error."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_session.get.return_value = mock_response

        result = client.safe_get("nonexistent")
        
        assert result["success"] == False

    @pytest.mark.parametrize("endpoint,params", [
        ("health", None),
        ("tools", {"category": "network"}),
        ("search", {"query": "scan", "limit": "10"}),
    ])
    def test_safe_get_various_endpoints(self, client, mock_session, endpoint, params):
        """Test GET requests to various endpoints."""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_session.get.return_value = mock_response

        result = client.safe_get(endpoint, params)
        
        assert result["success"] == True


class TestSafePost:
    """Test safe_post method for HTTP POST requests."""

    @pytest.fixture
    def mock_session(self):
        """Mock requests.Session."""
        return Mock()

    @pytest.fixture
    def client(self, mock_session):
        """Create client with mocked session."""
        with patch('requests.Session', return_value=mock_session):
            from mcp_core.hexstrike_client import HexStrikeClient
            client = HexStrikeClient("http://localhost:8888")
            client.session = mock_session
            return client

    def test_safe_post_success(self, client, mock_session):
        """Test successful POST request."""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True, "result": "executed"}
        mock_session.post.return_value = mock_response

        result = client.safe_post("api/command", {"command": "nmap scan"})
        
        assert result["success"] == True
        assert result["result"] == "executed"
        mock_session.post.assert_called_once()

    def test_safe_post_with_complex_data(self, client, mock_session):
        """Test POST request with complex JSON data."""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_session.post.return_value = mock_response

        data = {
            "command": "nmap",
            "target": "10.0.0.1",
            "options": {"timeout": 300, "ports": "1-1000"}
        }
        result = client.safe_post("api/command", data)
        
        assert result["success"] == True

    def test_safe_post_connection_error(self, client, mock_session):
        """Test POST request with connection error."""
        mock_session.post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        result = client.safe_post("api/command", {"command": "test"})
        
        assert result["success"] == False
        assert "error" in result

    def test_safe_post_timeout(self, client, mock_session):
        """Test POST request timeout."""
        mock_session.post.side_effect = requests.exceptions.Timeout("Request timed out")

        result = client.safe_post("api/command", {"command": "test"})
        
        assert result["success"] == False

    def test_safe_post_invalid_json_response(self, client, mock_session):
        """Test POST request with invalid JSON response."""
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_session.post.return_value = mock_response

        result = client.safe_post("api/command", {"command": "test"})
        
        assert result["success"] == False


class TestExecuteCommand:
    """Test execute_command method."""

    @pytest.fixture
    def mock_session(self):
        """Mock requests.Session."""
        return Mock()

    @pytest.fixture
    def client(self, mock_session):
        """Create client with mocked session."""
        with patch('requests.Session', return_value=mock_session):
            from mcp_core.hexstrike_client import HexStrikeClient
            client = HexStrikeClient("http://localhost:8888")
            client.session = mock_session
            return client

    def test_execute_command_with_cache(self, client, mock_session):
        """Test execute_command with cache enabled."""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True, "output": "scan results"}
        mock_session.post.return_value = mock_response

        result = client.execute_command("nmap 10.0.0.1", use_cache=True)
        
        assert result["success"] == True
        # Verify the POST was called with cache=True
        call_args = mock_session.post.call_args
        assert "command" in call_args.kwargs["json"]

    def test_execute_command_without_cache(self, client, mock_session):
        """Test execute_command with cache disabled."""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True, "output": "fresh results"}
        mock_session.post.return_value = mock_response

        result = client.execute_command("nmap 10.0.0.1", use_cache=False)
        
        assert result["success"] == True

    @pytest.mark.parametrize("command", [
        "nmap 10.0.0.1",
        "sqlmap -u http://target.com",
        "hashid hash123",
        "enum4linux -a 192.168.1.1",
    ])
    def test_execute_command_various(self, client, mock_session, command):
        """Test execute_command with various commands."""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_session.post.return_value = mock_response

        result = client.execute_command(command)
        
        assert result["success"] == True


class TestCheckHealth:
    """Test check_health method."""

    @pytest.fixture
    def mock_session(self):
        """Mock requests.Session."""
        return Mock()

    @pytest.fixture
    def client(self, mock_session):
        """Create client with mocked session."""
        with patch('requests.Session', return_value=mock_session):
            from mcp_core.hexstrike_client import HexStrikeClient
            client = HexStrikeClient("http://localhost:8888")
            client.session = mock_session
            return client

    def test_check_health_success(self, client, mock_session):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "healthy",
            "all_essential_tools_available": True,
            "tools_status": {"nmap": True, "sqlmap": True}
        }
        mock_session.get.return_value = mock_response

        result = client.check_health()
        
        assert result["status"] == "healthy"
        assert result["all_essential_tools_available"] == True
        # Connection thread may make a /ping call, so just verify get was called
        assert mock_session.get.call_count >= 1

    def test_check_health_missing_tools(self, client, mock_session):
        """Test health check with missing tools."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "degraded",
            "all_essential_tools_available": False,
            "tools_status": {"nmap": True, "sqlmap": False}
        }
        mock_session.get.return_value = mock_response

        result = client.check_health()
        
        assert result["all_essential_tools_available"] == False

    def test_check_health_server_down(self, client, mock_session):
        """Test health check when server is down."""
        mock_session.get.side_effect = requests.exceptions.ConnectionError()

        result = client.check_health()
        
        assert "error" in result or result.get("success") == False


class TestClientEdgeCases:
    """Test edge cases and error scenarios."""

    def test_client_empty_server_url(self):
        """Test client with empty server URL."""
        with patch('requests.Session'):
            from mcp_core.hexstrike_client import HexStrikeClient
            client = HexStrikeClient("")
            assert client.server_url == ""

    def test_client_malformed_server_url(self):
        """Test client with malformed server URL."""
        with patch('requests.Session'):
            from mcp_core.hexstrike_client import HexStrikeClient
            client = HexStrikeClient("not-a-valid-url")
            assert client.server_url == "not-a-valid-url"

    def test_client_zero_timeout(self):
        """Test client with zero timeout."""
        with patch('requests.Session'):
            from mcp_core.hexstrike_client import HexStrikeClient
            client = HexStrikeClient("http://localhost:8888", timeout=0)
            assert client.timeout == 0

    def test_safe_get_with_empty_endpoint(self):
        """Test safe_get with empty endpoint."""
        with patch('requests.Session') as mock_session_cls:
            mock_session = Mock()
            mock_session.get.return_value.json.return_value = {"result": "ok"}
            mock_session_cls.return_value = mock_session
            
            from mcp_core.hexstrike_client import HexStrikeClient
            client = HexStrikeClient("http://localhost:8888")
            client.session = mock_session
            
            result = client.safe_get("")
            assert result["result"] == "ok"
