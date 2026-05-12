"""Unit tests for mcp_core.hexstrike_client — 100% coverage."""

import pytest
from unittest.mock import patch, MagicMock
import requests
import threading

from mcp_core import hexstrike_client
from mcp_core.hexstrike_client import HexStrikeClient


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_session():
    return MagicMock()


@pytest.fixture
def client(mock_session):
    """HexStrikeClient with mocked session & patched threading.Thread
    so the background _verify_connection thread never actually runs."""
    with patch.object(hexstrike_client.threading, "Thread") as mock_t, \
         patch("requests.Session", return_value=mock_session):
        mock_t.return_value.start = MagicMock()
        c = HexStrikeClient("http://localhost:8888")
        return c


# ── __init__ ────────────────────────────────────────────────────────────────

class TestInit:
    def test_default_params(self):
        with patch.object(hexstrike_client.threading, "Thread"), \
             patch("requests.Session"):
            c = HexStrikeClient("http://localhost:8888")
            assert c.server_url == "http://localhost:8888"
            assert c.timeout == hexstrike_client.DEFAULT_REQUEST_TIMEOUT
            assert c._connected is False
            assert isinstance(c._connect_lock, threading.Lock)

    def test_custom_timeout(self):
        with patch.object(hexstrike_client.threading, "Thread"), \
             patch("requests.Session"):
            c = HexStrikeClient("http://localhost:8888", timeout=600)
            assert c.timeout == 600

    def test_auth_token_sets_header(self):
        with patch.object(hexstrike_client.threading, "Thread"), \
             patch("requests.Session") as mock_sess_cls:
            sess = MagicMock()
            mock_sess_cls.return_value = sess
            c = HexStrikeClient("http://localhost:8888", auth_token="tok123")
            sess.headers.update.assert_called_once_with(
                {"Authorization": "Bearer tok123"}
            )

    def test_verify_ssl_false_disables_verification(self):
        with patch.object(hexstrike_client.threading, "Thread"), \
             patch("requests.Session") as mock_sess_cls:
            sess = MagicMock()
            mock_sess_cls.return_value = sess
            c = HexStrikeClient("http://localhost:8888", verify_ssl=False)
            assert sess.verify is False

    def test_trailing_slash_stripped(self):
        with patch.object(hexstrike_client.threading, "Thread"), \
             patch("requests.Session"):
            c = HexStrikeClient("http://localhost:8888/")
            assert c.server_url == "http://localhost:8888"

    def test_background_thread_started(self):
        with patch.object(hexstrike_client.threading, "Thread") as mock_t, \
             patch("requests.Session"):
            c = HexStrikeClient("http://localhost:8888")
            mock_t.assert_called_once_with(target=c._verify_connection, daemon=True)
            mock_t.return_value.start.assert_called_once()


# ── _verify_connection ──────────────────────────────────────────────────────

class TestVerifyConnection:
    def test_success_first_try(self, client):
        mock_resp = MagicMock()
        client.session.get.return_value = mock_resp
        client._verify_connection()
        client.session.get.assert_called_once_with(
            "http://localhost:8888/ping", timeout=5
        )
        mock_resp.raise_for_status.assert_called_once()
        assert client._connected is True

    def test_connection_error_then_success(self, client):
        mock_resp = MagicMock()
        client.session.get.side_effect = [
            requests.exceptions.ConnectionError(),
            mock_resp,
        ]
        with patch.object(hexstrike_client, "MAX_RETRIES", 2), \
             patch.object(hexstrike_client.time, "sleep"):
            client._verify_connection()
        assert client.session.get.call_count == 2
        assert client._connected is True

    def test_generic_exception_then_success(self, client):
        """Cover except Exception in _verify_connection (lines 47-48)."""
        mock_resp = MagicMock()
        client.session.get.side_effect = [
            Exception("unexpected failure"),
            mock_resp,
        ]
        with patch.object(hexstrike_client, "MAX_RETRIES", 2), \
             patch.object(hexstrike_client.time, "sleep"):
            client._verify_connection()
        assert client.session.get.call_count == 2
        assert client._connected is True

    def test_all_retries_exhausted(self, client):
        """Cover critical log (line 52) + sleep skipped on last attempt (line 49 False branch)."""
        client.session.get.side_effect = requests.exceptions.ConnectionError()
        with patch.object(hexstrike_client, "MAX_RETRIES", 1), \
             patch.object(hexstrike_client.time, "sleep") as mock_sleep, \
             patch.object(hexstrike_client.logging, "critical") as mock_crit:
            client._verify_connection()
        assert client.session.get.call_count == 1
        mock_sleep.assert_not_called()
        mock_crit.assert_called_once()
        assert "offline" in mock_crit.call_args[0][0].lower()
        assert client._connected is False

    def test_all_connection_errors_with_retries(self, client):
        """Multiple ConnectionErrors with MAX_RETRIES=3 — covers sleep on early attempts + critical on last."""
        client.session.get.side_effect = requests.exceptions.ConnectionError()
        with patch.object(hexstrike_client, "MAX_RETRIES", 3), \
             patch.object(hexstrike_client.time, "sleep"), \
             patch.object(hexstrike_client.logging, "critical") as mock_crit:
            client._verify_connection()
        assert client.session.get.call_count == 3
        mock_crit.assert_called_once()
        assert client._connected is False

    def test_all_generic_exceptions(self, client):
        """All attempts raise generic Exception."""
        client.session.get.side_effect = Exception("persistent failure")
        with patch.object(hexstrike_client, "MAX_RETRIES", 2), \
             patch.object(hexstrike_client.time, "sleep"), \
             patch.object(hexstrike_client.logging, "critical") as mock_crit:
            client._verify_connection()
        assert client.session.get.call_count == 2
        mock_crit.assert_called_once()
        assert client._connected is False


# ── safe_get ────────────────────────────────────────────────────────────────

class TestSafeGet:
    def test_success(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ok"}
        client.session.get.return_value = mock_resp
        result = client.safe_get("health")
        assert result == {"status": "ok"}

    def test_with_params(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": "yes"}
        client.session.get.return_value = mock_resp
        result = client.safe_get("api/tools", {"filter": "nmap"})
        assert result == {"data": "yes"}

    def test_params_none_uses_empty_dict(self, client):
        """params=None hits the `if params is None` branch."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True}
        client.session.get.return_value = mock_resp
        result = client.safe_get("health", None)
        assert result == {"ok": True}
        client.session.get.assert_called_once_with(
            "http://localhost:8888/health", params={}, timeout=300
        )

    def test_request_exception(self, client):
        """Cover RequestException handler (lines 63-65)."""
        client.session.get.side_effect = requests.exceptions.Timeout("timed out")
        result = client.safe_get("health")
        assert result == {"error": "Request failed: timed out", "success": False}

    def test_generic_exception(self, client):
        """Cover generic Exception handler in safe_get (lines 66-68)."""
        client.session.get.side_effect = ValueError("bad value")
        result = client.safe_get("health")
        assert result == {"error": "Unexpected error: bad value", "success": False}


# ── safe_post ───────────────────────────────────────────────────────────────

class TestSafePost:
    def test_success(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"result": "done"}
        client.session.post.return_value = mock_resp
        result = client.safe_post("api/command", {"cmd": "nmap"})
        assert result == {"result": "done"}

    def test_request_exception(self, client):
        """Cover RequestException handler in safe_post (lines 77-79)."""
        client.session.post.side_effect = requests.exceptions.ConnectionError("refused")
        result = client.safe_post("api/command", {"cmd": "nmap"})
        assert result == {"error": "Request failed: refused", "success": False}

    def test_generic_exception(self, client):
        """Cover generic Exception handler in safe_post (lines 80-82)."""
        mock_resp = MagicMock()
        mock_resp.json.side_effect = ValueError("bad json")
        client.session.post.return_value = mock_resp
        result = client.safe_post("api/command", {"cmd": "nmap"})
        assert result == {"error": "Unexpected error: bad json", "success": False}


# ── execute_command ─────────────────────────────────────────────────────────

class TestExecuteCommand:
    def test_with_cache_default(self, client):
        with patch.object(client, "safe_post", return_value={"ok": True}) as mp:
            assert client.execute_command("nmap target") == {"ok": True}
            mp.assert_called_once_with(
                "api/command", {"command": "nmap target", "use_cache": True}
            )

    def test_without_cache(self, client):
        with patch.object(client, "safe_post", return_value={"ok": True}) as mp:
            assert client.execute_command("nmap target", use_cache=False) == {"ok": True}
            mp.assert_called_once_with(
                "api/command", {"command": "nmap target", "use_cache": False}
            )


# ── check_health ────────────────────────────────────────────────────────────

class TestCheckHealth:
    def test_delegates_to_safe_get(self, client):
        with patch.object(client, "safe_get", return_value={"status": "up"}) as mp:
            assert client.check_health() == {"status": "up"}
            mp.assert_called_once_with("health")
