"""Tests for raw_tcp — raw TCP socket primitive."""

import socket
from unittest.mock import patch, MagicMock

import pytest

from mcp_core.misc_direct import _raw_tcp


def test_requires_host_and_port():
    result = _raw_tcp({"payload_hex": "00"})
    assert result["success"] is False
    assert "host and port are required" in result.get("error", "")


def test_payload_optional_for_push_banner():
    """Empty payload_hex is allowed — connect + recv only, no send."""
    result = _raw_tcp({"host": "127.0.0.1", "port": 1, "payload_hex": "", "timeout": 2})
    assert not result["success"]  # will fail (connection refused / timeout), but not on validation
    # If we somehow connected, bytes_sent would be 0
    # Just verify the code path didn't reject at validation
    assert "required" not in result.get("error", "")


def test_invalid_hex():
    result = _raw_tcp({"host": "127.0.0.1", "port": 80, "payload_hex": "ZZ"})
    assert result["success"] is False
    assert "invalid payload_hex" in result.get("error", "")


def test_connection_refused():
    result = _raw_tcp({"host": "127.0.0.1", "port": 1, "payload_hex": "00", "timeout": 2})
    assert result["success"] is False
    # error may be "connection refused" or OS-specific variant
    assert "refused" in result.get("error", "").lower() or "denied" in result.get("error", "").lower() or result["error"]


def test_dns_failure():
    result = _raw_tcp({"host": "nonexistent-domain-xyz.invalid", "port": 80, "payload_hex": "00", "timeout": 2})
    assert result["success"] is False


def test_empty_host():
    result = _raw_tcp({"host": "", "port": 80, "payload_hex": "00"})
    assert result["success"] is False


def test_zero_port():
    result = _raw_tcp({"host": "127.0.0.1", "port": 0, "payload_hex": "00"})
    assert result["success"] is False


@patch("mcp_core.misc_direct.socket.socket")
def test_successful_send_recv(mock_socket_cls):
    mock_sock = MagicMock()
    mock_socket_cls.return_value = mock_sock
    mock_sock.recv.side_effect = [b"HTTP/1.1 200 OK\r\n", socket.timeout("timeout")]

    result = _raw_tcp({
        "host": "example.com",
        "port": 80,
        "payload_hex": "474554202f20485454502f312e310d0a0d0a",
        "timeout": 5,
    })

    assert result["success"] is True
    assert result["bytes_sent"] == 18
    assert result["bytes_recv"] == 17
    assert "response_hex" in result
    assert result["duration_ms"] >= 0


@patch("mcp_core.misc_direct.socket.socket")
def test_no_response_empty_hex(mock_socket_cls):
    mock_sock = MagicMock()
    mock_socket_cls.return_value = mock_sock
    mock_sock.recv.side_effect = [b"", socket.timeout("timeout")]

    result = _raw_tcp({
        "host": "example.com",
        "port": 80,
        "payload_hex": "00",
        "timeout": 3,
    })

    assert result["success"] is True
    assert result["response_hex"] == ""
    assert result["bytes_recv"] == 0


@patch("mcp_core.misc_direct.socket.socket")
def test_socket_error_handled(mock_socket_cls):
    mock_sock = MagicMock()
    mock_socket_cls.return_value = mock_sock
    mock_sock.sendall.side_effect = ConnectionResetError("connection reset")

    result = _raw_tcp({
        "host": "example.com",
        "port": 80,
        "payload_hex": "00",
        "timeout": 3,
    })

    assert result["success"] is False
    assert "connection reset" in result.get("error", "").lower()
