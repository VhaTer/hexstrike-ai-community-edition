"""Tests for http_request — generic HTTP client primitive."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from mcp_core.misc_direct import _http_request


def _make_mock_namedtemp(name):
    """Helper: create a MagicMock that behaves like a closed NamedTemporaryFile."""
    m = MagicMock()
    m.name = name
    m.close = MagicMock()
    return m


def test_get_returns_body():
    """GET example.com returns HTML body with success=True."""
    result = _http_request({"url": "http://example.com", "method": "GET"})
    assert result["success"] is True
    assert result["status_code"] == 200
    assert result["ok"] is True
    assert "Example Domain" in result["body"]
    assert "headers" in result
    assert result["body_truncated"] is False


def test_body_truncation():
    """max_body_size truncates body and sets body_truncated flag."""
    result = _http_request({
        "url": "http://example.com",
        "method": "GET",
        "max_body_size": 50,
    })
    assert result["body_truncated"] is True
    assert len(result["body"]) <= 50


def test_invalid_url_returns_error():
    """Invalid URL returns success=False, not an exception."""
    result = _http_request({
        "url": "http://nonexistent.invalid",
        "method": "GET",
    })
    assert result["success"] is False


def test_empty_url_returns_error():
    """Empty url returns early error."""
    result = _http_request({"url": "", "method": "GET"})
    assert result["success"] is False
    assert "url is required" in result.get("error", "")


def test_post_returns_structured():
    """POST with form data returns structured response."""
    result = _http_request({
        "url": "https://httpbin.org/post",
        "method": "POST",
        "data": "key=value",
    })
    if not result.get("ok"):
        pytest.skip("httpbin.org unavailable")
    assert result["status_code"] == 200
    assert "form" in result["body"] or "key" in result["body"]


def test_custom_headers():
    """Custom headers are sent and reflected in response."""
    result = _http_request({
        "url": "https://httpbin.org/headers",
        "method": "GET",
        "headers": "X-Test: hello\nX-Another: world",
    })
    if not result.get("ok"):
        pytest.skip("httpbin.org unavailable")
    body = result["body"]
    assert "X-Test" in body or "x-test" in body.lower()


def test_cookie_passthrough():
    """Sending a cookie includes it in the request."""
    result = _http_request({
        "url": "https://httpbin.org/cookies",
        "method": "GET",
        "cookie": "PHPSESSID=abc123",
    })
    if not result.get("ok"):
        pytest.skip("httpbin.org unavailable")
    body = result["body"]
    assert "PHPSESSID" in body and "abc123" in body


def test_no_follow_redirects():
    """follow_redirects=False stops at redirect status."""
    result = _http_request({
        "url": "http://httpbin.org/redirect/1",
        "method": "GET",
        "follow_redirects": False,
    })
    if not result.get("success"):
        pytest.skip("httpbin.org unavailable")
    if not result.get("ok") and result["status_code"] == 0:
        pytest.skip("httpbin.org unavailable")
    if result.get("status_code") == 503:
        pytest.skip("httpbin.org returned 503")
    assert result["status_code"] in (302, 301, 307, 308)


def test_get_sets_ok_for_2xx():
    """ok is True for 2xx, False otherwise."""
    ok_result = _http_request({"url": "http://example.com", "method": "GET"})
    assert ok_result["ok"] is True


def test_headers_parsed_to_dict():
    """Response headers returned as dict."""
    result = _http_request({"url": "http://example.com", "method": "GET"})
    assert isinstance(result["headers"], dict)
    assert "Content-Type" in result["headers"] or "content-type" in {k.lower(): v for k, v in result["headers"].items()}


def test_binary_body_returns_hex():
    """Binary response populates body_hex for non-UTF-8 content."""
    hf_path = tempfile.mktemp(suffix=".hdr")
    bf_path = tempfile.mktemp(suffix=".bdy")
    try:
        with open(hf_path, "w") as f:
            f.write("Content-Type: image/png\r\n")
        with open(bf_path, "wb") as f:
            f.write(bytes(range(256)))

        with patch("mcp_core.misc_direct.execute_command") as mock_exec:
            mock_exec.return_value = {"stdout": "200", "stderr": ""}
            with patch("mcp_core.misc_direct.tempfile.NamedTemporaryFile") as mock_tmp:
                mock_tmp.side_effect = [
                    _make_mock_namedtemp(hf_path),
                    _make_mock_namedtemp(bf_path),
                ]
                result = _http_request({"url": "http://example.com/img", "method": "GET"})

        assert result["success"] is True
        assert result["body_hex"] != ""
        assert result["body_hex"] == bytes(range(256)).hex()
    finally:
        for p in (hf_path, bf_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def test_timeout_configurable():
    """timeout parameter is passed to execute_command."""
    hf_path = tempfile.mktemp(suffix=".hdr")
    bf_path = tempfile.mktemp(suffix=".bdy")
    try:
        with open(hf_path, "w") as f:
            f.write("Content-Type: text/plain\r\n")
        with open(bf_path, "wb") as f:
            f.write(b"hello")

        with patch("mcp_core.misc_direct.execute_command") as mock_exec:
            mock_exec.return_value = {"stdout": "200", "stderr": ""}
            with patch("mcp_core.misc_direct.tempfile.NamedTemporaryFile") as mock_tmp:
                mock_tmp.side_effect = [
                    _make_mock_namedtemp(hf_path),
                    _make_mock_namedtemp(bf_path),
                ]
                _http_request({"url": "http://example.com", "timeout": 120})

            assert mock_exec.called
            _, kwargs = mock_exec.call_args
            assert kwargs.get("timeout") == 120
    finally:
        for p in (hf_path, bf_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def test_output_key_present():
    """Response includes 'output' key for _normalize_tool_result."""
    result = _http_request({"url": "http://example.com", "method": "GET"})
    assert "output" in result
    assert "HTTP 200" in result["output"]
