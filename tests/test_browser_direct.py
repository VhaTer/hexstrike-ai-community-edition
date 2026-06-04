"""Tests for browser_direct — optional Playwright browser primitives."""

from unittest.mock import patch, MagicMock

import pytest

from mcp_core.browser_direct import (
    browser_exec, browser_fetch, browser_screenshot, browser_eval,
    _HANDLERS,
)


def test_unknown_tool():
    result = browser_exec("nonexistent", {})
    assert result["success"] is False
    assert "Unknown browser tool" in result.get("error", "")


def test_browser_not_available_without_playwright():
    with patch("mcp_core.browser_direct.HAS_PLAYWRIGHT", False):
        result = browser_fetch({"url": "http://example.com"})
        assert result["success"] is False
        assert "browser not available" in result.get("error", "")


def test_screenshot_not_available_without_playwright():
    with patch("mcp_core.browser_direct.HAS_PLAYWRIGHT", False):
        result = browser_screenshot({"url": "http://example.com"})
        assert result["success"] is False
        assert "browser not available" in result.get("error", "")


def test_eval_not_available_without_playwright():
    with patch("mcp_core.browser_direct.HAS_PLAYWRIGHT", False):
        result = browser_eval({"url": "http://example.com", "js": "1+1"})
        assert result["success"] is False
        assert "browser not available" in result.get("error", "")


def test_fetch_requires_url():
    with patch("mcp_core.browser_direct.HAS_PLAYWRIGHT", True):
        result = browser_fetch({"url": ""})
        assert result["success"] is False
        assert "url is required" in result.get("error", "")


def test_screenshot_requires_url():
    with patch("mcp_core.browser_direct.HAS_PLAYWRIGHT", True):
        result = browser_screenshot({"url": ""})
        assert result["success"] is False
        assert "url is required" in result.get("error", "")


def test_eval_requires_url():
    with patch("mcp_core.browser_direct.HAS_PLAYWRIGHT", True):
        result = browser_eval({"url": "", "js": "1+1"})
        assert result["success"] is False
        assert "url is required" in result.get("error", "")


def test_eval_requires_js():
    with patch("mcp_core.browser_direct.HAS_PLAYWRIGHT", True):
        result = browser_eval({"url": "http://example.com", "js": ""})
        assert result["success"] is False
        assert "js is required" in result.get("error", "")


def test_hint_contains_install_instructions():
    from mcp_core.browser_direct import _HINT
    assert "pip install playwright" in _HINT
    assert "playwright install chromium" in _HINT


@patch("mcp_core.browser_direct.sync_playwright", create=True)
@patch("mcp_core.browser_direct.HAS_PLAYWRIGHT", True)
def test_fetch_successful(mock_pw):
    mock_page = MagicMock()
    mock_page.content.return_value = "<html><body>Hello</body></html>"
    mock_page.title.return_value = "Test Page"
    mock_browser = MagicMock()
    mock_browser.new_page.return_value = mock_page
    mock_p_instance = MagicMock()
    mock_p_instance.chromium.launch.return_value = mock_browser
    mock_pw.return_value.start.return_value = mock_p_instance

    result = browser_fetch({"url": "http://example.com"})

    assert result["success"] is True
    assert "Hello" in result["html"]
    assert result["title"] == "Test Page"


@patch("mcp_core.browser_direct.sync_playwright", create=True)
@patch("mcp_core.browser_direct.HAS_PLAYWRIGHT", True)
def test_screenshot_successful(mock_pw):
    mock_page = MagicMock()
    mock_page.screenshot.return_value = b"PNG-DATA"
    mock_page.title.return_value = "Screenshot Page"
    mock_browser = MagicMock()
    mock_browser.new_page.return_value = mock_page
    mock_p_instance = MagicMock()
    mock_p_instance.chromium.launch.return_value = mock_browser
    mock_pw.return_value.start.return_value = mock_p_instance

    result = browser_screenshot({"url": "http://example.com"})

    assert result["success"] is True
    assert "screenshot_base64" in result
    assert result["title"] == "Screenshot Page"


@patch("mcp_core.browser_direct.sync_playwright", create=True)
@patch("mcp_core.browser_direct.HAS_PLAYWRIGHT", True)
def test_eval_successful(mock_pw):
    mock_page = MagicMock()
    mock_page.evaluate.return_value = 42
    mock_page.title.return_value = "Eval Page"
    mock_browser = MagicMock()
    mock_browser.new_page.return_value = mock_page
    mock_p_instance = MagicMock()
    mock_p_instance.chromium.launch.return_value = mock_browser
    mock_pw.return_value.start.return_value = mock_p_instance

    result = browser_eval({"url": "http://example.com", "js": "1+1"})

    assert result["success"] is True
    assert result["result"] == "42"
    assert result["title"] == "Eval Page"


@patch("mcp_core.browser_direct.sync_playwright", create=True)
@patch("mcp_core.browser_direct.HAS_PLAYWRIGHT", True)
def test_eval_with_wait_selector(mock_pw):
    mock_page = MagicMock()
    mock_page.evaluate.return_value = "token-abc"
    mock_page.title.return_value = "Auth Page"
    mock_browser = MagicMock()
    mock_browser.new_page.return_value = mock_page
    mock_p_instance = MagicMock()
    mock_p_instance.chromium.launch.return_value = mock_browser
    mock_pw.return_value.start.return_value = mock_p_instance

    result = browser_eval({"url": "http://example.com", "js": "document.querySelector('#token').innerText"})

    assert result["success"] is True
    assert result["result"] == "token-abc"
