"""
mcp_core/browser_direct.py

Browser primitives using Playwright (optional, not installed by default).
3 primitives: browser_fetch, browser_screenshot, browser_eval.

Install with: pip install playwright && playwright install chromium
"""

from typing import Any, Dict

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

_HINT = (
    "browser not available — install with: "
    "pip install playwright && playwright install chromium"
)


def _check_playwright() -> dict:
    if not HAS_PLAYWRIGHT:
        return {"success": False, "error": _HINT}
    return {}


def _ensure_page(url: str, wait_selector: str | None = None):
    p = sync_playwright().start()
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(url, wait_until="networkidle")
    if wait_selector:
        page.wait_for_selector(wait_selector, timeout=10000)
    return p, browser, page


def browser_fetch(data: dict) -> dict:
    err = _check_playwright()
    if err:
        return err
    url = data.get("url", "")
    wait_selector = data.get("wait_selector", None)
    if not url:
        return {"success": False, "error": "url is required"}
    try:
        p, browser, page = _ensure_page(url, wait_selector)
        html = page.content()
        title = page.title()
        browser.close()
        p.stop()
        return {"success": True, "html": html, "title": title, "url": url}
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_screenshot(data: dict) -> dict:
    err = _check_playwright()
    if err:
        return err
    url = data.get("url", "")
    selector = data.get("selector", None)
    if not url:
        return {"success": False, "error": "url is required"}
    try:
        p, browser, page = _ensure_page(url)
        if selector:
            el = page.wait_for_selector(selector, timeout=10000)
            screenshot = el.screenshot(full_page=True)
        else:
            screenshot = page.screenshot(full_page=True)
        title = page.title()
        browser.close()
        p.stop()
        import base64
        return {
            "success": True,
            "screenshot_base64": base64.b64encode(screenshot).decode(),
            "title": title,
            "url": url,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_eval(data: dict) -> dict:
    err = _check_playwright()
    if err:
        return err
    url = data.get("url", "")
    js = data.get("js", "")
    if not url:
        return {"success": False, "error": "url is required"}
    if not js:
        return {"success": False, "error": "js is required"}
    try:
        p, browser, page = _ensure_page(url)
        result = page.evaluate(js)
        title = page.title()
        browser.close()
        p.stop()
        return {"success": True, "result": str(result), "title": title, "url": url}
    except Exception as e:
        return {"success": False, "error": str(e)}


_HANDLERS = {
    "browser_fetch": browser_fetch,
    "browser_screenshot": browser_screenshot,
    "browser_eval": browser_eval,
}


def browser_exec(tool: str, data: dict) -> dict:
    handler = _HANDLERS.get(tool)
    if handler is None:
        return {"success": False, "error": f"Unknown browser tool: '{tool}'"}
    return handler(data)
