"""
mcp_core/web_probe_direct.py

Phase 2 — Direct execution layer for web probing tools:
  - whatweb   : web technology fingerprinting
  - commix    : command injection exploitation
  - joomscan  : Joomla vulnerability scanner

Bypasses Flask/HTTP entirely. Calls tools directly via execute_command().

Usage:
    import mcp_core.web_probe_direct as _web_probe_direct
    result = await loop.run_in_executor(
        None, lambda: _web_probe_direct.web_probe_exec("whatweb", data)
    )
"""

import logging
from typing import Any, Dict

from server_core.command_executor import execute_command

logger = logging.getLogger(__name__)


def _require(data: dict, *keys: str) -> Dict[str, Any]:
    for key in keys:
        if not data.get(key, ""):
            return {"success": False, "error": f"'{key}' is required"}
    return {}


# ---------------------------------------------------------------------------
# whatweb handler
# ---------------------------------------------------------------------------

def _whatweb(data: dict) -> dict:
    """
    WhatWeb — web technology fingerprinting.

    Parameters:
        url:             Target URL (required)
        aggression:      Aggression level 1-4 (default 1 — stealthy)
        log_brief:       Output brief results (default True)
        no_errors:       Suppress error messages (default False)
        additional_args: Raw extra flags
    """
    err = _require(data, "url")
    if err:
        return err

    url        = data["url"].strip()
    aggression = data.get("aggression", 1)
    additional_args = data.get("additional_args", "")

    command = f"whatweb --aggression={aggression}"

    if data.get("log_brief", True):
        command += " --log-brief=/dev/stdout"

    if data.get("no_errors", False):
        command += " --no-errors"

    if additional_args:
        command += f" {additional_args}"

    command += f" {url}"

    logger.debug(f"🔍 whatweb → {url}")
    return execute_command(command)


# ---------------------------------------------------------------------------
# commix handler
# ---------------------------------------------------------------------------

def _commix(data: dict) -> dict:
    """
    Commix — automated command injection exploitation.

    Parameters:
        url:             Target URL (required)
        level:           Testing level (1-3, default 1)
        technique:       Injection technique (classic/time/file-based)
        os:              Target OS (unix/windows)
        batch:           Non-interactive mode (default True)
        crawl:           Crawl depth (0 = disabled)
        cookie:          HTTP Cookie header
        header:          Extra HTTP header
        proxy:           HTTP proxy URL
        additional_args: Raw extra flags
    """
    err = _require(data, "url")
    if err:
        return err

    url   = data["url"].strip()
    level = data.get("level", "")
    additional_args = data.get("additional_args", "")

    command = "commix"

    # Non-interactive by default
    if data.get("batch", True):
        command += " --batch"

    if level:
        command += f" --level={level}"

    technique = data.get("technique", "")
    if technique:
        command += f" --technique={technique}"

    os_target = data.get("os", "")
    if os_target:
        command += f" --os={os_target}"

    crawl = data.get("crawl", 0)
    if crawl:
        command += f" --crawl={crawl}"

    cookie = data.get("cookie", "")
    if cookie:
        command += f" --cookie='{cookie}'"

    header = data.get("header", "")
    if header:
        command += f" --header='{header}'"

    proxy = data.get("proxy", "")
    if proxy:
        command += f" --proxy={proxy}"

    if additional_args:
        command += f" {additional_args}"

    command += f" --url={url}"

    logger.debug(f"💉 commix → {url}")
    return execute_command(command)


# ---------------------------------------------------------------------------
# joomscan handler
# ---------------------------------------------------------------------------

def _joomscan(data: dict) -> dict:
    """
    JoomScan — Joomla CMS vulnerability scanner.

    Parameters:
        url:             Target Joomla URL (required)
        enum_components: Enumerate installed components (default True)
        random_agent:    Use random user agent (default False)
        cookie:          HTTP Cookie header
        additional_args: Raw extra flags
    """
    err = _require(data, "url")
    if err:
        return err

    url = data["url"].strip()
    additional_args = data.get("additional_args", "")

    command = f"joomscan --url {url}"

    if data.get("enum_components", True):
        command += " --enumerate-components"

    if data.get("random_agent", False):
        command += " --random-agent"

    cookie = data.get("cookie", "")
    if cookie:
        command += f" --cookie '{cookie}'"

    if additional_args:
        command += f" {additional_args}"

    logger.debug(f"🔍 joomscan → {url}")
    return execute_command(command)


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

_HANDLERS = {
    "whatweb":  _whatweb,
    "commix":   _commix,
    "joomscan": _joomscan,
}


def web_probe_exec(tool: str, data: dict) -> dict:
    """Entry point for run_security_tool() dispatch."""
    handler = _HANDLERS.get(tool)
    if handler is None:
        return {"success": False, "error": f"Unknown web_probe tool: '{tool}'"}
    return handler(data)
