"""
mcp_core/web_scan_direct.py

Phase 2 — Direct execution layer for web_scan tools.

Bypasses Flask/HTTP entirely. Extracts pure logic from
server_api/web_scan/ and calls execute_command() directly.

Note: burpsuite_alternative is excluded — it depends on
browser_agent and http_framework internal instances.
It will be migrated in Phase 3.

Usage:
    import mcp_core.web_scan_direct as _web_scan_direct
    result = await loop.run_in_executor(
        None, lambda: _web_scan_direct.web_scan_exec("nikto", data)
    )
"""

import logging
from typing import Any, Dict

from server_core.command_executor import execute_command

logger = logging.getLogger(__name__)


def _sanitize(value: str) -> str:
    """Strip shell-dangerous characters from user-supplied free-text parameters."""
    for ch in ("'", "`", ";", "\n", "\r"):
        value = value.replace(ch, "")
    for seq in ("&&", "||", "$(", "${"):
        value = value.replace(seq, "")
    return value


def _require(data: dict, *keys: str) -> Dict[str, Any]:
    for key in keys:
        if not data.get(key, ""):
            return {"success": False, "error": f"'{key}' is required"}
    return {}


# ---------------------------------------------------------------------------
# web_scan/ handlers
# ---------------------------------------------------------------------------

def _nikto(data: dict) -> dict:
    err = _require(data, "target")
    if err:
        return err

    target          = data["target"].strip()
    additional_args = data.get("additional_args", "")

    command = f"nikto -h {target}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _sqlmap(data: dict) -> dict:
    err = _require(data, "url")
    if err:
        return err

    url             = data["url"].strip()
    post_data       = data.get("data", "")
    additional_args = data.get("additional_args", "")

    command = f"sqlmap -u {url} --batch"
    if post_data:       command += f" --data=\"{post_data}\""
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _wpscan(data: dict) -> dict:
    err = _require(data, "url")
    if err:
        return err

    url             = data["url"].strip()
    additional_args = data.get("additional_args", "")

    command = f"wpscan --url {url}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _dalfox(data: dict) -> dict:
    url       = data.get("url", "").strip()
    pipe_mode = data.get("pipe_mode", False)

    if not url and not pipe_mode:
        return {"success": False, "error": "url or pipe_mode is required"}

    mining_dom      = data.get("mining_dom", True)
    mining_dict     = data.get("mining_dict", True)
    blind           = data.get("blind", False)
    custom_payload  = _sanitize(data.get("custom_payload", ""))
    additional_args = data.get("additional_args", "")

    if pipe_mode:
        command = "dalfox pipe"
    else:
        command = f"dalfox url {url}"

    if blind:          command += " --blind"
    if mining_dom:     command += " --mining-dom"
    if mining_dict:    command += " --mining-dict"
    if custom_payload: command += f" --custom-payload '{custom_payload}'"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _jaeles(data: dict) -> dict:
    err = _require(data, "url")
    if err:
        return err

    url             = data["url"].strip()
    signatures      = data.get("signatures", "")
    config          = data.get("config", "")
    threads         = data.get("threads", 20)
    timeout         = data.get("timeout", 20)
    additional_args = data.get("additional_args", "")

    command = f"jaeles scan -u {url} -c {threads} --timeout {timeout}"
    if signatures:      command += f" -s {signatures}"
    if config:          command += f" --config {config}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _xsser(data: dict) -> dict:
    err = _require(data, "url")
    if err:
        return err

    url             = data["url"].strip()
    params_str      = data.get("params", "")
    additional_args = data.get("additional_args", "")

    command = f"xsser --url '{url}'"
    if params_str:      command += f" --param='{params_str}'"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _zap(data: dict) -> dict:
    target      = data.get("target", "").strip()
    scan_type   = data.get("scan_type", "baseline")
    api_key     = data.get("api_key", "")
    daemon      = data.get("daemon", False)
    port        = data.get("port", "8090")
    host        = data.get("host", "0.0.0.0")
    format_type = data.get("format", "xml")
    output_file = data.get("output_file", "")
    additional_args = data.get("additional_args", "")

    if not target and scan_type != "daemon":
        return {"success": False, "error": "target is required for scans"}

    if daemon:
        command = f"zaproxy -daemon -host {host} -port {port}"
        if api_key: command += f" -config api.key={api_key}"
    else:
        command = f"zaproxy -cmd -quickurl {target}"
        if format_type:  command += f" -quickout {format_type}"
        if output_file:  command += f" -quickprogress -dir \"{output_file}\""
        if api_key:      command += f" -config api.key={api_key}"

    if additional_args: command += f" {additional_args}"

    return execute_command(command)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_HANDLERS = {
    "nikto":   _nikto,
    "sqlmap":  _sqlmap,
    "wpscan":  _wpscan,
    "dalfox":  _dalfox,
    "jaeles":  _jaeles,
    "xsser":   _xsser,
    "zap":     _zap,
    # burpsuite-alternative: excluded — depends on browser_agent/http_framework
    # will be migrated in Phase 3
}


def web_scan_exec(tool: str, data: dict) -> dict:
    """
    Execute a web_scan tool directly — no Flask, no HTTP.

    Args:
        tool: tool name (e.g. "nikto", "sqlmap", "dalfox")
        data: parameter dict

    Returns:
        {"success": bool, "output": str, "returncode": int, ...}
    """
    handler = _HANDLERS.get(tool)
    if handler is None:
        return {"success": False, "error": f"Unknown web_scan tool: '{tool}'"}
    return handler(data)
