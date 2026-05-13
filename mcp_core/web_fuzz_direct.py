"""
mcp_core/web_fuzz_direct.py

Phase 2 — Direct execution layer for web_fuzz tools.

Bypasses Flask/HTTP entirely. Extracts pure logic from
server_api/web_fuzz/ and calls execute_command() directly.

Usage:
    import mcp_core.web_fuzz_direct as _web_fuzz_direct
    result = await loop.run_in_executor(
        None, lambda: _web_fuzz_direct.web_fuzz_exec("gobuster", data)
    )
"""

from typing import Any, Dict

from server_core.command_executor import execute_command
from server_core.singletons import COMMON_DIRB_PATH, COMMON_DIRSEARCH_PATH


def _require(data: dict, *keys: str) -> Dict[str, Any]:
    for key in keys:
        if not data.get(key, ""):
            return {"success": False, "error": f"'{key}' is required"}
    return {}


# ---------------------------------------------------------------------------
# web_fuzz/ handlers
# ---------------------------------------------------------------------------

def _gobuster(data: dict) -> dict:
    err = _require(data, "url")
    if err:
        return err

    url             = data["url"].strip()
    mode            = data.get("mode", "dir")
    wordlist        = data.get("wordlist", COMMON_DIRB_PATH)
    additional_args = data.get("additional_args", "")

    if mode not in ["dir", "dns", "fuzz", "vhost"]:
        return {"success": False, "error": f"Invalid mode: {mode}. Must be: dir, dns, fuzz, vhost"}

    command = f"gobuster {mode} -u {url} -w {wordlist}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _ffuf(data: dict) -> dict:
    err = _require(data, "url")
    if err:
        return err

    url             = data["url"].strip()
    wordlist        = data.get("wordlist", COMMON_DIRB_PATH)
    mode            = data.get("mode", "directory")
    match_codes     = data.get("match_codes", "200,204,301,302,307,401,403")
    additional_args = data.get("additional_args", "")

    command = "ffuf"
    if mode == "directory":
        fuzz_url = url if "/FUZZ" in url else f"{url}/FUZZ"
        command += f" -u {fuzz_url} -w {wordlist}"
    elif mode == "vhost":
        command += f" -u {url} -H 'Host: FUZZ' -w {wordlist}"
    elif mode == "parameter":
        command += f" -u {url}?FUZZ=value -w {wordlist}"
    else:
        command += f" -u {url} -w {wordlist}"

    command += f" -mc {match_codes}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _feroxbuster(data: dict) -> dict:
    err = _require(data, "url")
    if err:
        return err

    url             = data["url"].strip()
    wordlist        = data.get("wordlist", COMMON_DIRB_PATH)
    threads         = data.get("threads", 10)
    additional_args = data.get("additional_args", "")

    command = f"feroxbuster -u {url} -w {wordlist} -t {threads}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _dirsearch(data: dict) -> dict:
    err = _require(data, "url")
    if err:
        return err

    url             = data["url"].strip()
    extensions      = data.get("extensions", "php,html,js,txt,xml,json")
    wordlist        = data.get("wordlist", COMMON_DIRSEARCH_PATH)
    threads         = data.get("threads", 30)
    recursive       = data.get("recursive", False)
    additional_args = data.get("additional_args", "")

    command = f"dirsearch -u {url} -e {extensions} -w {wordlist} -t {threads}"
    if recursive:       command += " -r"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _dirb(data: dict) -> dict:
    err = _require(data, "url")
    if err:
        return err

    url             = data["url"].strip()
    wordlist        = data.get("wordlist", COMMON_DIRB_PATH)
    additional_args = data.get("additional_args", "")

    command = f"dirb {url} {wordlist}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _wfuzz(data: dict) -> dict:
    err = _require(data, "url")
    if err:
        return err

    url             = data["url"].strip()
    wordlist        = data.get("wordlist", COMMON_DIRB_PATH)
    additional_args = data.get("additional_args", "")

    command = f"wfuzz -w {wordlist} '{url}'"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _dotdotpwn(data: dict) -> dict:
    err = _require(data, "target")
    if err:
        return err

    target          = data["target"].strip()
    module          = data.get("module", "http")
    additional_args = data.get("additional_args", "")

    command = f"dotdotpwn -m {module} -h {target}"
    if additional_args: command += f" {additional_args}"
    command += " -b"

    return execute_command(command)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_HANDLERS = {
    "gobuster":    _gobuster,
    "ffuf":        _ffuf,
    "feroxbuster": _feroxbuster,
    "dirsearch":   _dirsearch,
    "dirb":        _dirb,
    "wfuzz":       _wfuzz,
    "dotdotpwn":   _dotdotpwn,
}


def web_fuzz_exec(tool: str, data: dict) -> dict:
    """
    Execute a web_fuzz tool directly — no Flask, no HTTP.

    Args:
        tool: tool name (e.g. "gobuster", "ffuf", "feroxbuster")
        data: parameter dict

    Returns:
        {"success": bool, "output": str, "returncode": int, ...}
    """
    handler = _HANDLERS.get(tool)
    if handler is None:
        return {"success": False, "error": f"Unknown web_fuzz tool: '{tool}'"}
    return handler(data)
