"""
mcp_core/osint_direct.py

Phase 2 — Direct execution layer for OSINT tools.
Covers: sherlock, spiderfoot, sublist3r, parsero

Usage:
    import mcp_core.osint_direct as _osint_direct
    result = await loop.run_in_executor(
        None, lambda: _osint_direct.osint_exec("sherlock", data)
    )
"""

from server_core.command_executor import execute_command


def _require(data: dict, *keys: str) -> dict:
    _HINTS = {"url": "use http://host[:port]", "target": "use an IP or hostname", "domain": "use a domain name like example.com"}
    for key in keys:
        if not data.get(key, ""):
            hint = _HINTS.get(key, "")
            msg = f"'{key}' is required" + (f" ({hint})" if hint else "")
            return {"success": False, "error": msg}
    return {}


def _sherlock(data: dict) -> dict:
    err = _require(data, "username")
    if err:
        return err
    username = data["username"].strip()
    command = f"sherlock {username} --output sherlock_results/{username}.json --json"
    return execute_command(command)


def _spiderfoot(data: dict) -> dict:
    err = _require(data, "target")
    if err:
        return err
    target = data["target"].strip()
    command = f"spiderfoot -s {target}"
    return execute_command(command)


def _sublist3r(data: dict) -> dict:
    err = _require(data, "domain")
    if err:
        return err
    domain  = data["domain"].strip()
    threads = data.get("threads", 3)
    engine  = data.get("engine", "")

    command = f"sublist3r -d {domain} -t {threads}"
    if engine:
        command += f" -e {engine}"
    return execute_command(command, use_cache=True)


def _parsero(data: dict) -> dict:
    err = _require(data, "target")
    if err:
        return err
    target          = data["target"].strip()
    additional_args = data.get("additional_args", "")

    command = f"parsero -u {target}"
    if additional_args:
        command += f" {additional_args}"
    return execute_command(command, use_cache=True)


_HANDLERS = {
    "sherlock":   _sherlock,
    "spiderfoot": _spiderfoot,
    "sublist3r":  _sublist3r,
    "parsero":    _parsero,
}


def osint_exec(tool: str, data: dict) -> dict:
    """Execute an OSINT tool directly"""
    handler = _HANDLERS.get(tool)
    if handler is None:
        return {"success": False, "error": f"Unknown OSINT tool: '{tool}'"}
    return handler(data)
