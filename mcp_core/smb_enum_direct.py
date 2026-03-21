"""
mcp_core/smb_enum_direct.py

Phase 2 — Direct execution layer for smb_enum tools.
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


def _enum4linux(data: dict) -> dict:
    err = _require(data, "target")
    if err: return err
    target          = data["target"].strip()
    additional_args = data.get("additional_args", "-a")
    return execute_command(f"enum4linux {additional_args} {target}")


def _enum4linux_ng(data: dict) -> dict:
    err = _require(data, "target")
    if err: return err
    target          = data["target"].strip()
    username        = data.get("username", "")
    password        = data.get("password", "")
    domain          = data.get("domain", "")
    shares          = data.get("shares", True)
    users           = data.get("users", True)
    groups          = data.get("groups", True)
    policy          = data.get("policy", True)
    additional_args = data.get("additional_args", "")

    command = f"enum4linux-ng {target}"
    if username: command += f" -u {username}"
    if password: command += f" -p {password}"
    if domain:   command += f" -d {domain}"

    opts = []
    if shares: opts.append("S")
    if users:  opts.append("U")
    if groups: opts.append("G")
    if policy: opts.append("P")
    if opts: command += f" -A {','.join(opts)}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _nbtscan(data: dict) -> dict:
    err = _require(data, "target")
    if err: return err
    target          = data["target"].strip()
    verbose         = data.get("verbose", False)
    timeout         = data.get("timeout", 2)
    additional_args = data.get("additional_args", "")

    command = f"nbtscan -t {timeout}"
    if verbose: command += " -v"
    command += f" {target}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _netexec(data: dict) -> dict:
    err = _require(data, "target")
    if err: return err
    target          = data["target"].strip()
    protocol        = data.get("protocol", "smb")
    username        = data.get("username", "")
    password        = data.get("password", "")
    hash_value      = data.get("hash", "")
    module          = data.get("module", "")
    additional_args = data.get("additional_args", "")

    command = f"nxc {protocol} {target}"
    if username:    command += f" -u {username}"
    if password:    command += f" -p {password}"
    if hash_value:  command += f" -H {hash_value}"
    if module:      command += f" -M {module}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _rpcclient(data: dict) -> dict:
    err = _require(data, "target")
    if err: return err
    target          = data["target"].strip()
    username        = data.get("username", "")
    password        = data.get("password", "")
    domain          = data.get("domain", "")
    commands        = data.get("commands", "enumdomusers;enumdomgroups;querydominfo")
    additional_args = data.get("additional_args", "")

    if username and password:
        auth = f"-U {username}%{password}"
    elif username:
        auth = f"-U {username}"
    else:
        auth = "-U ''"
    if domain: auth += f" -W {domain}"

    command_sequence = commands.replace(";", "\n")
    command = f"echo -e '{command_sequence}' | rpcclient {auth} {target}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _smbmap(data: dict) -> dict:
    err = _require(data, "target")
    if err: return err
    target          = data["target"].strip()
    username        = data.get("username", "")
    password        = data.get("password", "")
    domain          = data.get("domain", "")
    additional_args = data.get("additional_args", "")

    command = f"smbmap -H {target}"
    if username: command += f" -u {username}"
    if password: command += f" -p {password}"
    if domain:   command += f" -d {domain}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


_HANDLERS = {
    "enum4linux":    _enum4linux,
    "enum4linux-ng": _enum4linux_ng,
    "nbtscan":       _nbtscan,
    "netexec":       _netexec,
    "rpcclient":     _rpcclient,
    "smbmap":        _smbmap,
}


def smb_enum_exec(tool: str, data: dict) -> dict:
    """Execute a smb_enum tool directly — no Flask, no HTTP."""
    handler = _HANDLERS.get(tool)
    if handler is None:
        return {"success": False, "error": f"Unknown smb_enum tool: '{tool}'"}
    return handler(data)
