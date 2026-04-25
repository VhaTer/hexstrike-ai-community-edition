"""
mcp_core/password_cracking_direct.py

Phase 2 — Direct execution layer for password_cracking tools.

Bypasses Flask/HTTP entirely. Extracts pure logic from
server_api/password_cracking/ and calls execute_command() directly.

Usage:
    import mcp_core.password_cracking_direct as _pwdcrack_direct
    result = await loop.run_in_executor(
        None, lambda: _pwdcrack_direct.pwdcrack_exec("hydra", data)
    )
"""

import logging
import os
from typing import Any, Dict

from server_core.command_executor import execute_command
from server_core.singletons import ROCKYOU_PATH

logger = logging.getLogger(__name__)


def _require(data: dict, *keys: str) -> Dict[str, Any]:
    for key in keys:
        if not data.get(key, ""):
            return {"success": False, "error": f"'{key}' is required"}
    return {}


# ---------------------------------------------------------------------------
# password_cracking/ handlers
# ---------------------------------------------------------------------------

def _hydra(data: dict) -> dict:
    target   = data.get("target", "").strip()
    service  = data.get("service", "").strip()

    if not target or not service:
        return {"success": False, "error": "target and service are required"}

    username      = data.get("username", "")
    username_file = data.get("username_file", "")
    password      = data.get("password", "")
    password_file = data.get("password_file", "")
    additional_args = data.get("additional_args", "")

    if not (username or username_file) or not (password or password_file):
        return {"success": False, "error": "username/username_file and password/password_file are required"}

    command = "hydra -t 4"
    if username:       command += f" -l {username}"
    elif username_file: command += f" -L {username_file}"
    if password:       command += f" -p {password}"
    elif password_file: command += f" -P {password_file}"
    if additional_args: command += f" {additional_args}"
    command += f" {target} {service}"

    return execute_command(command)


def _hashcat(data: dict) -> dict:
    hash_file   = data.get("hash_file", "").strip()
    hash_type   = data.get("hash_type", "").strip()

    if not hash_file:
        return {"success": False, "error": "hash_file is required"}
    if not hash_type:
        return {"success": False, "error": "hash_type is required"}

    attack_mode     = data.get("attack_mode", "0")
    wordlist        = data.get("wordlist", ROCKYOU_PATH)
    mask            = data.get("mask", "")
    additional_args = data.get("additional_args", "")

    command = f"hashcat -m {hash_type} -a {attack_mode} {hash_file}"
    if attack_mode == "0" and wordlist: command += f" {wordlist}"
    elif attack_mode == "3" and mask:   command += f" {mask}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _john(data: dict) -> dict:
    err = _require(data, "hash_file")
    if err:
        return err

    hash_file       = data["hash_file"].strip()
    wordlist        = data.get("wordlist", ROCKYOU_PATH)
    format_type     = data.get("format", "")
    additional_args = data.get("additional_args", "")

    command = "john"
    if format_type: command += f" --format={format_type}"
    if wordlist:    command += f" --wordlist={wordlist}"
    if additional_args: command += f" {additional_args}"
    command += f" {hash_file}"

    return execute_command(command)


def _medusa(data: dict) -> dict:
    target = data.get("target", "").strip()
    module = data.get("module", "").strip()

    if not target or not module:
        return {"success": False, "error": "target and module are required"}

    username      = data.get("username", "")
    username_file = data.get("username_file", "")
    password      = data.get("password", "")
    password_file = data.get("password_file", "")
    additional_args = data.get("additional_args", "")

    command = f"medusa -h {target} -M {module}"
    if username:        command += f" -u {username}"
    elif username_file: command += f" -U {username_file}"
    if password:        command += f" -p {password}"
    elif password_file: command += f" -P {password_file}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _patator(data: dict) -> dict:
    module = data.get("module", "").strip()
    target = data.get("target", "").strip()

    if not module or not target:
        return {"success": False, "error": "module and target are required"}

    username      = data.get("username", "")
    username_file = data.get("username_file", "")
    password      = data.get("password", "")
    password_file = data.get("password_file", "")
    additional_args = data.get("additional_args", "")

    if username and username_file:
        return {"success": False, "error": "Specify only one of username or username_file"}
    if password and password_file:
        return {"success": False, "error": "Specify only one of password or password_file"}

    if username_file and not os.path.isfile(username_file):
        return {"success": False, "error": f"username_file not found: {username_file}"}
    if password_file and not os.path.isfile(password_file):
        return {"success": False, "error": f"password_file not found: {password_file}"}

    command = f"patator {module} host={target}"
    if username:        command += f" user={username}"
    elif username_file: command += f" user=FILE:{username_file}"
    if password:        command += f" password={password}"
    elif password_file: command += f" password=FILE:{password_file}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _hashid(data: dict) -> dict:
    hash_value = (data.get("hash_value") or data.get("hash") or "").strip()
    if not hash_value:
        return {"success": False, "error": "'hash_value' is required"}

    additional_args = data.get("additional_args", "")

    command = f"hashid {hash_value} {additional_args}"
    return execute_command(command, use_cache=True)


def _ophcrack(data: dict) -> dict:
    hash_file  = data.get("hash_file", "").strip()
    tables_dir = data.get("tables_dir", "")
    tables     = data.get("tables", "")
    additional_args = data.get("additional_args", "")

    if not hash_file:
        return {"success": False, "error": "hash_file is required"}
    if not os.path.isfile(hash_file):
        return {"success": False, "error": f"Hash file not found: {hash_file}"}
    if tables_dir and not os.path.isdir(tables_dir):
        return {"success": False, "error": f"tables_dir not found: {tables_dir}"}

    command = "ophcrack -g"
    if tables_dir: command += f" -d {tables_dir}"
    if tables:     command += f" -t {tables}"
    command += f" -f {hash_file}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _aircrack_ng(data: dict) -> dict:
    capture_files = data.get("capture_files", [])
    wordlist      = data.get("wordlist", "")
    bssid         = data.get("bssid", "")

    if not capture_files:
        return {"success": False, "error": "capture_files is required"}
    if not wordlist:
        return {"success": False, "error": "wordlist is required"}

    command = f"aircrack-ng {' '.join(capture_files)} -w {wordlist}"
    if bssid: command += f" -b {bssid}"

    return execute_command(command, use_cache=False)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_HANDLERS = {
    "hydra":       _hydra,
    "hashcat":     _hashcat,
    "john":        _john,
    "medusa":      _medusa,
    "patator":     _patator,
    "hashid":      _hashid,
    "ophcrack":    _ophcrack,
    "aircrack_ng": _aircrack_ng,
}


def pwdcrack_exec(tool: str, data: dict) -> dict:
    """
    Execute a password_cracking tool directly — no Flask, no HTTP.

    Args:
        tool: tool name (e.g. "hydra", "hashcat", "john")
        data: parameter dict

    Returns:
        {"success": bool, "output": str, "returncode": int, ...}
    """
    handler = _HANDLERS.get(tool)
    if handler is None:
        return {"success": False, "error": f"Unknown password_cracking tool: '{tool}'"}
    return handler(data)
