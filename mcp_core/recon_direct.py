"""
mcp_core/recon_direct.py

Phase 2 — Direct execution layer for recon, dns_enum, and net_lookup tools.

Bypasses Flask/HTTP entirely. Each function extracts the pure logic from
server_api/recon/, server_api/dns_enum/, and server_api/net_lookup/
and calls execute_command() directly.

Usage:
    from mcp_core.recon_direct import recon_exec

    result = await recon_exec("amass", {
        "domain": "example.com",
        "mode": "enum",
    })

All functions return the same dict shape as before:
    {"success": bool, "output": str, "returncode": int, ...}
"""

import logging
import subprocess
from typing import Any, Dict

from server_core.command_executor import execute_command

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _require(data: dict, *keys: str) -> Dict[str, Any]:
    for key in keys:
        if not data.get(key, ""):
            return {"success": False, "error": f"'{key}' is required"}
    return {}


# ---------------------------------------------------------------------------
# recon/ handlers
# ---------------------------------------------------------------------------

def _amass(data: dict) -> dict:
    err = _require(data, "domain")
    if err:
        return err

    domain = data["domain"].strip()
    mode   = data.get("mode", "enum")
    additional_args = data.get("additional_args", "")

    command = f"amass {mode} -d {domain}"
    if additional_args:
        command += f" {additional_args}"

    return execute_command(command)


def _subfinder(data: dict) -> dict:
    err = _require(data, "domain")
    if err:
        return err

    domain      = data["domain"].strip()
    silent      = data.get("silent", True)
    all_sources = data.get("all_sources", False)
    additional_args = data.get("additional_args", "")

    command = f"subfinder -d {domain}"
    if silent:      command += " -silent"
    if all_sources: command += " -all"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _autorecon(data: dict) -> dict:
    err = _require(data, "target")
    if err:
        return err

    target       = data["target"].strip()
    output_dir   = data.get("output_dir", "/tmp/autorecon")
    port_scans   = data.get("port_scans", "default")
    service_scans = data.get("service_scans", "default")
    heartbeat    = data.get("heartbeat", 60)
    timeout      = data.get("timeout", 300)
    additional_args = data.get("additional_args", "")

    command = f"autorecon {target} -o {output_dir} --heartbeat {heartbeat} --timeout {timeout}"
    if port_scans != "default":   command += f" --port-scans {port_scans}"
    if service_scans != "default": command += f" --service-scans {service_scans}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _theharvester(data: dict) -> dict:
    err = _require(data, "domain")
    if err:
        return err

    domain = data["domain"].strip()
    additional_args = data.get("additional_args", "")

    # theHarvester — capital H, sync with upstream fix
    command = f"theHarvester -d {domain} {additional_args}"
    return execute_command(command, use_cache=True)


# ---------------------------------------------------------------------------
# dns_enum/ handlers
# ---------------------------------------------------------------------------

def _dnsenum(data: dict) -> dict:
    err = _require(data, "domain")
    if err:
        return err

    domain     = data["domain"].strip()
    dns_server = data.get("dns_server", "")
    wordlist   = data.get("wordlist", "")
    additional_args = data.get("additional_args", "")

    command = f"dnsenum {domain}"
    if dns_server: command += f" --dnsserver {dns_server}"
    if wordlist:   command += f" --file {wordlist}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _fierce(data: dict) -> dict:
    err = _require(data, "domain")
    if err:
        return err

    domain     = data["domain"].strip()
    dns_server = data.get("dns_server", "")
    additional_args = data.get("additional_args", "")

    command = f"fierce --domain {domain}"
    if dns_server: command += f" --dns-servers {dns_server}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


# ---------------------------------------------------------------------------
# net_lookup/ handlers
# ---------------------------------------------------------------------------

def _whois(data: dict) -> dict:
    err = _require(data, "target")
    if err:
        return err

    target = data["target"].strip()

    try:
        result = subprocess.run(
            ["whois", target],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            text=True
        )
        output = result.stdout if result.returncode == 0 else result.stderr
        return {"success": result.returncode == 0, "output": output}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_HANDLERS = {
    # recon
    "amass":         _amass,
    "subfinder":     _subfinder,
    "autorecon":     _autorecon,
    "theharvester":  _theharvester,
    # dns_enum
    "dnsenum":       _dnsenum,
    "fierce":        _fierce,
    # net_lookup
    "whois":         _whois,
}


def recon_exec(tool: str, data: dict) -> dict:
    """
    Execute a recon/dns_enum/net_lookup tool directly — no Flask, no HTTP.

    Args:
        tool: tool name (e.g. "amass", "subfinder", "whois")
        data: parameter dict (same shape as the old safe_post payload)

    Returns:
        {"success": bool, "output": str, "returncode": int, ...}
    """
    handler = _HANDLERS.get(tool)
    if handler is None:
        return {"success": False, "error": f"Unknown recon tool: '{tool}'"}
    return handler(data)
