"""
mcp_core/net_scan_direct.py

Phase 2 — Direct execution layer for net_scan tools.

Bypasses Flask/HTTP entirely. Extracts pure logic from
server_api/net_scan/ and calls execute_command() directly.

Usage:
    import mcp_core.net_scan_direct as _net_scan_direct
    result = await loop.run_in_executor(
        None, lambda: _net_scan_direct.net_scan_exec("nmap", data)
    )
"""

from typing import Any, Dict

from server_core.command_executor import execute_command


def _require(data: dict, *keys: str) -> Dict[str, Any]:
    _HINTS = {"url": "use http://host[:port]", "target": "use an IP or hostname", "domain": "use a domain name like example.com"}
    for key in keys:
        if not data.get(key, ""):
            hint = _HINTS.get(key, "")
            msg = f"'{key}' is required" + (f" ({hint})" if hint else "")
            return {"success": False, "error": msg}
    return {}


# ---------------------------------------------------------------------------
# net_scan/ handlers
# ---------------------------------------------------------------------------

def _nmap(data: dict) -> dict:
    err = _require(data, "target")
    if err:
        return err

    target          = data["target"].strip()
    scan_type       = data.get("scan_type", "-sCV")
    ports           = data.get("ports", "")
    additional_args = data.get("additional_args", "-T4 -Pn")

    command = f"nmap {scan_type}"
    if ports:           command += f" -p {ports}"
    if additional_args: command += f" {additional_args}"
    command += f" {target}"

    return execute_command(command)


def _nmap_advanced(data: dict) -> dict:
    err = _require(data, "target")
    if err:
        return err

    target           = data["target"].strip()
    scan_type        = data.get("scan_type", "-sS")
    ports            = data.get("ports", "")
    timing           = data.get("timing", "T4")
    nse_scripts      = data.get("nse_scripts", "")
    os_detection     = data.get("os_detection", False)
    version_detection = data.get("version_detection", False)
    aggressive       = data.get("aggressive", False)
    stealth          = data.get("stealth", False)
    additional_args  = data.get("additional_args", "")

    command = f"nmap {scan_type} {target}"
    if ports: command += f" -p {ports}"

    if stealth:
        command += " -T2 -f --mtu 24"
    else:
        command += f" -{timing}"

    if os_detection:      command += " -O"
    if version_detection: command += " -sV"
    if aggressive:        command += " -A"

    if nse_scripts:
        command += f" --script={nse_scripts}"
    elif not aggressive:
        command += " --script=default,discovery,safe"

    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _masscan(data: dict) -> dict:
    err = _require(data, "target")
    if err:
        return err

    target          = data["target"].strip()
    ports           = data.get("ports", "1-65535")
    rate            = data.get("rate", 1000)
    interface       = data.get("interface", "")
    router_mac      = data.get("router_mac", "")
    source_ip       = data.get("source_ip", "")
    banners         = data.get("banners", False)
    additional_args = data.get("additional_args", "")

    command = f"masscan {target} -p{ports} --rate={rate}"
    if interface:       command += f" -e {interface}"
    if router_mac:      command += f" --router-mac {router_mac}"
    if source_ip:       command += f" --source-ip {source_ip}"
    if banners:         command += " --banners"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _rustscan(data: dict) -> dict:
    err = _require(data, "target")
    if err:
        return err

    target          = data["target"].strip()
    ports           = data.get("ports", "")
    ulimit          = data.get("ulimit", 5000)
    batch_size      = data.get("batch_size", 4500)
    timeout         = data.get("timeout", 1500)
    scripts         = data.get("scripts", "")
    additional_args = data.get("additional_args", "")

    command = f"rustscan -a {target} --ulimit {ulimit} -b {batch_size} -t {timeout}"
    if ports:           command += f" -p {ports}"
    if scripts:         command += " -- -sC -sV"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _arp_scan(data: dict) -> dict:
    target        = data.get("target", "").strip()
    local_network = data.get("local_network", False)

    if not target and not local_network:
        return {"success": False, "error": "target or local_network flag is required"}

    interface       = data.get("interface", "")
    timeout         = data.get("timeout", 500)
    retry           = data.get("retry", 3)
    additional_args = data.get("additional_args", "")

    command = f"arp-scan -t {timeout} -r {retry}"
    if interface:     command += f" -I {interface}"
    if local_network: command += " -l"
    else:             command += f" {target}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_HANDLERS = {
    "nmap":          _nmap,
    "nmap-advanced": _nmap_advanced,
    "masscan":       _masscan,
    "rustscan":      _rustscan,
    "arp-scan":      _arp_scan,
}


def net_scan_exec(tool: str, data: dict) -> dict:
    """
    Execute a net_scan tool directly — no Flask, no HTTP.

    Args:
        tool: tool name (e.g. "nmap", "masscan", "rustscan")
        data: parameter dict

    Returns:
        {"success": bool, "output": str, "returncode": int, ...}
    """
    handler = _HANDLERS.get(tool)
    if handler is None:
        return {"success": False, "error": f"Unknown net_scan tool: '{tool}'"}
    return handler(data)
