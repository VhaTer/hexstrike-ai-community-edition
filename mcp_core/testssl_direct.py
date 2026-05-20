"""
mcp_core/testssl_direct.py

Phase 2 — Direct execution layer for testssl.sh SSL/TLS analysis.

Bypasses Flask/HTTP entirely. Calls testssl.sh directly via execute_command().

Usage:
    import mcp_core.testssl_direct as _testssl_direct
    result = await loop.run_in_executor(
        None, lambda: _testssl_direct.testssl_exec("testssl", data)
    )

testssl.sh installation:
    apt install testssl.sh
    or: https://testssl.sh/
"""

import logging
from typing import Any, Dict

from server_core.command_executor import execute_command

logger = logging.getLogger(__name__)


def _require(data: dict, *keys: str) -> Dict[str, Any]:
    _HINTS = {"url": "use http://host[:port]", "target": "use an IP or hostname", "domain": "use a domain name like example.com"}
    for key in keys:
        if not data.get(key, ""):
            hint = _HINTS.get(key, "")
            msg = f"'{key}' is required" + (f" ({hint})" if hint else "")
            return {"success": False, "error": msg}
    return {}


# ---------------------------------------------------------------------------
# testssl.sh handler
# ---------------------------------------------------------------------------

def _testssl(data: dict) -> dict:
    """
    Run testssl.sh SSL/TLS analysis on a target host.

    Common parameters:
        target:          Host[:port] to scan (e.g. 'example.com' or 'example.com:443')
        protocols:       Test protocols (default True)
        server_defaults: Test server defaults (default True)
        vulnerable:      Test for known vulnerabilities (default False)
        full:            Run all checks (default False)
        headers:         Test HTTP security headers (default False)
        severity:        Minimum severity to report (LOW/MEDIUM/HIGH/CRITICAL)
        starttls:        STARTTLS protocol (smtp/pop3/imap/ftp/xmpp/telnet/ldap/nntp/postgres/mysql)
        json_output:     Output results as JSON (default False)
        quiet:           Suppress banner (default True)
        additional_args: Raw extra flags passed to testssl.sh
    """
    err = _require(data, "target")
    if err:
        return err

    target = data["target"].strip()

    # Build command
    command = "testssl.sh"

    # Quiet mode — suppress banner by default
    if data.get("quiet", True):
        command += " --quiet"

    # Protocol checks
    if data.get("protocols", True):
        command += " --protocols"

    # Server defaults
    if data.get("server_defaults", True):
        command += " --server-defaults"

    # Optional checks
    if data.get("server_preference", False):
        command += " --server-preference"

    if data.get("forward_secrecy", False):
        command += " --fs"

    if data.get("headers", False):
        command += " --headers"

    if data.get("vulnerable", False):
        command += " --vulnerable"

    if data.get("full", False):
        command += " --full"

    if data.get("client_simulation", False):
        command += " --client-simulation"

    if data.get("rating_only", False):
        command += " --rating"

    # STARTTLS
    starttls = data.get("starttls", "")
    if starttls:
        command += f" --starttls {starttls}"

    # Severity filter
    severity = data.get("severity", "")
    if severity:
        command += f" --severity {severity.upper()}"

    # JSON output
    if data.get("json_output", False):
        jsonfile = data.get("jsonfile", "/tmp/testssl_output.json")
        command += f" --jsonfile {jsonfile}"

    # Proxy
    proxy = data.get("proxy", "")
    if proxy:
        command += f" --proxy {proxy}"

    # IP version
    if data.get("ipv4_only", False):
        command += " -4"
    elif data.get("ipv6_only", False):
        command += " -6"

    # Specific IP
    ip = data.get("ip", "")
    if ip:
        command += f" --ip {ip}"

    # Additional raw args
    additional_args = data.get("additional_args", "")
    if additional_args:
        command += f" {additional_args}"

    # Target is always last
    command += f" {target}"

    logger.debug(f"🔐 testssl.sh → {target}")
    return execute_command(command)


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

_HANDLERS = {
    "testssl": _testssl,
}


def testssl_exec(tool: str, data: dict) -> dict:
    """Entry point for run_security_tool() dispatch."""
    handler = _HANDLERS.get(tool)
    if handler is None:
        return {"success": False, "error": f"Unknown testssl tool: '{tool}'"}
    return handler(data)
