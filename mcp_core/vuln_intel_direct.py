"""
mcp_core/vuln_intel_direct.py

Phase 2 — Direct execution layer for vulnerability intelligence tools:
  - vulnx : CVE vulnerability intelligence and analysis

Bypasses Flask/HTTP entirely. Calls tools directly via execute_command().

Usage:
    import mcp_core.vuln_intel_direct as _vuln_intel_direct
    result = await loop.run_in_executor(
        None, lambda: _vuln_intel_direct.vuln_intel_exec("vulnx", data)
    )

vulnx installation:
    pip install vulnx
    or: https://github.com/anouarbensaad/vulnx
"""

import logging
from typing import Any, Dict

from server_core.command_executor import execute_command

logger = logging.getLogger(__name__)


def _require_one(data: dict, *keys: str) -> Dict[str, Any]:
    """Require at least one of the given keys to be non-empty."""
    for key in keys:
        if data.get(key, ""):
            return {}
    return {"success": False, "error": f"At least one of {list(keys)} is required"}


# ---------------------------------------------------------------------------
# vulnx handler
# ---------------------------------------------------------------------------

def _vulnx(data: dict) -> dict:
    """
    Vulnx — CVE vulnerability intelligence and analysis.

    Parameters:
        cve_id:          CVE identifier (e.g. 'CVE-2021-44228') — required if no search
        search:          Search keyword (e.g. 'apache log4j') — required if no cve_id
        auth_key:        API authentication key (optional)
        output:          Output format (json/table, default table)
        additional_args: Raw extra flags
    """
    err = _require_one(data, "cve_id", "search")
    if err:
        return err

    cve_id   = data.get("cve_id", "").strip()
    search   = data.get("search", "").strip()
    auth_key = data.get("auth_key", "").strip()
    additional_args = data.get("additional_args", "")

    command = "vulnx"

    if cve_id:
        command += f" --cve-id {cve_id}"

    if search:
        command += f" --search {search}"

    if auth_key:
        command += f" --auth-key {auth_key}"

    output = data.get("output", "")
    if output:
        command += f" --output {output}"

    if additional_args:
        command += f" {additional_args}"

    logger.debug(f"🔎 vulnx → cve={cve_id or 'N/A'} search={search or 'N/A'}")
    return execute_command(command)


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

_HANDLERS = {
    "vulnx": _vulnx,
}


def vuln_intel_exec(tool: str, data: dict) -> dict:
    """Entry point for run_security_tool() dispatch."""
    handler = _HANDLERS.get(tool)
    if handler is None:
        return {"success": False, "error": f"Unknown vuln_intel tool: '{tool}'"}
    return handler(data)
