"""
Shared validation and utility helpers for mcp_core/*_direct.py modules.

Kept minimal — extracted from duplicated _require() in 12+ files.
"""

from typing import Any, Dict


_HINTS = {
    "url": "use http://host[:port]",
    "target": "use an IP or hostname",
    "domain": "use a domain name like example.com",
}


def require(data: dict, *keys: str) -> Dict[str, Any]:
    """Return error dict if any required key is missing or empty.

    Usage::

        err = require(data, "target")
        if err:
            return err

    Returns {"success": False, "error": msg} on failure, {} on success.
    """
    for key in keys:
        val = data.get(key)
        if val is None or val == "":
            hint = _HINTS.get(key, "")
            msg = f"'{key}' is required" + (f" ({hint})" if hint else "")
            return {"success": False, "error": msg}
    return {}


def require_one(data: dict, *keys: str) -> Dict[str, Any]:
    """Return error dict if none of the given keys are present.

    Usage::

        err = require_one(data, "cve_id", "search")
        if err:
            return err

    Returns {"success": False, "error": msg} if all keys are empty,
    {} if at least one key is present.
    """
    for key in keys:
        val = data.get(key)
        if val is not None and val != "":
            return {}
    keys_list = list(keys)
    return {"success": False, "error": f"At least one of {keys_list} is required"}
