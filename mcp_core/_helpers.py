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


# Shell metacharacters that would trigger EnhancedCommandExecutor's shell=True
# fallback (see server_core/enhanced_command_executor.py _SHELL_OPERATORS).
# Any of these in a target-like field (target/url/domain/interface/etc.) is
# either an accident or an injection attempt — neither should silently
# promote the whole command to shell=True.
_SHELL_METACHARS = ('|', '>', '<', '&&', '||', ';', '`', '$(', '${', '\n', '\r')


def reject_shell_metachars(data: dict, *keys: str) -> Dict[str, Any]:
    """Return error dict if any of the given fields contain shell metacharacters.

    Call this AFTER require() for any field that gets interpolated directly
    into a command string in a *_direct.py handler that does not itself rely
    on shell operators (pipes, &&, ||) by design. For handlers that DO need
    shell features intentionally, quote the user-controlled value with
    shlex.quote() at the interpolation site instead of calling this.

    Usage::

        err = require(data, "target")
        if err:
            return err
        err = reject_shell_metachars(data, "target")
        if err:
            return err

    Returns {"success": False, "error": msg} on failure, {} on success.
    """
    for key in keys:
        val = data.get(key)
        if not isinstance(val, str):
            continue
        for ch in _SHELL_METACHARS:
            if ch in val:
                return {
                    "success": False,
                    "error": f"'{key}' contains disallowed character sequence: {ch!r}",
                }
    return {}
