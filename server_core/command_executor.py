from typing import Any, Optional
from beartype import beartype
from server_core import config_core
from server_core.enhanced_command_executor import EnhancedCommandExecutor
from server_core.singletons import cache as _cache

COMMAND_TIMEOUT = config_core.get("COMMAND_TIMEOUT", 300)  # Default to 5 minutes if not set


def _cache_get(active_cache, command: str, params: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Read from either AdvancedCache(key) or legacy HexStrikeCache(command, params)."""
    try:
        return active_cache.get(command)
    except TypeError:
        return active_cache.get(command, params or {})


def _cache_set(active_cache, command: str, params: Optional[dict[str, Any]], result: dict[str, Any]) -> None:
    """Write to either AdvancedCache(key, value) or legacy HexStrikeCache(command, params, result)."""
    try:
        active_cache.set(command, result)
    except TypeError:
        active_cache.set(command, params or {}, result)


def _normalize_result(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize EnhancedCommandExecutor output to the canonical HexStrike result shape.

    EnhancedCommandExecutor returns:
        {stdout, stderr, return_code, success, timed_out, partial_results,
         execution_time, timestamp}

    All consumers (run_security_tool, RateLimitDetector, _scan_cache, tests)
    expect the canonical shape:
        {success, output, error, returncode, timed_out, partial_results,
         execution_time, timestamp}

    Both key sets are preserved so legacy callers that still read stdout/stderr
    directly do not break.
    """
    # Already normalized/minimal (e.g. _require() early-return or unknown-tool error)
    if "stdout" not in raw and "return_code" not in raw:
        return raw

    stdout = raw.get("stdout", "")
    stderr = raw.get("stderr", "")
    success = raw.get("success", False)
    timed_out = raw.get("timed_out", False)

    # Canonical output: prefer stdout; fall back to stderr when stdout is empty
    output = stdout if stdout.strip() else stderr

    # Canonical error: non-empty stderr on failure, or timeout message
    if timed_out:
        error = f"Command timed out after {raw.get('execution_time', '?'):.0f}s"
        if stderr.strip():
            error += f": {stderr.strip()[:200]}"
    elif not success and stderr.strip():
        error = stderr.strip()
    else:
        error = ""

    return {
        # Canonical keys consumed by run_security_tool / RateLimitDetector / cache
        "success":         success,
        "output":          output,
        "error":           error,
        "returncode":      raw.get("return_code", -1),
        "timed_out":       timed_out,
        "partial_results": raw.get("partial_results", False),
        "execution_time":  raw.get("execution_time", 0.0),
        "timestamp":       raw.get("timestamp", ""),
        # Keep raw keys for any legacy callers still reading stdout/stderr directly
        "stdout":          stdout,
        "stderr":          stderr,
        "return_code":     raw.get("return_code", -1),
    }


@beartype
def execute_command(
  command: str,
  use_cache: bool = True,
  cache=None,
  timeout: int = COMMAND_TIMEOUT,
  tool: Optional[str] = None,
  endpoint: Optional[str] = None,
  params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
  """
  Execute a shell command with enhanced features.

  Args:
      command:    The command to execute
      use_cache:  Whether to use caching for this command
      cache:      Optional cache instance (falls back to the module-level singleton)
      timeout:    Command execution timeout in seconds
      tool:       Reserved — tool name (unused; recording is done in the after_request hook)
      endpoint:   Reserved — API endpoint (unused; recording is done in the after_request hook)
      params:     Reserved — request params (unused; recording is done in the after_request hook)

  Returns:
      A dictionary containing the stdout, stderr, return code, and metadata
  """
  active_cache = cache if cache is not None else (_cache if use_cache else None)

  if active_cache is not None:
    cached_result = _cache_get(active_cache, command, params)
    if cached_result:
      return cached_result

  # Create a fresh executor per call — avoids shared mutable state race condition
  # under concurrent asyncio.run_in_executor() calls (CODEX P0 fix)
  executor = EnhancedCommandExecutor(command, timeout=timeout)
  result = _normalize_result(executor.execute())

  if active_cache is not None and result.get("success", False):
    _cache_set(active_cache, command, params, result)

  return result
