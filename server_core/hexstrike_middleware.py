"""
server_core/hexstrike_middleware.py

HexStrike-specific FastMCP middleware.

Provides two middleware classes:

1. HexStrikeLoggingMiddleware
   Structured JSON logging at the FastMCP framework level.
   Captures wall-clock duration for every tools/call, resource read,
   and prompt get — without duplicating the semantic telemetry already
   recorded by run_security_tool's finalize() closure.

2. HexStrikeSessionMiddleware
   Captures session_id on initialize and logs new session starts.
   Useful for correlating tool call logs with session lifecycle.

Usage in setup_mcp_server_standalone():

    from server_core.hexstrike_middleware import (
        HexStrikeLoggingMiddleware,
        HexStrikeSessionMiddleware,
    )

    mcp.add_middleware(HexStrikeSessionMiddleware())
    mcp.add_middleware(HexStrikeLoggingMiddleware())

Design notes:
    - Middleware operates at the MCP protocol layer — before FastMCP dispatches
      to the tool function. This means it captures the full wall-clock time
      including FastMCP overhead, parameter validation, and response serialization.
    - run_security_tool's finalize() operates inside the tool function and
      captures only the tool execution time. Both views are complementary.
    - Do NOT record to _op_metrics from middleware — that would double-count
      runs. _op_metrics is owned by finalize() inside run_security_tool.
    - Middleware logs use the 'hexstrike.middleware' logger — separate from
      the '[telemetry]' JSON lines produced by finalize().
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastmcp.server.middleware.middleware import CallNext, Middleware, MiddlewareContext
from server_core.request_context import generate_request_id, set_request_id, get_request_id

_mw_logger = logging.getLogger("hexstrike.middleware")


class HexStrikeLoggingMiddleware(Middleware):
    """
    Structured JSON logging middleware for HexStrike.

    Emits one log line per tools/call (and optionally resources/read and
    prompts/get) with:

        {
          "event":       "tool_call",
          "tool":        "nmap",
          "session_id":  "abc123",
          "request_id":  "a1b2c3d4e5f6",
          "duration_ms": 1234.56,
          "status":      "success" | "error",
          "error":       "..." (only on error)
        }

    Kept intentionally minimal — semantic details (cache_hit, timed_out,
    confirmation, opt_profile, etc.) are in the [telemetry] JSON lines
    emitted by finalize() inside run_security_tool.
    """

    def __init__(
        self,
        *,
        log_resources: bool = False,
        log_prompts: bool = False,
        log_level: int = logging.INFO,
    ) -> None:
        self._log_resources = log_resources
        self._log_prompts = log_prompts
        self._log_level = log_level

    # ── tools/call ────────────────────────────────────────────────────────────

    async def on_call_tool(
        self,
        context: MiddlewareContext,
        call_next: CallNext,
    ) -> Any:
        tool_name = getattr(context.message, "name", "unknown")
        session_id = _get_session_id(context)
        request_id = generate_request_id()
        set_request_id(request_id)
        t0 = time.perf_counter()

        try:
            result = await call_next(context)
            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            _mw_logger.log(
                self._log_level,
                json.dumps({
                    "event":       "tool_call",
                    "tool":        tool_name,
                    "session_id":  session_id,
                    "request_id":  request_id,
                    "duration_ms": duration_ms,
                    "status":      "success",
                }),
            )
            return result

        except Exception as exc:
            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            _mw_logger.error(
                json.dumps({
                    "event":       "tool_call",
                    "tool":        tool_name,
                    "session_id":  session_id,
                    "request_id":  request_id,
                    "duration_ms": duration_ms,
                    "status":      "error",
                    "error":       str(exc)[:200],
                }),
            )
            raise

    # ── resources/read (optional) ─────────────────────────────────────────────

    async def on_read_resource(
        self,
        context: MiddlewareContext,
        call_next: CallNext,
    ) -> Any:
        if not self._log_resources:
            return await call_next(context)

        uri = getattr(context.message, "uri", "unknown")
        session_id = _get_session_id(context)
        request_id = get_request_id()
        t0 = time.perf_counter()

        try:
            result = await call_next(context)
            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            _mw_logger.log(
                self._log_level,
                json.dumps({
                    "event":       "resource_read",
                    "uri":         str(uri),
                    "session_id":  session_id,
                    "request_id":  request_id,
                    "duration_ms": duration_ms,
                    "status":      "success",
                }),
            )
            return result
        except Exception as exc:
            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            _mw_logger.error(
                json.dumps({
                    "event":       "resource_read",
                    "uri":         str(uri),
                    "session_id":  session_id,
                    "request_id":  request_id,
                    "duration_ms": duration_ms,
                    "status":      "error",
                    "error":       str(exc)[:200],
                }),
            )
            raise

    # ── prompts/get (optional) ────────────────────────────────────────────────

    async def on_get_prompt(
        self,
        context: MiddlewareContext,
        call_next: CallNext,
    ) -> Any:
        if not self._log_prompts:
            return await call_next(context)

        prompt_name = getattr(context.message, "name", "unknown")
        session_id = _get_session_id(context)
        request_id = get_request_id()
        t0 = time.perf_counter()

        try:
            result = await call_next(context)
            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            _mw_logger.log(
                self._log_level,
                json.dumps({
                    "event":       "prompt_get",
                    "prompt":      prompt_name,
                    "session_id":  session_id,
                    "request_id":  request_id,
                    "duration_ms": duration_ms,
                    "status":      "success",
                }),
            )
            return result
        except Exception as exc:
            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            _mw_logger.error(
                json.dumps({
                    "event":       "prompt_get",
                    "prompt":      prompt_name,
                    "session_id":  session_id,
                    "request_id":  request_id,
                    "duration_ms": duration_ms,
                    "status":      "error",
                    "error":       str(exc)[:200],
                }),
            )
            raise


class HexStrikeSessionMiddleware(Middleware):
    """
    Logs new MCP session initialization.

    Emits one structured log line per client session:

        {"event": "session_start", "session_id": "...", "client": "..."}

    Useful for correlating tool call logs with session lifecycle and
    diagnosing multi-client issues.
    """

    async def on_initialize(
        self,
        context: MiddlewareContext,
        call_next: CallNext,
    ) -> Any:
        result = await call_next(context)

        session_id = _get_session_id(context)
        client_info = getattr(context.message, "params", None)
        client_name = ""
        if client_info:
            client_impl = getattr(client_info, "clientInfo", None)
            if client_impl:
                client_name = getattr(client_impl, "name", "")

        _mw_logger.info(
            json.dumps({
                "event":      "session_start",
                "session_id": session_id,
                "client":     client_name or "unknown",
            })
        )
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_session_id(context: MiddlewareContext) -> str:
    """Extract session_id from middleware context safely."""
    fctx = getattr(context, "fastmcp_context", None)
    if fctx is None:
        return "unknown"
    try:
        return fctx.session_id
    except (RuntimeError, AttributeError):
        return "unknown"
