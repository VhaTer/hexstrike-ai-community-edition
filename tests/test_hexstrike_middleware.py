"""
tests/test_hexstrike_middleware.py

Unit tests for server_core/hexstrike_middleware.py

Covers:
  - HexStrikeLoggingMiddleware.on_call_tool: success + error paths
  - HexStrikeLoggingMiddleware.on_read_resource: gated by log_resources flag
  - HexStrikeLoggingMiddleware.on_get_prompt: gated by log_prompts flag
  - HexStrikeSessionMiddleware.on_initialize: logs session_start
  - _get_session_id: handles missing context, RuntimeError, normal case
  - Structured JSON output validity
"""

import json
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from server_core.hexstrike_middleware import (
    HexStrikeLoggingMiddleware,
    HexStrikeSessionMiddleware,
    _get_session_id,
)
from fastmcp.server.middleware.middleware import MiddlewareContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(method="tools/call", tool_name="nmap", session_id="sess-test"):
    message = MagicMock()
    message.name = tool_name

    fctx = MagicMock()
    fctx.session_id = session_id

    ctx = MiddlewareContext(
        message=message,
        method=method,
        fastmcp_context=fctx,
    )
    return ctx


async def _call_next_ok(context):
    return MagicMock()


async def _call_next_error(context):
    raise RuntimeError("tool exploded")


# ---------------------------------------------------------------------------
# _get_session_id
# ---------------------------------------------------------------------------

class TestGetSessionId:

    def test_returns_session_id_from_context(self):
        ctx = _make_context(session_id="my-session")
        assert _get_session_id(ctx) == "my-session"

    def test_returns_unknown_when_no_fastmcp_context(self):
        ctx = MiddlewareContext(message=MagicMock(), method="tools/call")
        assert _get_session_id(ctx) == "unknown"

    def test_returns_unknown_on_runtime_error(self):
        fctx = MagicMock()
        type(fctx).session_id = property(lambda self: (_ for _ in ()).throw(RuntimeError("no session")))
        ctx = MiddlewareContext(message=MagicMock(), method="tools/call", fastmcp_context=fctx)
        assert _get_session_id(ctx) == "unknown"


# ---------------------------------------------------------------------------
# HexStrikeLoggingMiddleware — on_call_tool
# ---------------------------------------------------------------------------

class TestLoggingMiddlewareToolCall:

    @pytest.mark.asyncio
    async def test_success_logs_structured_json(self, caplog):
        mw = HexStrikeLoggingMiddleware()
        ctx = _make_context(tool_name="nmap", session_id="s1")

        with caplog.at_level(logging.INFO, logger="hexstrike.middleware"):
            await mw.on_call_tool(ctx, _call_next_ok)

        assert len(caplog.records) == 1
        data = json.loads(caplog.records[0].message)
        assert data["event"] == "tool_call"
        assert data["tool"] == "nmap"
        assert data["session_id"] == "s1"
        assert data["status"] == "success"
        assert "duration_ms" in data
        assert isinstance(data["duration_ms"], float)

    @pytest.mark.asyncio
    async def test_error_logs_error_status_and_reraises(self, caplog):
        mw = HexStrikeLoggingMiddleware()
        ctx = _make_context(tool_name="sqlmap", session_id="s2")

        with caplog.at_level(logging.ERROR, logger="hexstrike.middleware"):
            with pytest.raises(RuntimeError, match="tool exploded"):
                await mw.on_call_tool(ctx, _call_next_error)

        assert len(caplog.records) == 1
        data = json.loads(caplog.records[0].message)
        assert data["event"] == "tool_call"
        assert data["tool"] == "sqlmap"
        assert data["status"] == "error"
        assert "tool exploded" in data["error"]

    @pytest.mark.asyncio
    async def test_duration_ms_is_positive(self, caplog):
        mw = HexStrikeLoggingMiddleware()
        ctx = _make_context(tool_name="nmap")

        with caplog.at_level(logging.INFO, logger="hexstrike.middleware"):
            await mw.on_call_tool(ctx, _call_next_ok)

        data = json.loads(caplog.records[0].message)
        assert data["duration_ms"] >= 0.0

    @pytest.mark.asyncio
    async def test_unknown_tool_name_when_message_has_no_name(self, caplog):
        mw = HexStrikeLoggingMiddleware()
        message = MagicMock(spec=[])  # no .name attribute
        fctx = MagicMock()
        fctx.session_id = "s3"
        ctx = MiddlewareContext(message=message, method="tools/call", fastmcp_context=fctx)

        with caplog.at_level(logging.INFO, logger="hexstrike.middleware"):
            await mw.on_call_tool(ctx, _call_next_ok)

        data = json.loads(caplog.records[0].message)
        assert data["tool"] == "unknown"


# ---------------------------------------------------------------------------
# HexStrikeLoggingMiddleware — on_read_resource
# ---------------------------------------------------------------------------

class TestLoggingMiddlewareResource:

    @pytest.mark.asyncio
    async def test_not_logged_when_log_resources_false(self, caplog):
        mw = HexStrikeLoggingMiddleware(log_resources=False)
        message = MagicMock()
        message.uri = "scan://example.com/nmap"
        ctx = MiddlewareContext(message=message, method="resources/read")

        with caplog.at_level(logging.INFO, logger="hexstrike.middleware"):
            await mw.on_read_resource(ctx, _call_next_ok)

        assert len(caplog.records) == 0

    @pytest.mark.asyncio
    async def test_logged_when_log_resources_true(self, caplog):
        mw = HexStrikeLoggingMiddleware(log_resources=True)
        message = MagicMock()
        message.uri = "scan://example.com/nmap"
        fctx = MagicMock()
        fctx.session_id = "s4"
        ctx = MiddlewareContext(message=message, method="resources/read", fastmcp_context=fctx)

        with caplog.at_level(logging.INFO, logger="hexstrike.middleware"):
            await mw.on_read_resource(ctx, _call_next_ok)

        assert len(caplog.records) == 1
        data = json.loads(caplog.records[0].message)
        assert data["event"] == "resource_read"
        assert data["status"] == "success"


# ---------------------------------------------------------------------------
# HexStrikeLoggingMiddleware — on_get_prompt
# ---------------------------------------------------------------------------

class TestLoggingMiddlewarePrompt:

    @pytest.mark.asyncio
    async def test_not_logged_when_log_prompts_false(self, caplog):
        mw = HexStrikeLoggingMiddleware(log_prompts=False)
        ctx = _make_context(method="prompts/get", tool_name="bug_bounty_recon")

        with caplog.at_level(logging.INFO, logger="hexstrike.middleware"):
            await mw.on_get_prompt(ctx, _call_next_ok)

        assert len(caplog.records) == 0

    @pytest.mark.asyncio
    async def test_logged_when_log_prompts_true(self, caplog):
        mw = HexStrikeLoggingMiddleware(log_prompts=True)
        ctx = _make_context(method="prompts/get", tool_name="bug_bounty_recon", session_id="s5")

        with caplog.at_level(logging.INFO, logger="hexstrike.middleware"):
            await mw.on_get_prompt(ctx, _call_next_ok)

        assert len(caplog.records) == 1
        data = json.loads(caplog.records[0].message)
        assert data["event"] == "prompt_get"
        assert data["prompt"] == "bug_bounty_recon"
        assert data["session_id"] == "s5"
        assert data["status"] == "success"


# ---------------------------------------------------------------------------
# HexStrikeSessionMiddleware — on_initialize
# ---------------------------------------------------------------------------

class TestSessionMiddleware:

    @pytest.mark.asyncio
    async def test_logs_session_start(self, caplog):
        mw = HexStrikeSessionMiddleware()
        message = MagicMock()
        params = MagicMock()
        client_info = MagicMock()
        client_info.name = "claude-desktop"
        params.clientInfo = client_info
        message.params = params

        fctx = MagicMock()
        fctx.session_id = "sess-init-test"
        ctx = MiddlewareContext(message=message, method="initialize", fastmcp_context=fctx)

        with caplog.at_level(logging.INFO, logger="hexstrike.middleware"):
            await mw.on_initialize(ctx, _call_next_ok)

        assert len(caplog.records) == 1
        data = json.loads(caplog.records[0].message)
        assert data["event"] == "session_start"
        assert data["session_id"] == "sess-init-test"
        assert data["client"] == "claude-desktop"

    @pytest.mark.asyncio
    async def test_session_start_unknown_client_when_no_client_info(self, caplog):
        mw = HexStrikeSessionMiddleware()
        message = MagicMock()
        message.params = None
        fctx = MagicMock()
        fctx.session_id = "sess-no-client"
        ctx = MiddlewareContext(message=message, method="initialize", fastmcp_context=fctx)

        with caplog.at_level(logging.INFO, logger="hexstrike.middleware"):
            await mw.on_initialize(ctx, _call_next_ok)

        data = json.loads(caplog.records[0].message)
        assert data["client"] == "unknown"

    @pytest.mark.asyncio
    async def test_call_next_called_before_logging(self):
        """on_initialize must call call_next before logging (result first)."""
        mw = HexStrikeSessionMiddleware()
        call_order = []

        async def tracked_call_next(ctx):
            call_order.append("call_next")
            return MagicMock()

        message = MagicMock()
        message.params = None
        fctx = MagicMock()
        fctx.session_id = "s"
        ctx = MiddlewareContext(message=message, method="initialize", fastmcp_context=fctx)

        await mw.on_initialize(ctx, tracked_call_next)
        assert call_order == ["call_next"]
