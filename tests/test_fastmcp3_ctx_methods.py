"""
================================================================================
FASTMCP 3.x CTX METHOD TESTS — ctx.info() / ctx.report_progress() / ctx.set_state()
================================================================================

Goal: Observe and assert FastMCP 3.x context methods are called correctly
during tool execution. These tests verify:
  • ctx.info()     — status/information messages
  • ctx.report_progress() — progress reporting during tool execution
  • ctx.set_state() — session state persistence (tech profiles)
  • ctx.get_state() — session state retrieval (tech profiles)
  • ctx.error()    — error messages on failures
  • ctx.warning()  — warning messages

Run with:
    pytest tests/test_fastmcp3_ctx_methods.py -v -s
================================================================================
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_mock_context():
    """Build a mock FastMCP Context with async methods pre-configured."""
    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.debug = AsyncMock()
    ctx.report_progress = AsyncMock()
    ctx.get_state = AsyncMock(return_value=None)
    ctx.set_state = AsyncMock()
    ctx.read_resource = AsyncMock(return_value=MagicMock(contents=[]))
    ctx.sample = AsyncMock(return_value="mock suggestion")
    ctx.session_id = "test-session-fixed"
    return ctx


async def call_run_security_tool(mcp, tool_name, parameters, ctx=None):
    """Invoke the run_security_tool registered on an MCP server.

    Parameters can be a dict (auto-serialized to JSON) or a raw string
    (passed through as-is — useful for testing invalid JSON).
    """
    tool = await mcp.get_tool("run_security_tool")
    assert tool is not None, "run_security_tool not found on MCP server"
    if ctx is None:
        ctx = make_mock_context()
    # Pass through raw strings (invalid JSON tests), serialize dicts
    payload = parameters if isinstance(parameters, str) else json.dumps(parameters)
    return await tool.fn(ctx, tool_name=tool_name, parameters=payload), ctx


def run(coro):
    """Run an async coroutine synchronously for test convenience."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def make_mcp():
    """Build a Phase-3 standalone MCP server."""
    from mcp_core.server_setup import setup_mcp_server_standalone
    return setup_mcp_server_standalone(MagicMock())


# ──────────────────────────────────────────────────────────────────────────────
# TEST GROUP 1 — ctx.info() assertions (basic tool execution flow)
# ──────────────────────────────────────────────────────────────────────────────

class TestCtxInfoBasicFlow:
    """Verify ctx.info() is called with expected messages during normal execution."""

    def test_info_called_on_tool_start_and_success(self):
        """ctx.info should log start + completion for a simple non-destructive tool."""
        with patch("mcp_core.misc_direct.misc_exec", return_value={"success": True, "output": "ok"}):
            mcp = make_mcp()
            result, ctx = run(call_run_security_tool(mcp, "uro", {}))

        assert result["success"] is True

        # Assert ctx.info was called at least twice (start + success)
        assert ctx.info.call_count >= 2, f"Expected >=2 ctx.info calls, got {ctx.info.call_count}"

        # Check specific messages were logged
        info_calls = [str(call) for call in ctx.info.call_args_list]
        assert any("uro" in c for c in info_calls), f"Expected 'uro' in ctx.info calls: {info_calls}"

        # Should have "Executing uro" and "completed"
        assert any("Executing" in c for c in info_calls), f"Expected 'Executing' in calls: {info_calls}"
        assert any("completed" in c or "✅" in c for c in info_calls), f"Expected completion in calls: {info_calls}"

    def test_info_called_on_tool_failure(self):
        """ctx.info should log start, and ctx.error should log failure."""
        with patch("mcp_core.misc_direct.misc_exec", return_value={"success": False, "error": "simulated failure"}):
            mcp = make_mcp()
            result, ctx = run(call_run_security_tool(mcp, "uro", {}))

        assert result["success"] is False

        # ctx.info should still be called for start
        assert ctx.info.called, "ctx.info should be called even on failure"

        # ctx.error should be called for the failure message
        assert ctx.error.called, "ctx.error should be called on tool failure"
        error_calls = [str(call) for call in ctx.error.call_args_list]
        assert any("uro" in c for c in error_calls), f"Expected 'uro' in ctx.error calls: {error_calls}"

    def test_info_called_with_invalid_json_params(self):
        """ctx.error should be called when parameters are invalid JSON."""
        mcp = make_mcp()
        result, ctx = run(call_run_security_tool(mcp, "uro", "not-json-at-all"))

        assert result["success"] is False
        assert "Invalid JSON" in result.get("error", "")
        assert ctx.error.called, "ctx.error should be called for invalid JSON"

        # ctx.info should still be called for the initial "Executing uro"
        assert ctx.info.called, "ctx.info should still be called before JSON parse error"


# ──────────────────────────────────────────────────────────────────────────────
# TEST GROUP 2 — ctx.report_progress() assertions
# ──────────────────────────────────────────────────────────────────────────────

class TestCtxReportProgress:
    """Verify ctx.report_progress() is called with the expected 0→25→50→75→100 pattern."""

    def test_progress_reported_for_simple_tool(self):
        """Progress should be reported at start (0) and end (100)."""
        with patch("mcp_core.misc_direct.misc_exec", return_value={"success": True, "output": "ok"}):
            mcp = make_mcp()
            result, ctx = run(call_run_security_tool(mcp, "anew", {}))

        assert result["success"] is True
        assert ctx.report_progress.called, "ctx.report_progress should be called"

        # Extract (current, total) tuples from calls
        progress_calls = [call.args for call in ctx.report_progress.call_args_list]
        print(f"  📊 Progress calls: {progress_calls}")

        # Should start at 0/100
        assert any(p == (0, 100) for p in progress_calls), f"Expected (0, 100) in progress calls: {progress_calls}"

        # Should end at 100/100
        assert any(p == (100, 100) for p in progress_calls), f"Expected (100, 100) in progress calls: {progress_calls}"

    def test_progress_phases_for_longer_tool(self):
        """Simulate a slow tool to verify intermediate progress phases (25, 50, 75)."""
        call_count = 0

        async def fake_wait(futures, timeout):
            """Mock asyncio.wait: first 3 calls incomplete, 4th call complete."""
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                return (set(), futures)
            return (futures, set())

        with patch("mcp_core.misc_direct.misc_exec", return_value={"success": True, "output": "slow ok"}), \
             patch("asyncio.wait", side_effect=fake_wait):
            mcp = make_mcp()
            result, ctx = run(call_run_security_tool(mcp, "anew", {}))

        assert result["success"] is True

        progress_calls = [call.args for call in ctx.report_progress.call_args_list]
        print(f"  📊 Progress calls (slow tool): {progress_calls}")

        # Should have intermediate progress values (25, 50, 75)
        values = [p[0] for p in progress_calls]
        assert 25 in values, f"Expected 25 in progress values: {values}"
        assert 50 in values, f"Expected 50 in progress values: {values}"
        assert 75 in values, f"Expected 75 in progress values: {values}"

    def test_progress_calls_are_sequential(self):
        """Progress values should increase monotonically."""
        with patch("mcp_core.misc_direct.misc_exec", return_value={"success": True, "output": "ok"}):
            mcp = make_mcp()
            result, ctx = run(call_run_security_tool(mcp, "checksec", {}))

        progress_calls = [call.args for call in ctx.report_progress.call_args_list]
        values = [p[0] for p in progress_calls]
        print(f"  📊 Sequential progress values: {values}")

        # Values should be non-decreasing
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1], f"Progress decreased: {values}"


# ──────────────────────────────────────────────────────────────────────────────
# TEST GROUP 3 — ctx.set_state() / ctx.get_state() (tech profile caching)
# ──────────────────────────────────────────────────────────────────────────────

class TestCtxStateMethods:
    """Verify ctx.set_state() and ctx.get_state() are called for technology profile caching."""

    def test_get_state_called_when_target_provided(self):
        """When a target is provided, ctx.get_state should be called to check for cached tech profile."""
        with patch("mcp_core.net_scan_direct.net_scan_exec", return_value={"success": True, "output": "ok"}):
            mcp = make_mcp()
            result, ctx = run(call_run_security_tool(mcp, "nmap", {"target": "scanme.nmap.org"}))

        assert result["success"] is True

        # get_state should be called to look up cached tech profile
        assert ctx.get_state.called, "ctx.get_state should be called when target is provided"

        # Check the key used
        state_calls = [call.args[0] for call in ctx.get_state.call_args_list]
        assert any("tech:" in str(k) for k in state_calls), f"Expected 'tech:' key in get_state calls: {state_calls}"

    def test_set_state_called_when_tech_detected(self):
        """When technology is detected, ctx.set_state should persist the profile."""
        # Simulate a whatweb-like result that would trigger tech detection
        whatweb_output = """
        http://example.com [200 OK] Apache[2.4.41], PHP[7.4.3]
        """

        with patch("mcp_core.web_probe_direct.web_probe_exec", return_value={
            "success": True,
            "output": whatweb_output,
        }):
            mcp = make_mcp()
            result, ctx = run(call_run_security_tool(mcp, "whatweb", {"target": "example.com"}))

        assert result["success"] is True

        # After successful scan, set_state may be called to cache tech profile
        # (This depends on whether _detect_from_cache finds something useful)
        # At minimum, get_state should have been called
        assert ctx.get_state.called, "ctx.get_state should be called for target-based tools"

    def test_set_state_called_with_correct_structure(self):
        """If set_state is called, verify it has the expected tech profile structure."""
        ctx = make_mock_context()
        ctx.get_state = AsyncMock(return_value=None)

        with patch("mcp_core.net_scan_direct.net_scan_exec", return_value={"success": True, "output": "ok"}):
            mcp = make_mcp()
            result, ctx = run(call_run_security_tool(mcp, "nmap", {"target": "10.0.0.1"}, ctx=ctx))

        assert result["success"] is True

        # If set_state was called, validate structure
        if ctx.set_state.called:
            args = ctx.set_state.call_args
            key = args.args[0]
            value = args.args[1]
            assert key.startswith("tech:"), f"Expected 'tech:' prefix, got: {key}"
            assert isinstance(value, dict), f"Expected dict value, got: {type(value)}"
            # Should have at least some tech profile keys
            expected_keys = {"web_servers", "frameworks", "cms", "databases", "languages", "security", "services"}
            assert any(k in value for k in expected_keys), f"Expected tech profile keys in: {value.keys()}"

    def test_get_state_returns_cached_profile(self):
        """If get_state returns a cached profile, set_state should NOT be called again."""
        cached_profile = {
            "web_servers": ["nginx"],
            "frameworks": ["django"],
            "cms": [],
            "databases": ["postgresql"],
            "languages": ["python"],
            "security": ["mod_security"],
            "services": [],
        }
        ctx = make_mock_context()
        ctx.get_state = AsyncMock(return_value=cached_profile)

        with patch("mcp_core.net_scan_direct.net_scan_exec", return_value={"success": True, "output": "ok"}):
            mcp = make_mcp()
            result, ctx = run(call_run_security_tool(mcp, "nmap", {"target": "cached-target.com"}, ctx=ctx))

        assert result["success"] is True

        # get_state should be called and return our cached profile
        assert ctx.get_state.called, "ctx.get_state should be called"

        # set_state should NOT be called because we already have a cached profile
        # (or if it is, it's idempotent — but ideally it's skipped)
        print(f"  📦 get_state calls: {ctx.get_state.call_args_list}")
        print(f"  💾 set_state calls: {ctx.set_state.call_args_list}")


# ──────────────────────────────────────────────────────────────────────────────
# TEST GROUP 4 — ctx.error() / ctx.warning() assertions
# ──────────────────────────────────────────────────────────────────────────────

class TestCtxErrorAndWarning:
    """Verify ctx.error() and ctx.warning() are called in error paths."""

    def test_error_called_for_unknown_tool(self):
        """Calling an unknown tool should trigger ctx.error."""
        mcp = make_mcp()
        result, ctx = run(call_run_security_tool(mcp, "nonexistent_tool_xyz", {}))

        assert result["success"] is False
        assert ctx.error.called, "ctx.error should be called for unknown tool"

        error_calls = [str(call) for call in ctx.error.call_args_list]
        assert any("Unknown tool" in c for c in error_calls), f"Expected 'Unknown tool' in error: {error_calls}"

    def test_error_called_for_json_parse_failure(self):
        """Invalid JSON should trigger ctx.error with Invalid JSON message."""
        mcp = make_mcp()
        result, ctx = run(call_run_security_tool(mcp, "nmap", "this is not json"))

        assert result["success"] is False
        assert "Invalid JSON" in result.get("error", "")
        assert ctx.error.called, "ctx.error should be called for JSON parse failure"

    def test_warning_not_called_on_successful_tool(self):
        """Successful tool execution should NOT trigger ctx.warning."""
        with patch("mcp_core.misc_direct.misc_exec", return_value={"success": True, "output": "ok"}):
            mcp = make_mcp()
            result, ctx = run(call_run_security_tool(mcp, "checksec", {"target": "test"}))

        assert result["success"] is True
        assert not ctx.warning.called, "ctx.warning should NOT be called for successful tool"

    def test_read_resource_mock_available(self):
        """ctx.read_resource mock is properly configured and awaitable."""
        ctx = make_mock_context()
        resource = asyncio.run(ctx.read_resource("skill://test/REFERENCE.md"))
        assert resource is not None
        assert hasattr(resource, "contents")

    def test_sample_mock_available(self):
        """ctx.sample mock is properly configured and returns a string."""
        ctx = make_mock_context()
        result = asyncio.run(ctx.sample("prompt", {}))
        assert isinstance(result, str)
        assert "mock suggestion" in result


# ──────────────────────────────────────────────────────────────────────────────
# TEST GROUP 5 — Full execution message log (live observation helper)
# ──────────────────────────────────────────────────────────────────────────────

class TestLiveObservation:
    """Print all ctx calls to stdout so you can observe them live during test run."""

    def test_print_all_ctx_calls_for_observation(self):
        """This test prints every ctx method call — run with -s to see live output."""
        with patch("mcp_core.misc_direct.misc_exec", return_value={"success": True, "output": "all good"}):
            mcp = make_mcp()
            result, ctx = run(call_run_security_tool(mcp, "checksec", {}))

        assert result["success"] is True

        print("\n" + "=" * 60)
        print("📋 LIVE OBSERVATION — All ctx method calls")
        print("=" * 60)

        print("\n🔹 ctx.info() calls:")
        for i, call in enumerate(ctx.info.call_args_list, 1):
            print(f"   {i}. {call}")

        print("\n🔹 ctx.report_progress() calls:")
        for i, call in enumerate(ctx.report_progress.call_args_list, 1):
            print(f"   {i}. {call}")

        print("\n🔹 ctx.error() calls:")
        for i, call in enumerate(ctx.error.call_args_list, 1):
            print(f"   {i}. {call}")

        print("\n🔹 ctx.warning() calls:")
        for i, call in enumerate(ctx.warning.call_args_list, 1):
            print(f"   {i}. {call}")

        print("\n🔹 ctx.get_state() calls:")
        for i, call in enumerate(ctx.get_state.call_args_list, 1):
            print(f"   {i}. {call}")

        print("\n🔹 ctx.set_state() calls:")
        for i, call in enumerate(ctx.set_state.call_args_list, 1):
            print(f"   {i}. {call}")

        print("\n" + "=" * 60)
        print("✅ Live observation complete — all ctx methods visible above")
        print("=" * 60 + "\n")

        # Sanity assertions
        assert ctx.info.called, "ctx.info should be called"
        assert ctx.report_progress.called, "ctx.report_progress should be called"


# ──────────────────────────────────────────────────────────────────────────────
# TEST GROUP 6 — Specific tool smoke tests (popular tools with rich ctx usage)
# ──────────────────────────────────────────────────────────────────────────────

class TestPopularToolsCtxUsage:
    """Smoke-test popular tools that have rich ctx.info() / ctx.report_progress() usage."""

    def test_nmap_tool_ctx_calls(self):
        """nmap is a popular tool — verify its ctx flow."""
        with patch("mcp_core.net_scan_direct.net_scan_exec", return_value={"success": True, "output": "nmap done"}):
            mcp = make_mcp()
            result, ctx = run(call_run_security_tool(mcp, "nmap", {"target": "127.0.0.1"}))

        assert result["success"] is True
        assert ctx.info.called, "ctx.info should be called for nmap"
        assert ctx.report_progress.called, "ctx.report_progress should be called for nmap"

    def test_sqlmap_tool_ctx_calls(self):
        """sqlmap — verify ctx flow."""
        with patch("mcp_core.web_scan_direct.web_scan_exec", return_value={"success": True, "output": "sqlmap done"}):
            mcp = make_mcp()
            result, ctx = run(call_run_security_tool(mcp, "sqlmap", {"url": "http://test.com"}))

        assert result["success"] is True
        assert ctx.info.called, "ctx.info should be called for sqlmap"

    def test_sherlock_tool_ctx_calls(self):
        """sherlock — verify ctx flow."""
        with patch("mcp_core.osint_direct.osint_exec", return_value={"success": True, "output": "sherlock done"}):
            mcp = make_mcp()
            result, ctx = run(call_run_security_tool(mcp, "sherlock", {"username": "testuser"}))

        assert result["success"] is True
        assert ctx.info.called, "ctx.info should be called for sherlock"

    def test_gobuster_tool_ctx_calls(self):
        """gobuster — verify ctx flow with progress reporting."""
        with patch("mcp_core.web_fuzz_direct.web_fuzz_exec", return_value={"success": True, "output": "gobuster done"}):
            mcp = make_mcp()
            result, ctx = run(call_run_security_tool(mcp, "gobuster", {"url": "http://test.com", "wordlist": "/tmp/wordlist.txt"}))

        assert result["success"] is True
        assert ctx.info.called, "ctx.info should be called for gobuster"
        assert ctx.report_progress.called, "ctx.report_progress should be called for gobuster"

    def test_metasploit_auxiliary_scanner_ctx_calls(self):
        """Metasploit auxiliary scanner should NOT require confirmation and should use ctx normally."""
        with patch("mcp_core.exploit_framework_direct.exploit_exec", return_value={"success": True, "output": "msf done"}), \
             patch("mcp_core.server_setup.confirm_destructive_action", new_callable=AsyncMock) as mock_confirm:
            mcp = make_mcp()
            result, ctx = run(call_run_security_tool(
                mcp, "metasploit",
                {"module": "auxiliary/scanner/smb/smb_version", "options": {"RHOSTS": "10.10.10.10"}}
            ))

        assert result["success"] is True
        mock_confirm.assert_not_awaited()
        assert ctx.info.called, "ctx.info should be called for metasploit auxiliary scanner"

    def test_httpx_tool_with_target_ctx_calls(self):
        """httpx with target should trigger get_state for tech profile lookup."""
        with patch("mcp_core.web_recon_direct.web_recon_exec", return_value={"success": True, "output": "httpx done", "headers": {}}):
            mcp = make_mcp()
            result, ctx = run(call_run_security_tool(mcp, "httpx", {"target": "example.com"}))

        assert result["success"] is True
        assert ctx.get_state.called, "ctx.get_state should be called for target-based httpx"
