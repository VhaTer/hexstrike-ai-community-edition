
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Auto-use fixture: clear global scan cache before every test to prevent
# cross-test state leakage (same session_id + target + tool = cache key collision)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_scan_cache():
    from mcp_core.server_setup import _scan_cache
    _scan_cache.cache.clear()
    _scan_cache.ttl_times.clear()


# ---------------------------------------------------------------------------
# Test helpers (duplicated from conftest.py for correct patch() target resolution)
# patch() requires the module to be imported under the same sys.modules key
# that production code uses. Importing via 'from tests.conftest import ...' or
# importlib creates a second module entry and breaks patch() path resolution.
# ---------------------------------------------------------------------------

def make_mock_context():
    ctx = SimpleNamespace()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.debug = AsyncMock()
    ctx.report_progress = AsyncMock()
    ctx.read_resource = AsyncMock(return_value=SimpleNamespace(contents=[]))
    ctx.get_state = AsyncMock(return_value=None)
    ctx.set_state = AsyncMock()
    ctx.get_prompt = AsyncMock(return_value=SimpleNamespace(messages=[]))
    ctx.session_id = "test-session-fixed"
    return ctx


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def make_mcp():
    from mcp_core.server_setup import setup_mcp_server_standalone
    return setup_mcp_server_standalone()


async def call_run_security_tool(mcp, tool_name, parameters):
    tool = await mcp.get_tool("run_security_tool")
    assert tool is not None
    ctx = make_mock_context()
    payload = parameters if isinstance(parameters, str) else json.dumps(parameters)
    return await tool.fn(ctx, tool_name=tool_name, parameters=payload)


def test_run_security_tool_registered():
    mcp = make_mcp()
    tool = run(mcp.get_tool("run_security_tool"))
    assert tool is not None


def test_aireplay_mode_9_skips_confirmation():
    with patch("mcp_core.wifi_direct.wifi_exec", return_value={"success": True, "output": "ok"}) as mock_exec, \
         patch("mcp_core.server_setup.confirm_destructive_action", new_callable=AsyncMock) as mock_confirm:
        mcp = make_mcp()
        result = run(call_run_security_tool(
            mcp,
            "aireplay_ng",
            {"interface": "wlan0mon", "attack_mode": 9},
        ))
    mock_confirm.assert_not_awaited()
    mock_exec.assert_called_once()
    assert result["success"] is True


def test_aireplay_mode_0_requires_confirmation():
    with patch("mcp_core.wifi_direct.wifi_exec", return_value={"success": True, "output": "ok"}) as mock_exec, \
         patch("mcp_core.server_setup.confirm_destructive_action", new_callable=AsyncMock, return_value=False) as mock_confirm:
        mcp = make_mcp()
        result = run(call_run_security_tool(
            mcp,
            "aireplay_ng",
            {"interface": "wlan0mon", "attack_mode": 0, "bssid": "AA:BB:CC:DD:EE:FF"},
        ))
    mock_confirm.assert_awaited_once()
    mock_exec.assert_not_called()
    assert result["success"] is False


def test_responder_analyze_skips_confirmation():
    with patch("mcp_core.misc_direct.misc_exec", return_value={"success": True, "output": "ok"}) as mock_exec, \
         patch("mcp_core.server_setup.confirm_destructive_action", new_callable=AsyncMock) as mock_confirm:
        mcp = make_mcp()
        result = run(call_run_security_tool(
            mcp,
            "responder",
            {"interface": "eth0", "analyze": True},
        ))
    mock_confirm.assert_not_awaited()
    mock_exec.assert_called_once()
    assert result["success"] is True


def test_metasploit_auxiliary_scanner_skips_confirmation():
    with patch("mcp_core.exploit_framework_direct.exploit_exec", return_value={"success": True, "output": "ok"}) as mock_exec, \
         patch("mcp_core.server_setup.confirm_destructive_action", new_callable=AsyncMock) as mock_confirm:
        mcp = make_mcp()
        result = run(call_run_security_tool(
            mcp,
            "metasploit",
            {"module": "auxiliary/scanner/smb/smb_version", "options": {"RHOSTS": "10.10.10.10"}},
        ))
    mock_confirm.assert_not_awaited()
    mock_exec.assert_called_once()
    assert result["success"] is True


def test_metasploit_exploit_requires_confirmation():
    with patch("mcp_core.exploit_framework_direct.exploit_exec", return_value={"success": True, "output": "ok"}) as mock_exec, \
         patch("mcp_core.server_setup.confirm_destructive_action", new_callable=AsyncMock, return_value=False) as mock_confirm:
        mcp = make_mcp()
        result = run(call_run_security_tool(
            mcp,
            "metasploit",
            {"module": "exploit/windows/smb/ms17_010_eternalblue", "options": {"RHOSTS": "10.10.10.10"}},
        ))
    mock_confirm.assert_awaited_once()
    mock_exec.assert_not_called()
    assert result["success"] is False


def test_mitm6_requires_confirmation():
    with patch("mcp_core.active_directory_direct.ad_exec", return_value={"success": True, "output": "ok"}) as mock_exec, \
         patch("mcp_core.server_setup.confirm_destructive_action", new_callable=AsyncMock, return_value=False) as mock_confirm:
        mcp = make_mcp()
        result = run(call_run_security_tool(
            mcp,
            "mitm6",
            {"interface": "eth0", "domain": "corp.local"},
        ))
    mock_confirm.assert_awaited_once()
    mock_exec.assert_not_called()
    assert result["success"] is False


def assert_canonical_result(result):
    for key in ("success", "output", "error", "returncode", "timed_out", "partial_results", "execution_time", "timestamp"):
        assert key in result


def test_invalid_json_records_metrics_and_returns_canonical_shape():
    with patch("mcp_core.server_setup._op_metrics.record") as mock_record:
        mcp = make_mcp()
        result = run(call_run_security_tool(mcp, "nmap", "not-json"))

    assert result["success"] is False
    assert "Invalid JSON" in result["error"]
    assert_canonical_result(result)
    mock_record.assert_called_once()
    telemetry = mock_record.call_args[0][0]
    assert telemetry["tool"] == "nmap"
    assert telemetry["success"] is False


def test_non_object_json_records_metrics_and_returns_canonical_shape():
    with patch("mcp_core.server_setup._op_metrics.record") as mock_record:
        mcp = make_mcp()
        result = run(call_run_security_tool(mcp, "nmap", "[]"))

    assert result["success"] is False
    assert "expected JSON object" in result["error"]
    assert_canonical_result(result)
    mock_record.assert_called_once()


def test_unknown_tool_records_metrics_and_returns_canonical_shape():
    with patch("mcp_core.server_setup._op_metrics.record") as mock_record:
        mcp = make_mcp()
        result = run(call_run_security_tool(mcp, "does_not_exist", {}))

    assert result["success"] is False
    assert "Unknown tool" in result["error"]
    assert_canonical_result(result)
    mock_record.assert_called_once()
    telemetry = mock_record.call_args[0][0]
    assert telemetry["tool"] == "does_not_exist"
    assert telemetry["success"] is False


def test_destructive_denial_records_metrics():
    with patch("mcp_core.wifi_direct.wifi_exec", return_value={"success": True, "output": "ok"}) as mock_exec, \
         patch("mcp_core.server_setup.confirm_destructive_action", new_callable=AsyncMock, return_value=False), \
         patch("mcp_core.server_setup._op_metrics.record") as mock_record:
        mcp = make_mcp()
        result = run(call_run_security_tool(
            mcp,
            "aireplay_ng",
            {"interface": "wlan0mon", "attack_mode": 0, "bssid": "AA:BB:CC:DD:EE:FF"},
        ))

    mock_exec.assert_not_called()
    assert result["success"] is False
    assert_canonical_result(result)
    mock_record.assert_called_once()
    telemetry = mock_record.call_args[0][0]
    assert telemetry["confirmation"] == "denied"


def test_cache_hit_records_metrics_and_normalizes_cached_result():
    from mcp_core.server_setup import _scan_cache

    # Session-scoped key — must match ctx.session_id = "test-session-fixed"
    cache_key = "test-session-fixed:nmap:cached-phase2.example"
    _scan_cache.set(cache_key, {
        "tool": "nmap",
        "target": "cached-phase2.example",
        "result": {
            "success": True,
            "stdout": "cached stdout",
            "return_code": 0,
        },
        "timestamp": 0,
    })

    with patch("mcp_core.net_scan_direct.net_scan_exec") as mock_exec, \
         patch("mcp_core.server_setup._op_metrics.record") as mock_record:
        mcp = make_mcp()
        result = run(call_run_security_tool(mcp, "nmap", {"target": "cached-phase2.example"}))

    mock_exec.assert_not_called()
    assert result["success"] is True
    assert result["output"] == "cached stdout"
    assert result["returncode"] == 0
    assert_canonical_result(result)
    mock_record.assert_called_once()
    telemetry = mock_record.call_args[0][0]
    assert telemetry["cache_hit"] is True
    assert telemetry["success"] is True


# ---------------------------------------------------------------------------
# Phase 5 — _normalize_tool_result coverage
# ---------------------------------------------------------------------------

class TestNormalizeToolResult:
    """Unit tests for mcp_core/server_setup.py::_normalize_tool_result()."""

    def _fn(self):
        from mcp_core.server_setup import _normalize_tool_result
        return _normalize_tool_result

    def test_non_dict_becomes_error(self):
        r = self._fn()(None)
        assert r["success"] is False
        assert "Invalid tool result type" in r["error"]

    def test_non_dict_string_becomes_error(self):
        r = self._fn()("unexpected string")
        assert r["success"] is False
        assert "str" in r["error"]

    def test_already_canonical_passthrough(self):
        canonical = {
            "success": True, "output": "data", "error": "",
            "returncode": 0, "timed_out": False, "partial_results": False,
            "execution_time": 1.0, "timestamp": "ts",
        }
        r = self._fn()(canonical)
        assert r["success"] is True
        assert r["output"] == "data"
        assert r["returncode"] == 0

    def test_stdout_fallback_when_output_missing(self):
        r = self._fn()({"success": True, "stdout": "from stdout"})
        assert r["output"] == "from stdout"

    def test_stderr_fallback_when_output_and_stdout_missing(self):
        r = self._fn()({"success": False, "stderr": "err msg"})
        assert r["error"] == "err msg"

    def test_missing_keys_defaulted(self):
        REQUIRED = {"success", "output", "error", "returncode",
                    "timed_out", "partial_results", "execution_time", "timestamp"}
        r = self._fn()({})
        missing = REQUIRED - set(r.keys())
        assert not missing, f"Missing: {missing}"

    def test_return_code_alias(self):
        r = self._fn()({
            "success": True, "output": "x", "return_code": 42,
        })
        assert r["returncode"] == 42

    def test_timed_out_preserved(self):
        r = self._fn()({"success": False, "timed_out": True})
        assert r["timed_out"] is True

    def test_existing_keys_preserved(self):
        r = self._fn()({
            "success": True, "output": "x", "extra_field": "keep_me",
        })
        assert r["extra_field"] == "keep_me"

    def test_idempotent(self):
        """Calling _normalize_tool_result twice must not corrupt the result."""
        fn = self._fn()
        once = fn({"success": True, "output": "data", "returncode": 0})
        twice = fn(once)
        assert twice["output"] == "data"
        assert twice["returncode"] == 0


# ---------------------------------------------------------------------------
# Phase 5 — typed wrapper vs generic call equivalence
# ---------------------------------------------------------------------------

def test_typed_wrapper_and_generic_produce_same_normalized_output():
    """
    A typed wrapper call and a direct run_security_tool() call for the same
    tool must produce identical canonical result shapes when given the same
    mocked direct-module output.

    Verifies that _create_typed_tool_wrapper() delegates faithfully to
    run_security_tool() and that no extra normalization or key stripping
    happens in the wrapper layer.
    """
    fake_output = {
        "success": True,
        "stdout": "nmap scan done\n",
        "stderr": "",
        "return_code": 0,
        "timed_out": False,
        "partial_results": False,
        "execution_time": 3.1,
        "timestamp": "2026-01-01T00:00:00",
    }

    target = "phase5-equiv.example"

    # Ensure cache is clear for this target so both calls execute
    from mcp_core.server_setup import _scan_cache
    _scan_cache.cache.pop(f"test-session-fixed:nmap:{target}", None)

    with patch("mcp_core.net_scan_direct.net_scan_exec", return_value=fake_output):
        # --- generic call ---
        mcp_generic = make_mcp()
        generic_result = run(call_run_security_tool(
            mcp_generic, "nmap", {"target": target},
        ))

    # Clear cache between the two calls so the typed wrapper also executes
    _scan_cache.cache.pop(f"test-session-fixed:nmap:{target}", None)

    async def call_typed_wrapper(mcp):
        tool = await mcp.get_tool("nmap")
        if tool is None:
            return None
        ctx = make_mock_context()
        ctx.get_state = AsyncMock(return_value=None)
        ctx.set_state = AsyncMock()
        # get_context() requires an active FastMCP request context — patch it
        # to return our mock ctx so typed wrappers can call run_security_tool
        with patch("mcp_core.server_setup.get_context", return_value=ctx):
            return await tool.fn(target=target)

    with patch("mcp_core.net_scan_direct.net_scan_exec", return_value=fake_output):
        mcp_typed = make_mcp()
        typed_result = run(call_typed_wrapper(mcp_typed))

    if typed_result is None:
        import pytest
        pytest.skip("nmap typed wrapper not registered (no registry entry) — skipping equivalence test")

    CANONICAL = {"success", "output", "error", "returncode", "timed_out",
                 "partial_results", "execution_time", "timestamp"}

    for key in CANONICAL:
        assert key in generic_result, f"generic result missing: {key}"
        assert key in typed_result,   f"typed result missing: {key}"

    assert generic_result["success"]  == typed_result["success"]
    assert generic_result["output"]   == typed_result["output"]
    assert generic_result["returncode"] == typed_result["returncode"]
    assert generic_result["timed_out"] == typed_result["timed_out"]


# ---------------------------------------------------------------------------
# Track 1 item 4 — ctx.sample() AI suggestion (opt-in)
# ---------------------------------------------------------------------------

def test_ai_suggest_calls_ctx_sample_on_success():
    """_ai_suggest=True triggers ctx.sample() on successful tool execution."""
    fake_output = {
        "success": True, "stdout": "nmap scan results\n22/tcp open ssh\n80/tcp open http",
        "stderr": "", "return_code": 0, "timed_out": False,
        "partial_results": False, "execution_time": 2.0, "timestamp": "",
    }

    async def run_with_sample():
        tool = await make_mcp().get_tool("run_security_tool")
        ctx = make_mock_context()
        ctx.get_state = AsyncMock(return_value=None)
        ctx.set_state = AsyncMock()
        sample_result = MagicMock()
        sample_result.text = "Next tool: nikto — HTTP port 80 found, web vuln scan recommended."
        ctx.sample = AsyncMock(return_value=sample_result)
        payload = json.dumps({"target": "10.0.0.1", "_ai_suggest": True})
        return await tool.fn(ctx, tool_name="nmap", parameters=payload), ctx

    with patch("mcp_core.net_scan_direct.net_scan_exec", return_value=fake_output):
        result, ctx = run(run_with_sample())

    ctx.sample.assert_awaited_once()
    assert result["success"] is True
    assert "ai_suggestion" in result
    assert "nikto" in result["ai_suggestion"]


def test_ai_suggest_silent_when_sampling_unsupported():
    """ctx.sample() exception must be swallowed — result still returned normally."""
    fake_output = {
        "success": True, "stdout": "scan done", "stderr": "", "return_code": 0,
        "timed_out": False, "partial_results": False, "execution_time": 1.0, "timestamp": "",
    }

    async def run_with_failing_sample():
        tool = await make_mcp().get_tool("run_security_tool")
        ctx = make_mock_context()
        ctx.get_state = AsyncMock(return_value=None)
        ctx.set_state = AsyncMock()
        ctx.sample = AsyncMock(side_effect=RuntimeError("Client does not support sampling"))
        payload = json.dumps({"target": "10.0.0.1", "_ai_suggest": True})
        return await tool.fn(ctx, tool_name="nmap", parameters=payload)

    with patch("mcp_core.net_scan_direct.net_scan_exec", return_value=fake_output):
        result = run(run_with_failing_sample())

    assert result["success"] is True
    assert "ai_suggestion" not in result


def test_ai_suggest_skipped_when_not_requested():
    """Without _ai_suggest=True, ctx.sample() must never be called."""
    fake_output = {
        "success": True, "stdout": "scan done", "stderr": "", "return_code": 0,
        "timed_out": False, "partial_results": False, "execution_time": 1.0, "timestamp": "",
    }

    async def run_without_suggest():
        tool = await make_mcp().get_tool("run_security_tool")
        ctx = make_mock_context()
        ctx.get_state = AsyncMock(return_value=None)
        ctx.set_state = AsyncMock()
        ctx.sample = AsyncMock()
        payload = json.dumps({"target": "10.0.0.1"})
        return await tool.fn(ctx, tool_name="nmap", parameters=payload), ctx

    with patch("mcp_core.net_scan_direct.net_scan_exec", return_value=fake_output):
        result, ctx = run(run_without_suggest())

    ctx.sample.assert_not_awaited()
    assert "ai_suggestion" not in result


def test_health_resource_has_tool_and_metrics_data():
    mcp = make_mcp()
    resource = run(mcp.get_resource("health://server"))
    assert resource is not None
    text = run(resource.fn())
    parsed = json.loads(text)
    assert parsed["status"] == "healthy"
    assert parsed["server"] == "hexstrike-ai-pulse"
    assert "fastmcp" in parsed
    assert parsed["tools_count"] > 0
    assert "cached_scans" in parsed
    assert "cache_stats" in parsed
    assert "op_metrics" in parsed
    assert "uptime_seconds" in parsed


def test_server_health_tool_is_registered():
    mcp = make_mcp()
    tool = run(mcp.get_tool("server_health"))
    assert tool is not None
    assert tool.name == "server_health"


def test_server_health_tool_returns_health_data():
    mcp = make_mcp()
    tool = run(mcp.get_tool("server_health"))
    assert tool is not None
    result = run(tool.fn())
    assert result["status"] == "healthy"
    assert result["server"] == "hexstrike-ai-pulse"
    assert result["tools_count"] > 0
    assert "op_metrics" in result
    assert "uptime_seconds" in result
