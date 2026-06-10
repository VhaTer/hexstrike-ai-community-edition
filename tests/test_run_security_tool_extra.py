import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


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
    from mcp_core.server_setup import run_security_tool as _run_security_tool
    ctx = make_mock_context()
    payload = parameters if isinstance(parameters, str) else json.dumps(parameters)
    return await _run_security_tool(ctx, tool_name, payload)


def test_confirmation_accepted_executes_and_records_telemetry():
    with patch("mcp_core.wifi_direct.wifi_exec", return_value={"success": True, "output": "ok"}) as mock_exec, \
         patch("mcp_core.server_setup.confirm_destructive_action", new_callable=AsyncMock, return_value=True) as mock_confirm, \
         patch("mcp_core.server_setup._op_metrics.record") as mock_record:
        mcp = make_mcp()
        result = run(call_run_security_tool(
            mcp,
            "aireplay_ng",
            {"interface": "wlan0mon", "attack_mode": 0, "bssid": "AA:BB:CC:DD:EE:FF"},
        ))

    mock_confirm.assert_awaited_once()
    mock_exec.assert_called_once()
    assert result["success"] is True
    mock_record.assert_called_once()
    telemetry = mock_record.call_args[0][0]
    assert telemetry["confirmation"] == "accepted"


def test_success_writes_cache_and_records_metrics():
    from mcp_core.server_setup import _scan_cache

    target = "cache-write.example"
    cache_key = f"test-session-fixed:nmap:{target}"
    # ensure cache is clear
    _scan_cache.cache.pop(cache_key, None)

    fake_output = {"success": True, "stdout": "done", "return_code": 0, "execution_time": 1.5}

    with patch("mcp_core.net_scan_direct.net_scan_exec", return_value=fake_output) as mock_exec, \
         patch("mcp_core.server_setup._op_metrics.record") as mock_record:
        mcp = make_mcp()
        result = run(call_run_security_tool(mcp, "nmap", {"target": target}))

    assert result["success"] is True
    # cache now contains the entry
    entry = _scan_cache.get(cache_key)
    assert entry is not None
    assert entry["tool"] == "nmap"
    mock_record.assert_called_once()
    telemetry = mock_record.call_args[0][0]
    assert telemetry.get("cache_hit") in (False, True)


def test_rate_limit_sets_state_and_records_telemetry():
    fake_output = {"success": True, "stdout": "ok", "return_code": 0}

    # Use a custom ctx directly in call to verify set_state
    async def _call_with_ctx(mcp):
        from mcp_core.server_setup import run_security_tool as _run_security_tool
        ctx = make_mock_context()
        payload = json.dumps({"target": "ratelimit.example"})
        return await _run_security_tool(ctx, "nmap", payload), ctx

    with patch("mcp_core.net_scan_direct.net_scan_exec", return_value=fake_output), \
         patch("mcp_core.server_setup._rate_limiter.detect_rate_limiting",
               return_value={"detected": True, "recommended_profile": "stealth", "confidence": 0.9}) as mock_rl, \
         patch("mcp_core.server_setup._op_metrics.record") as mock_record:
        mcp = make_mcp()
        result, ctx = run(_call_with_ctx(mcp))

    mock_rl.assert_called()
    ctx.set_state.assert_awaited()  # ratelimit profile persisted to session
    mock_record.assert_called_once()
    telemetry = mock_record.call_args[0][0]
    assert "rate_limit" in telemetry
    assert telemetry["rate_limit"] == "stealth"
    assert result["success"] is True


def test_optimizer_forced_stealth_applies_and_is_recorded():
    # Patch the optimizer to simulate forcing stealth/profile changes
    fake_output = {"success": True, "stdout": "ok", "return_code": 0}

    with patch("mcp_core.net_scan_direct.net_scan_exec", return_value=fake_output) as mock_exec, \
         patch("mcp_core.server_setup._optimizer.optimize") as mock_optimize, \
         patch("mcp_core.server_setup._op_metrics.record") as mock_record:

        # simulate optimizer modifying params and returning meta information
        mock_optimize.return_value = {"opt_profile": "stealth", "optimizer_meta": {"forced_stealth": True}}

        mcp = make_mcp()
        result = run(call_run_security_tool(mcp, "nmap", {"target": "opt.example"}))

    mock_optimize.assert_called()
    mock_record.assert_called_once()
    assert result["success"] is True


def test_timeout_and_partial_results_recorded():
    # Simulate an execution that timed out and produced partial results
    fake_output = {"success": False, "stdout": "", "return_code": 1, "timed_out": True, "partial_results": True}

    with patch("mcp_core.net_scan_direct.net_scan_exec", return_value=fake_output), \
         patch("mcp_core.server_setup._op_metrics.record") as mock_record:
        mcp = make_mcp()
        result = run(call_run_security_tool(mcp, "nmap", {"target": "timeout.example"}))

    mock_record.assert_called_once()
    telemetry = mock_record.call_args[0][0]
    # timed_out is copied to telemetry from result
    assert telemetry.get("timed_out") is True
    assert telemetry["success"] is False
    # partial_results lives in the result dict, not telemetry
    assert result.get("partial_results") is True
