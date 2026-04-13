import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch


def make_mock_context():
    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.debug = AsyncMock()
    ctx.report_progress = AsyncMock()
    ctx.read_resource = AsyncMock(return_value=MagicMock(contents=[]))
    return ctx


async def call_run_security_tool(mcp, tool_name, parameters):
    tool = await mcp.get_tool("run_security_tool")
    assert tool is not None
    ctx = make_mock_context()
    return await tool.fn(ctx, tool_name=tool_name, parameters=json.dumps(parameters))


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def make_mcp():
    from mcp_core.server_setup import setup_mcp_server_standalone
    return setup_mcp_server_standalone(MagicMock())


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
