"""
tests/test_wifi_mcp_tools.py

Unit tests for mcp_tools/wifi_pentest/* — no hardware, no Flask server needed.

Strategy:
- Patch mcp_core.wifi_direct.wifi_exec to intercept direct calls (Phase 1)
- Register tools against a real FastMCP instance
- Use FastMCP 3.1 public API: await mcp.get_tool()
- Inject a mock Context for ctx: Context param
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from fastmcp import FastMCP


# ---------------------------------------------------------------------------
# Mock Context — satisfies ctx: Context parameter
# ---------------------------------------------------------------------------

def make_mock_context():
    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.debug = AsyncMock()
    ctx.report_progress = AsyncMock()
    return ctx


# ---------------------------------------------------------------------------
# Shared mock wifi_exec
# ---------------------------------------------------------------------------

def make_mock_wifi_exec(success=True, extra=None):
    response = {"success": success, "output": "mocked", "returncode": 0}
    if extra:
        response.update(extra)
    return MagicMock(return_value=response)


def make_mcp_with(register_fn, logger=None):
    mcp = FastMCP("test-hexstrike")
    if logger is None:
        logger = MagicMock()
    # hexstrike_client is no longer used by wifi tools — pass a dummy
    register_fn(mcp, MagicMock(), logger)
    return mcp, logger


async def call_tool(mcp, name, **kwargs):
    """Call a registered tool — injects a mock ctx as first positional arg."""
    tool = await mcp.get_tool(name)
    assert tool is not None, f"Tool '{name}' not registered."
    ctx = make_mock_context()
    return await tool.fn(ctx, **kwargs)


async def assert_tool_registered(mcp, name):
    tool = await mcp.get_tool(name)
    assert tool is not None, f"Tool '{name}' not registered."


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ===========================================================================
# airmon_ng
# ===========================================================================

class TestAirmonNg:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mcp_tools.wifi_pentest.airmon_ng import register_airmon_ng_tools
        self.mcp, self.logger = make_mcp_with(register_airmon_ng_tools)

    def test_tool_registered(self):
        run(assert_tool_registered(self.mcp, "airmon_ng"))

    def test_check_kill_calls_correct_tool(self):
        mock = make_mock_wifi_exec()
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            run(call_tool(self.mcp, "airmon_ng", interface="wlan0", action="check kill"))
        mock.assert_called_once()
        tool, data = mock.call_args[0]
        assert tool == "airmon_ng"
        assert data["interface"] == "wlan0"
        assert data["action"] == "check kill"

    def test_start_monitor_mode(self):
        mock = make_mock_wifi_exec()
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            run(call_tool(self.mcp, "airmon_ng", interface="wlan0", action="start"))
        _, data = mock.call_args[0]
        assert data["action"] == "start"

    def test_start_with_channel(self):
        mock = make_mock_wifi_exec()
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            run(call_tool(self.mcp, "airmon_ng", interface="wlan0", action="start", channel="6"))
        _, data = mock.call_args[0]
        assert data["channel"] == "6"

    def test_stop_monitor_mode(self):
        mock = make_mock_wifi_exec()
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            run(call_tool(self.mcp, "airmon_ng", interface="wlan0mon", action="stop"))
        _, data = mock.call_args[0]
        assert data["action"] == "stop"
        assert data["interface"] == "wlan0mon"

    def test_returns_success_result(self):
        mock = make_mock_wifi_exec(success=True)
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            result = run(call_tool(self.mcp, "airmon_ng", interface="wlan0", action="start"))
        assert result["success"] is True

    def test_returns_failure_result(self):
        mock = make_mock_wifi_exec(success=False, extra={"error": "permission denied"})
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            result = run(call_tool(self.mcp, "airmon_ng", interface="wlan0", action="start"))
        assert result["success"] is False


# ===========================================================================
# airodump_ng
# ===========================================================================

class TestAirodumpNg:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mcp_tools.wifi_pentest.airodump_ng import register_airodump_ng_tools
        self.mcp, self.logger = make_mcp_with(register_airodump_ng_tools)

    def test_tool_registered(self):
        run(assert_tool_registered(self.mcp, "airodump_ng"))

    def test_calls_correct_tool(self):
        mock = make_mock_wifi_exec()
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            run(call_tool(self.mcp, "airodump_ng", interface="wlan0mon"))
        tool, data = mock.call_args[0]
        assert tool == "airodump_ng"
        assert data["interface"] == "wlan0mon"

    def test_bssid_filter(self):
        mock = make_mock_wifi_exec()
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            run(call_tool(self.mcp, "airodump_ng", interface="wlan0mon", bssid="AA:BB:CC:DD:EE:FF"))
        _, data = mock.call_args[0]
        assert data["bssid"] == "AA:BB:CC:DD:EE:FF"

    def test_channel_filter(self):
        mock = make_mock_wifi_exec()
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            run(call_tool(self.mcp, "airodump_ng", interface="wlan0mon", channel="6"))
        _, data = mock.call_args[0]
        assert data["channel"] == "6"


# ===========================================================================
# aireplay_ng
# ===========================================================================

class TestAireplayNg:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mcp_tools.wifi_pentest.aireplay_ng import register_aireplay_ng_tools
        self.mcp, self.logger = make_mcp_with(register_aireplay_ng_tools)

    def test_tool_registered(self):
        run(assert_tool_registered(self.mcp, "aireplay_ng"))

    def test_deauth_calls_correct_tool(self):
        mock = make_mock_wifi_exec()
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            run(call_tool(self.mcp, "aireplay_ng",
                interface="wlan0mon", attack_mode="deauth",
                bssid="AA:BB:CC:DD:EE:FF", count="10"))
        tool, data = mock.call_args[0]
        assert tool == "aireplay_ng"
        assert data["attack_mode"] == "deauth"
        assert data["bssid"] == "AA:BB:CC:DD:EE:FF"

    def test_arp_replay_attack(self):
        mock = make_mock_wifi_exec()
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            run(call_tool(self.mcp, "aireplay_ng",
                interface="wlan0mon", attack_mode="arp-replay",
                bssid="AA:BB:CC:DD:EE:FF"))
        _, data = mock.call_args[0]
        assert data["attack_mode"] == "arp-replay"


# ===========================================================================
# aircrack_ng
# ===========================================================================

class TestAircrackNg:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mcp_tools.wifi_pentest.aircrack_ng import register_aircrack_ng_tools
        self.mcp, self.logger = make_mcp_with(register_aircrack_ng_tools)

    def test_tool_registered(self):
        run(assert_tool_registered(self.mcp, "aircrack_ng"))

    def test_calls_correct_tool(self):
        mock = make_mock_wifi_exec()
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            run(call_tool(self.mcp, "aircrack_ng",
                capture_files=["/tmp/capture.cap"],
                wordlist="/usr/share/wordlists/rockyou.txt"))
        tool, data = mock.call_args[0]
        assert tool == "aircrack_ng"
        assert data["capture_files"] == ["/tmp/capture.cap"]
        assert data["wordlist"] == "/usr/share/wordlists/rockyou.txt"

    def test_multiple_capture_files(self):
        mock = make_mock_wifi_exec()
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            run(call_tool(self.mcp, "aircrack_ng",
                capture_files=["/tmp/cap1.cap", "/tmp/cap2.cap"],
                wordlist="/usr/share/wordlists/rockyou.txt"))
        _, data = mock.call_args[0]
        assert len(data["capture_files"]) == 2

    def test_with_bssid_filter(self):
        mock = make_mock_wifi_exec()
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            run(call_tool(self.mcp, "aircrack_ng",
                capture_files=["/tmp/capture.cap"],
                wordlist="/usr/share/wordlists/rockyou.txt",
                bssid="AA:BB:CC:DD:EE:FF"))
        _, data = mock.call_args[0]
        assert data["bssid"] == "AA:BB:CC:DD:EE:FF"


# ===========================================================================
# hcxdumptool
# ===========================================================================

class TestHcxdumptool:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mcp_tools.wifi_pentest.hcxdumptool import register_hcxdumptool_tools
        self.mcp, self.logger = make_mcp_with(register_hcxdumptool_tools)

    def test_tool_registered(self):
        run(assert_tool_registered(self.mcp, "hcxdumptool"))

    def test_calls_correct_tool(self):
        mock = make_mock_wifi_exec()
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            run(call_tool(self.mcp, "hcxdumptool", interface="wlan0mon"))
        tool, data = mock.call_args[0]
        assert tool == "hcxdumptool"
        assert data["interface"] == "wlan0mon"


# ===========================================================================
# hcxpcapngtool
# ===========================================================================

class TestHcxpcapngtool:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mcp_tools.wifi_pentest.hcxpcapngtool import register_hcxpcapngtool_tools
        self.mcp, self.logger = make_mcp_with(register_hcxpcapngtool_tools)

    def test_tool_registered(self):
        run(assert_tool_registered(self.mcp, "hcxpcapngtool"))

    def test_calls_correct_tool(self):
        mock = make_mock_wifi_exec()
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            run(call_tool(self.mcp, "hcxpcapngtool",
                input_file="/tmp/capture.pcapng",
                output_file="/tmp/hashes.hc22000"))
        tool, data = mock.call_args[0]
        assert tool == "hcxpcapngtool"
        assert data["input_file"] == "/tmp/capture.pcapng"


# ===========================================================================
# wifite2
# ===========================================================================

class TestWifite2:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mcp_tools.wifi_pentest.wifite2 import register_wifite2_tools
        self.mcp, self.logger = make_mcp_with(register_wifite2_tools)

    def test_tool_registered(self):
        run(assert_tool_registered(self.mcp, "wifite2"))

    def test_calls_correct_tool(self):
        mock = make_mock_wifi_exec()
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            run(call_tool(self.mcp, "wifite2", interface="wlan0mon"))
        tool, data = mock.call_args[0]
        assert tool == "wifite2"
        assert data["interface"] == "wlan0mon"


# ===========================================================================
# eaphammer
# ===========================================================================

class TestEaphammer:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mcp_tools.wifi_pentest.eaphammer import register_eaphammer_tools
        self.mcp, self.logger = make_mcp_with(register_eaphammer_tools)

    def test_tool_registered(self):
        run(assert_tool_registered(self.mcp, "eaphammer"))

    def test_calls_correct_tool(self):
        mock = make_mock_wifi_exec()
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            run(call_tool(self.mcp, "eaphammer",
                interface="wlan0mon", essid="TargetCorp", attack_type="creds"))
        tool, data = mock.call_args[0]
        assert tool == "eaphammer"
        assert data["essid"] == "TargetCorp"
        assert data["attack_type"] == "creds"

    def test_wpa_enterprise_mode(self):
        mock = make_mock_wifi_exec()
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            run(call_tool(self.mcp, "eaphammer",
                interface="wlan0mon", essid="CorpNet", auth_mode="wpa-eap"))
        _, data = mock.call_args[0]
        assert data["auth_mode"] == "wpa-eap"


# ===========================================================================
# bettercap_wifi
# ===========================================================================

class TestBettercapWifi:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mcp_tools.wifi_pentest.bettercap_wifi import register_bettercap_wifi_tools
        self.mcp, self.logger = make_mcp_with(register_bettercap_wifi_tools)

    def test_tool_registered(self):
        run(assert_tool_registered(self.mcp, "bettercap_wifi"))

    def test_calls_correct_tool(self):
        mock = make_mock_wifi_exec()
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            run(call_tool(self.mcp, "bettercap_wifi", interface="wlan0mon"))
        tool, data = mock.call_args[0]
        assert tool == "bettercap_wifi"
        assert data["interface"] == "wlan0mon"


# ===========================================================================
# mdk4
# ===========================================================================

class TestMdk4:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mcp_tools.wifi_pentest.mdk4 import register_mdk4_tools
        self.mcp, self.logger = make_mcp_with(register_mdk4_tools)

    def test_tool_registered(self):
        run(assert_tool_registered(self.mcp, "mdk4"))

    def test_calls_correct_tool(self):
        mock = make_mock_wifi_exec()
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            run(call_tool(self.mcp, "mdk4", interface="wlan0mon", attack_mode="b"))
        tool, data = mock.call_args[0]
        assert tool == "mdk4"
        assert data["attack_mode"] == "b"


# ===========================================================================
# airbase_ng
# ===========================================================================

class TestAirbaseNg:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mcp_tools.wifi_pentest.airbase_ng import register_airbase_ng_tools
        self.mcp, self.logger = make_mcp_with(register_airbase_ng_tools)

    def test_tool_registered(self):
        run(assert_tool_registered(self.mcp, "airbase_ng"))

    def test_calls_correct_tool(self):
        mock = make_mock_wifi_exec()
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            run(call_tool(self.mcp, "airbase_ng", interface="wlan0mon", essid="EvilTwin"))
        tool, data = mock.call_args[0]
        assert tool == "airbase_ng"
        assert data["essid"] == "EvilTwin"

    def test_wpa2_mode(self):
        mock = make_mock_wifi_exec()
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            run(call_tool(self.mcp, "airbase_ng",
                interface="wlan0mon", essid="EvilTwin", wpa_mode="wpa2"))
        _, data = mock.call_args[0]
        assert data["wpa_mode"] == "wpa2"


# ===========================================================================
# airdecap_ng
# ===========================================================================

class TestAirdecapNg:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mcp_tools.wifi_pentest.airdecap_ng import register_airdecap_ng_tools
        self.mcp, self.logger = make_mcp_with(register_airdecap_ng_tools)

    def test_tool_registered(self):
        run(assert_tool_registered(self.mcp, "airdecap_ng"))

    def test_calls_correct_tool(self):
        mock = make_mock_wifi_exec()
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            run(call_tool(self.mcp, "airdecap_ng",
                capture_file="/tmp/capture.cap",
                password="password123"))
        tool, data = mock.call_args[0]
        assert tool == "airdecap_ng"
        assert data["capture_file"] == "/tmp/capture.cap"
        assert data["password"] == "password123"

    def test_wep_decryption(self):
        mock = make_mock_wifi_exec()
        with patch("mcp_core.wifi_direct.wifi_exec", mock):
            run(call_tool(self.mcp, "airdecap_ng",
                capture_file="/tmp/capture.cap",
                wep_key="AABBCCDDEE"))
        _, data = mock.call_args[0]
        assert data["wep_key"] == "AABBCCDDEE"
