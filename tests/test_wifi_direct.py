import pytest
from unittest.mock import patch
from mcp_core.wifi_direct import wifi_exec, _require


@pytest.fixture
def mock_exec():
    with patch("mcp_core.wifi_direct.execute_command") as m:
        m.return_value = {"success": True, "output": "ok", "returncode": 0}
        yield m


class TestWifiRequire:
    def test_require_all_present(self):
        assert _require({"a": "x", "b": "y"}, "a", "b") == {}

    def test_require_missing(self):
        result = _require({"a": "x"}, "a", "b")
        assert result == {"success": False, "error": "'b' is required"}

    def test_require_empty_value(self):
        result = _require({"a": "", "b": "y"}, "a")
        assert result == {"success": False, "error": "'a' is required"}


class TestWifiExecRouting:
    def test_unknown_tool(self):
        result = wifi_exec("nonexistent", {})
        assert result["success"] is False
        assert "Unknown" in result["error"]

    def test_airmon_ng_routing(self, mock_exec):
        result = wifi_exec("airmon_ng", {"interface": "wlan0", "action": "start"})
        assert result["success"] is True

    def test_airodump_ng_routing(self, mock_exec):
        result = wifi_exec("airodump_ng", {"interface": "wlan0mon", "bssid": "aa:bb:cc:dd:ee:ff"})
        assert result["success"] is True

    def test_aireplay_routing(self, mock_exec):
        result = wifi_exec("aireplay_ng", {"interface": "wlan0mon", "attack_mode": 3, "bssid": "aa:bb:cc:dd:ee:ff"})
        assert result["success"] is True

    def test_aircrack_routing(self, mock_exec):
        result = wifi_exec("aircrack_ng", {"capture_files": ["capture.cap"], "wordlist": "rockyou.txt"})
        assert result["success"] is True

    def test_tcpdump_routing(self, mock_exec):
        result = wifi_exec("tcpdump", {"interface": "eth0"})
        assert result["success"] is True

    def test_tshark_routing(self, mock_exec):
        result = wifi_exec("tshark", {"interface": "eth0"})
        assert result["success"] is True

    def test_all_tools_covered(self):
        from mcp_core.wifi_direct import _HANDLERS
        expected = {
            "airmon_ng", "airodump_ng", "aireplay_ng", "aircrack_ng",
            "airbase_ng", "airdecap_ng", "hcxdumptool", "hcxpcapngtool",
            "wifite2", "eaphammer", "bettercap_wifi", "bettercap",
            "tcpdump", "tshark", "mdk4",
        }
        assert set(_HANDLERS.keys()) == expected


class TestAirmonNg:
    def test_missing_interface(self, mock_exec):
        result = wifi_exec("airmon_ng", {"action": "start"})
        assert result["success"] is False
        assert "interface" in result["error"]

    def test_missing_action(self, mock_exec):
        result = wifi_exec("airmon_ng", {"interface": "wlan0"})
        assert result["success"] is False
        assert "action" in result["error"]

    def test_invalid_action(self, mock_exec):
        result = wifi_exec("airmon_ng", {"interface": "wlan0", "action": "invalid"})
        assert result["success"] is False
        assert "action must be" in result["error"]

    def test_start_monitor(self, mock_exec):
        wifi_exec("airmon_ng", {"interface": "wlan0", "action": "start"})
        cmd = mock_exec.call_args[0][0]
        assert "airmon-ng start wlan0" in cmd

    def test_stop_monitor(self, mock_exec):
        wifi_exec("airmon_ng", {"interface": "wlan0", "action": "stop"})
        cmd = mock_exec.call_args[0][0]
        assert "airmon-ng stop wlan0" in cmd

    def test_check_kill(self, mock_exec):
        wifi_exec("airmon_ng", {"interface": "wlan0", "action": "check kill"})
        cmd = mock_exec.call_args[0][0]
        assert cmd == "airmon-ng check kill"

    def test_start_with_channel(self, mock_exec):
        wifi_exec("airmon_ng", {"interface": "wlan0", "action": "start", "channel": "6"})
        cmd = mock_exec.call_args[0][0]
        assert "airmon-ng start wlan0 6" in cmd

    def test_no_cache_for_airmon(self, mock_exec):
        wifi_exec("airmon_ng", {"interface": "wlan0", "action": "start"})
        assert mock_exec.call_args[1].get("use_cache") is False
