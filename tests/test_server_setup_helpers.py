import pytest

from mcp_core import server_setup


def test_normalize_tool_result_passthrough():
    res = {"success": True, "output": "ok", "return_code": 0}
    norm = server_setup._normalize_tool_result(res)
    assert norm["success"] is True
    assert norm["output"] == "ok"
    assert norm["returncode"] == 0


def test_normalize_tool_result_legacy_keys():
    res = {"stdout": "out", "stderr": "err", "return_code": 2}
    norm = server_setup._normalize_tool_result(res)
    assert norm["output"] == "out"
    assert norm["error"] == "err"
    assert norm["returncode"] == 2


def test_build_destructive_confirmation_aireplay():
    params = {"interface": "wlan0", "attack_mode": 0, "bssid": "AA:BB:CC:DD:EE:FF"}
    msg = server_setup._build_destructive_confirmation("aireplay_ng", params)
    assert msg is not None
    assert "aireplay-ng" in msg["action"] or "aireplay" in msg["action"]


def test_build_destructive_confirmation_metasploit_aux_allowed():
    params = {"module": "auxiliary/scanner/ssh/ssh_version", "options": {"RHOSTS": "10.0.0.1"}}
    msg = server_setup._build_destructive_confirmation("metasploit", params)
    assert msg is None  # auxiliary modules allowed without confirmation
