import json
from unittest.mock import patch

from mcp_core.password_cracking_direct import pwdcrack_exec
from server_core.advanced_cache import AdvancedCache


def test_advanced_cache_supports_len_and_items():
    cache = AdvancedCache(max_size=10, default_ttl=60)
    cache.set("one", {"value": 1})
    cache.set("two", {"value": 2})

    assert len(cache) == 2
    assert dict(cache.items()) == {
        "one": {"value": 1},
        "two": {"value": 2},
    }
    assert set(cache.keys()) == {"one", "two"}
    assert cache.values()


def test_hashid_accepts_typed_tool_param_name():
    mock_result = {"success": True, "output": "MD5", "return_code": 0}

    with patch("mcp_core.password_cracking_direct.execute_command", return_value=mock_result) as mock_exec:
        result = pwdcrack_exec("hashid", {"hash": "5f4dcc3b5aa765d61d8327deb882cf99"})

    mock_exec.assert_called_once()
    assert "hashid 5f4dcc3b5aa765d61d8327deb882cf99" in mock_exec.call_args.args[0]
    assert result["success"] is True


def test_hashid_still_accepts_legacy_hash_value_param():
    mock_result = {"success": True, "output": "MD5", "return_code": 0}

    with patch("mcp_core.password_cracking_direct.execute_command", return_value=mock_result) as mock_exec:
        result = pwdcrack_exec("hashid", {"hash_value": "5f4dcc3b5aa765d61d8327deb882cf99"})

    mock_exec.assert_called_once()
    assert "hashid 5f4dcc3b5aa765d61d8327deb882cf99" in mock_exec.call_args.args[0]
    assert result["success"] is True


def test_hashid_requires_hash_input():
    result = pwdcrack_exec("hashid", {})

    assert result == {"success": False, "error": "'hash_value' is required"}
