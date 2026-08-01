import pytest
from unittest.mock import patch
from pulse.tools.smb_enum_direct import smb_enum_exec


@pytest.fixture
def mock_exec():
    with patch("pulse.tools.smb_enum_direct.execute_command") as m:
        m.return_value = {"success": True, "output": "ok", "returncode": 0}
        yield m


class TestSMBRouting:
    def test_unknown_tool(self):
        result = smb_enum_exec("nonexistent", {})
        assert result["success"] is False
        assert "Unknown" in result["error"]

    @pytest.mark.parametrize("tool", [
        "enum4linux", "enum4linux-ng", "nbtscan", "netexec",
        "nxc", "rpcclient", "smbmap", "evil_winrm",
    ])
    def test_routing_all_tools(self, mock_exec, tool):
        data = {"target": "10.0.0.1"}
        if tool == "evil_winrm":
            data["username"] = "admin"
        result = smb_enum_exec(tool, data)
        assert result["success"] is True
        mock_exec.assert_called_once()


class TestSMBMissingTarget:
    def test_missing_target(self, mock_exec):
        result = smb_enum_exec("enum4linux", {})
        assert result["success"] is False
        assert "'target' is required" in result["error"]
        mock_exec.assert_not_called()

    def test_missing_username_evil_winrm(self, mock_exec):
        result = smb_enum_exec("evil_winrm", {"target": "10.0.0.1"})
        assert result["success"] is False
        assert "'username' is required" in result["error"]
        mock_exec.assert_not_called()


class TestSMBShellMetachars:
    @pytest.mark.parametrize("tool", [
        "enum4linux", "enum4linux-ng", "nbtscan", "netexec",
        "smbmap", "evil_winrm",
    ])
    @pytest.mark.parametrize("bad", ["10.0.0.1;id", "10.0.0.1 && ls", "10.0.0.1$(id)"])
    def test_rejects_metachars(self, mock_exec, tool, bad):
        data = {"target": bad}
        if tool == "evil_winrm":
            data["username"] = "admin"
        result = smb_enum_exec(tool, data)
        assert result["success"] is False
        assert "disallowed character sequence" in result["error"]
        mock_exec.assert_not_called()

    def test_rpcclient_allows_pipe_commands(self, mock_exec):
        """rpcclient builds an echo | rpcclient pipeline by design —
        commands are shlex.quote()d, not rejected."""
        result = smb_enum_exec("rpcclient", {
            "target": "10.0.0.1",
            "commands": "enumdomusers;querydominfo",
        })
        assert result["success"] is True
        called = mock_exec.call_args[0][0]
        assert "|" in called

    @pytest.mark.parametrize("tool", [
        "enum4linux", "enum4linux-ng", "nbtscan", "netexec",
        "nxc", "rpcclient", "smbmap", "evil_winrm",
    ])
    def test_clean_target_passes(self, mock_exec, tool):
        data = {"target": "10.0.0.1"}
        if tool == "evil_winrm":
            data["username"] = "admin"
        result = smb_enum_exec(tool, data)
        assert result["success"] is True
