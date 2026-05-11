import pytest
from unittest.mock import patch
from mcp_core.active_directory_direct import ad_exec, _require


@pytest.fixture
def mock_exec():
    with patch("mcp_core.active_directory_direct.execute_command") as m:
        m.return_value = {"success": True, "output": "ok", "returncode": 0}
        yield m


class TestADRequire:
    def test_require_all_present(self):
        assert _require({"a": "x", "b": "y"}, "a", "b") == {}

    def test_require_missing(self):
        result = _require({"a": "x"}, "a", "b")
        assert result == {"success": False, "error": "'b' is required"}

    def test_require_empty(self):
        result = _require({"a": ""}, "a")
        assert result == {"success": False, "error": "'a' is required"}


class TestADExecRouting:
    def test_unknown_tool(self):
        result = ad_exec("nonexistent", {})
        assert result["success"] is False
        assert "Unknown" in result["error"]

    def test_impacket_routing(self, mock_exec):
        result = ad_exec("impacket", {"script": "GetADUsers", "target": "dc.example.com"})
        assert result["success"] is True

    def test_ldapdomaindump_routing(self, mock_exec):
        result = ad_exec("ldapdomaindump", {"hostname": "dc.example.com"})
        assert result["success"] is True

    def test_adidnsdump_routing(self, mock_exec):
        result = ad_exec("adidnsdump", {"target": "dc.example.com"})
        assert result["success"] is True

    def test_certipy_ad_routing(self, mock_exec):
        result = ad_exec("certipy_ad", {"action": "find", "target": "dc.example.com"})
        assert result["success"] is True

    def test_mitm6_routing(self, mock_exec):
        result = ad_exec("mitm6", {"interface": "eth0", "domain": "example.local"})
        assert result["success"] is True

    def test_pywerview_routing(self, mock_exec):
        result = ad_exec("pywerview", {"request": "get-netuser", "target": "dc.example.com"})
        assert result["success"] is True

    def test_bloodhound_routing(self, mock_exec):
        result = ad_exec("bloodhound", {"domain": "example.local", "username": "admin", "password": "pass", "dc_ip": "10.0.0.1"})
        assert result["success"] is True

    def test_all_tools_covered(self):
        from mcp_core.active_directory_direct import _HANDLERS
        expected = {"impacket", "ldapdomaindump", "adidnsdump", "certipy_ad", "certipy", "mitm6", "pywerview", "bloodhound", "bloodhound_python"}
        assert set(_HANDLERS.keys()) == expected


class TestImpacket:
    def test_missing_script(self, mock_exec):
        result = ad_exec("impacket", {"target": "dc.example.com"})
        assert result["success"] is False
        assert "script" in result["error"]

    def test_missing_target(self, mock_exec):
        result = ad_exec("impacket", {"script": "GetADUsers"})
        assert result["success"] is False
        assert "target" in result["error"]

    def test_unsupported_script(self, mock_exec):
        result = ad_exec("impacket", {"script": "FakeScript", "target": "dc.example.com"})
        assert result["success"] is False
        assert "Unsupported" in result["error"]

    def test_command_with_options(self, mock_exec):
        ad_exec("impacket", {
            "script": "GetADUsers", "target": "dc.example.com",
            "options": {"dc-ip": "10.0.0.1", "all": True},
        })
        cmd = mock_exec.call_args[0][0]
        assert "impacket-GetADUsers" in cmd
        assert "-dc-ip 10.0.0.1" in cmd
        assert "-all" in cmd
        assert "dc.example.com" in cmd

    def test_command_with_extra_args(self, mock_exec):
        ad_exec("impacket", {
            "script": "GetADUsers", "target": "dc.example.com",
            "extra_args": "-debug -no-pass",
        })
        cmd = mock_exec.call_args[0][0]
        assert "-debug" in cmd
        assert "-no-pass" in cmd


class TestLdapdomaindump:
    def test_missing_hostname(self, mock_exec):
        result = ad_exec("ldapdomaindump", {})
        assert result["success"] is False
        assert "hostname" in result["error"]

    def test_default_authtype(self, mock_exec):
        ad_exec("ldapdomaindump", {"hostname": "dc.example.com"})
        cmd = mock_exec.call_args[0][0]
        assert "ldapdomaindump dc.example.com --authtype NTLM" in cmd

    def test_custom_authtype(self, mock_exec):
        ad_exec("ldapdomaindump", {"hostname": "dc.example.com", "authtype": "Basic"})
        cmd = mock_exec.call_args[0][0]
        assert "--authtype Basic" in cmd


class TestMitm6:
    def test_defaults(self, mock_exec):
        ad_exec("mitm6", {"interface": "eth0"})
        cmd = mock_exec.call_args[0][0]
        assert "mitm6 -i eth0" in cmd
        assert "-d" not in cmd

    def test_with_domain(self, mock_exec):
        ad_exec("mitm6", {"interface": "eth0", "domain": "example.local"})
        cmd = mock_exec.call_args[0][0]
        assert "-i eth0" in cmd
        assert "-d example.local" in cmd


class TestBloodhound:
    def test_command_format(self, mock_exec):
        ad_exec("bloodhound", {
            "domain": "example.local", "username": "admin",
            "password": "secret", "dc_ip": "10.0.0.1",
        })
        cmd = mock_exec.call_args[0][0]
        assert "bloodhound-python" in cmd
        assert "-u admin" in cmd
        assert "-p secret" in cmd
        assert "-d example.local" in cmd
        assert "--dc 10.0.0.1" in cmd

    def test_missing_domain(self, mock_exec):
        result = ad_exec("bloodhound", {"username": "admin", "password": "pass", "dc_ip": "10.0.0.1"})
        assert result["success"] is False
        assert "domain" in result["error"]

    def test_missing_dc_ip(self, mock_exec):
        result = ad_exec("bloodhound", {"domain": "example.local", "username": "admin", "password": "pass"})
        assert result["success"] is False
        assert "dc_ip" in result["error"]
