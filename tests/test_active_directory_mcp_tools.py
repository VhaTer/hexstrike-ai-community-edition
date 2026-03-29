"""
tests/test_active_directory_mcp_tools.py

Unit tests for mcp_tools/active_directory/* — no network, no Flask needed.

Strategy:
- Patch mcp_core.active_directory_direct.ad_exec to intercept direct calls
- Register tools against a real FastMCP instance
- Use FastMCP 3.x public API: await mcp.get_tool()
- Inject a mock Context for ctx: Context param
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from fastmcp import FastMCP


# ---------------------------------------------------------------------------
# Helpers — same pattern as test_wifi/osint_mcp_tools.py
# ---------------------------------------------------------------------------

def make_elicit_result(accepted=True):
    result = MagicMock()
    result.action = "accept" if accepted else "decline"
    result.data = accepted
    return result


def make_mock_context(elicit_accepted=True):
    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.debug = AsyncMock()
    ctx.report_progress = AsyncMock()
    ctx.elicit = AsyncMock(return_value=make_elicit_result(elicit_accepted))
    return ctx


def make_mock_ad_exec(success=True, extra=None):
    response = {"success": success, "output": "mocked", "returncode": 0}
    if extra:
        response.update(extra)
    return MagicMock(return_value=response)


def make_mcp_with(register_fn, logger=None):
    mcp = FastMCP("test-hexstrike-ad")
    if logger is None:
        logger = MagicMock()
    register_fn(mcp, MagicMock(), logger)
    return mcp, logger


async def call_tool(mcp, name, **kwargs):
    tool = await mcp.get_tool(name)
    assert tool is not None, f"Tool '{name}' not registered."
    ctx = make_mock_context(elicit_accepted=True)
    return await tool.fn(ctx, **kwargs)


async def _call_tool_with_elicit(mcp, name, elicit_accepted=True, **kwargs):
    tool = await mcp.get_tool(name)
    assert tool is not None, f"Tool '{name}' not registered."
    ctx = make_mock_context(elicit_accepted=elicit_accepted)
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
# impacket_run
# ===========================================================================

class TestImpacket:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mcp_tools.active_directory.impacket_scripts import register_impacket
        with patch("mcp_core.active_directory_direct.ad_exec", make_mock_ad_exec()) as mock:
            self.mock_exec = mock
            self.mcp, _ = make_mcp_with(register_impacket)
            yield

    def test_tool_registered(self):
        run(assert_tool_registered(self.mcp, "impacket_run"))

    def test_calls_correct_tool(self):
        result = run(call_tool(
            self.mcp, "impacket_run",
            script="GetADUsers",
            target="corp.local/user:pass",
        ))
        self.mock_exec.assert_called_once_with("impacket", {
            "script":     "GetADUsers",
            "target":     "corp.local/user:pass",
            "options":    {},
            "extra_args": "",
        })
        assert result["success"] is True

    def test_calls_with_options(self):
        result = run(call_tool(
            self.mcp, "impacket_run",
            script="GetNPUsers",
            target="corp.local/",
            options={"dc-ip": "10.10.10.1", "all": True},
        ))
        self.mock_exec.assert_called_once_with("impacket", {
            "script":     "GetNPUsers",
            "target":     "corp.local/",
            "options":    {"dc-ip": "10.10.10.1", "all": True},
            "extra_args": "",
        })
        assert result["success"] is True

    def test_returns_failure_on_error(self):
        with patch("mcp_core.active_directory_direct.ad_exec", make_mock_ad_exec(success=False)):
            mcp, _ = make_mcp_with(
                __import__("mcp_tools.active_directory.impacket_scripts",
                           fromlist=["register_impacket"]).register_impacket
            )
            result = run(call_tool(mcp, "impacket_run", script="GetADUsers", target="x"))
            assert result["success"] is False


# ===========================================================================
# ldapdomaindump
# ===========================================================================

class TestLdapdomaindump:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mcp_tools.active_directory.ldapdomaindump import register_ldapdomaindump_tool
        with patch("mcp_core.active_directory_direct.ad_exec", make_mock_ad_exec()) as mock:
            self.mock_exec = mock
            self.mcp, _ = make_mcp_with(register_ldapdomaindump_tool)
            yield

    def test_tool_registered(self):
        run(assert_tool_registered(self.mcp, "ldapdomaindump"))

    def test_calls_correct_tool_defaults(self):
        result = run(call_tool(self.mcp, "ldapdomaindump", hostname="10.10.10.1"))
        self.mock_exec.assert_called_once_with("ldapdomaindump", {
            "hostname": "10.10.10.1",
            "username": "",
            "password": "",
            "authtype": "NTLM",
        })
        assert result["success"] is True

    def test_calls_with_credentials(self):
        result = run(call_tool(
            self.mcp, "ldapdomaindump",
            hostname="10.10.10.1",
            username="admin",
            password="pass123",
            authtype="SIMPLE",
        ))
        self.mock_exec.assert_called_once_with("ldapdomaindump", {
            "hostname": "10.10.10.1",
            "username": "admin",
            "password": "pass123",
            "authtype": "SIMPLE",
        })
        assert result["success"] is True


# ===========================================================================
# adidnsdump
# ===========================================================================

class TestAdidnsdump:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mcp_tools.active_directory.adidnsdump import register_adidnsdump_tool
        with patch("mcp_core.active_directory_direct.ad_exec", make_mock_ad_exec()) as mock:
            self.mock_exec = mock
            self.mcp, _ = make_mcp_with(register_adidnsdump_tool)
            yield

    def test_tool_registered(self):
        run(assert_tool_registered(self.mcp, "adidnsdump"))

    def test_calls_correct_tool_defaults(self):
        result = run(call_tool(self.mcp, "adidnsdump", target="10.10.10.1"))
        self.mock_exec.assert_called_once_with("adidnsdump", {
            "target":          "10.10.10.1",
            "username":        "",
            "password":        "",
            "zone":            "",
            "additional_args": "",
        })
        assert result["success"] is True

    def test_calls_with_zone(self):
        result = run(call_tool(
            self.mcp, "adidnsdump",
            target="10.10.10.1",
            username="user",
            password="pass",
            zone="corp.local",
        ))
        self.mock_exec.assert_called_once_with("adidnsdump", {
            "target":          "10.10.10.1",
            "username":        "user",
            "password":        "pass",
            "zone":            "corp.local",
            "additional_args": "",
        })
        assert result["success"] is True


# ===========================================================================
# certipy_ad
# ===========================================================================

class TestCertipyAd:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mcp_tools.active_directory.certipy_ad import register_certipy_ad_tool
        with patch("mcp_core.active_directory_direct.ad_exec", make_mock_ad_exec()) as mock:
            self.mock_exec = mock
            self.mcp, _ = make_mcp_with(register_certipy_ad_tool)
            yield

    def test_tool_registered(self):
        run(assert_tool_registered(self.mcp, "certipy_ad"))

    def test_calls_find_action(self):
        result = run(call_tool(
            self.mcp, "certipy_ad",
            action="find",
            target="corp.local",
            username="user@corp.local",
            password="pass",
            dc_ip="10.10.10.1",
        ))
        self.mock_exec.assert_called_once_with("certipy_ad", {
            "action":          "find",
            "target":          "corp.local",
            "username":        "user@corp.local",
            "password":        "pass",
            "dc_ip":           "10.10.10.1",
            "additional_args": "",
        })
        assert result["success"] is True

    def test_calls_req_action(self):
        result = run(call_tool(self.mcp, "certipy_ad", action="req"))
        self.mock_exec.assert_called_once_with("certipy_ad", {
            "action":          "req",
            "target":          "",
            "username":        "",
            "password":        "",
            "dc_ip":           "",
            "additional_args": "",
        })
        assert result["success"] is True


# ===========================================================================
# mitm6 — uses ctx.elicit() (destructive tool)
# ===========================================================================

class TestMitm6:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mcp_tools.active_directory.mitm6 import register_mitm6_tool
        with patch("mcp_core.active_directory_direct.ad_exec", make_mock_ad_exec()) as mock:
            self.mock_exec = mock
            self.mcp, _ = make_mcp_with(register_mitm6_tool)
            yield

    def test_tool_registered(self):
        run(assert_tool_registered(self.mcp, "mitm6"))

    def test_calls_correct_tool_defaults(self):
        """Elicitation accepted → mitm6 executes."""
        result = run(call_tool(self.mcp, "mitm6", interface="eth0"))
        self.mock_exec.assert_called_once_with("mitm6", {
            "interface":       "eth0",
            "domain":          "",
            "additional_args": "",
        })
        assert result["success"] is True

    def test_calls_with_domain(self):
        """Elicitation accepted → mitm6 executes with domain."""
        result = run(call_tool(self.mcp, "mitm6", interface="eth0", domain="corp.local"))
        self.mock_exec.assert_called_once_with("mitm6", {
            "interface":       "eth0",
            "domain":          "corp.local",
            "additional_args": "",
        })
        assert result["success"] is True

    def test_cancelled_does_not_execute(self):
        """Elicitation declined → mitm6 does NOT execute."""
        result = run(_call_tool_with_elicit(self.mcp, "mitm6",
            elicit_accepted=False, interface="eth0"))
        self.mock_exec.assert_not_called()
        assert result["success"] is False


# ===========================================================================
# pywerview
# ===========================================================================

class TestPywerview:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mcp_tools.active_directory.pywerview import register_pywerview_tool
        with patch("mcp_core.active_directory_direct.ad_exec", make_mock_ad_exec()) as mock:
            self.mock_exec = mock
            self.mcp, _ = make_mcp_with(register_pywerview_tool)
            yield

    def test_tool_registered(self):
        run(assert_tool_registered(self.mcp, "pywerview"))

    def test_calls_get_netuser(self):
        result = run(call_tool(
            self.mcp, "pywerview",
            request="get-netuser",
            target="10.10.10.1",
        ))
        self.mock_exec.assert_called_once_with("pywerview", {
            "request":         "get-netuser",
            "target":          "10.10.10.1",
            "username":        "",
            "password":        "",
            "dc_ip":           "",
            "additional_args": "",
        })
        assert result["success"] is True

    def test_calls_with_credentials(self):
        result = run(call_tool(
            self.mcp, "pywerview",
            request="get-netgroup",
            target="10.10.10.1",
            username="admin",
            password="pass",
            dc_ip="10.10.10.1",
        ))
        self.mock_exec.assert_called_once_with("pywerview", {
            "request":         "get-netgroup",
            "target":          "10.10.10.1",
            "username":        "admin",
            "password":        "pass",
            "dc_ip":           "10.10.10.1",
            "additional_args": "",
        })
        assert result["success"] is True


# ===========================================================================
# bloodhound_python
# ===========================================================================

class TestBloodhoundPython:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mcp_tools.active_directory.bloodhound_ce_python import register_bloodhound_tool
        with patch("mcp_core.active_directory_direct.ad_exec", make_mock_ad_exec()) as mock:
            self.mock_exec = mock
            self.mcp, _ = make_mcp_with(register_bloodhound_tool)
            yield

    def test_tool_registered(self):
        run(assert_tool_registered(self.mcp, "bloodhound_python"))

    def test_calls_correct_tool(self):
        result = run(call_tool(
            self.mcp, "bloodhound_python",
            domain="corp.local",
            username="admin",
            password="pass123",
            dc_ip="10.10.10.1",
        ))
        self.mock_exec.assert_called_once_with("bloodhound", {
            "domain":          "corp.local",
            "username":        "admin",
            "password":        "pass123",
            "dc_ip":           "10.10.10.1",
            "additional_args": "",
        })
        assert result["success"] is True

    def test_returns_failure_on_error(self):
        with patch("mcp_core.active_directory_direct.ad_exec", make_mock_ad_exec(success=False)):
            mcp, _ = make_mcp_with(
                __import__("mcp_tools.active_directory.bloodhound_ce_python",
                           fromlist=["register_bloodhound_tool"]).register_bloodhound_tool
            )
            result = run(call_tool(
                mcp, "bloodhound_python",
                domain="corp.local", username="u", password="p", dc_ip="10.0.0.1"
            ))
            assert result["success"] is False


# ===========================================================================
# active_directory_direct.py — handler unit tests
# ===========================================================================

class TestAdDirectHandlers:
    """Test _direct.py handlers directly — no FastMCP overhead."""

    def test_impacket_requires_script(self):
        with patch("mcp_core.active_directory_direct.execute_command") as mock:
            from mcp_core.active_directory_direct import ad_exec
            result = ad_exec("impacket", {"target": "corp.local"})
            mock.assert_not_called()
            assert result["success"] is False
            assert "script" in result["error"]

    def test_impacket_requires_target(self):
        with patch("mcp_core.active_directory_direct.execute_command") as mock:
            from mcp_core.active_directory_direct import ad_exec
            result = ad_exec("impacket", {"script": "GetADUsers"})
            mock.assert_not_called()
            assert result["success"] is False
            assert "target" in result["error"]

    def test_impacket_rejects_unknown_script(self):
        with patch("mcp_core.active_directory_direct.execute_command") as mock:
            from mcp_core.active_directory_direct import ad_exec
            result = ad_exec("impacket", {"script": "notreal", "target": "corp.local"})
            mock.assert_not_called()
            assert result["success"] is False
            assert "Unsupported" in result["error"]

    def test_ldapdomaindump_requires_hostname(self):
        with patch("mcp_core.active_directory_direct.execute_command") as mock:
            from mcp_core.active_directory_direct import ad_exec
            result = ad_exec("ldapdomaindump", {})
            mock.assert_not_called()
            assert result["success"] is False
            assert "hostname" in result["error"]

    def test_adidnsdump_requires_target(self):
        with patch("mcp_core.active_directory_direct.execute_command") as mock:
            from mcp_core.active_directory_direct import ad_exec
            result = ad_exec("adidnsdump", {})
            mock.assert_not_called()
            assert result["success"] is False
            assert "target" in result["error"]

    def test_certipy_requires_action(self):
        with patch("mcp_core.active_directory_direct.execute_command") as mock:
            from mcp_core.active_directory_direct import ad_exec
            result = ad_exec("certipy_ad", {})
            mock.assert_not_called()
            assert result["success"] is False
            assert "action" in result["error"]

    def test_mitm6_requires_interface(self):
        with patch("mcp_core.active_directory_direct.execute_command") as mock:
            from mcp_core.active_directory_direct import ad_exec
            result = ad_exec("mitm6", {})
            mock.assert_not_called()
            assert result["success"] is False
            assert "interface" in result["error"]

    def test_pywerview_requires_request_and_target(self):
        with patch("mcp_core.active_directory_direct.execute_command") as mock:
            from mcp_core.active_directory_direct import ad_exec
            result = ad_exec("pywerview", {"target": "10.0.0.1"})
            mock.assert_not_called()
            assert result["success"] is False
            assert "request" in result["error"]

    def test_bloodhound_requires_all_params(self):
        with patch("mcp_core.active_directory_direct.execute_command") as mock:
            from mcp_core.active_directory_direct import ad_exec
            result = ad_exec("bloodhound", {"domain": "corp.local"})
            mock.assert_not_called()
            assert result["success"] is False

    def test_unknown_tool_returns_error(self):
        from mcp_core.active_directory_direct import ad_exec
        result = ad_exec("nonexistent", {})
        assert result["success"] is False
        assert "Unknown" in result["error"]

    def test_ldapdomaindump_builds_correct_command(self):
        with patch("mcp_core.active_directory_direct.execute_command",
                   return_value={"success": True, "output": ""}) as mock:
            from mcp_core.active_directory_direct import ad_exec
            ad_exec("ldapdomaindump", {
                "hostname": "10.10.10.1",
                "username": "admin",
                "password": "pass",
                "authtype": "NTLM",
            })
            cmd = mock.call_args[0][0]
            assert "ldapdomaindump" in cmd
            assert "10.10.10.1" in cmd
            assert "admin" in cmd

    def test_impacket_builds_correct_command(self):
        with patch("mcp_core.active_directory_direct.execute_command",
                   return_value={"success": True, "output": ""}) as mock:
            from mcp_core.active_directory_direct import ad_exec
            ad_exec("impacket", {
                "script":  "GetADUsers",
                "target":  "corp.local/user:pass",
                "options": {"dc-ip": "10.10.10.1", "all": True},
                "extra_args": "",
            })
            cmd = mock.call_args[0][0]
            assert "impacket-GetADUsers" in cmd
            assert "corp.local/user:pass" in cmd
            assert "10.10.10.1" in cmd
