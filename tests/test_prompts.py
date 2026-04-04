"""
tests/test_prompts.py

Unit tests for mcp_core/prompts.py — HexStrike workflow prompts.

Strategy:
- Register prompts against a real FastMCP instance
- Render each prompt with test arguments
- Verify message count, tool names, and argument injection
"""

import pytest
import asyncio
from fastmcp import FastMCP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mcp_with_prompts():
    from mcp_core.prompts import register_prompts
    mcp = FastMCP("test-hexstrike-prompts")
    register_prompts(mcp)
    return mcp


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def render(mcp, name, **kwargs):
    prompt = await mcp.get_prompt(name)
    assert prompt is not None, f"Prompt '{name}' not registered"
    result = await prompt.render(arguments=kwargs)
    return result


def messages_text(result) -> str:
    return "\n".join(m.content.text for m in result.messages)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestPromptsRegistered:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mcp = make_mcp_with_prompts()

    def _check(self, name):
        async def go():
            p = await self.mcp.get_prompt(name)
            assert p is not None, f"'{name}' not registered"
        run(go())

    def test_bug_bounty_recon_registered(self):    self._check("bug_bounty_recon")
    def test_wifi_attack_chain_registered(self):   self._check("wifi_attack_chain")
    def test_ctf_web_challenge_registered(self):   self._check("ctf_web_challenge")
    def test_smb_lateral_movement_registered(self):self._check("smb_lateral_movement")
    def test_cloud_security_audit_registered(self):self._check("cloud_security_audit")


# ---------------------------------------------------------------------------
# bug_bounty_recon
# ---------------------------------------------------------------------------

class TestBugBountyRecon:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mcp = make_mcp_with_prompts()

    def test_renders_without_error(self):
        result = run(render(self.mcp, "bug_bounty_recon", target="example.com"))
        assert len(result.messages) > 0

    def test_target_injected(self):
        result = run(render(self.mcp, "bug_bounty_recon", target="example.com"))
        assert "example.com" in messages_text(result)

    def test_contains_subfinder(self):
        result = run(render(self.mcp, "bug_bounty_recon", target="example.com"))
        assert "subfinder" in messages_text(result)

    def test_contains_nuclei(self):
        result = run(render(self.mcp, "bug_bounty_recon", target="example.com"))
        assert "nuclei" in messages_text(result)

    def test_contains_nmap(self):
        result = run(render(self.mcp, "bug_bounty_recon", target="example.com"))
        assert "nmap" in messages_text(result)

    def test_uses_run_security_tool_syntax(self):
        result = run(render(self.mcp, "bug_bounty_recon", target="example.com"))
        assert "run_security_tool" in messages_text(result)

    def test_minimum_message_count(self):
        result = run(render(self.mcp, "bug_bounty_recon", target="example.com"))
        assert len(result.messages) >= 7


# ---------------------------------------------------------------------------
# wifi_attack_chain
# ---------------------------------------------------------------------------

class TestWifiAttackChain:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mcp = make_mcp_with_prompts()

    def test_renders_without_error(self):
        result = run(render(self.mcp, "wifi_attack_chain",
            interface="wlan0", bssid="AA:BB:CC:DD:EE:FF"))
        assert len(result.messages) > 0

    def test_interface_injected(self):
        result = run(render(self.mcp, "wifi_attack_chain",
            interface="wlan0", bssid="AA:BB:CC:DD:EE:FF"))
        assert "wlan0" in messages_text(result)

    def test_bssid_injected(self):
        result = run(render(self.mcp, "wifi_attack_chain",
            interface="wlan0", bssid="AA:BB:CC:DD:EE:FF"))
        assert "AA:BB:CC:DD:EE:FF" in messages_text(result)

    def test_contains_airmon_ng(self):
        result = run(render(self.mcp, "wifi_attack_chain",
            interface="wlan0", bssid="AA:BB:CC:DD:EE:FF"))
        assert "airmon_ng" in messages_text(result)

    def test_contains_aireplay_ng(self):
        result = run(render(self.mcp, "wifi_attack_chain",
            interface="wlan0", bssid="AA:BB:CC:DD:EE:FF"))
        assert "aireplay_ng" in messages_text(result)

    def test_contains_aircrack_ng(self):
        result = run(render(self.mcp, "wifi_attack_chain",
            interface="wlan0", bssid="AA:BB:CC:DD:EE:FF"))
        assert "aircrack_ng" in messages_text(result)

    def test_elicitation_warning_present(self):
        result = run(render(self.mcp, "wifi_attack_chain",
            interface="wlan0", bssid="AA:BB:CC:DD:EE:FF"))
        text = messages_text(result)
        assert "confirmation" in text.lower() or "⚠️" in text

    def test_custom_channel(self):
        result = run(render(self.mcp, "wifi_attack_chain",
            interface="wlan0", bssid="AA:BB:CC:DD:EE:FF", channel="11"))
        assert "11" in messages_text(result)


# ---------------------------------------------------------------------------
# ctf_web_challenge
# ---------------------------------------------------------------------------

class TestCtfWebChallenge:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mcp = make_mcp_with_prompts()

    def test_renders_without_error(self):
        result = run(render(self.mcp, "ctf_web_challenge",
            url="http://challenge.ctf.local:8080"))
        assert len(result.messages) > 0

    def test_url_injected(self):
        result = run(render(self.mcp, "ctf_web_challenge",
            url="http://challenge.ctf.local:8080"))
        assert "challenge.ctf.local" in messages_text(result)

    def test_contains_gobuster(self):
        result = run(render(self.mcp, "ctf_web_challenge",
            url="http://challenge.ctf.local:8080"))
        assert "gobuster" in messages_text(result)

    def test_contains_sqlmap(self):
        result = run(render(self.mcp, "ctf_web_challenge",
            url="http://challenge.ctf.local:8080"))
        assert "sqlmap" in messages_text(result)

    def test_contains_dalfox(self):
        result = run(render(self.mcp, "ctf_web_challenge",
            url="http://challenge.ctf.local:8080"))
        assert "dalfox" in messages_text(result)

    def test_contains_nuclei(self):
        result = run(render(self.mcp, "ctf_web_challenge",
            url="http://challenge.ctf.local:8080"))
        assert "nuclei" in messages_text(result)


# ---------------------------------------------------------------------------
# smb_lateral_movement
# ---------------------------------------------------------------------------

class TestSmbLateralMovement:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mcp = make_mcp_with_prompts()

    def test_renders_without_error(self):
        result = run(render(self.mcp, "smb_lateral_movement", target="10.10.10.10"))
        assert len(result.messages) > 0

    def test_target_injected(self):
        result = run(render(self.mcp, "smb_lateral_movement", target="10.10.10.10"))
        assert "10.10.10.10" in messages_text(result)

    def test_contains_enum4linux(self):
        result = run(render(self.mcp, "smb_lateral_movement", target="10.10.10.10"))
        assert "enum4linux" in messages_text(result)

    def test_contains_netexec(self):
        result = run(render(self.mcp, "smb_lateral_movement", target="10.10.10.10"))
        assert "netexec" in messages_text(result)

    def test_contains_metasploit(self):
        result = run(render(self.mcp, "smb_lateral_movement", target="10.10.10.10"))
        assert "metasploit" in messages_text(result)

    def test_exploitation_warning_present(self):
        result = run(render(self.mcp, "smb_lateral_movement", target="10.10.10.10"))
        text = messages_text(result)
        assert "confirmation" in text.lower() or "⚠️" in text

    def test_cidr_range_works(self):
        result = run(render(self.mcp, "smb_lateral_movement", target="192.168.1.0/24"))
        assert "192.168.1.0/24" in messages_text(result)


# ---------------------------------------------------------------------------
# cloud_security_audit
# ---------------------------------------------------------------------------

class TestCloudSecurityAudit:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mcp = make_mcp_with_prompts()

    def test_renders_without_error(self):
        result = run(render(self.mcp, "cloud_security_audit"))
        assert len(result.messages) > 0

    def test_default_provider_aws(self):
        result = run(render(self.mcp, "cloud_security_audit"))
        assert "aws" in messages_text(result)

    def test_provider_injected(self):
        result = run(render(self.mcp, "cloud_security_audit",
            provider="azure", profile="prod"))
        assert "azure" in messages_text(result)

    def test_contains_prowler(self):
        result = run(render(self.mcp, "cloud_security_audit"))
        assert "prowler" in messages_text(result)

    def test_contains_trivy(self):
        result = run(render(self.mcp, "cloud_security_audit"))
        assert "trivy" in messages_text(result)

    def test_contains_kube_hunter(self):
        result = run(render(self.mcp, "cloud_security_audit"))
        assert "kube_hunter" in messages_text(result)
