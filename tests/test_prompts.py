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
from unittest.mock import patch, MagicMock, AsyncMock
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


async def render(mcp, prompt_name, **kwargs):
    prompt = await mcp.get_prompt(prompt_name)
    assert prompt is not None, f"Prompt '{prompt_name}' not registered"
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
    def test_ctf_challenge_registered(self):       self._check("ctf_challenge")
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
# Fixtures for CTF prompt tests
# ---------------------------------------------------------------------------

def _build_workflow(with_steps: bool = True):
    """Build a CTFWorkflowManager return dict for testing."""
    wf = {
        "workflow_steps": [],
        "strategies": [],
        "fallback_strategies": [],
        "parallel_tasks": [],
        "tools": ["nmap", "gobuster", "sqlmap"],
        "estimated_time": 1800,
        "success_probability": 0.75,
        "validation_steps": [],
        "expected_artifacts": [],
        "resource_requirements": {},
    }
    if with_steps:
        wf["workflow_steps"] = [
            {"step": 1, "action": "recon", "description": "Network recon", "tools": ["nmap"], "parallel": True, "estimated_time": 120},
            {"step": 2, "action": "enum", "description": "Service enum", "tools": ["gobuster"], "parallel": False, "estimated_time": 180},
            {"step": 3, "action": "scan", "description": "Vuln scan", "tools": ["nuclei"], "parallel": False, "estimated_time": 300},
            {"step": 4, "action": "manual_review", "description": "Manual code review", "tools": ["manual"], "parallel": False, "estimated_time": 600},
            {"step": 5, "action": "exploit", "description": "Exploit phase", "tools": ["sqlmap"], "parallel": False, "estimated_time": 300},
        ]
        wf["strategies"] = [
            {"strategy": "s1", "description": "Strategy one"},
            {"strategy": "s2", "description": "Strategy two"},
        ]
        wf["fallback_strategies"] = [
            {"strategy": "f1", "description": "Fallback one"},
        ]
        wf["parallel_tasks"] = [
            {"task_group": "recon", "tasks": ["nmap", "gobuster"], "max_concurrent": 2},
        ]
        wf["validation_steps"] = [
            {"step": "flag_format_check"},
            {"step": "solution_verification"},
        ]
        wf["expected_artifacts"] = [
            {"type": "capture_file"},
            {"type": "screenshot"},
        ]
        wf["resource_requirements"] = {
            "cpu_cores": 4,
            "memory_mb": 4096,
            "gpu_required": True,
        }
    return wf


@pytest.fixture
def mock_ctf_manager():
    """Mock get_ctf_manager + CTFToolManager for prompt tests."""
    mgr = MagicMock()
    mgr.create_ctf_challenge_workflow.return_value = _build_workflow(with_steps=True)
    with patch("server_core.singletons.get_ctf_manager") as mock_get:
        mock_get.return_value = mgr
        with patch("server_core.workflows.ctf.toolManager.CTFToolManager") as mock_tm_cls:
            mock_tm = MagicMock()
            mock_tm.get_tool_command.return_value = "nmap target"
            mock_tm_cls.return_value = mock_tm
            yield mgr, mock_tm


@pytest.fixture
def mock_ctf_manager_empty():
    """Mock CTF manager returning empty/no-step workflow."""
    mgr = MagicMock()
    mgr.create_ctf_challenge_workflow.return_value = _build_workflow(with_steps=False)
    with patch("server_core.singletons.get_ctf_manager") as mock_get:
        mock_get.return_value = mgr
        yield mgr


# ---------------------------------------------------------------------------
# ctf_web_challenge — extended step_label branch coverage
# ---------------------------------------------------------------------------

class TestCtfWebChallengeExtended:
    @pytest.fixture(autouse=True)
    def setup(self, mock_ctf_manager):
        self.mcp = make_mcp_with_prompts()
        self.mock_mgr = mock_ctf_manager[0]

    def test_step_label_with_steps(self):
        """step_label returns step action/description when steps exist."""
        result = run(render(self.mcp, "ctf_web_challenge",
            url="http://test.ctf:8080"))
        text = messages_text(result)
        assert "Network recon" in text
        assert "recon" in text

    def test_step_label_fallback_without_steps(self):
        """step_label falls back to default string when not enough steps."""
        # Re-register with empty workflow
        empty_wf = _build_workflow(with_steps=False)
        mock_mgr = MagicMock()
        mock_mgr.create_ctf_challenge_workflow.return_value = empty_wf
        with patch("server_core.singletons.get_ctf_manager") as mock_get:
            mock_get.return_value = mock_mgr
            mcp2 = make_mcp_with_prompts()
            result = run(render(mcp2, "ctf_web_challenge",
                url="http://test.ctf:8080"))
        text = messages_text(result)
        assert "Reconnaissance" in text or "reconnaissance" in text.lower()

    def test_step_label_partial_coverage(self):
        """step_label(0) works but step_label(2) falls back with only 1 step."""
        wf = _build_workflow(with_steps=True)
        wf["workflow_steps"] = [wf["workflow_steps"][0]]  # only 1 step
        mock_mgr = MagicMock()
        mock_mgr.create_ctf_challenge_workflow.return_value = wf
        with patch("server_core.singletons.get_ctf_manager") as mock_get:
            mock_get.return_value = mock_mgr
            mcp2 = make_mcp_with_prompts()
            result = run(render(mcp2, "ctf_web_challenge",
                url="http://test.ctf:8080"))
        assert len(result.messages) > 0

    def test_strategies_displayed(self):
        """Strategies and fallback summaries are included."""
        result = run(render(self.mcp, "ctf_web_challenge",
            url="http://test.ctf:8080"))
        text = messages_text(result)
        assert "Strategy one" in text or "s1" in text or "Strategies" in text

    def test_fallback_displayed(self):
        """Fallback strategies are included."""
        result = run(render(self.mcp, "ctf_web_challenge",
            url="http://test.ctf:8080"))
        text = messages_text(result)
        assert "Fallback" in text or "fallback" in text.lower()

    def test_validation_in_final(self):
        result = run(render(self.mcp, "ctf_web_challenge",
            url="http://test.ctf:8080"))
        text = messages_text(result)
        assert "flag_format_check" in text or "validation" in text.lower()

    def test_suggested_tools_in_context(self):
        result = run(render(self.mcp, "ctf_web_challenge",
            url="http://test.ctf:8080"))
        text = messages_text(result)
        assert "nmap" in text


# ---------------------------------------------------------------------------
# ctf_challenge — full branch coverage
# ---------------------------------------------------------------------------

class TestCtfChallenge:
    @pytest.fixture(autouse=True)
    def setup(self, mock_ctf_manager):
        self.mcp = make_mcp_with_prompts()
        self.mock_mgr, self.mock_tm = mock_ctf_manager

    def test_renders_without_error(self):
        result = run(render(self.mcp, "ctf_challenge",
            name="test", category="web", description="A web challenge"))
        assert len(result.messages) > 0

    def test_basic_fields_injected(self):
        result = run(render(self.mcp, "ctf_challenge",
            name="chall1", category="pwn", description="pwn me",
            difficulty="hard", target="10.0.0.1", points=500))
        text = messages_text(result)
        assert "chall1" in text
        assert "PWN" in text
        assert "hard" in text
        assert "10.0.0.1" in text
        assert "500" in text

    def test_description_truncated(self):
        long_desc = "A" * 300
        result = run(render(self.mcp, "ctf_challenge",
            name="test", category="web", description=long_desc))
        text = messages_text(result)
        assert "AAA" in text
        assert "..." in text

    def test_workflow_steps_in_messages(self):
        result = run(render(self.mcp, "ctf_challenge",
            name="test", category="web", description="test"))
        text = messages_text(result)
        assert "STEP 1" in text
        assert "STEP 2" in text
        assert "recon" in text
        assert "enum" in text

    def test_workflow_steps_with_tools(self):
        """Tools in steps generate run_security_tool calls."""
        result = run(render(self.mcp, "ctf_challenge",
            name="test", category="web", description="test"))
        text = messages_text(result)
        assert "run_security_tool" in text
        assert "nmap" in text
        assert "gobuster" in text

    def test_manual_step_does_not_generate_rst_call(self):
        """Tools named 'manual' skip run_security_tool generation."""
        result = run(render(self.mcp, "ctf_challenge",
            name="test", category="web", description="test"))
        text = messages_text(result)
        assert "Manual code review" in text
        assert 'run_security_tool(tool_name="manual"' not in text

    def test_parallel_note_in_step(self):
        """Parallel steps include [PARALLEL] note."""
        result = run(render(self.mcp, "ctf_challenge",
            name="test", category="web", description="test"))
        text = messages_text(result)
        assert "PARALLEL" in text or "[PARALLEL]" in text

    def test_strategies_and_fallback(self):
        """Strategies and fallback are included as a separate message."""
        result = run(render(self.mcp, "ctf_challenge",
            name="test", category="web", description="test"))
        text = messages_text(result)
        assert "Strategy one" in text
        assert "Fallback one" in text
        assert "FALLBACK" in text

    def test_resource_requirements(self):
        """GPU requirement is noted when present."""
        result = run(render(self.mcp, "ctf_challenge",
            name="test", category="web", description="test"))
        text = messages_text(result)
        assert "GPU" in text
        assert "CPU" in text

    def test_parallel_tasks_summary(self):
        """Parallel task groups are summarized."""
        result = run(render(self.mcp, "ctf_challenge",
            name="test", category="web", description="test"))
        text = messages_text(result)
        assert "recon" in text
        assert "Parallel execution" in text

    def test_final_message_artifacts(self):
        """Final message includes expected artifacts and validation."""
        result = run(render(self.mcp, "ctf_challenge",
            name="test", category="web", description="test"))
        text = messages_text(result)
        assert "capture_file" in text
        assert "flag_format_check" in text

    def test_final_message_default_artifacts(self):
        """When no artifacts/validation, show defaults."""
        mgr = MagicMock()
        mgr.create_ctf_challenge_workflow.return_value = _build_workflow(with_steps=True)
        mgr.create_ctf_challenge_workflow.return_value["expected_artifacts"] = []
        mgr.create_ctf_challenge_workflow.return_value["validation_steps"] = []
        with patch("server_core.singletons.get_ctf_manager") as mock_get:
            mock_get.return_value = mgr
            with patch("server_core.workflows.ctf.toolManager.CTFToolManager"):
                mcp2 = make_mcp_with_prompts()
                result = run(render(mcp2, "ctf_challenge",
                    name="test", category="web", description="test"))
        text = messages_text(result)
        assert "flag, solution artifacts" in text
        assert "flag_format_check" in text


class TestCtfChallengeEmpty:
    """Test ctf_challenge with empty/edge-case workflow responses."""

    def test_no_steps_still_renders(self, mock_ctf_manager_empty):
        mcp = make_mcp_with_prompts()
        result = run(render(mcp, "ctf_challenge",
            name="test", category="web", description="test"))
        assert len(result.messages) >= 2
        text = messages_text(result)
        assert "WEB" in text

    def test_no_tools_no_commands_section(self):
        """When suggested_tools is empty, tool commands section is omitted."""
        mgr = MagicMock()
        wf = _build_workflow(with_steps=True)
        wf["tools"] = []
        mgr.create_ctf_challenge_workflow.return_value = wf
        with patch("server_core.singletons.get_ctf_manager") as mock_get:
            mock_get.return_value = mgr
            with patch("server_core.workflows.ctf.toolManager.CTFToolManager"):
                mcp2 = make_mcp_with_prompts()
                result = run(render(mcp2, "ctf_challenge",
                    name="test", category="web", description="test"))
        text = messages_text(result)
        assert "Optimized commands" not in text

    def test_no_resources_no_resource_info(self):
        """When resources are empty, resource info is omitted."""
        mgr = MagicMock()
        wf = _build_workflow(with_steps=True)
        wf["resource_requirements"] = {}
        mgr.create_ctf_challenge_workflow.return_value = wf
        with patch("server_core.singletons.get_ctf_manager") as mock_get:
            mock_get.return_value = mgr
            with patch("server_core.workflows.ctf.toolManager.CTFToolManager"):
                mcp2 = make_mcp_with_prompts()
                result = run(render(mcp2, "ctf_challenge",
                    name="test", category="web", description="test"))
        text = messages_text(result)
        assert "CPU" not in text

    def test_no_parallel_tasks_no_parallel_info(self):
        """When parallel_tasks is empty, parallel info is omitted."""
        mgr = MagicMock()
        wf = _build_workflow(with_steps=True)
        wf["parallel_tasks"] = []
        mgr.create_ctf_challenge_workflow.return_value = wf
        with patch("server_core.singletons.get_ctf_manager") as mock_get:
            mock_get.return_value = mgr
            with patch("server_core.workflows.ctf.toolManager.CTFToolManager"):
                mcp2 = make_mcp_with_prompts()
                result = run(render(mcp2, "ctf_challenge",
                    name="test", category="web", description="test"))
        text = messages_text(result)
        assert "Parallel execution" not in text

    def test_asyncio_mode_not_required(self):
        """Synchronous render works without event loop issues."""
        mgr = MagicMock()
        mgr.create_ctf_challenge_workflow.return_value = _build_workflow(with_steps=False)
        with patch("server_core.singletons.get_ctf_manager") as mock_get:
            mock_get.return_value = mgr
            with patch("server_core.workflows.ctf.toolManager.CTFToolManager"):
                mcp2 = make_mcp_with_prompts()
                result = run(render(mcp2, "ctf_challenge",
                    name="async_test", category="crypto", description="test"))
        assert len(result.messages) > 0

    def test_get_tool_command_exception(self):
        """Exception in get_tool_command is caught and shows fallback text."""
        mgr = MagicMock()
        wf = _build_workflow(with_steps=True)
        wf["tools"] = ["nmap"]
        mgr.create_ctf_challenge_workflow.return_value = wf
        with patch("server_core.singletons.get_ctf_manager") as mock_get:
            mock_get.return_value = mgr
            with patch("server_core.workflows.ctf.toolManager.CTFToolManager") as mock_tm_cls:
                mock_tm = MagicMock()
                mock_tm.get_tool_command.side_effect = ValueError("no tool")
                mock_tm_cls.return_value = mock_tm
                mcp2 = make_mcp_with_prompts()
                result = run(render(mcp2, "ctf_challenge",
                    name="test", category="web", description="test"))
        text = messages_text(result)
        assert "(see tool docs)" in text

    def test_strategies_no_fallback(self):
        """Strategies shown without fallback section when fallback is empty."""
        mgr = MagicMock()
        wf = _build_workflow(with_steps=True)
        wf["strategies"] = [{"strategy": "s1", "description": "d1"}]
        wf["fallback_strategies"] = []
        mgr.create_ctf_challenge_workflow.return_value = wf
        with patch("server_core.singletons.get_ctf_manager") as mock_get:
            mock_get.return_value = mgr
            with patch("server_core.workflows.ctf.toolManager.CTFToolManager"):
                mcp2 = make_mcp_with_prompts()
                result = run(render(mcp2, "ctf_challenge",
                    name="test", category="web", description="test"))
        text = messages_text(result)
        assert "s1" in text or "d1" in text
        assert "FALLBACK" not in text


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

    def test_contains_kube_hunter_with_ip(self):
        """kube_hunter only runs when k8s_api_ip is provided."""
        result = run(render(self.mcp, "cloud_security_audit",
            k8s_api_ip="10.10.10.1"))
        assert "kube_hunter" in messages_text(result)

    def test_kube_hunter_skipped_without_ip(self):
        """Without k8s_api_ip, kube_hunter step shows skip message."""
        result = run(render(self.mcp, "cloud_security_audit"))
        assert "k8s_api_ip" in messages_text(result)
