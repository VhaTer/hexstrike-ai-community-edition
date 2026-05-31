"""
FastMCP 3.x native workflow prompt tests.

Goal: Validate prompt structure (message count, format, tool references),
not the underlying CTFManager/CTFToolManager intelligence.

Functions are tested directly at module level (no FastMCP instance needed)
since FastMCP 3.2.4 has a bug in get_prompt() that causes version mismatch.

What we test:
  4 simple prompts: bug_bounty_recon, wifi_attack_chain, smb_lateral_movement, cloud_security_audit
  2 CTF prompts:    ctf_web_challenge, ctf_challenge

Simple prompts:  len ≥ 2, run_security_tool present, first message role == "user"
CTF prompts:     len ≥ 3, "ctf" context injected, mock CTFWorkflowManager

Run:
    pytest tests/test_prompts.py -v -q --tb=short
"""

from unittest.mock import MagicMock, patch

import pytest

from mcp_core.prompts import (
    PromptResult,
    bug_bounty_recon,
    cloud_security_audit,
    ctf_challenge,
    ctf_web_challenge,
    pulse_dashboards,
    register_prompts,
    smb_lateral_movement,
    wifi_attack_chain,
)


def run(coro):
    """Run async coroutine synchronously and normalize to list[Message]."""
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(coro)
        if isinstance(result, PromptResult):
            return result.messages
        return result
    finally:
        loop.close()


def _ctf_workflow_stub():
    """Minimal workflow dict returned by mock CTFWorkflowManager.

    Only includes keys accessed by ctf_web_challenge / ctf_challenge prompt builders.
    The CTFWorkflowManager itself is tested elsewhere.
    """
    return {
        "workflow_steps": [
            {
                "step": 1,
                "action": "recon",
                "description": "Initial recon",
                "tools": ["nmap", "whatweb"],
                "estimated_time": 300,
            },
        ],
        "strategies": [
            {"strategy": "port_scan", "description": "Scan all ports"},
        ],
        "fallback_strategies": [
            {"strategy": "full_tcp", "description": "Full TCP connect scan"},
        ],
        "parallel_tasks": [
            {"task_group": "recon", "tasks": ["nmap", "whatweb"], "max_concurrent": 2},
        ],
        "tools": ["nmap", "whatweb", "gobuster"],
        "estimated_time": 1800,
        "success_probability": 0.75,
        "validation_steps": [
            {"step": "flag_check"},
        ],
        "expected_artifacts": [
            {"type": "flag"},
        ],
        "resource_requirements": {
            "cpu_cores": 2,
            "memory_mb": 1024,
        },
    }


# ── Simple prompts ────────────────────────────────────────────────────────


class TestSimplePrompts:
    """bug_bounty_recon, wifi_attack_chain, smb_lateral_movement, cloud_security_audit.

    These return static messages with no external service calls.
    """

    def test_bug_bounty_recon_structure(self):
        """Message count ≥ 2, first role user, run_security_tool present."""
        messages = run(bug_bounty_recon(target="example.com"))
        assert len(messages) >= 2
        assert messages[0].role == "user"
        combined = " ".join(str(m) for m in messages)
        assert "run_security_tool" in combined

    def test_bug_bounty_recon_tools(self):
        """All expected tools mentioned."""
        messages = run(bug_bounty_recon(target="example.com"))
        combined = " ".join(str(m) for m in messages)
        for tool in ["wafw00f", "httpx", "subfinder", "amass", "rustscan", "nmap", "katana", "gobuster", "nuclei", "nikto"]:
            assert tool in combined, f"Missing tool: {tool}"

    def test_bug_bounty_recon_final_message(self):
        """Last message contains FINAL."""
        messages = run(bug_bounty_recon(target="example.com"))
        assert "FINAL" in str(messages[-1])

    def test_bug_bounty_recon_meta(self):
        """PromptResult includes meta with version and tools_count."""
        import asyncio
        result = asyncio.new_event_loop().run_until_complete(
            bug_bounty_recon(target="example.com")
        )
        assert isinstance(result, PromptResult)
        assert result.meta["version"] == "0.10.1"
        assert result.meta["tools_count"] >= 8
        assert result.description == "Bug bounty recon workflow for example.com"

    def test_wifi_attack_chain_structure(self):
        """Message count ≥ 2, first role user, run_security_tool present."""
        messages = run(wifi_attack_chain(interface="wlan0", bssid="AA:BB:CC:DD:EE:FF"))
        assert len(messages) >= 2
        assert messages[0].role == "user"
        combined = " ".join(str(m) for m in messages)
        assert "run_security_tool" in combined

    def test_wifi_attack_chain_tools(self):
        """All expected tools mentioned."""
        messages = run(wifi_attack_chain(interface="wlan0", bssid="AA:BB:CC:DD:EE:FF"))
        combined = " ".join(str(m) for m in messages)
        for tool in ["airmon_ng", "airodump_ng", "aireplay_ng", "aircrack_ng"]:
            assert tool in combined, f"Missing tool: {tool}"

    def test_wifi_attack_chain_final_message(self):
        """Last message contains FINAL."""
        messages = run(wifi_attack_chain(interface="wlan0", bssid="AA:BB:CC:DD:EE:FF"))
        assert "FINAL" in str(messages[-1])

    def test_wifi_attack_chain_meta(self):
        """PromptResult includes meta."""
        import asyncio
        result = asyncio.new_event_loop().run_until_complete(
            wifi_attack_chain(interface="wlan0", bssid="AA:BB:CC:DD:EE:FF")
        )
        assert isinstance(result, PromptResult)
        assert "WiFi" in result.description
        assert result.meta["tools_count"] == 4

    def test_smb_lateral_movement_structure(self):
        """Message count ≥ 2, first role user, run_security_tool present."""
        messages = run(smb_lateral_movement(target="10.10.10.10"))
        assert len(messages) >= 2
        assert messages[0].role == "user"
        combined = " ".join(str(m) for m in messages)
        assert "run_security_tool" in combined

    def test_smb_lateral_movement_tools(self):
        """All expected tools mentioned."""
        messages = run(smb_lateral_movement(target="10.10.10.10"))
        combined = " ".join(str(m) for m in messages)
        for tool in ["nbtscan", "nmap", "enum4linux", "smbmap", "netexec", "metasploit"]:
            assert tool in combined, f"Missing tool: {tool}"

    def test_smb_lateral_movement_final_message(self):
        """Last message contains FINAL."""
        messages = run(smb_lateral_movement(target="10.10.10.10"))
        assert "FINAL" in str(messages[-1])

    def test_smb_lateral_movement_meta(self):
        """PromptResult includes meta."""
        import asyncio
        result = asyncio.new_event_loop().run_until_complete(
            smb_lateral_movement(target="10.10.10.10")
        )
        assert isinstance(result, PromptResult)
        assert "SMB" in result.description
        assert result.meta["tools_count"] == 6

    def test_cloud_security_audit_structure(self):
        """Message count ≥ 2, first role user, run_security_tool present."""
        messages = run(cloud_security_audit(provider="aws"))
        assert len(messages) >= 2
        assert messages[0].role == "user"
        combined = " ".join(str(m) for m in messages)
        assert "run_security_tool" in combined

    def test_cloud_security_audit_tools(self):
        """All expected tools mentioned."""
        messages = run(cloud_security_audit(provider="aws"))
        combined = " ".join(str(m) for m in messages)
        for tool in ["prowler", "trivy"]:
            assert tool in combined, f"Missing tool: {tool}"

    def test_cloud_security_audit_final_message(self):
        """Last message contains FINAL."""
        messages = run(cloud_security_audit(provider="aws"))
        assert "FINAL" in str(messages[-1])

    def test_cloud_security_audit_with_k8s(self):
        """k8s_api_ip adds kube_hunter to messages."""
        messages = run(cloud_security_audit(provider="aws", k8s_api_ip="192.168.1.100"))
        combined = " ".join(str(m) for m in messages)
        assert "kube_hunter" in combined

    def test_cloud_security_audit_default_image(self):
        """Default image is nginx:latest."""
        messages = run(cloud_security_audit(provider="aws"))
        combined = " ".join(str(m) for m in messages)
        assert "nginx" in combined

    def test_cloud_security_audit_custom_image(self):
        """Custom target_image appears in messages."""
        messages = run(cloud_security_audit(provider="aws", target_image="myapp:latest"))
        combined = " ".join(str(m) for m in messages)
        assert "myapp" in combined

    def test_cloud_security_audit_meta(self):
        """PromptResult includes meta."""
        import asyncio
        result = asyncio.new_event_loop().run_until_complete(
            cloud_security_audit(provider="aws")
        )
        assert isinstance(result, PromptResult)
        assert "cloud" in result.description.lower()
        assert result.meta["tools_count"] == 3

    def test_register_prompts_wires_title_and_tags(self):
        """register_prompts passes title+tags to mcp.prompt()."""
        from unittest.mock import MagicMock
        mock = MagicMock()
        mock.prompt.return_value = lambda f: f
        register_prompts(mock)
        calls = {c[1]["name"]: c[1] for c in mock.prompt.call_args_list}
        assert calls["bug_bounty_recon"]["title"] == "Bug Bounty Reconnaissance"
        assert "recon" in calls["bug_bounty_recon"]["tags"]
        assert calls["wifi_attack_chain"]["title"] == "WiFi WPA/WPA2 Attack Chain"
        assert "wifi" in calls["wifi_attack_chain"]["tags"]
        assert calls["smb_lateral_movement"]["title"] == "SMB Lateral Movement"
        assert "smb" in calls["smb_lateral_movement"]["tags"]
        assert calls["cloud_security_audit"]["title"] == "Cloud Security Audit"
        assert "cloud" in calls["cloud_security_audit"]["tags"]
        assert calls["ctf_web_challenge"]["title"] == "CTF Web Challenge"
        assert calls["ctf_challenge"]["title"] == "CTF Challenge (Universal)"


# ── CTF prompts ──────────────────────────────────────────────────────────


class TestCtfPrompts:
    """ctf_web_challenge and ctf_challenge — mock CTFWorkflowManager.

    What we validate: message structure (len ≥ 3, CTF context injected).
    Not validating: CTFWorkflowManager intelligence (tested separately).
    """

    CTF_WEB_KWARGS = {"url": "http://challenge.ctf.local:8080"}
    CTF_CHALLENGE_KWARGS = {
        "name": "test_challenge",
        "category": "web",
        "description": "A web challenge",
        "difficulty": "easy",
        "target": "http://10.0.0.1/",
        "points": "100",
    }

    # -- ctf_web_challenge --

    def test_ctf_web_structure(self):
        """Message len ≥ 3, CTF context, run_security_tool."""
        with patch("server_core.singletons.get_ctf_manager") as gm:
            gm.return_value.create_ctf_challenge_workflow.return_value = _ctf_workflow_stub()
            messages = run(ctf_web_challenge(**self.CTF_WEB_KWARGS))

        assert len(messages) >= 3
        assert messages[0].role == "user"
        combined = " ".join(str(m) for m in messages)
        assert "CTF" in combined

    def test_ctf_web_run_security_tool(self):
        """Messages contain run_security_tool references."""
        with patch("server_core.singletons.get_ctf_manager") as gm:
            gm.return_value.create_ctf_challenge_workflow.return_value = _ctf_workflow_stub()
            messages = run(ctf_web_challenge(**self.CTF_WEB_KWARGS))

        combined = " ".join(str(m) for m in messages)
        assert "run_security_tool" in combined

    def test_ctf_web_final_message(self):
        """Final message references flag/validation."""
        with patch("server_core.singletons.get_ctf_manager") as gm:
            gm.return_value.create_ctf_challenge_workflow.return_value = _ctf_workflow_stub()
            messages = run(ctf_web_challenge(**self.CTF_WEB_KWARGS))

        last = str(messages[-1])
        assert "FINAL" in last or "flag" in last.lower()

    def test_ctf_web_empty_workflow(self):
        """Empty workflow doesn't crash — uses fallback defaults."""
        with patch("server_core.singletons.get_ctf_manager") as gm:
            gm.return_value.create_ctf_challenge_workflow.return_value = {
                "workflow_steps": [],
                "strategies": [],
                "fallback_strategies": [],
                "tools": [],
                "estimated_time": 3600,
                "success_probability": 0.5,
                "validation_steps": [],
            }
            messages = run(ctf_web_challenge(**self.CTF_WEB_KWARGS))

        assert len(messages) >= 3
        combined = " ".join(str(m) for m in messages)
        assert "CTF" in combined
        assert "run_security_tool" in combined

    # -- ctf_challenge (universal) --

    def test_ctf_challenge_structure(self):
        """Message len ≥ 3, CTF context, run_security_tool."""
        with patch("server_core.singletons.get_ctf_manager") as gm:
            gm.return_value.create_ctf_challenge_workflow.return_value = _ctf_workflow_stub()
            with patch("server_core.workflows.ctf.toolManager.CTFToolManager") as tm_cls:
                tm_cls.return_value.get_tool_command.return_value = "nmap -sV target"
                messages = run(ctf_challenge(**self.CTF_CHALLENGE_KWARGS))

        assert len(messages) >= 3
        assert messages[0].role == "user"
        combined = " ".join(str(m) for m in messages)
        assert "CTF" in combined

    def test_ctf_challenge_run_security_tool(self):
        """Messages contain run_security_tool from workflow steps."""
        with patch("server_core.singletons.get_ctf_manager") as gm:
            gm.return_value.create_ctf_challenge_workflow.return_value = _ctf_workflow_stub()
            with patch("server_core.workflows.ctf.toolManager.CTFToolManager") as tm_cls:
                tm_cls.return_value.get_tool_command.return_value = "nmap -sV target"
                messages = run(ctf_challenge(**self.CTF_CHALLENGE_KWARGS))

        combined = " ".join(str(m) for m in messages)
        assert "run_security_tool" in combined

    def test_ctf_challenge_category_in_first_message(self):
        """First message includes category marker (e.g. [WEB])."""
        with patch("server_core.singletons.get_ctf_manager") as gm:
            gm.return_value.create_ctf_challenge_workflow.return_value = _ctf_workflow_stub()
            with patch("server_core.workflows.ctf.toolManager.CTFToolManager") as tm_cls:
                tm_cls.return_value.get_tool_command.return_value = "nmap -sV target"
                messages = run(ctf_challenge(**self.CTF_CHALLENGE_KWARGS))

        first = str(messages[0])
        assert "[WEB]" in first

    def test_ctf_challenge_strategies_section(self):
        """Strategies from stub appear in messages."""
        with patch("server_core.singletons.get_ctf_manager") as gm:
            gm.return_value.create_ctf_challenge_workflow.return_value = _ctf_workflow_stub()
            with patch("server_core.workflows.ctf.toolManager.CTFToolManager") as tm_cls:
                tm_cls.return_value.get_tool_command.return_value = "nmap -sV target"
                messages = run(ctf_challenge(**self.CTF_CHALLENGE_KWARGS))

        combined = " ".join(str(m) for m in messages)
        assert "KEY STRATEGIES" in combined

    def test_ctf_challenge_fallback_section(self):
        """Fallback strategies appear when strategies exist."""
        with patch("server_core.singletons.get_ctf_manager") as gm:
            gm.return_value.create_ctf_challenge_workflow.return_value = _ctf_workflow_stub()
            with patch("server_core.workflows.ctf.toolManager.CTFToolManager") as tm_cls:
                tm_cls.return_value.get_tool_command.return_value = "nmap -sV target"
                messages = run(ctf_challenge(**self.CTF_CHALLENGE_KWARGS))

        combined = " ".join(str(m) for m in messages)
        assert "FALLBACK" in combined

    def test_ctf_challenge_empty_workflow(self):
        """Empty workflow doesn't crash — fallback defaults used."""
        with patch("server_core.singletons.get_ctf_manager") as gm:
            gm.return_value.create_ctf_challenge_workflow.return_value = {
                "workflow_steps": [],
                "strategies": [],
                "fallback_strategies": [],
                "parallel_tasks": [],
                "tools": [],
                "estimated_time": 3600,
                "success_probability": 0.5,
                "validation_steps": [],
                "expected_artifacts": [],
                "resource_requirements": {},
            }
            with patch("server_core.workflows.ctf.toolManager.CTFToolManager") as tm_cls:
                tm_cls.return_value.get_tool_command.side_effect = Exception("tool not found")
                messages = run(ctf_challenge(**self.CTF_CHALLENGE_KWARGS))

        assert len(messages) >= 1
        first = str(messages[0])
        assert "[WEB]" in first

    def test_ctf_challenge_points_int_parsing(self):
        """Points as string '100' is parsed to int."""
        with patch("server_core.singletons.get_ctf_manager") as gm:
            gm.return_value.create_ctf_challenge_workflow.return_value = _ctf_workflow_stub()
            with patch("server_core.workflows.ctf.toolManager.CTFToolManager") as tm_cls:
                tm_cls.return_value.get_tool_command.return_value = "nmap -sV target"
                messages = run(ctf_challenge(**{**self.CTF_CHALLENGE_KWARGS, "points": "500"}))

        first = str(messages[0])
        assert "500 pts" in first

    def test_ctf_challenge_target_used_in_step_calls(self):
        """Target parameter is used in run_security_tool calls within steps."""
        with patch("server_core.singletons.get_ctf_manager") as gm:
            gm.return_value.create_ctf_challenge_workflow.return_value = _ctf_workflow_stub()
            with patch("server_core.workflows.ctf.toolManager.CTFToolManager") as tm_cls:
                tm_cls.return_value.get_tool_command.return_value = "nmap -sV target"
                messages = run(ctf_challenge(**self.CTF_CHALLENGE_KWARGS))

        combined = " ".join(str(m) for m in messages)
        assert "http://10.0.0.1/" in combined or '"target": "http://10.0.0.1/"' in combined


class TestPulseDashboards:
    """pulse_dashboards prompt — discovers CTF/pentest/recon UI entry points."""

    def test_single_user_message(self):
        result = run(pulse_dashboards(target="http://192.168.1.165/DVWA/"))
        assert len(result) == 1
        assert result[0].role == "user"

    def test_description_contains_expected_terms(self):
        result = run(pulse_dashboards(target="http://192.168.1.165/DVWA/"))
        desc = str(result[0]).lower()
        assert "ctf" in desc
        assert "pentest" in desc
        assert "recon" in desc

    def test_mentions_call_tool(self):
        result = run(pulse_dashboards(target="test.local"))
        desc = str(result[0])
        assert "call_tool" in desc

    def test_includes_target_in_output(self):
        result = run(pulse_dashboards(target="http://target.test"))
        assert "http://target.test" in str(result[0])

    def test_empty_target_placeholder(self):
        result = run(pulse_dashboards())
        assert "<target>" in str(result[0])

    def test_meta_present(self):
        result = run(pulse_dashboards(target="x"))
        assert len(result) == 1
