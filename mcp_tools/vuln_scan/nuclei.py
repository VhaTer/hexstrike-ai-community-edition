# mcp_tools/vuln_scan/nuclei.py

from typing import Dict, Any
from fastmcp import Context

def register_nuclei(mcp, hexstrike_client, logger=None, HexStrikeColors=None):

    @mcp.tool()
    async def nuclei_scan(
        ctx: Context,
        target: str,
        severity: str = "",
        tags: str = "",
        template: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Template-based vulnerability scanner using Nuclei — 4000+ templates for CVEs and misconfigs.

        Workflow position: broad vulnerability sweep — use early after httpx confirms
        live targets. Nuclei gives the widest signal for the least effort.
        Run before targeted tools (sqlmap, dalfox) to prioritize attack surface.

        Parameters:
        - target: URL, IP, or file of targets (e.g. 'https://example.com', '192.168.1.1',
                  '/tmp/targets.txt' for bulk scanning)
        - severity: filter by severity (comma-separated)
            'critical'              — RCE, auth bypass, critical CVEs only
            'critical,high'         — recommended starting point
            'critical,high,medium'  — broader coverage
            ''                      — all severities (default)
        - tags: filter by template tags (comma-separated)
            'cve'          — all CVE templates
            'rce'          — remote code execution
            'lfi'          — local file inclusion
            'sqli'         — SQL injection
            'xss'          — cross-site scripting
            'wordpress'    — WordPress-specific templates
            'apache,nginx' — specific server templates
            'misconfig'    — misconfiguration checks
        - template: path to custom template file or directory
        - additional_args: extra nuclei flags
            '-c <n>'              — concurrency (default 25)
            '-rl <n>'             — rate limit requests/sec
            '-timeout <sec>'      — timeout per request
            '-H "header: value"'  — custom headers
            '-proxy <url>'        — route through proxy
            '-o <file>'           — save output to file
            '-json'               — JSON output format
            '-update-templates'   — update template library first

        Prerequisites: nuclei templates installed (run `nuclei -update-templates`).

        Output: matched vulnerabilities with template ID, severity, evidence,
        and remediation reference. CRITICAL findings warrant immediate escalation.

        nuclei vs nikto vs jaeles:
        - nuclei  — best coverage, 4000+ templates, fast, preferred first scanner
        - nikto   — web server specific, older checks, good complement
        - jaeles  — custom signatures, complex chains, research use

        Typical sequence:
            1. nuclei_scan(target='https://example.com',
                           severity='critical,high')    — priority sweep
            2. nuclei_scan(target='https://example.com',
                           tags='cve,misconfig')        — broader sweep
            3. Follow up with sqlmap/dalfox on injection findings
        """
        data: Dict[str, Any] = {
            "target": target,
            "severity": severity,
            "tags": tags,
            "template": template,
            "additional_args": additional_args,
            "use_recovery": True
        }
        await ctx.info(f"🔬 Starting nuclei scan: {target}")
        if severity:
            await ctx.info(f"Severity filter: {severity}")
        if tags:
            await ctx.info(f"Tags filter: {tags}")

        result = hexstrike_client.safe_post("api/tools/nuclei", data)

        if result.get("success"):
            await ctx.info(f"✅ nuclei completed for {target}")
            output = result.get("stdout", "")
            if "CRITICAL" in output:
                await ctx.warning("🚨 CRITICAL vulnerabilities detected!")
            elif "HIGH" in output:
                await ctx.warning("⚠️ HIGH severity vulnerabilities found!")
            if result.get("recovery_info", {}).get("recovery_applied"):
                attempts = result["recovery_info"].get("attempts_made", 1)
                await ctx.info(f"⚠️ Recovery applied: {attempts} attempt(s)")
        else:
            await ctx.error(f"❌ nuclei failed for {target}")
        return result
