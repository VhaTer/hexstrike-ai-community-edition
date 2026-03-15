# mcp_tools/web_scan/nikto.py

from typing import Dict, Any
from fastmcp import Context

def register_nikto_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def nikto_scan(
        ctx: Context,
        target: str,
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Web server vulnerability scanner using Nikto.

        Workflow position: broad web server check — run early after httpx
        confirms target is live. Catches misconfigurations, default files,
        outdated server software, and common vulnerabilities quickly.

        Parameters:
        - target: target URL or IP (e.g. 'https://example.com' or '192.168.1.1')
        - additional_args: extra nikto flags
            '-p <port>'        — specify port
            '-ssl'             — force SSL
            '-nossl'           — disable SSL
            '-Tuning <x>'      — tune scan type (1=files, 2=config, 4=injection...)
            '-maxtime <sec>'   — max scan duration
            '-o <file>'        — save output to file
            '-Format <fmt>'    — output format (txt, xml, html, csv)
            '-useproxy'        — use defined proxy

        Prerequisites: target accessible on HTTP/HTTPS.

        Output: discovered vulnerabilities, misconfigurations, outdated software,
        interesting files/directories, and default credentials.

        nikto vs nuclei:
        - nikto   — broad web server checks, good for quick overview
        - nuclei  — template-based, more precise, broader CVE coverage

        Typical sequence:
            1. nikto_scan(target='https://example.com')       — broad check
            2. nuclei — targeted CVE/vuln scanning
            3. sqlmap/dalfox on discovered parameters
        """
        data = {"target": target, "additional_args": additional_args}
        await ctx.info(f"🔬 Starting nikto scan: {target}")
        result = hexstrike_client.safe_post("api/tools/nikto", data)
        if result.get("success"):
            await ctx.info(f"✅ nikto completed for {target}")
        else:
            await ctx.error(f"❌ nikto failed for {target}")
        return result
