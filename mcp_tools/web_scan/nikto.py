# mcp_tools/web_scan/nikto.py

from typing import Dict, Any
import asyncio
from fastmcp import Context
import mcp_core.web_scan_direct as _web_scan_direct

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
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _web_scan_direct.web_scan_exec("nikto", data)
        )

        phases = [
            (15, "🔌 Connecting to web server..."),
            (35, "🔍 Scanning for vulnerabilities..."),
            (60, "🔍 Running vulnerability checks..."),
            (85, "📋 Compiling findings..."),
        ]
        for progress, message in phases:
            done, _ = await asyncio.wait([future], timeout=15)
            if done:
                break
            await ctx.report_progress(progress, 100)
            await ctx.info(message)

        result = await future
        await ctx.report_progress(100, 100)

        if result.get("success"):
            await ctx.info("✅ Completed successfully")
        else:
            await ctx.error(f"❌ Failed: {result.get('error', 'unknown')}")
        return result
