# mcp_tools/web_scan/wpscan.py

from typing import Dict, Any
import asyncio
from fastmcp import Context
import mcp_core.web_scan_direct as _web_scan_direct

def register_wpscan_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def wpscan_analyze(
        ctx: Context,
        url: str,
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        WordPress security scanner using WPScan.

        Workflow position: use when httpx or nikto identifies WordPress
        (X-Powered-By: WordPress, /wp-login.php, /wp-admin/).
        WPScan is the definitive tool for WordPress-specific vulnerabilities.

        Parameters:
        - url: WordPress site URL (e.g. 'https://example.com')
        - additional_args: extra wpscan flags
            '--enumerate u'    — enumerate users (useful for brute force)
            '--enumerate p'    — enumerate plugins with vulnerabilities
            '--enumerate t'    — enumerate themes with vulnerabilities
            '--enumerate ap'   — enumerate all plugins (slow)
            '--api-token <t>'  — WPVulnDB API token for vuln data
            '--passwords <f>'  — wordlist for credential brute force
            '--usernames <u>'  — usernames to brute force
            '--random-agent'   — random User-Agent
            '--proxy <url>'    — route through proxy
            '-f cli'           — output format (cli, json, cli-no-colour)

        Prerequisites: target must be a WordPress site.
        WPVulnDB API token recommended for up-to-date vulnerability data
        (free tier: 75 requests/day).

        Output: WordPress version, installed plugins/themes with CVEs,
        user enumeration, and security misconfigurations.

        Typical sequence:
            1. httpx detects WordPress → wpscan_analyze
            2. wpscan_analyze(url='https://example.com',
                               additional_args='--enumerate u,p,t --api-token <key>')
            3. sqlmap/dalfox on vulnerable plugin parameters
            4. hydra or netexec for credential spray if users found
        """
        data = {"url": url, "additional_args": additional_args}
        await ctx.info(f"🔍 Starting wpscan: {url}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _web_scan_direct.web_scan_exec("wpscan", data)
        )

        phases = [
            (20, "🔌 Fingerprinting WordPress..."),
            (45, "🔍 Enumerating plugins and themes..."),
            (70, "🛡️ Checking for vulnerabilities..."),
            (88, "📋 Generating report..."),
        ]
        for progress, message in phases:
            done, _ = await asyncio.wait([future], timeout=12)
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
