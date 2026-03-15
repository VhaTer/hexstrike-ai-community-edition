# mcp_tools/web_scan/wpscan.py

from typing import Dict, Any
from fastmcp import Context

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
        result = hexstrike_client.safe_post("api/tools/wpscan", data)
        if result.get("success"):
            await ctx.info(f"✅ wpscan completed for {url}")
        else:
            await ctx.error(f"❌ wpscan failed for {url}")
        return result
