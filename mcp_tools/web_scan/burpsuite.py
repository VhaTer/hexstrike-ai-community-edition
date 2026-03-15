# mcp_tools/web_scan/burpsuite.py

from typing import Dict, Any
from fastmcp import Context

def register_burpsuite_tool(mcp, hexstrike_client, logger=None, HexStrikeColors=None):

    @mcp.tool()
    async def burpsuite_scan(
        ctx: Context,
        project_file: str = "",
        config_file: str = "",
        target: str = "",
        headless: bool = False,
        scan_type: str = "",
        scan_config: str = "",
        output_file: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Automated web vulnerability scanning using Burp Suite.

        Workflow position: comprehensive web app scanning — use after
        manual recon to run Burp's automated scanner against a target.
        Particularly powerful for authenticated scanning with session tokens.

        Parameters:
        - project_file: path to existing Burp project file (.burp)
        - config_file: path to Burp user/project config file
        - target: target URL for automated scanning
        - headless: run Burp in headless mode (no GUI, for automation)
        - scan_type: type of scan to perform (e.g. 'crawl_and_audit')
        - scan_config: scan configuration name or path
        - output_file: path to save scan results
        - additional_args: extra Burp CLI arguments

        Prerequisites: Burp Suite installed. Pro license required for
        automated scanner. Headless mode requires valid license.

        Output: discovered vulnerabilities with evidence, severity,
        and remediation guidance in Burp's format.

        Note: Burp Suite is most powerful when used interactively.
        For automated scanning, nuclei + dalfox + sqlmap often cover
        the same ground without requiring a license.
        """
        data = {
            "project_file": project_file,
            "config_file": config_file,
            "target": target,
            "headless": headless,
            "scan_type": scan_type,
            "scan_config": scan_config,
            "output_file": output_file,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting Burp Suite scan: {target or 'configured target'}")
        result = hexstrike_client.safe_post("api/tools/burpsuite", data)
        if result.get("success"):
            await ctx.info("✅ Burp Suite scan completed")
        else:
            await ctx.error("❌ Burp Suite scan failed")
        return result

    @mcp.tool()
    async def burpsuite_alternative_scan(
        ctx: Context,
        target: str,
        scan_type: str = "comprehensive",
        headless: bool = True,
        max_depth: int = 3,
        max_pages: int = 50
    ) -> Dict[str, Any]:
        """
        Comprehensive web security scan combining HTTP framework and browser agent.

        Use this as a Burp Suite alternative when Burp Pro is not available.
        Combines crawling and active vulnerability testing in one call.

        Parameters:
        - target: target URL (e.g. 'https://example.com')
        - scan_type: 'comprehensive' (all checks) or 'quick' (essential only)
        - headless: use headless browser for JS-heavy targets (default True)
        - max_depth: crawl depth limit (default 3)
        - max_pages: max pages to crawl (default 50)

        Prerequisites: target accessible.
        """
        data = {
            "target": target,
            "scan_type": scan_type,
            "headless": headless,
            "max_depth": max_depth,
            "max_pages": max_pages
        }
        await ctx.info(f"🔍 Starting burpsuite alternative scan: {target}")
        result = hexstrike_client.safe_post("api/tools/burpsuite-alt", data)
        if result.get("success"):
            await ctx.info(f"✅ Alternative scan completed for {target}")
        else:
            await ctx.error(f"❌ Alternative scan failed for {target}")
        return result
