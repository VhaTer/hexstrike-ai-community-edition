# mcp_tools/web_scan/xsser.py

from typing import Dict, Any
from fastmcp import Context

def register_xsser_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def xsser_scan(
        ctx: Context,
        url: str,
        params: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        XSS vulnerability testing using XSSer.

        Workflow position: XSS testing alongside or after dalfox.
        Use xsser when dalfox misses findings or as a second opinion —
        different payload sets and injection techniques.

        Parameters:
        - url: target URL (e.g. 'https://example.com/search?q=test')
        - params: specific parameters to test (comma-separated)
        - additional_args: extra xsser flags
            '-s'              — statistics output
            '-v'              — verbose
            '--auto'          — automatic XSS discovery
            '--Cem <payload>' — custom XSS payload
            '--cookie <c>'    — set cookies
            '--proxy <url>'   — route through proxy
            '--threads <n>'   — concurrent threads
            '--timeout <sec>' — request timeout
            '--payload'       — show used payloads
            '--reverse-check' — check if injected code is reflected

        Prerequisites: target URL with injectable parameters.

        xsser vs dalfox:
        - dalfox  — more comprehensive, DOM mining, actively maintained, preferred
        - xsser   — good alternative, different payload set, useful for edge cases

        Typical sequence:
            1. dalfox_xss_scan — primary XSS scanner
            2. xsser_scan — secondary check for missed findings
        """
        data = {"url": url, "params": params, "additional_args": additional_args}
        await ctx.info(f"🔍 Starting xsser: {url}")
        result = hexstrike_client.safe_post("api/tools/xsser", data)
        if result.get("success"):
            await ctx.info(f"✅ xsser completed for {url}")
        else:
            await ctx.error(f"❌ xsser failed for {url}")
        return result
