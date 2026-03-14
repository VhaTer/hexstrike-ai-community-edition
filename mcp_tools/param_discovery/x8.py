# mcp_tools/param_discovery/x8.py

from typing import Dict, Any
from fastmcp import Context

def register_x8_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def x8_parameter_discovery(
        ctx: Context,
        url: str,
        wordlist: str = "/usr/share/wordlists/x8/params.txt",
        method: str = "GET",
        body: str = "",
        headers: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Fast hidden parameter discovery using x8.

        Workflow position: active parameter discovery alongside arjun.
        x8 is faster than arjun for large wordlists — use for broad sweeps,
        then verify interesting findings with arjun stable mode.

        Parameters:
        - url: target URL (e.g. 'https://example.com/search')
        - wordlist: parameter wordlist path
                    (default: /usr/share/wordlists/x8/params.txt)
        - method: HTTP method ('GET', 'POST')
        - body: request body for POST requests
        - headers: custom headers (e.g. 'Authorization: Bearer token')
        - additional_args: extra x8 flags
            '-c <n>'          — concurrency (default: 1)
            '-t <ms>'         — timeout in milliseconds
            '--output-format' — output format (json, text)

        Prerequisites: target URL accessible.

        Output: discovered parameters with response differences
        that indicate the parameter affects server behavior.

        x8 vs arjun:
        - x8    — faster, better for large wordlists, concurrency support
        - arjun — slower but more accurate, multiple HTTP methods, stable mode

        Typical sequence:
            1. x8_parameter_discovery(url='endpoint')      — fast broad sweep
            2. arjun_parameter_discovery(url='endpoint',
                                         stable=True)       — verify findings
            3. Test confirmed params with sqlmap/dalfox
        """
        data = {
            "url": url,
            "wordlist": wordlist,
            "method": method,
            "body": body,
            "headers": headers,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting x8 parameter discovery: {url}")
        result = hexstrike_client.safe_post("api/tools/x8", data)
        if result.get("success"):
            await ctx.info(f"✅ x8 completed for {url}")
        else:
            await ctx.error(f"❌ x8 failed for {url}")
        return result
