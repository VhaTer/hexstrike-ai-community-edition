# mcp_tools/param_discovery/arjun.py

from typing import Dict, Any
from fastmcp import Context

def register_arjun_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def arjun_parameter_discovery(
        ctx: Context,
        url: str,
        method: str = "GET",
        wordlist: str = "",
        delay: int = 0,
        threads: int = 25,
        stable: bool = False,
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Discover hidden HTTP parameters using Arjun.

        Workflow position: parameter discovery after URL recon (gau/waybackurls).
        Use on interesting endpoints to find hidden parameters before
        injection testing (SQLi, XSS, SSRF, etc.).

        Parameters:
        - url: target URL (e.g. 'https://example.com/api/search')
        - method: HTTP method ('GET', 'POST', 'JSON', 'XML')
        - wordlist: custom parameter wordlist path (uses built-in if empty)
        - delay: delay between requests in seconds (0 = no delay)
        - threads: concurrent threads (default 25 — reduce if rate-limited)
        - stable: use stable mode (slower, more accurate, fewer false positives)
        - additional_args: extra arjun flags

        Prerequisites: target URL must be accessible.

        Output: list of valid hidden parameters with their reflected values.
        Feed discovered parameters into sqlmap, dalfox, or manual testing.

        arjun vs paramspider vs x8:
        - arjun      — active probing, most accurate, finds hidden params
        - paramspider — passive mining from archives, no direct contact
        - x8          — fast active discovery, good for large wordlists

        Typical sequence:
            1. gau_discovery + waybackurls_discovery        — find endpoints
            2. arjun_parameter_discovery(url='endpoint')    — find params
            3. sqlmap or dalfox on endpoint + params
        """
        data = {
            "url": url,
            "method": method,
            "wordlist": wordlist,
            "delay": delay,
            "threads": threads,
            "stable": stable,
            "additional_args": additional_args
        }
        await ctx.info(f"🎯 Starting arjun parameter discovery: {url}")
        result = hexstrike_client.safe_post("api/tools/arjun", data)
        if result.get("success"):
            await ctx.info(f"✅ arjun completed for {url}")
        else:
            await ctx.error(f"❌ arjun failed for {url}")
        return result

    @mcp.tool()
    async def arjun_scan(
        ctx: Context,
        url: str,
        method: str = "GET",
        data: str = "",
        headers: str = "",
        timeout: str = "",
        output_file: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Advanced Arjun scan with custom headers, body data, and output options.

        Use this when you need full control over the request — custom headers
        (auth tokens, cookies), POST body, or saving results to file.
        Use arjun_parameter_discovery for standard discovery.

        Parameters:
        - url: target URL
        - method: HTTP method ('GET', 'POST', 'JSON', 'XML', 'HEADERS')
        - data: POST body data (e.g. 'key=value&other=test')
        - headers: custom headers as JSON string
                   (e.g. '{"Authorization": "Bearer token"}')
        - timeout: request timeout in seconds
        - output_file: save results to this file path
        - additional_args: extra arjun flags

        Prerequisites: target URL accessible, valid auth if required.
        """
        payload = {
            "url": url,
            "method": method,
            "data": data,
            "headers": headers,
            "timeout": timeout,
            "output_file": output_file,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting arjun scan: {url}")
        result = hexstrike_client.safe_post("api/tools/arjun", payload)
        if result.get("success"):
            await ctx.info(f"✅ arjun scan completed for {url}")
        else:
            await ctx.error(f"❌ arjun scan failed for {url}")
        return result
