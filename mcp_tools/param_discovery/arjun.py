# mcp_tools/param_discovery/arjun.py

from typing import Dict, Any
import asyncio
from fastmcp import Context
import mcp_core.web_recon_direct as _web_recon_direct

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
        - threads: concurrent threads (default 25)
        - stable: use stable mode (slower, more accurate)
        - additional_args: extra arjun flags

        Typical sequence:
            1. gau_discovery + waybackurls_discovery — find endpoints
            2. arjun_parameter_discovery(url='endpoint') — find params
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
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _web_recon_direct.web_recon_exec("arjun", data)
        )

        phases = [
            (20, "🔍 Loading parameter wordlist..."),
            (50, "💥 Testing parameters..."),
            (80, "📋 Processing discovered parameters..."),
        ]
        for progress, message in phases:
            done, _ = await asyncio.wait([future], timeout=10)
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

        Use when you need full control — custom headers (auth tokens, cookies),
        POST body, or saving results to file.

        Parameters:
        - url: target URL
        - method: HTTP method ('GET', 'POST', 'JSON', 'XML', 'HEADERS')
        - data: POST body data
        - headers: custom headers as JSON string
        - timeout: request timeout in seconds
        - output_file: save results to this file path
        - additional_args: extra arjun flags
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
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _web_recon_direct.web_recon_exec("arjun", payload)
        )
        if result.get("success"):
            await ctx.info(f"✅ arjun scan completed for {url}")
        else:
            await ctx.error(f"❌ arjun scan failed for {url}")
        return result
