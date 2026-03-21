# mcp_tools/web_probe/httpx.py

from typing import Dict, Any
import asyncio
from fastmcp import Context
import mcp_core.web_recon_direct as _web_recon_direct

def register_httpx_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def httpx_probe(
        ctx: Context,
        target: str,
        probe: bool = True,
        tech_detect: bool = False,
        status_code: bool = False,
        content_length: bool = False,
        title: bool = False,
        web_server: bool = False,
        threads: int = 50,
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Fast HTTP probing and technology detection using httpx.

        Workflow position: post-subdomain-discovery — filter live hosts
        before running heavier web scanners (nikto, nuclei, gobuster).

        Parameters:
        - target: single URL, IP, or path to file containing hosts
        - probe: enable HTTP/HTTPS probing (default True)
        - tech_detect: detect web technologies
        - status_code: include HTTP status codes
        - content_length: include response content length
        - title: include page titles
        - web_server: show web server header
        - threads: concurrent probing threads (default 50)
        - additional_args: extra httpx flags

        Typical subdomain pipeline:
            1. subfinder_scan + amass_scan — enumerate subdomains
            2. httpx_probe(target='/tmp/subdomains.txt',
                           status_code=True, title=True) — filter live
            3. nuclei or gobuster on live hosts only
        """
        data = {
            "target": target,
            "probe": probe,
            "tech_detect": tech_detect,
            "status_code": status_code,
            "content_length": content_length,
            "title": title,
            "web_server": web_server,
            "threads": threads,
            "additional_args": additional_args
        }
        await ctx.info(f"🌍 Starting httpx probe: {target}")
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _web_recon_direct.web_recon_exec("httpx", data)
        )
        if result.get("success"):
            await ctx.info(f"✅ httpx probe completed for {target}")
        else:
            await ctx.error(f"❌ httpx probe failed for {target}")
        return result
