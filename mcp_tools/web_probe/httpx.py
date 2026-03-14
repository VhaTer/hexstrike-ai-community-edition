# mcp_tools/web_probe/httpx.py

from typing import Dict, Any
from fastmcp import Context

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
        Essential step to avoid wasting time on dead subdomains.

        Parameters:
        - target: single URL, IP, or path to file containing hosts
                  (e.g. 'https://example.com', '192.168.1.1',
                   '/tmp/subdomains.txt', 'example.com:8443')
        - probe: enable HTTP/HTTPS probing (default True)
        - tech_detect: detect web technologies (framework, CMS, server)
        - status_code: include HTTP status codes in output
        - content_length: include response content length
        - title: include page titles (useful for quick triage)
        - web_server: show web server header (Apache, nginx, IIS...)
        - threads: concurrent probing threads (default 50)
        - additional_args: extra httpx flags
            '-p 80,443,8080,8443'  — probe specific ports
            '-follow-redirects'    — follow HTTP redirects
            '-mc 200,301,302'      — match specific status codes
            '-fc 404'              — filter out status codes
            '-o <file>'            — save output to file
            '-json'                — JSON output format

        Prerequisites: none — HTTP requests only.

        Output: live hosts with requested metadata (status, title, tech, server).
        Feed live hosts into gobuster, nikto, or nuclei for deeper scanning.

        Typical subdomain pipeline:
            1. subfinder_scan(domain='example.com')
            2. amass_scan(domain='example.com')
            3. httpx_probe(target='/tmp/subdomains.txt',
                           status_code=True, title=True,
                           tech_detect=True)        — filter live + triage
            4. nuclei or gobuster on live hosts only
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
        result = hexstrike_client.safe_post("api/tools/httpx", data)
        if result.get("success"):
            await ctx.info(f"✅ httpx probe completed for {target}")
        else:
            await ctx.error(f"❌ httpx probe failed for {target}")
        return result
