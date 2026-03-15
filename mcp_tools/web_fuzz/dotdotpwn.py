# mcp_tools/web_fuzz/dotdotpwn.py

from typing import Dict, Any
from fastmcp import Context

def register_dotdotpwn_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def dotdotpwn_scan(
        ctx: Context,
        target: str,
        module: str = "http",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Directory traversal vulnerability testing using DotDotPwn.

        Workflow position: targeted vulnerability test — use after content
        discovery reveals file inclusion or path parameters. Specifically
        tests for path traversal (../../etc/passwd) vulnerabilities.

        Parameters:
        - target: target hostname or IP (e.g. 'example.com' or '192.168.1.1')
                  do NOT include http:// — dotdotpwn handles the protocol
        - module: protocol module to test:
            'http'      — HTTP path traversal (most common)
            'http-url'  — HTTP URL-based traversal
            'ftp'       — FTP traversal
            'tftp'      — TFTP traversal
            'payload'   — generate payloads only (no active testing)
        - additional_args: extra dotdotpwn flags
            '-d <n>'     — traversal depth (default 6)
            '-f <file>'  — target file to read (default /etc/passwd)
            '-b'         — break after first match found
            '-q'         — quiet mode
            '-k <str>'   — string to search in response (confirms traversal)
            '-s'         — enable URL encoding of payloads
            '-o'         — use only ../ without encoding variations

        Prerequisites: target must have a path/file parameter that could be
        vulnerable to directory traversal. Identify candidates with katana
        or dirb first (look for ?file=, ?path=, ?page= parameters).

        Output: list of successful traversal payloads and their responses.

        Typical sequence:
            1. katana_crawl — find ?file= or ?page= parameters
            2. dotdotpwn_scan(target='example.com', module='http') — test traversal
        """
        data = {
            "target": target,
            "module": module,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting dotdotpwn: {target} [{module}]")
        result = hexstrike_client.safe_post("api/tools/dotdotpwn", data)
        if result.get("success"):
            await ctx.info(f"✅ dotdotpwn completed for {target}")
        else:
            await ctx.error(f"❌ dotdotpwn failed for {target}")
        return result
