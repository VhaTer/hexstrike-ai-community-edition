# mcp_tools/web_scan/jaeles.py

from typing import Dict, Any
from fastmcp import Context

def register_jaeles_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def jaeles_vulnerability_scan(
        ctx: Context,
        url: str,
        signatures: str = "",
        config: str = "",
        threads: int = 20,
        timeout: int = 20,
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Advanced web vulnerability scanning with custom signatures using Jaeles.

        Workflow position: targeted vulnerability scanning with custom signature sets.
        Jaeles covers complex multi-step vulnerability chains that template scanners
        like nuclei may miss — ideal for specific vuln classes or custom targets.

        Parameters:
        - url: target URL (e.g. 'https://example.com')
        - signatures: path to signature file or directory
                      (uses built-in signatures if empty)
                      common: '/root/jaeles-signatures/'
        - config: path to jaeles config file (optional)
        - threads: concurrent threads (default 20)
        - timeout: request timeout in seconds (default 20)
        - additional_args: extra jaeles flags
            '-s <sig>'        — specific signature(s) to run
            '--level <1-3>'   — signature level (1=safe, 3=aggressive)
            '--proxy <url>'   — route through proxy
            '--no-db'         — disable database logging

        Prerequisites: target accessible. Jaeles signatures installed
        (run `jaeles config update` to get latest signatures).

        Output: matched vulnerabilities with request/response evidence,
        severity rating, and remediation notes.

        jaeles vs nuclei:
        - nuclei  — large public template library, faster, broad CVE coverage
        - jaeles  — custom signatures, complex multi-step chains, research use

        Typical sequence:
            1. nuclei — broad template-based scan
            2. jaeles_vulnerability_scan — targeted with custom signatures
        """
        data = {
            "url": url,
            "signatures": signatures,
            "config": config,
            "threads": threads,
            "timeout": timeout,
            "additional_args": additional_args
        }
        await ctx.info(f"🔬 Starting jaeles scan: {url}")
        result = hexstrike_client.safe_post("api/tools/jaeles", data)
        if result.get("success"):
            await ctx.info(f"✅ jaeles completed for {url}")
        else:
            await ctx.error(f"❌ jaeles failed for {url}")
        return result
