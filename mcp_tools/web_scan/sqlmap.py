# mcp_tools/web_scan/sqlmap.py

from typing import Dict, Any
from fastmcp import Context

def register_sqlmap_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def sqlmap_scan(
        ctx: Context,
        url: str,
        data: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Automatic SQL injection detection and exploitation using sqlmap.

        Workflow position: targeted injection testing after parameter discovery.
        Use on URLs/endpoints with parameters — especially those found via
        arjun, paramspider, gau, or katana.

        Parameters:
        - url: target URL with parameters (e.g. 'https://example.com/search?q=test')
               or just the base URL if using data for POST
        - data: POST body data (e.g. 'username=test&password=test')
                if provided, sqlmap tests POST injection
        - additional_args: extra sqlmap flags
            '--batch'          — non-interactive, use defaults (important for automation)
            '--dbs'            — enumerate databases
            '--tables'         — enumerate tables (use with -D <db>)
            '--dump'           — dump table data (use with -D <db> -T <table>)
            '--level <1-5>'    — test level (default 1, increase for deeper testing)
            '--risk <1-3>'     — risk level (default 1, increase with caution)
            '--tamper <script>'— use tamper script for WAF bypass
            '--random-agent'   — random User-Agent to avoid detection
            '--proxy <url>'    — route through proxy
            '--technique BEUSTQ' — injection techniques to test
            '-p <param>'       — test specific parameter only
            '--forms'          — auto-detect and test forms

        Prerequisites: target URL must have injectable parameters.
        Always use --batch in automated/agent workflows to prevent interactive prompts.

        ⚠️ sqlmap can be destructive with --risk 3 — use low risk levels first.

        Typical sequence:
            1. arjun_parameter_discovery(url='endpoint') — find params
            2. sqlmap_scan(url='endpoint?param=test',
                           additional_args='--batch --dbs') — detect + enumerate
            3. sqlmap_scan(additional_args='--batch --dump -D <db> -T <table>') — extract
        """
        data_payload = {"url": url, "data": data, "additional_args": additional_args}
        await ctx.info(f"💉 Starting sqlmap: {url}")
        result = hexstrike_client.safe_post("api/tools/sqlmap", data_payload)
        if result.get("success"):
            await ctx.info(f"✅ sqlmap completed for {url}")
        else:
            await ctx.error(f"❌ sqlmap failed for {url}")
        return result
