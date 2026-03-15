# mcp_tools/web_scan/dalfox.py

from typing import Dict, Any
from fastmcp import Context

def register_dalfox_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def dalfox_xss_scan(
        ctx: Context,
        url: str,
        pipe_mode: bool = False,
        blind: bool = False,
        mining_dom: bool = True,
        mining_dict: bool = True,
        custom_payload: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Advanced XSS vulnerability scanning using Dalfox.

        Workflow position: XSS testing after parameter discovery.
        Dalfox is the most capable XSS scanner — DOM analysis, blind XSS,
        and dictionary-based mining make it comprehensive.

        Parameters:
        - url: target URL with parameter(s) to test
               (e.g. 'https://example.com/search?q=test')
        - pipe_mode: accept URLs from stdin pipe (for bulk scanning)
        - blind: enable blind XSS testing — injects payloads that call
                 back to a listener when executed (requires dalfox listener)
        - mining_dom: analyze DOM for XSS sinks (default True)
        - mining_dict: use dictionary-based payload mining (default True)
        - custom_payload: inject a specific custom XSS payload
        - additional_args: extra dalfox flags
            '--cookie <c>'     — set cookies for authenticated testing
            '--header <h>'     — custom headers
            '--proxy <url>'    — route through proxy
            '--timeout <sec>'  — request timeout
            '--delay <ms>'     — delay between requests (ms)
            '-w <threads>'     — concurrent workers
            '--output <file>'  — save results to file
            '--format <fmt>'   — output format (plain, json, markdown)

        Prerequisites: target must reflect input somewhere in the response.
        Use arjun first to discover which parameters reflect input.

        dalfox vs xsser:
        - dalfox — more comprehensive, DOM mining, blind XSS, actively maintained
        - xsser  — alternative, good for edge cases dalfox misses

        Typical XSS testing sequence:
            1. arjun_parameter_discovery — find reflected parameters
            2. dalfox_xss_scan(url='endpoint?param=test') — test each param
            3. Manual verification of confirmed findings
        """
        data = {
            "url": url,
            "pipe_mode": pipe_mode,
            "blind": blind,
            "mining_dom": mining_dom,
            "mining_dict": mining_dict,
            "custom_payload": custom_payload,
            "additional_args": additional_args
        }
        await ctx.info(f"🎯 Starting dalfox XSS scan: {url if url else 'pipe mode'}")
        result = hexstrike_client.safe_post("api/tools/dalfox", data)
        if result.get("success"):
            await ctx.info("✅ dalfox XSS scan completed")
        else:
            await ctx.error("❌ dalfox XSS scan failed")
        return result
