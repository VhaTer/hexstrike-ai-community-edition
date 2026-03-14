# mcp_tools/param_discovery/paramspider.py

from typing import Dict, Any
from fastmcp import Context

def register_paramspider_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def paramspider_scan(
        ctx: Context,
        domain: str,
        level: int = 2,
        exclude: str = "png,jpg,gif,jpeg,swf,woff,svg,pdf,css,ico",
        output: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Mine parameterized URLs from web archives using ParamSpider.

        Workflow position: passive parameter discovery — no direct contact
        with target. Use before arjun/x8 to build a list of known
        parameterized endpoints from historical data.

        Parameters:
        - domain: target domain (e.g. 'example.com')
        - level: crawl depth level (1=low, 2=medium default, 3=high)
        - exclude: comma-separated file extensions to exclude
        - output: save results to this file path (optional)
        - additional_args: extra paramspider flags

        Prerequisites: none — queries web archives passively.

        Output: list of URLs with parameter placeholders (FUZZ markers).
        Ready for direct use with ffuf, sqlmap, or dalfox.

        paramspider vs arjun vs x8:
        - paramspider — passive, no contact, mines known params from archives
        - arjun       — active, discovers hidden params not in archives
        - x8          — active, fast large-wordlist scanning

        Best practice: run paramspider first (passive), then arjun on
        interesting endpoints (active verification).

        Typical pipeline:
            1. paramspider_scan(domain='example.com')         — passive mining
            2. arjun_parameter_discovery(url='<endpoint>')    — active discovery
            3. sqlmap/dalfox on parameterized URLs
        """
        data = {
            "domain": domain,
            "level": level,
            "exclude": exclude,
            "output": output,
            "additional_args": additional_args
        }
        await ctx.info(f"🕷️ Starting paramspider: {domain}")
        result = hexstrike_client.safe_post("api/tools/paramspider", data)
        if result.get("success"):
            await ctx.info(f"✅ paramspider completed for {domain}")
        else:
            await ctx.error(f"❌ paramspider failed for {domain}")
        return result
