# mcp_tools/data_processing/anew.py

from typing import Dict, Any
from fastmcp import Context

def register_anew_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def anew_data_processing(
        ctx: Context,
        input_data: str,
        output_file: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Deduplicate and append new unique lines using anew.

        Workflow position: utility tool — use between discovery steps
        to deduplicate combined results before feeding into scanners.
        Prevents running the same URL/host through multiple tools twice.

        Parameters:
        - input_data: newline-separated data to deduplicate
        - output_file: append unique lines to this file (optional)
                       if omitted, returns deduplicated lines in result
        - additional_args: extra anew flags

        Prerequisites: none — pure text processing.

        Output: unique lines not previously seen in output_file,
        or the full deduplicated set if no file specified.

        Common use cases:
        - Combine gau + waybackurls output and remove duplicates
        - Merge subfinder + amass subdomain lists
        - Build a growing unique URL list across multiple scan sessions

        Typical pipeline:
            1. gau_discovery(domain='example.com')          → urls1
            2. waybackurls_discovery(domain='example.com')  → urls2
            3. anew_data_processing(input_data=urls1+urls2,
                                    output_file='all_urls.txt')
            4. Feed all_urls.txt into dalfox/sqlmap/ffuf
        """
        data = {
            "input_data": input_data,
            "output_file": output_file,
            "additional_args": additional_args
        }
        await ctx.info("📝 Starting anew deduplication")
        result = hexstrike_client.safe_post("api/tools/anew", data)
        if result.get("success"):
            await ctx.info("✅ anew completed")
        else:
            await ctx.error("❌ anew failed")
        return result
