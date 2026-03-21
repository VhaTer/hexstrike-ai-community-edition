# mcp_tools/api_fuzz/api_fuzzer.py

from typing import Dict, Any
import asyncio
from fastmcp import Context
import mcp_core.misc_direct as _misc_direct

def register_api_fuzzer_tool(mcp, hexstrike_client, logger):
    
    @mcp.tool()
    async def api_fuzzer(ctx: Context, base_url: str, endpoints: str = "", methods: str = "GET,POST,PUT,DELETE", wordlist: str = "/usr/share/wordlists/api/api-endpoints.txt") -> Dict[str, Any]:
        """
        Advanced API endpoint fuzzing with intelligent parameter discovery.

        Args:
            base_url: Base URL of the API
            endpoints: Comma-separated list of specific endpoints to test
            methods: HTTP methods to test (comma-separated)
            wordlist: Wordlist for endpoint discovery

        Returns:
            API fuzzing results with endpoint discovery and vulnerability assessment
        """
        data = {
            "base_url": base_url,
            "endpoints": [e.strip() for e in endpoints.split(",") if e.strip()] if endpoints else [],
            "methods": [m.strip() for m in methods.split(",")],
            "wordlist": wordlist
        }

        await ctx.info(f"🔍 Starting API fuzzing: {base_url}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _misc_direct.misc_exec("api_fuzzer", data)
        )
        done, _ = await asyncio.wait([future], timeout=30)
        if not done:
            await ctx.report_progress(50, 100)
            await ctx.info("⏳ Still running...")
        result = await future
        await ctx.report_progress(100, 100)

        if result.get("success"):
            fuzzing_type = result.get("fuzzing_type", "unknown")
            if fuzzing_type == "endpoint_testing":
                endpoint_count = len(result.get("results", []))
                await ctx.info(f"✅ API endpoint testing completed: {endpoint_count} endpoints tested")
            else:
                await ctx.info(f"✅ API endpoint discovery completed")
        else:
            await ctx.error("❌ API fuzzing failed")

        return result
