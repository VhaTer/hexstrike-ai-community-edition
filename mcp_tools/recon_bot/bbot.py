import asyncio
from fastmcp import Context
# mcp_tools/recon_bot/bbot.py

def register_bbot_tools(mcp, hexstrike_client):

    @mcp.tool()
    async def bbot_scan(ctx: Context, target: str, parameters: dict) -> dict:
        """
        Run BBot scan via HexStrike server.

        Endpoint:
            POST /api/bot/bbot

        Description:
            Interacts with the BBot module on the HexStrike server for reconnaissance and enumeration tasks.

        Parameters:
            target (str): The domain or IP address to scan.
            parameters (dict): BBot flags and module options.
                - f: Enable these flags (e.g. "subdomain-enum")
                - rf: Require modules to have this flag (e.g. "safe")
                - ef: Exclude these flags (e.g. "slow")
                - em: Exclude these individual modules (e.g. "ipneighbor")

        Returns:
            Query results as JSON

        Example:
            bbot_scan(
                target="example.com",
                parameters={
                    "f": "subdomain-enum",
                    "rf": "safe",
                    "ef": "slow",
                    "em": "ipneighbor"
                }
            )

        Usage:
            - Use for subdomain enumeration, module filtering, and safe/fast scanning.
            - Combine flags for advanced control.
            - Returns JSON with BBot response or error details.
        """
        await ctx.info(f"🤖 Starting BBOT scan: {target}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: hexstrike_client.safe_post("api/bot/bbot", {
                "target": target,
                "parameters": parameters
            })
        )

        phases = [
            (15, "🤖 Initializing BBOT scan..."),
            (35, "🌐 Running recon modules..."),
            (60, "🔍 Aggregating intelligence..."),
            (85, "📋 Building target graph..."),
        ]

        for progress, message in phases:
            done, _ = await asyncio.wait([future], timeout=20)
            if done:
                break
            await ctx.report_progress(progress, 100)
            await ctx.info(message)

        result = await future
        await ctx.report_progress(100, 100)

        if result.get("success"):
            await ctx.info("✅ BBOT scan completed")
        else:
            await ctx.error(f"❌ BBOT scan failed: {result.get('error', 'unknown')}")
        return result
