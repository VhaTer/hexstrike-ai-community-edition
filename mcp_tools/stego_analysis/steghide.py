# mcp_tools/stego_analysis/steghide.py

from typing import Dict, Any
import asyncio
from fastmcp import Context

def register_steghide_tool(mcp, hexstrike_client, logger):
    
    @mcp.tool()
    async def steghide_analysis(ctx: Context, action: str, cover_file: str, embed_file: str = "", passphrase: str = "", output_file: str = "", additional_args: str = "") -> Dict[str, Any]:
        """
        Execute Steghide for steganography analysis with enhanced logging.

        Args:
            action: Action to perform (extract, embed, info)
            cover_file: Cover file for steganography
            embed_file: File to embed (for embed action)
            passphrase: Passphrase for steganography
            output_file: Output file path
            additional_args: Additional Steghide arguments

        Returns:
            Steganography analysis results
        """
        data = {
            "action": action,
            "cover_file": cover_file,
            "embed_file": embed_file,
            "passphrase": passphrase,
            "output_file": output_file,
            "additional_args": additional_args
        }
        await ctx.info(f"🖼️ Starting Steghide {action}: {cover_file}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: hexstrike_client.safe_postsafe_post("api/tools/steghide", data)
        )

        done, _ = await asyncio.wait([future], timeout=30)
        if not done:
            await ctx.report_progress(50, 100)
            await ctx.info("⏳ Still running...")

        result = await future
        await ctx.report_progress(100, 100)

        if result.get("success"):
            await ctx.info("✅ Completed successfully")
        else:
            await ctx.error(f"❌ Failed: {result.get('error', 'unknown')}")
        return result
