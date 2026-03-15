# mcp_tools/ops/auto_install.py

from typing import Dict, Any
from fastmcp import Context

def register_auto_install_tools(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def auto_install_missing_apt_tools(ctx: Context) -> Dict[str, Any]:
        """
        Detect and install missing apt-installable tools on the HexStrike server.
        Run after deployment or when a tool fails with 'command not found'.
        """
        await ctx.info("🔧 Triggering auto-install of missing apt tools")
        result = hexstrike_client.safe_post("api/tools/auto-install-missing-apt", {})
        if result.get("success"):
            await ctx.info(f"✅ Auto-install attempted for: {result.get('attempted_tools', [])}")
        else:
            await ctx.error("❌ Auto-install failed")
        return result
