# mcp_tools/password_cracking/aircrack_ng.py

from typing import Dict, Any, Optional, List
import asyncio
from fastmcp import Context
import mcp_core.password_cracking_direct as _pwdcrack_direct

def register_aircrack_ng_tools(mcp, hexstrike_client, logger):

    @mcp.tool()
    async def aircrack_ng_analysis(
        ctx: Context,
        capture_files: List[str],
        wordlist: Optional[str] = None,
        bssid: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute Aircrack-ng for Wi-Fi password cracking.

        Args:
            capture_files (List[str]): List of capture file paths (.cap, .ivs, etc.).
            wordlist (str): Path to a wordlist file.
            bssid (str, optional): Target BSSID (AP MAC address).

        Returns:
            dict: Results of the Aircrack-ng analysis or error information.
        """
        if not capture_files:
            return {"success": False, "error": "At least one capture file must be provided."}
        if not wordlist:
            return {"success": False, "error": "A wordlist file must be provided."}

        await ctx.info(f"🔍 Starting Aircrack-ng on {len(capture_files)} capture file(s)")

        data = {
            "capture_files": capture_files,
            "bssid": bssid,
            "wordlist": wordlist
        }

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _pwdcrack_direct.pwdcrack_exec("aircrack_ng", data)
        )
        if result.get("success"):
            await ctx.info("✅ Aircrack-ng analysis completed")
        else:
            await ctx.error("❌ Aircrack-ng analysis failed")
        return result
