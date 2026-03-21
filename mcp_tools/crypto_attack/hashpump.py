# mcp_tools/crypto_attack/hashpump.py

from fastmcp import Context
import mcp_core.misc_direct as _misc_direct
from typing import Dict, Any
import asyncio

def register_hashpump_tool(mcp, hexstrike_client, logger):

    @mcp.tool()
    async def hashpump_attack(signature: str, data: str, key_length: str, append_data: str, additional_args: str = "") -> Dict[str, Any]:
        """
        Execute HashPump for hash length extension attacks with enhanced logging.

        Args:
            signature: Original hash signature
            data: Original data
            key_length: Length of secret key
            append_data: Data to append
            additional_args: Additional HashPump arguments

        Returns:
            Hash length extension attack results
        """
        payload = {
            "signature": signature,
            "data": data,
            "key_length": key_length,
            "append_data": append_data,
            "additional_args": additional_args
        }
        await ctx.info(f"🔐 Starting HashPump attack")
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _misc_direct.misc_exec("hashpump", data)
        )
        if result.get("success"):
            await ctx.info(f"✅ HashPump attack completed")
        else:
            await ctx.error(f"❌ HashPump attack failed")
        return result