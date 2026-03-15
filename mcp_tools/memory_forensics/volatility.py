# mcp_tools/memory_forensics/volatility.py

from typing import Dict, Any
from fastmcp import Context

def register_volatility_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def volatility_analyze(
        ctx: Context,
        memory_file: str,
        plugin: str,
        profile: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Memory forensics using Volatility 2 (use volatility3_analyze for modern systems).

        Parameters:
        - memory_file: path to memory dump (.raw, .vmem, .mem, .dmp)
        - plugin: volatility plugin to run
            'pslist'    — list processes       'netscan'  — network connections
            'pstree'    — process tree         'malfind'  — injected code
            'psscan'    — hidden processes     'hashdump' — password hashes
            'cmdline'   — process cmd args     'hivelist' — registry hives
            'dlllist'   — loaded DLLs          'dumpfiles'— extract files
        - profile: OS profile (e.g. 'Win7SP1x64') — use volatility3 for auto-detect
        - additional_args: extra flags (-p <pid>, -D <dir>)

        volatility vs volatility3: prefer volatility3 — auto-detects profile.
        """
        data = {
            "memory_file": memory_file,
            "plugin": plugin,
            "profile": profile,
            "additional_args": additional_args
        }
        await ctx.info(f"🧠 Starting volatility [{plugin}]: {memory_file}")
        result = hexstrike_client.safe_post("api/tools/volatility", data)
        if result.get("success"):
            await ctx.info(f"✅ volatility [{plugin}] completed")
        else:
            await ctx.error(f"❌ volatility [{plugin}] failed")
        return result
