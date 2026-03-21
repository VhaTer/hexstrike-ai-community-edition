# mcp_tools/memory_forensics/volatility3.py

from typing import Dict, Any
import asyncio
from fastmcp import Context
import mcp_core.misc_direct as _misc_direct

def register_volatility3(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def volatility3_analyze(
        ctx: Context,
        memory_file: str,
        plugin: str,
        output_file: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Advanced memory forensics using Volatility3 — auto-detects OS profile.

        Parameters:
        - memory_file: path to memory dump (.raw, .vmem, .mem, .dmp)
        - plugin: volatility3 plugin (namespace.PluginName format)
            Windows: windows.pslist, windows.pstree, windows.psscan,
                     windows.cmdline, windows.netscan, windows.malfind,
                     windows.dlllist, windows.hashdump, windows.dumpfiles
            Linux:   linux.pslist, linux.bash, linux.check_syscall
        - output_file: save output to file (optional)
        - additional_args: extra flags (-p <pid>, --output-dir <dir>)

        Preferred over volatility — no manual profile selection needed.

        Typical sequence:
            1. volatility3_analyze(memory_file='dump.raw', plugin='windows.pslist')
            2. volatility3_analyze(plugin='windows.netscan')
            3. volatility3_analyze(plugin='windows.malfind')
            4. volatility3_analyze(plugin='windows.hashdump') → hashcat mode 1000
        """
        data = {
            "memory_file": memory_file,
            "plugin": plugin,
            "output_file": output_file,
            "additional_args": additional_args
        }
        await ctx.info(f"🧠 Starting volatility3 [{plugin}]: {memory_file}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _misc_direct.misc_exec("volatility3", data)
        )

        phases = [
            (15, "📂 Loading memory image..."),
            (40, "🔍 Auto-detecting OS profile..."),
            (65, "🧠 Running analysis plugins..."),
            (88, "📋 Processing artifacts..."),
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
            await ctx.info("✅ Completed successfully")
        else:
            await ctx.error(f"❌ Failed: {result.get('error', 'unknown')}")
        return result
