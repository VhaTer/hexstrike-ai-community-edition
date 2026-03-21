# mcp_tools/binary_debug/gdb.py

from typing import Dict, Any
import asyncio
from fastmcp import Context
import mcp_core.misc_direct as _misc_direct

def register_gdb_tools(mcp, hexstrike_client, logger):
    
    @mcp.tool()
    async def gdb_analyze(ctx: Context, binary: str, commands: str = "", script_file: str = "", additional_args: str = "") -> Dict[str, Any]:
        """
        Execute GDB for binary analysis and debugging with enhanced logging.

        Args:
            binary: Path to the binary file
            commands: GDB commands to execute
            script_file: Path to GDB script file
            additional_args: Additional GDB arguments

        Returns:
            Binary analysis results
        """
        data = {
            "binary": binary,
            "commands": commands,
            "script_file": script_file,
            "additional_args": additional_args
        }
        await ctx.info(f"🔧 Starting GDB analysis: {binary}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _misc_direct.misc_exec("gdb", data)
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

    @mcp.tool()
    async def gdb_peda_debug(ctx: Context, binary: str = "", commands: str = "", attach_pid: int = 0,
                      core_file: str = "", additional_args: str = "") -> Dict[str, Any]:
        """
        Execute GDB with PEDA for enhanced debugging and exploitation.

        Args:
            binary: Binary to debug
            commands: GDB commands to execute
            attach_pid: Process ID to attach to
            core_file: Core dump file to analyze
            additional_args: Additional GDB arguments

        Returns:
            Enhanced debugging results with PEDA
        """
        data = {
            "binary": binary,
            "commands": commands,
            "attach_pid": attach_pid,
            "core_file": core_file,
            "additional_args": additional_args
        }
        await ctx.info(f"🔧 Starting GDB-PEDA analysis: {binary or f'PID {attach_pid}' or core_file}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _misc_direct.misc_exec("gdb", data)
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