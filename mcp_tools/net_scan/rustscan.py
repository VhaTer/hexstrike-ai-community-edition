# mcp_tools/net_scan/rustscan.py

from typing import Dict, Any
import asyncio
from fastmcp import Context
import mcp_core.net_scan_direct as _net_scan_direct

def register_rustscan_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def rustscan_fast_scan(
        ctx: Context,
        target: str,
        ports: str = "",
        ulimit: int = 5000,
        batch_size: int = 4500,
        timeout: int = 1500,
        scripts: bool = False,
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Ultra-fast port scanning using rustscan — finds open ports then hands off to nmap.

        Workflow position: FIRST step for single-target port discovery.
        Faster than nmap for full port sweeps — use to identify open ports,
        then pass results to nmap_scan for service/version detection.

        Parameters:
        - target: IP address or hostname (e.g. '192.168.1.1')
        - ports: specific port range (e.g. '1-65535', '80,443,8080')
                 omit for default top ports scan
        - ulimit: file descriptor limit — increase for faster scanning
                  (default 5000 — set to 65535 if system allows)
        - batch_size: ports scanned per batch (default 4500)
        - timeout: connection timeout in milliseconds (default 1500)
                   increase for slow/unstable targets
        - scripts: run nmap scripts on discovered ports (slower, more detail)
        - additional_args: extra rustscan flags

        Prerequisites: none — designed as the entry point of the scan chain.

        Output: list of open ports — minimal info, fast delivery.
        Pass open ports directly to nmap_scan for detailed enumeration.

        Recommended sequence:
            1. rustscan_fast_scan(target='192.168.1.1', ports='1-65535')
            2. nmap_scan(target='192.168.1.1', scan_type='-sV -sC',
                         ports='<comma-separated open ports from step 1>')
        """
        data = {
            "target": target,
            "ports": ports,
            "ulimit": ulimit,
            "batch_size": batch_size,
            "timeout": timeout,
            "scripts": scripts,
            "additional_args": additional_args
        }
        await ctx.info(f"⚡ Starting rustscan: {target}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _net_scan_direct.net_scan_exec("rustscan", data)
        )

        phases = [
            (25, "⚡ Fast port discovery..."),
            (60, "🔍 Identifying open ports..."),
            (88, "📋 Passing to nmap for service detection..."),
        ]
        for progress, message in phases:
            done, _ = await asyncio.wait([future], timeout=8)
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
