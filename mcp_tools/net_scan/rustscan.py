# mcp_tools/net_scan/rustscan.py

from typing import Dict, Any
from fastmcp import Context

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
        result = hexstrike_client.safe_post("api/tools/rustscan", data)
        if result.get("success"):
            await ctx.info(f"✅ rustscan completed for {target}")
        else:
            await ctx.error(f"❌ rustscan failed for {target}")
        return result
