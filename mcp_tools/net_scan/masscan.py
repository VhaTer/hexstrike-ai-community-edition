# mcp_tools/net_scan/masscan.py

from typing import Dict, Any
import asyncio
from fastmcp import Context
import mcp_core.net_scan_direct as _net_scan_direct

def register_masscan_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def masscan_high_speed(
        ctx: Context,
        target: str,
        ports: str = "1-65535",
        rate: int = 1000,
        interface: str = "",
        router_mac: str = "",
        source_ip: str = "",
        banners: bool = False,
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        High-speed Internet-scale port scanning using masscan.

        Workflow position: FIRST step for large CIDR ranges or full port sweeps.
        Use instead of nmap for initial discovery when scanning many hosts
        or all 65535 ports — then hand off results to nmap for service detection.

        Parameters:
        - target: IP address, hostname, or CIDR range (e.g. '10.0.0.0/8')
        - ports: port range to scan (default: 1-65535 full sweep)
                 examples: '80,443', '1-1024', '22,80,443,8080,8443'
        - rate: packets per second (default 1000 — increase carefully)
                warning: high rates (>10000) may trigger IDS/IPS or drop packets
        - interface: network interface to use
        - router_mac: router MAC address (required for some network setups)
        - source_ip: custom source IP address
        - banners: enable basic banner grabbing (slower)
        - additional_args: extra masscan flags

        Prerequisites: requires root/sudo for raw packet sending.

        Output: list of open ports per host — no service info (use nmap after).

        Recommended sequence for large ranges:
            1. masscan_high_speed(target='10.0.0.0/24', ports='1-65535', rate=1000)
            2. nmap_scan(target='<live hosts>', ports='<open ports>',
                         scan_type='-sV -sC')
        """
        data = {
            "target": target,
            "ports": ports,
            "rate": rate,
            "interface": interface,
            "router_mac": router_mac,
            "source_ip": source_ip,
            "banners": banners,
            "additional_args": additional_args
        }
        await ctx.info(f"🚀 Starting masscan: {target} at {rate} pps")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _net_scan_direct.net_scan_exec("masscan", data)
        )

        phases = [
            (20, "📡 Starting high-speed scan..."),
            (50, "📡 Scanning port ranges..."),
            (80, "📋 Processing discovered ports..."),
        ]
        for progress, message in phases:
            done, _ = await asyncio.wait([future], timeout=10)
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
