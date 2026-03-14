# mcp_tools/net_scan/arp_scan.py

from typing import Dict, Any
from fastmcp import Context

def register_arp_scan_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def arp_scan_discovery(
        ctx: Context,
        target: str = "",
        interface: str = "",
        local_network: bool = False,
        timeout: int = 500,
        retry: int = 3,
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Discover live hosts on a local network using ARP scanning.

        Workflow position: FIRST step for local network recon.
        Use before nmap/rustscan to identify which hosts are actually up
        on the same subnet — faster and more reliable than ICMP ping on LAN.

        Parameters:
        - target: IP range to scan (e.g. '192.168.1.0/24') — omit if local_network=True
        - interface: network interface to use (e.g. 'eth0', 'wlan0')
        - local_network: scan the entire local subnet automatically
        - timeout: ARP reply timeout in milliseconds (default 500)
        - retry: number of ARP request retries per host (default 3)
        - additional_args: extra arp-scan flags

        Prerequisites: must be on the same network segment as targets.
        ARP does not cross router boundaries — LAN only.

        Output: list of live hosts with IP address and MAC address.
        MAC vendor prefix reveals device type (useful for IoT/embedded targets).

        Typical sequence:
            1. arp_scan_discovery(local_network=True)          — discover live hosts
            2. rustscan_fast_scan(target='192.168.1.x', ...)   — port scan each host
            3. nmap_scan(target='192.168.1.x', ...)            — service enumeration
        """
        data = {
            "target": target,
            "interface": interface,
            "local_network": local_network,
            "timeout": timeout,
            "retry": retry,
            "additional_args": additional_args
        }
        scan_target = target if target else "local network"
        await ctx.info(f"🔍 Starting arp-scan: {scan_target}")
        result = hexstrike_client.safe_post("api/tools/arp-scan", data)
        if result.get("success"):
            await ctx.info("✅ arp-scan completed")
        else:
            await ctx.error("❌ arp-scan failed")
        return result
