# mcp_tools/smb_enum/nbtscan.py

from typing import Dict, Any
from fastmcp import Context

def register_nbtscan_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def nbtscan_netbios(
        ctx: Context,
        target: str,
        verbose: bool = False,
        timeout: int = 2,
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        NetBIOS name scanning and Windows host discovery using nbtscan.

        Workflow position: FIRST step in Windows/SMB enumeration.
        Use before enum4linux/smbmap to identify which hosts are Windows
        and discover domain names, workgroup membership, and DC candidates.

        Parameters:
        - target: IP address, hostname, or CIDR range (e.g. '192.168.1.0/24')
        - verbose: enable verbose output with all NetBIOS names (default False)
        - timeout: response timeout in seconds (default 2)
        - additional_args: extra nbtscan flags

        Prerequisites: UDP port 137 accessible (NetBIOS Name Service).
        Works on same subnet without routing, or cross-subnet if firewall allows.

        Output per host:
        - IP address
        - NetBIOS name (often the hostname)
        - Workgroup or domain name
        - MAC address
        - Host type flags (server, DC, file server, etc.)

        Domain controller identification: look for hosts with
        '<1C>' group name = domain controller candidates.

        Typical SMB enum sequence:
            1. nbtscan_netbios(target='192.168.1.0/24')      — discover Windows hosts
            2. enum4linux_scan(target='<DC IP>',
                               additional_args='-P')          — get password policy
            3. enum4linux_scan(target='<DC IP>')              — full enum
            4. smbmap_scan(target='<DC IP>')                  — share permissions
            5. netexec_scan(target='<DC IP>')                 — credential testing
        """
        data = {
            "target": target,
            "verbose": verbose,
            "timeout": timeout,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting nbtscan: {target}")
        result = hexstrike_client.safe_post("api/tools/nbtscan", data)
        if result.get("success"):
            await ctx.info(f"✅ nbtscan completed for {target}")
        else:
            await ctx.error(f"❌ nbtscan failed for {target}")
        return result
