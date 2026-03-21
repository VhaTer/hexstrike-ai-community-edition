# mcp_tools/smb_enum/nbtscan.py

from typing import Dict, Any
import asyncio
from fastmcp import Context
import mcp_core.smb_enum_direct as _smb_direct

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

        Output per host:
        - IP address, NetBIOS name, workgroup/domain, MAC address, host type flags

        Typical SMB enum sequence:
            1. nbtscan_netbios(target='192.168.1.0/24')      — discover Windows hosts
            2. enum4linux_scan(target='<DC IP>')              — full enum
            3. smbmap_scan(target='<DC IP>')                  — share permissions
            4. netexec_scan(target='<DC IP>')                 — credential testing
        """
        data = {
            "target": target,
            "verbose": verbose,
            "timeout": timeout,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting nbtscan: {target}")
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _smb_direct.smb_enum_exec("nbtscan", data)
        )
        if result.get("success"):
            await ctx.info(f"✅ nbtscan completed for {target}")
        else:
            await ctx.error(f"❌ nbtscan failed for {target}")
        return result
