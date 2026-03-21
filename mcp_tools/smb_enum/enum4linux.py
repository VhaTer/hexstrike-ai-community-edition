# mcp_tools/smb_enum/enum4linux.py

from typing import Dict, Any
import asyncio
from fastmcp import Context
import mcp_core.smb_enum_direct as _smb_direct

def register_enum4linux_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def enum4linux_scan(
        ctx: Context,
        target: str,
        additional_args: str = "-a"
    ) -> Dict[str, Any]:
        """
        Full SMB/RPC enumeration using enum4linux.

        Workflow position: after nbtscan confirms Windows/SMB hosts,
        before netexec credential testing. Noisy but comprehensive.

        Parameters:
        - target: target IP address (e.g. '192.168.1.10')
        - additional_args: enum4linux flags (default: '-a' = full auto)
            '-a'  — all enumeration (default, recommended)
            '-U'  — users only
            '-G'  — groups only
            '-S'  — shares only
            '-P'  — password policy only
            '-o'  — OS information only
            '-i'  — printer information

        Prerequisites: SMB port 445 or 139 open on target.

        Output:
        - Users and RIDs (valuable for password spraying)
        - Groups and memberships
        - Share names and permissions
        - Password policy (lockout threshold — critical before spraying)
        - OS version and domain info

        ⚠️ enum4linux is noisy — generates many SMB/RPC connections.
        Always check password policy (-P) before credential spraying
        to avoid account lockouts.

        Typical SMB enum sequence:
            1. nbtscan_netbios(target='192.168.1.0/24')      — discover hosts
            2. enum4linux_scan(target='192.168.1.10',
                               additional_args='-P')          — check lockout policy
            3. enum4linux_scan(target='192.168.1.10')        — full enum
            4. smbmap_scan(target='192.168.1.10')            — share permissions
            5. netexec_scan(target='192.168.1.10')           — credential testing
        """
        data = {
            "target": target,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting enum4linux: {target}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _smb_direct.smb_enum_exec("enum4linux", data)
        )

        phases = [
            (20, "🔌 Connecting to SMB target..."),
            (45, "👥 Enumerating users and groups..."),
            (70, "📂 Enumerating shares..."),
            (88, "📋 Gathering policy info..."),
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

    @mcp.tool()
    async def enum4linux_ng_advanced(
        ctx: Context,
        target: str,
        username: str = "",
        password: str = "",
        domain: str = "",
        shares: bool = True,
        users: bool = True,
        groups: bool = True,
        policy: bool = True,
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Advanced SMB enumeration using enum4linux-ng (modern rewrite).

        Use this instead of enum4linux_scan when you have credentials,
        or when you need cleaner JSON-parseable output.
        enum4linux-ng is faster and supports more modern Windows targets.

        Parameters:
        - target: target IP address
        - username: username for authenticated enumeration (optional)
        - password: password for authenticated enumeration (optional)
        - domain: Windows domain name (optional)
        - shares: enumerate shares (default True)
        - users: enumerate users (default True)
        - groups: enumerate groups (default True)
        - policy: enumerate password policy (default True)
        - additional_args: extra enum4linux-ng flags

        Prerequisites: SMB open. Credentials optional but give more results.

        enum4linux vs enum4linux-ng:
        - enum4linux    — older, wider compatibility, noisier
        - enum4linux-ng — faster, cleaner output, better modern Windows support
        """
        data = {
            "target": target,
            "username": username,
            "password": password,
            "domain": domain,
            "shares": shares,
            "users": users,
            "groups": groups,
            "policy": policy,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting enum4linux-ng: {target}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _smb_direct.smb_enum_exec("enum4linux", data)
        )

        phases = [
            (20, "🔌 Connecting to SMB target..."),
            (45, "👥 Enumerating users and groups..."),
            (70, "📂 Enumerating shares..."),
            (88, "📋 Gathering policy info..."),
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
