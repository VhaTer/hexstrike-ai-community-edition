from typing import Dict, Any
import asyncio
import mcp_core.active_directory_direct as _ad_direct
from fastmcp import Context


def register_ldapdomaindump_tool(mcp, hexstrike_client, logger):
    @mcp.tool()
    async def ldapdomaindump(
        ctx: Context,
        hostname: str,
        username: str = "",
        password: str = "",
        authtype: str = "NTLM",
    ) -> Dict[str, Any]:
        """
        Run ldapdomaindump against an Active Directory domain.

        Args:
            hostname: Target hostname or IP of the AD domain controller
            username: Optional username for authentication
            password: Optional password for authentication
            authtype: Authentication type (default: NTLM)

        Returns:
            ldapdomaindump output
        """
        data = {
            "hostname": hostname,
            "username": username,
            "password": password,
            "authtype": authtype,
        }

        await ctx.info(f"🔍 Starting ldapdomaindump against {hostname}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _ad_direct.ad_exec("ldapdomaindump", data)
        )

        phases = [(50, "Dumping LDAP data...")]
        tick = 20
        for progress, message in phases:
            done, _ = await asyncio.wait([future], timeout=tick)
            if done:
                break
            await ctx.report_progress(progress, 100)
            await ctx.info(message)

        result = await future
        await ctx.report_progress(100, 100)

        if result.get("success"):
            await ctx.info(f"✅ ldapdomaindump completed for {hostname}")
        else:
            await ctx.error(f"❌ ldapdomaindump failed: {result.get('error')}")
        return result
