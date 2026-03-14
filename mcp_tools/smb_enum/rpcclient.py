# mcp_tools/smb_enum/rpcclient.py

from typing import Dict, Any
from fastmcp import Context

def register_rpcclient_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def rpcclient_enumeration(
        ctx: Context,
        target: str,
        username: str = "",
        password: str = "",
        domain: str = "",
        commands: str = "enumdomusers;enumdomgroups;querydominfo",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Low-level Windows RPC enumeration using rpcclient.

        Workflow position: targeted deep enumeration after enum4linux
        identifies interesting RPC endpoints. Best for SID lookups,
        specific user queries, and manual AD exploration.

        Parameters:
        - target: target IP address (e.g. '192.168.1.10')
        - username: username for authentication (empty = null session attempt)
        - password: password for authentication
        - domain: Windows domain name (optional)
        - commands: semicolon-separated RPC commands to execute
            Default runs: enumdomusers + enumdomgroups + querydominfo
            Common commands:
            'enumdomusers'          — list all domain users
            'enumdomgroups'         — list all domain groups
            'querydominfo'          — domain info (lockout policy, etc.)
            'queryuser <RID>'       — details for specific user RID
            'enumprinters'          — list printers (often misconfigured)
            'srvinfo'               — server OS info
            'lsaquery'              — LSA policy info
            'lookupnames <name>'    — resolve name to SID
        - additional_args: extra rpcclient flags

        Prerequisites: port 445 open. Null session may work on old systems.

        Output: raw RPC responses for the specified commands.
        Particularly useful for resolving specific RIDs to usernames
        after enum4linux gives you RID ranges.

        Typical use cases:
        - Null session user enumeration on older Windows targets
        - Resolving SIDs to usernames for targeted attacks
        - Querying specific AD attributes not shown by enum4linux
        """
        data = {
            "target": target,
            "username": username,
            "password": password,
            "domain": domain,
            "commands": commands,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting rpcclient: {target}")
        result = hexstrike_client.safe_post("api/tools/rpcclient", data)
        if result.get("success"):
            await ctx.info(f"✅ rpcclient completed for {target}")
        else:
            await ctx.error(f"❌ rpcclient failed for {target}")
        return result
