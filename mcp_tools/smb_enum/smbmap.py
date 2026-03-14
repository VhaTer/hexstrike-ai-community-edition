# mcp_tools/smb_enum/smbmap.py

from typing import Dict, Any
from fastmcp import Context

def register_smbmap_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def smbmap_scan(
        ctx: Context,
        target: str,
        username: str = "",
        password: str = "",
        domain: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        SMB share enumeration and permission mapping using smbmap.

        Workflow position: after nbtscan/enum4linux, before netexec.
        Best tool for quickly seeing share permissions and browsing content.

        Parameters:
        - target: target IP address (e.g. '192.168.1.10')
        - username: username (leave empty for null session attempt)
        - password: password (leave empty for null session)
        - domain: Windows domain name (optional)
        - additional_args: extra smbmap flags
            '-R <share>'          — recursive listing of share contents
            '-A <pattern>'        — download files matching pattern
            '-x "<command>"'      — execute command (if write access)
            '--download <path>'   — download specific file
            '-H <hash>'           — pass-the-hash (NTLM hash)

        Prerequisites: SMB port 445 open. Credentials optional.
        Null session (no creds) may reveal share names on older systems.

        Output: shares with READ/WRITE/NO ACCESS permissions per share.
        WRITE access = potential file upload for exploitation.

        Typical sequence:
            1. smbmap_scan(target='x.x.x.x')                     — null session
            2. smbmap_scan(target='x.x.x.x', username='user',
                           password='pass')                        — authenticated
            3. smbmap_scan(target='x.x.x.x', username='user',
                           password='pass',
                           additional_args='-R SHARENAME')         — list contents
        """
        data = {
            "target": target,
            "username": username,
            "password": password,
            "domain": domain,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting smbmap: {target}")
        result = hexstrike_client.safe_post("api/tools/smbmap", data)
        if result.get("success"):
            await ctx.info(f"✅ smbmap completed for {target}")
        else:
            await ctx.error(f"❌ smbmap failed for {target}")
        return result
