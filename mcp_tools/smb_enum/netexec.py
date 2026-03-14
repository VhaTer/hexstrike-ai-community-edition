# mcp_tools/smb_enum/netexec.py

from typing import Dict, Any
from fastmcp import Context

def register_netexec_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def netexec_scan(
        ctx: Context,
        target: str,
        protocol: str = "smb",
        username: str = "",
        password: str = "",
        hash_value: str = "",
        module: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Network credential testing and lateral movement using NetExec (formerly CrackMapExec).

        Workflow position: FINAL step in SMB enum — after discovering users
        and checking password policy. Used for credential verification,
        password spraying, pass-the-hash, and post-exploitation modules.

        Parameters:
        - target: IP, hostname, or CIDR range (e.g. '192.168.1.0/24')
        - protocol: target protocol:
            'smb'   — Windows file sharing (port 445) — most common
            'ssh'   — SSH (port 22)
            'winrm' — Windows Remote Management (port 5985/5986)
            'ldap'  — Active Directory LDAP (port 389/636)
            'mssql' — Microsoft SQL Server (port 1433)
            'rdp'   — Remote Desktop (port 3389)
        - username: username for authentication
        - password: password for authentication
        - hash_value: NTLM hash for pass-the-hash attack (format: LM:NTLM)
        - module: netexec module to run after authentication:
            '--sam'      — dump SAM database (local accounts)
            '--lsa'      — dump LSA secrets
            '--ntds'     — dump NTDS.dit (domain accounts, requires DC)
            '--users'    — enumerate domain users (ldap)
            '--shares'   — list accessible shares
            '--sessions' — list active sessions
        - additional_args: extra netexec flags

        Prerequisites: target reachable on protocol port.
        ⚠️ Password spraying: always check lockout policy first with
        enum4linux_scan(additional_args='-P') — one wrong spray can
        lock out every account in the domain.

        Typical post-enum sequence:
            1. enum4linux_scan — get users + password policy
            2. netexec_scan(target='x.x.x.x', protocol='smb',
                            username='admin', password='Password1')  — test creds
            3. netexec_scan(target='x.x.x.x', protocol='smb',
                            username='admin', password='Password1',
                            module='--sam')                          — dump hashes
            4. netexec_scan(target='x.x.x.x', protocol='smb',
                            username='admin', hash_value='<hash>')   — PTH
        """
        data = {
            "target": target,
            "protocol": protocol,
            "username": username,
            "password": password,
            "hash": hash_value,
            "module": module,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting netexec {protocol}: {target}")
        result = hexstrike_client.safe_post("api/tools/netexec", data)
        if result.get("success"):
            await ctx.info(f"✅ netexec completed for {target}")
        else:
            await ctx.error(f"❌ netexec failed for {target}")
        return result
