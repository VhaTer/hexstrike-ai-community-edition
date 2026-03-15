# mcp_tools/password_cracking/hydra.py

from typing import Dict, Any
from fastmcp import Context

def register_hydra_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def hydra_attack(
        ctx: Context,
        target: str,
        service: str,
        username: str = "",
        username_file: str = "",
        password: str = "",
        password_file: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Network login brute-force using Hydra.

        Workflow position: credential testing after user enumeration.
        Use enum4linux/theharvester to get usernames first, then hydra to spray.

        Parameters:
        - target: target IP or hostname (e.g. '192.168.1.10')
        - service: protocol to attack:
            'ssh'              — SSH (port 22)
            'ftp'              — FTP (port 21)
            'http-get'         — HTTP Basic Auth GET
            'http-post-form'   — HTTP form POST (see additional_args)
            'rdp'              — Remote Desktop
            'smb'              — SMB (use netexec for SMB — better)
            'mysql'            — MySQL
            'mssql'            — MSSQL
            'pop3'             — POP3
            'imap'             — IMAP
            'snmp'             — SNMP
        - username: single username to test
        - username_file: file with usernames (one per line)
        - password: single password to test
        - password_file: file with passwords (one per line)
        - additional_args: extra hydra flags
            '-t <n>'      — parallel tasks (default 16, reduce if target blocks)
            '-w <sec>'    — wait between retries
            '-f'          — stop after first valid credential found
            '-V'          — verbose (show each attempt)
            '-s <port>'   — custom port
            For http-post-form: '/login:user=^USER^&pass=^PASS^:Invalid credentials'

        Prerequisites: must provide username/username_file AND password/password_file.
        ⚠️ Check account lockout policy before spraying (enum4linux -P).

        hydra vs medusa vs netexec:
        - hydra    — widest protocol support, most common choice
        - medusa   — faster on some protocols (SSH, FTP), parallel design
        - netexec  — preferred for SMB/WinRM/LDAP (Windows environments)

        Typical sequence:
            1. enum4linux_scan(additional_args='-U -P') — get users + lockout policy
            2. hydra_attack(target='x.x.x.x', service='ssh',
                             username_file='users.txt',
                             password_file='/usr/share/wordlists/rockyou.txt',
                             additional_args='-t 4 -f')
        """
        data = {
            "target": target,
            "service": service,
            "username": username,
            "username_file": username_file,
            "password": password,
            "password_file": password_file,
            "additional_args": additional_args
        }
        await ctx.info(f"🔑 Starting hydra: {target}:{service}")
        result = hexstrike_client.safe_post("api/tools/hydra", data)
        if result.get("success"):
            await ctx.info(f"✅ hydra completed for {target}")
        else:
            await ctx.error(f"❌ hydra failed for {target}")
        return result
