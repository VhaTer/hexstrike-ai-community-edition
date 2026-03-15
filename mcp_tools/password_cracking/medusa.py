# mcp_tools/password_cracking/medusa.py

from typing import Dict, Any
from fastmcp import Context

def register_medusa_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def medusa_attack(
        ctx: Context,
        target: str,
        module: str,
        username: str = "",
        username_file: str = "",
        password: str = "",
        password_file: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Network login brute-force using Medusa — parallel, modular design.

        Workflow position: credential testing, alternative to hydra.
        Medusa is faster than hydra on SSH and FTP due to its parallel design.

        Parameters:
        - target: target hostname or IP (e.g. '192.168.1.10')
        - module: Medusa service module:
            'ssh'     — SSH (port 22)
            'ftp'     — FTP (port 21)
            'http'    — HTTP Basic Auth
            'smb'     — SMB (use netexec for better SMB support)
            'mysql'   — MySQL
            'mssql'   — MSSQL
            'rdp'     — RDP
            'telnet'  — Telnet
            'pop3'    — POP3
            'imap'    — IMAP
        - username: single username to test
        - username_file: file with usernames (one per line)
        - password: single password to test
        - password_file: file with passwords (one per line)
        - additional_args: extra medusa flags
            '-t <n>'  — parallel login attempts per host (default 16)
            '-T <n>'  — parallel hosts (default 1)
            '-f'      — stop after first valid credential
            '-v <n>'  — verbosity level (0-6)
            '-s'      — enable SSL

        Prerequisites: username AND password (or files) required.
        ⚠️ Check account lockout policy first.

        medusa vs hydra:
        - medusa   — faster for SSH/FTP, cleaner parallel design
        - hydra    — wider protocol support, http-post-form support
        """
        data = {
            "target": target,
            "module": module,
            "username": username,
            "username_file": username_file,
            "password": password,
            "password_file": password_file,
            "additional_args": additional_args
        }
        await ctx.info(f"🔑 Starting medusa: {target}:{module}")
        result = hexstrike_client.safe_post("api/tools/medusa", data)
        if result.get("success"):
            await ctx.info(f"✅ medusa completed for {target}")
        else:
            await ctx.error(f"❌ medusa failed for {target}")
        return result
