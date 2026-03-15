# mcp_tools/password_cracking/patator.py

from typing import Dict, Any
from fastmcp import Context

def register_patator_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def patator_attack(
        ctx: Context,
        module: str,
        target: str,
        username: str = "",
        username_file: str = "",
        password: str = "",
        password_file: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Multi-purpose brute-forcer using Patator — flexible module system.

        Workflow position: credential testing, alternative to hydra/medusa.
        Patator's module system handles edge cases that hydra/medusa struggle with.

        Parameters:
        - module: Patator module to use (note: patator uses underscore format):
            'ssh_login'    — SSH
            'ftp_login'    — FTP
            'http_fuzz'    — HTTP fuzzing
            'smtp_login'   — SMTP
            'mysql_login'  — MySQL
            'mssql_login'  — MSSQL
            'snmp_login'   — SNMP
            'dns_forward'  — DNS forward lookup
        - target: target host or address
        - username: single username
        - username_file: file with usernames
        - password: single password
        - password_file: file with passwords
        - additional_args: extra patator arguments
            '-x ignore:code=530'  — ignore specific responses
            '-t <n>'              — threads

        Prerequisites: username AND password (or files) required.

        patator vs hydra vs medusa:
        - patator  — most flexible module system, good for custom protocols
        - hydra    — most protocol support, most commonly used
        - medusa   — fastest for SSH/FTP parallel attacks
        """
        data = {
            "module": module,
            "target": target,
            "username": username,
            "username_file": username_file,
            "password": password,
            "password_file": password_file,
            "additional_args": additional_args
        }
        await ctx.info(f"🔑 Starting patator: {target}:{module}")
        result = hexstrike_client.safe_post("api/tools/patator", data)
        if result.get("success"):
            await ctx.info(f"✅ patator completed for {target}")
        else:
            await ctx.error(f"❌ patator failed for {target}")
        return result
