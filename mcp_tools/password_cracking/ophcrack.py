# mcp_tools/password_cracking/ophcrack.py

from typing import Dict, Any
import asyncio
from fastmcp import Context

def register_ophcrack_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def ophcrack_crack(
        ctx: Context,
        hash_file: str,
        tables_dir: str = "",
        tables: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Windows NTLM hash cracking using Ophcrack rainbow tables.

        Workflow position: NTLM hash cracking when wordlist/mask attacks fail.
        Rainbow tables trade disk space for instant cracking — no GPU needed.
        Best for simple/common Windows passwords when hashcat wordlist fails.

        Parameters:
        - hash_file: path to hash file in pwdump or session format
                     (format: username:uid:LM_hash:NTLM_hash:::)
        - tables_dir: path to directory containing rainbow tables
                      (download from https://ophcrack.sourceforge.io/tables.php)
        - tables: table set name to use:
            'VistaFree'   — free Vista/7 tables (368MB, ~50% success)
            'WinXP Free'  — free XP tables (388MB, most XP passwords)
            'Vista proba' — extended Vista tables (8GB, ~90% success)
        - additional_args: extra ophcrack flags
            '-v'          — verbose
            '-n <n>'      — number of threads

        Prerequisites: rainbow tables must be downloaded and available locally.
        Tables are large (hundreds of MB to GB) — download once, reuse.

        ophcrack vs hashcat:
        - ophcrack  — no GPU needed, instant if hash is in tables, but limited coverage
        - hashcat   — much better coverage with good wordlist + rules, GPU accelerated

        Typical sequence:
            1. hashcat_crack with rockyou.txt + best64 rules — try wordlist first
            2. ophcrack_crack if wordlist fails — rainbow table fallback
        """
        data = {
            "hash_file": hash_file,
            "tables_dir": tables_dir,
            "tables": tables,
            "additional_args": additional_args
        }
        await ctx.info(f"🔑 Starting ophcrack: {hash_file}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: hexstrike_client.safe_post("api/tools/password-cracking/ophcrack", data)
        )

        phases = [
            (20, "📂 Loading rainbow tables..."),
            (50, "💥 Cracking NTLM hashes..."),
            (80, "📋 Finalizing results..."),
        ]
        for progress, message in phases:
            done, _ = await asyncio.wait([future], timeout=15)
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
