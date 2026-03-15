# mcp_tools/password_cracking/john.py

from typing import Dict, Any
from fastmcp import Context

def register_john_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def john_crack(
        ctx: Context,
        hash_file: str,
        wordlist: str = "/usr/share/wordlists/rockyou.txt",
        format_type: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Password cracking using John the Ripper — best for format auto-detection.

        Workflow position: hash cracking after hashid identifies the type.
        John is best for automatic format detection and exotic hash types.
        Use hashcat for speed when GPU is available.

        Parameters:
        - hash_file: path to file containing hashes
        - wordlist: path to wordlist (default: rockyou.txt)
        - format_type: force specific hash format (e.g. 'NT', 'sha512crypt', 'bcrypt')
                       leave empty for auto-detection
                       run `john --list=formats` to see all supported formats
        - additional_args: extra john flags
            '--rules'              — enable word mangling rules
            '--rules=best64'       — use best64 ruleset
            '--incremental'        — brute force mode
            '--show'               — show cracked passwords
            '--pot=<file>'         — custom pot file
            '--session=<name>'     — save/restore session

        Prerequisites: hash file must contain hashes in a format John recognizes.

        john vs hashcat:
        - john     — better format auto-detection, many exotic formats, CPU-based
        - hashcat  — much faster (GPU), rule-based attacks, mask attacks

        Typical sequence:
            1. hashid(hash_value='<hash>', additional_args='-m')   — identify type
            2. john_crack(hash_file='hashes.txt',
                          wordlist='/usr/share/wordlists/rockyou.txt')
            3. john_crack(additional_args='--show')                 — show results
        """
        data = {
            "hash_file": hash_file,
            "wordlist": wordlist,
            "format": format_type,
            "additional_args": additional_args
        }
        await ctx.info(f"🔐 Starting john: {hash_file}")
        result = hexstrike_client.safe_post("api/tools/john", data)
        if result.get("success"):
            await ctx.info("✅ john completed")
        else:
            await ctx.error("❌ john failed")
        return result
