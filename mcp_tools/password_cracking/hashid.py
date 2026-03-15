# mcp_tools/password_cracking/hashid.py

from typing import Dict, Any
from fastmcp import Context

def register_hashid_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def hashid(
        ctx: Context,
        hash_value: str,
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Identify the type of an unknown hash using hashid.

        Workflow position: FIRST step — always identify hash type before cracking.
        Using the wrong hashcat mode wastes significant GPU time.

        Parameters:
        - hash_value: the hash string to identify
                      (e.g. '5f4dcc3b5aa765d61d8327deb882cf99')
        - additional_args: extra hashid flags
            '-m'  — show hashcat mode numbers (recommended)
            '-j'  — show john format names
            '-e'  — extended mode (more candidates)

        Prerequisites: none.

        Output: list of possible hash types with hashcat mode numbers.
        Use the mode number directly in hashcat_crack(hash_type='<mode>').

        Common hash → hashcat mode mapping:
            MD5          → 0       NTLM         → 1000
            SHA-1        → 100     Net-NTLMv2   → 5600
            SHA-256      → 1400    bcrypt        → 3200
            SHA-512      → 1700    WPA2          → 22000
            MySQL4.1+    → 300     Kerberos TGS  → 13100
            SHA-512crypt → 1800    Kerberos AS-REP → 18200

        Typical cracking sequence:
            1. hashid(hash_value='<hash>', additional_args='-m') — identify + mode
            2. hashcat_crack(hash_file='hashes.txt',
                             hash_type='<mode from step 1>',
                             wordlist='/usr/share/wordlists/rockyou.txt')
        """
        data = {"hash_value": hash_value, "additional_args": additional_args}
        await ctx.info(f"🔍 Identifying hash: {hash_value[:20]}...")
        result = hexstrike_client.safe_post("api/tools/password_cracking/hashid", data)
        if result.get("success"):
            await ctx.info("✅ Hash identification completed")
        else:
            await ctx.error("❌ Hash identification failed")
        return result
