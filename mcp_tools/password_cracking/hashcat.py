# mcp_tools/password_cracking/hashcat.py

from typing import Dict, Any
from fastmcp import Context

def register_hashcat_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def hashcat_crack(
        ctx: Context,
        hash_file: str,
        hash_type: str,
        attack_mode: str = "0",
        wordlist: str = "/usr/share/wordlists/rockyou.txt",
        mask: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        GPU-accelerated password cracking using hashcat.

        Workflow position: primary hash cracker — faster than john when GPU available.
        Use hashcat for NTLM, Net-NTLMv2, WPA2, and any high-volume cracking.

        Parameters:
        - hash_file: path to file containing hashes
        - hash_type: hashcat mode number (get from hashid with -m flag)
            Common modes: 0=MD5, 100=SHA1, 1000=NTLM, 1400=SHA256,
                          1700=SHA512, 3200=bcrypt, 5600=Net-NTLMv2,
                          13100=Kerberos-TGS, 18200=Kerberos-AS-REP, 22000=WPA2
        - attack_mode: hashcat attack mode:
            '0' — dictionary attack (wordlist)
            '1' — combination attack (two wordlists)
            '3' — mask/brute-force attack
            '6' — wordlist + mask hybrid
            '7' — mask + wordlist hybrid
        - wordlist: wordlist path for mode 0/1/6/7
        - mask: mask pattern for mode 3/6/7
            charset: ?l=lowercase ?u=uppercase ?d=digit ?s=special ?a=all
            example: '?u?l?l?l?l?d?d' = Capital + 4 lowercase + 2 digits
        - additional_args: extra hashcat flags
            '-r <rules>'      — apply rules file (e.g. best64.rule, rockyou-30000.rule)
            '--show'          — show cracked passwords from pot file
            '-O'              — optimized kernel (faster, some limitations)
            '--force'         — ignore warnings (use carefully)
            '-w 3'            — workload profile (1=low, 4=nightmare)
            '--status'        — enable status display

        Prerequisites: hashcat installed, GPU drivers for acceleration.

        Typical sequences:
            Dictionary: hashcat_crack(hash_file='ntlm.txt', hash_type='1000',
                                       wordlist='/usr/share/wordlists/rockyou.txt')
            Dictionary+rules: hashcat_crack(hash_file='ntlm.txt', hash_type='1000',
                                             additional_args='-r /usr/share/hashcat/rules/best64.rule')
            Mask: hashcat_crack(hash_file='ntlm.txt', hash_type='1000',
                                 attack_mode='3', mask='?u?l?l?l?l?d?d?s')
        """
        data = {
            "hash_file": hash_file,
            "hash_type": hash_type,
            "attack_mode": attack_mode,
            "wordlist": wordlist,
            "mask": mask,
            "additional_args": additional_args
        }
        await ctx.info(f"🔐 Starting hashcat: mode {attack_mode}, type {hash_type}")
        result = hexstrike_client.safe_post("api/tools/hashcat", data)
        if result.get("success"):
            await ctx.info("✅ hashcat completed")
        else:
            await ctx.error("❌ hashcat failed")
        return result
