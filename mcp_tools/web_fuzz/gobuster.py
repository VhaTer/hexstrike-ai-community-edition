# mcp_tools/web_fuzz/gobuster.py

from typing import Dict, Any
from fastmcp import Context

def register_gobuster(mcp, hexstrike_client, logger=None, HexStrikeColors=None):

    @mcp.tool()
    async def gobuster_scan(
        ctx: Context,
        url: str,
        mode: str = "dir",
        wordlist: str = "/usr/share/wordlists/dirb/common.txt",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Directory, DNS, and vhost discovery using gobuster.

        Workflow position: primary content discovery tool — fast, multi-threaded,
        supports directory, DNS subdomain, and virtual host fuzzing modes.

        Parameters:
        - url: target URL for dir/fuzz/vhost modes, or domain for dns mode
               dir mode:   'https://example.com'
               dns mode:   'example.com'
               vhost mode: 'https://example.com'
        - mode: gobuster mode:
            'dir'   — directory/file brute-force (most common)
            'dns'   — subdomain enumeration via DNS
            'fuzz'  — generic fuzzing with FUZZ placeholder
            'vhost' — virtual host discovery
        - wordlist: path to wordlist file
            common choices:
            '/usr/share/wordlists/dirb/common.txt'                    — ~4600 entries, fast
            '/usr/share/wordlists/dirbuster/medium.txt'               — ~220k entries
            '/usr/share/seclists/Discovery/Web-Content/raft-medium-directories.txt'
            '/usr/share/seclists/Discovery/DNS/subdomains-top1million-5000.txt' — for dns mode
        - additional_args: extra gobuster flags
            '-t <n>'       — threads (default 10, increase to 50 on stable networks)
            '-x php,txt'   — file extensions to append
            '-s 200,301'   — match only these status codes
            '-b 404,403'   — blacklist (hide) these status codes
            '-k'           — skip SSL verification
            '-c <cookie>'  — set cookies
            '-H <header>'  — custom header
            '-o <file>'    — output file

        Prerequisites: target accessible.

        Output includes recovery info if server-side error recovery was applied.

        gobuster vs ffuf vs feroxbuster:
        - gobuster    — fast, multi-mode (dir/dns/vhost), simple syntax
        - ffuf        — FUZZ anywhere, most flexible for custom fuzzing
        - feroxbuster — auto-recursive, Rust speed, best for deep crawls
        """
        data: Dict[str, Any] = {
            "url": url,
            "mode": mode,
            "wordlist": wordlist,
            "additional_args": additional_args,
            "use_recovery": True
        }
        await ctx.info(f"📁 Starting gobuster {mode}: {url}")
        result = hexstrike_client.safe_post("api/tools/gobuster", data)
        if result.get("success"):
            await ctx.info(f"✅ gobuster completed for {url}")
            if result.get("recovery_info", {}).get("recovery_applied"):
                attempts = result["recovery_info"].get("attempts_made", 1)
                await ctx.info(f"⚠️ Recovery applied: {attempts} attempt(s)")
        else:
            await ctx.error(f"❌ gobuster failed for {url}")
            if result.get("alternative_tool_suggested"):
                await ctx.info(f"💡 Alternative suggested: {result['alternative_tool_suggested']}")
        return result
