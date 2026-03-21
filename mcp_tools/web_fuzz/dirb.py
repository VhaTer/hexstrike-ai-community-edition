# mcp_tools/web_fuzz/dirb.py

from typing import Dict, Any
import asyncio
from fastmcp import Context
import mcp_core.web_fuzz_direct as _web_fuzz_direct

def register_dirb_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def dirb_scan(
        ctx: Context,
        url: str,
        wordlist: str = "/usr/share/wordlists/dirb/common.txt",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Directory brute-force using dirb — classic, reliable, simple.

        Workflow position: content discovery after httpx confirms target is live.
        Dirb is the simplest dir fuzzer — good for quick checks or
        when other tools are unavailable.

        Parameters:
        - url: target URL (e.g. 'https://example.com')
        - wordlist: path to wordlist (default: dirb common.txt ~4600 entries)
            common alternatives:
            '/usr/share/wordlists/dirb/big.txt'           — ~20k entries
            '/usr/share/wordlists/dirbuster/medium.txt'   — ~220k entries
            '/usr/share/seclists/Discovery/Web-Content/raft-medium-directories.txt'
        - additional_args: extra dirb flags
            '-r'          — don't search recursively
            '-z'          — add millisecond delay between requests
            '-a <agent>'  — custom user-agent
            '-c <cookie>' — set cookie
            '-H <header>' — add custom header
            '-o <file>'   — save output to file

        Prerequisites: target accessible on HTTP/HTTPS.

        dirb vs gobuster vs ffuf vs feroxbuster:
        - dirb        — simple, no threads config, reliable fallback
        - gobuster    — faster, multi-threaded, DNS/vhost modes
        - ffuf        — most flexible, FUZZ placeholder, vhost/param fuzzing
        - feroxbuster — recursive by default, Rust-based, very fast

        Typical sequence:
            1. dirb_scan or gobuster_scan — quick common paths
            2. ffuf_scan with larger wordlist — thorough sweep
            3. Follow up on interesting findings
        """
        data = {
            "url": url,
            "wordlist": wordlist,
            "additional_args": additional_args
        }
        await ctx.info(f"📁 Starting dirb scan: {url}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _web_fuzz_direct.web_fuzz_exec("dirb", data)
        )

        phases = [
            (20, "🔍 Loading wordlist..."),
            (45, "💥 Directory brute-forcing..."),
            (70, "💥 Still scanning..."),
            (88, "📋 Processing results..."),
        ]
        for progress, message in phases:
            done, _ = await asyncio.wait([future], timeout=10)
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
