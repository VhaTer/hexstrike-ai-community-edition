# mcp_tools/web_fuzz/dirsearch.py

from typing import Dict, Any
import asyncio
from fastmcp import Context
import mcp_core.web_fuzz_direct as _web_fuzz_direct

def register_dirsearch_tools(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def dirsearch_scan(
        ctx: Context,
        url: str,
        extensions: str = "php,html,js,txt,xml,json",
        wordlist: str = "/usr/share/wordlists/dirsearch/common.txt",
        threads: int = 30,
        recursive: bool = False,
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Advanced directory and file discovery using dirsearch.

        Workflow position: content discovery — particularly good at
        finding files by extension rather than just directories.
        Use when you know the target stack (PHP, Python, Java) and
        want extension-aware discovery.

        Parameters:
        - url: target URL (e.g. 'https://example.com')
        - extensions: comma-separated file extensions to append to each word
            PHP stack:   'php,php3,php5,bak,txt,xml'
            Python stack: 'py,pyc,txt,cfg,env'
            Java stack:  'jsp,jspx,do,action,xml'
            Generic:     'php,html,js,txt,xml,json,bak,zip'
        - wordlist: path to wordlist file
        - threads: concurrent threads (default 30)
        - recursive: scan discovered directories recursively (slower but thorough)
        - additional_args: extra dirsearch flags
            '-i 200,301,302' — include only these status codes
            '-x 404,403'     — exclude these status codes
            '--timeout <n>'  — request timeout in seconds
            '-H "header"'    — custom headers
            '--follow-redirects' — follow HTTP redirects

        Prerequisites: target accessible.

        dirsearch vs gobuster vs ffuf:
        - dirsearch   — best for extension-aware file discovery, recursive support
        - gobuster    — faster for pure directory discovery
        - ffuf        — most flexible (FUZZ anywhere)

        Typical sequence:
            1. gobuster_scan — fast directory sweep
            2. dirsearch_scan(extensions='php,bak,txt,zip') — file discovery
            3. Manual review of interesting findings
        """
        data = {
            "url": url,
            "extensions": extensions,
            "wordlist": wordlist,
            "threads": threads,
            "recursive": recursive,
            "additional_args": additional_args
        }
        await ctx.info(f"📁 Starting dirsearch: {url}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _web_fuzz_direct.web_fuzz_exec("dirsearch", data)
        )

        phases = [
            (20, "🔍 Loading wordlist..."),
            (45, "💥 Scanning directories..."),
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
