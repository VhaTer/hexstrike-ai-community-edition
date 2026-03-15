# mcp_tools/web_fuzz/feroxbuster.py

from typing import Dict, Any
from fastmcp import Context

def register_feroxbuster_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def feroxbuster_scan(
        ctx: Context,
        url: str,
        wordlist: str = "/usr/share/wordlists/dirb/common.txt",
        threads: int = 10,
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Recursive content discovery using feroxbuster — finds directories then digs deeper automatically.

        Workflow position: thorough content discovery when recursive scanning is needed.
        Feroxbuster's auto-recursion makes it ideal for complex web apps with
        deep directory structures — it finds /admin, then scans /admin/*, etc.

        Parameters:
        - url: target URL (e.g. 'https://example.com')
        - wordlist: path to wordlist file
        - threads: concurrent threads (default 10 — increase to 50 on stable targets)
        - additional_args: extra feroxbuster flags
            '-d <n>'              — max recursion depth (default unlimited)
            '-x php,txt,bak'      — file extensions to append
            '-s 200,301,302'      — filter by status codes
            '--filter-size <n>'   — hide responses of this size
            '--timeout <n>'       — request timeout
            '-H "Cookie: x=y"'   — custom headers
            '-k'                  — disable TLS verification
            '--dont-scan <path>'  — skip specific paths
            '-o <file>'           — save output to file

        Prerequisites: target accessible.

        feroxbuster vs gobuster vs ffuf:
        - feroxbuster — best for deep recursive discovery, auto-finds new dirs
        - gobuster    — faster for flat directory scans, multi-mode
        - ffuf        — most flexible (FUZZ anywhere), best for custom fuzzing

        Typical sequences:
            Quick check: gobuster_scan(url='https://example.com')
            Deep scan:   feroxbuster_scan(url='https://example.com',
                                           threads=50,
                                           additional_args='-x php,bak,txt -d 5')
        """
        data = {
            "url": url,
            "wordlist": wordlist,
            "threads": threads,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting feroxbuster: {url}")
        result = hexstrike_client.safe_post("api/tools/feroxbuster", data)
        if result.get("success"):
            await ctx.info(f"✅ feroxbuster completed for {url}")
        else:
            await ctx.error(f"❌ feroxbuster failed for {url}")
        return result
