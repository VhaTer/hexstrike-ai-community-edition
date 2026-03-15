# mcp_tools/web_fuzz/wfuzz.py

from typing import Dict, Any
from fastmcp import Context

def register_wfuzz_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def wfuzz_scan(
        ctx: Context,
        url: str,
        wordlist: str = "/usr/share/wordlists/dirb/common.txt",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Web application fuzzer using wfuzz — inject payloads via FUZZ placeholder.

        Workflow position: flexible fuzzing for any part of an HTTP request —
        directories, parameters, headers, cookies, or POST bodies.
        Particularly powerful with its filter system for response analysis.

        Parameters:
        - url: target URL with FUZZ placeholder where payloads are injected
               examples:
               'https://example.com/FUZZ'          — directory discovery
               'https://example.com/?id=FUZZ'      — parameter fuzzing
               'https://example.com/FUZZ.php'      — PHP file discovery
        - wordlist: path to wordlist file (payloads replace FUZZ)
        - additional_args: extra wfuzz flags — powerful filter system:
            '-c'                  — colorize output
            '--hc 404'            — hide responses with status code 404
            '--hc 404,403'        — hide multiple status codes
            '--hw <n>'            — hide responses with N words
            '--hl <n>'            — hide responses with N lines
            '--hs <n>'            — hide responses with size N bytes
            '--sc 200'            — show only status code 200
            '-t <n>'              — threads (default 10)
            '-z file,<wordlist>'  — explicit wordlist specification
            '-b "cookie=value"'   — set cookies
            '-H "header: value"'  — custom headers
            '-d "post=data"'      — POST body data
            '--follow'            — follow redirects

        Prerequisites: target accessible, FUZZ placeholder in URL.

        wfuzz vs ffuf:
        - wfuzz — powerful filter system (hc/hw/hl/hs), better for complex filtering
        - ffuf  — faster, simpler syntax, same FUZZ concept

        Typical sequences:
            Dir: wfuzz_scan(url='https://example.com/FUZZ',
                             additional_args='--hc 404')
            Param: wfuzz_scan(url='https://example.com/?id=FUZZ',
                               wordlist='/usr/share/seclists/Fuzzing/SQLi/quick-SQLi.txt',
                               additional_args='--hc 404 --hl 10')
        """
        data = {
            "url": url,
            "wordlist": wordlist,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting wfuzz: {url}")
        result = hexstrike_client.safe_post("api/tools/wfuzz", data)
        if result.get("success"):
            await ctx.info(f"✅ wfuzz completed for {url}")
        else:
            await ctx.error(f"❌ wfuzz failed for {url}")
        return result
