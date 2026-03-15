# mcp_tools/web_fuzz/ffuf.py

from typing import Dict, Any
from fastmcp import Context

def register_ffuf_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def ffuf_scan(
        ctx: Context,
        url: str,
        wordlist: str = "/usr/share/wordlists/dirb/common.txt",
        mode: str = "directory",
        match_codes: str = "200,204,301,302,307,401,403",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Fast web fuzzer using ffuf — most flexible fuzzer, supports FUZZ placeholder anywhere.

        Workflow position: content discovery and parameter fuzzing.
        The most versatile web_fuzz tool — use FUZZ in any part of the URL,
        headers, or POST body to fuzz any parameter.

        Parameters:
        - url: target URL with FUZZ placeholder
            directory mode: 'https://example.com/FUZZ'
            parameter mode: 'https://example.com/page?id=FUZZ'
            vhost mode:     'https://example.com' (ffuf handles Host header)
            header mode:    use -H in additional_args
        - wordlist: path to wordlist file
        - mode: fuzzing mode:
            'directory' — path/directory discovery (FUZZ in URL path)
            'vhost'     — virtual host discovery (fuzzes Host header)
            'parameter' — parameter value fuzzing (FUZZ in query string)
        - match_codes: comma-separated HTTP status codes to show (filter noise)
        - additional_args: extra ffuf flags
            '-t <n>'              — threads (default 40)
            '-fc 404,403'         — filter (hide) these status codes
            '-mc all'             — match all status codes
            '-fs <size>'          — filter by response size
            '-fw <words>'         — filter by word count
            '-e .php,.html,.txt'  — add extensions to each wordlist entry
            '-recursion'          — enable recursive fuzzing
            '-H "Cookie: x=y"'   — custom header

        Prerequisites: target accessible.

        ffuf vs other fuzzers:
        - ffuf        — most flexible (FUZZ anywhere), fast, best for param fuzzing
        - gobuster    — simpler, dir/dns/vhost modes, good for standard discovery
        - feroxbuster — recursive by default, auto-discovers new dirs
        - dirb        — simplest, good fallback

        Typical sequences:
            Directory: ffuf_scan(url='https://example.com/FUZZ',
                                  wordlist='/usr/share/seclists/.../raft-medium-directories.txt')
            Extensions: ffuf_scan(url='https://example.com/FUZZ',
                                   additional_args='-e .php,.bak,.txt')
            Parameters: ffuf_scan(url='https://example.com/page?FUZZ=test',
                                   wordlist='/usr/share/seclists/.../burp-parameter-names.txt')
        """
        data = {
            "url": url,
            "wordlist": wordlist,
            "mode": mode,
            "match_codes": match_codes,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting ffuf {mode} fuzzing: {url}")
        result = hexstrike_client.safe_post("api/tools/ffuf", data)
        if result.get("success"):
            await ctx.info(f"✅ ffuf completed for {url}")
        else:
            await ctx.error(f"❌ ffuf failed for {url}")
        return result
