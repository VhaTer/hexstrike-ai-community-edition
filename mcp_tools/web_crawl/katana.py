# mcp_tools/web_crawl/katana.py

from typing import Dict, Any
import asyncio
from fastmcp import Context
import mcp_core.web_recon_direct as _web_recon_direct

def register_katana_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def katana_crawl(
        ctx: Context,
        url: str,
        depth: int = 3,
        js_crawl: bool = True,
        form_extraction: bool = True,
        output_format: str = "json",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Next-generation web crawler using Katana — discovers endpoints, forms, and JS links.

        Workflow position: active crawling after httpx confirms a target is live.
        Use before web_fuzz tools to discover real endpoints rather than brute-forcing.
        Katana's JS rendering finds endpoints hidden in JavaScript that static crawlers miss.

        Parameters:
        - url: target URL to crawl (e.g. 'https://example.com')
        - depth: crawl depth — how many links deep to follow (default 3)
                 increase for large sites, decrease for focused crawls
        - js_crawl: enable headless JS rendering to discover dynamic endpoints (default True)
                    slower but finds significantly more endpoints on modern SPAs
        - form_extraction: extract and report HTML forms with their fields (default True)
                           forms are prime targets for injection testing
        - output_format: 'json' for structured output, 'txt' for plain URLs
        - additional_args: extra katana flags
            '-c <n>'         — concurrency (default 10)
            '-timeout <sec>' — timeout per request
            '-H "header"'    — custom headers (e.g. auth cookies)
            '-kf all'        — enable all known files discovery
            '-fx'            — match/filter extensions

        Prerequisites: target must be accessible. JS crawl requires headless browser.

        Output: discovered URLs, endpoints, forms, and JS-rendered paths.
        Feed into gobuster/ffuf for path fuzzing, or sqlmap/dalfox for injection testing.

        katana vs hakrawler:
        - katana    — JS rendering, form extraction, modern SPA support, slower
        - hakrawler — fast, lightweight, no JS, good for traditional sites

        Typical web recon sequence:
            1. httpx_probe(target='example.com')             — confirm live + tech
            2. katana_crawl(url='https://example.com')       — active crawl
            3. gau_discovery(domain='example.com')           — historical URLs
            4. arjun_parameter_discovery on interesting endpoints
            5. nuclei/dalfox/sqlmap on discovered endpoints
        """
        data = {
            "url": url,
            "depth": depth,
            "js_crawl": js_crawl,
            "form_extraction": form_extraction,
            "output_format": output_format,
            "additional_args": additional_args
        }
        await ctx.info(f"⚔️ Starting katana crawl: {url}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _web_recon_direct.web_recon_exec("katana", data)
        )

        phases = [
            (20, "🌐 Fetching initial page..."),
            (40, "🔍 Crawling discovered links..."),
            (65, "⚙️ Rendering JavaScript endpoints..."),
            (85, "📋 Extracting forms and parameters..."),
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
