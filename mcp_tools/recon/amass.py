# mcp_tools/recon/amass.py

from typing import Dict, Any
from fastmcp import Context

def register_amass_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def amass_scan(
        ctx: Context,
        domain: str,
        mode: str = "enum",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Subdomain enumeration and attack surface mapping using Amass.

        Workflow position: active subdomain discovery after subfinder passive scan.
        More thorough than subfinder — uses DNS brute-force, scraping, and graph analysis.

        Parameters:
        - domain: target domain (e.g. 'example.com')
        - mode: Amass operation mode:
            'enum'  — subdomain enumeration (default, most useful)
            'intel' — collect open source intelligence (WHOIS, ASN, CIDRs)
            'viz'   — visualize the enumeration graph (requires previous enum)
        - additional_args: extra amass flags
            '-passive'        — passive only, no direct DNS queries (stealthy)
            '-active'         — enable active recon (zone transfers, cert grabbing)
            '-brute'          — DNS brute-force using built-in wordlist
            '-w <wordlist>'   — custom wordlist for brute-force
            '-o <file>'       — save output to file

        Prerequisites: none — passive mode has zero direct contact with target.

        Output: discovered subdomains with IP addresses, ASN info, and data sources.

        subfinder vs amass:
        - subfinder — faster, passive only, good for quick overview
        - amass     — slower, more thorough, active options available

        Typical subdomain enum sequence:
            1. subfinder_scan(domain='example.com')              — fast passive sweep
            2. amass_scan(domain='example.com',
                          additional_args='-passive')            — deeper passive
            3. amass_scan(domain='example.com',
                          additional_args='-active -brute')      — active enum
            4. httpx probe on all discovered subdomains
        """
        data = {
            "domain": domain,
            "mode": mode,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting amass {mode}: {domain}")
        result = hexstrike_client.safe_post("api/tools/amass", data)
        if result.get("success"):
            await ctx.info(f"✅ amass completed for {domain}")
        else:
            await ctx.error(f"❌ amass failed for {domain}")
        return result
