# mcp_tools/dns_enum/fierce.py

from typing import Dict, Any
from fastmcp import Context

def register_fierce_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def fierce_scan(
        ctx: Context,
        domain: str,
        dns_server: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        DNS reconnaissance and subdomain brute-force using fierce.

        Workflow position: early DNS recon — discovers subdomains and
        identifies nearby IP ranges. Use after whois, before active scanning.

        Parameters:
        - domain: target domain (e.g. 'example.com')
        - dns_server: custom DNS resolver (e.g. '8.8.8.8') — omit to use system default
        - additional_args: extra fierce flags

        Prerequisites: none — passive/semi-passive DNS queries only.

        Output:
        - Discovered subdomains with IP addresses
        - Nearby IP ranges (non-contiguous ranges owned by same org)
        - Zone transfer results (if server misconfigured)

        Fierce vs dnsenum:
        - fierce    — faster, focuses on subdomain brute-force + nearby ranges
        - dnsenum   — slower, more thorough record enumeration + zone transfer

        Typical DNS enum sequence:
            1. whois_lookup(target='example.com')           — nameservers + registrar
            2. fierce_scan(domain='example.com')            — quick subdomain sweep
            3. dnsenum_scan(domain='example.com')           — full zone + record enum
            4. nmap_scan(target='<discovered IPs>')         — port scan live hosts
        """
        data = {
            "domain": domain,
            "dns_server": dns_server,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting fierce DNS recon: {domain}")
        result = hexstrike_client.safe_post("api/tools/fierce", data)
        if result.get("success"):
            await ctx.info(f"✅ fierce completed for {domain}")
        else:
            await ctx.error(f"❌ fierce failed for {domain}")
        return result
