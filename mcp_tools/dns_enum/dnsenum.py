# mcp_tools/dns_enum/dnsenum.py

from typing import Dict, Any
from fastmcp import Context

def register_dnsenum_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def dnsenum_scan(
        ctx: Context,
        domain: str,
        dns_server: str = "",
        wordlist: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Comprehensive DNS enumeration using dnsenum — records, zone transfers, brute-force.

        Workflow position: DNS recon after fierce_scan for deeper enumeration.
        Most valuable for zone transfer attempts and full record mapping.

        Parameters:
        - domain: target domain (e.g. 'example.com')
        - dns_server: custom DNS resolver (e.g. '8.8.8.8') — omit for system default
        - wordlist: path to subdomain wordlist for brute-force
                    (e.g. '/usr/share/wordlists/dnsmap.txt')
                    omit to use dnsenum's built-in list
        - additional_args: extra dnsenum flags

        Prerequisites: none — DNS queries only, no direct host contact.

        Output:
        - A, MX, NS, CNAME records
        - Zone transfer results (high value if server misconfigured)
        - Brute-forced subdomains with IP resolution
        - Reverse lookup on discovered IP ranges

        Zone transfers are rare but expose the full DNS zone — always attempt.
        If successful, skip brute-force — you already have everything.

        Typical DNS enum sequence:
            1. whois_lookup(target='example.com')           — nameservers
            2. fierce_scan(domain='example.com')            — fast subdomain sweep
            3. dnsenum_scan(domain='example.com',
                            wordlist='/usr/share/wordlists/dnsmap.txt')
            4. nmap_scan(target='<discovered IPs>')         — port scan
        """
        data = {
            "domain": domain,
            "dns_server": dns_server,
            "wordlist": wordlist,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting dnsenum: {domain}")
        result = hexstrike_client.safe_post("api/tools/dnsenum", data)
        if result.get("success"):
            await ctx.info(f"✅ dnsenum completed for {domain}")
        else:
            await ctx.error(f"❌ dnsenum failed for {domain}")
        return result
