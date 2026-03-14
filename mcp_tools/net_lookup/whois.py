# mcp_tools/net_lookup/whois.py

from typing import Dict, Any
from fastmcp import Context

def register_whois(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def whois_lookup(
        ctx: Context,
        target: str
    ) -> Dict[str, Any]:
        """
        Perform a WHOIS lookup for a domain or IP address.

        Workflow position: early recon — gather registration info before
        active scanning. Use alongside dns_enum for full domain intelligence.

        Parameters:
        - target: domain name or IP address (e.g. 'example.com', '1.2.3.4')

        Prerequisites: none — passive lookup, no direct contact with target.

        Output:
        - For domains: registrar, registration/expiry dates, nameservers,
                       registrant contact (if not privacy-protected), ASN
        - For IPs: ASN, network range, organization, country, abuse contact

        Useful for:
        - Identifying hosting provider and infrastructure owner
        - Finding nameservers for dns_enum zone transfer attempts
        - Discovering related IP ranges owned by the same organization

        Typical sequence:
            1. whois_lookup(target='example.com')        — registrar + nameservers
            2. fierce_scan(target='example.com')         — DNS brute-force
            3. dnsenum_scan(target='example.com')        — zone transfer attempt
        """
        await ctx.info(f"🔎 WHOIS lookup: {target}")
        try:
            result = hexstrike_client.safe_post("api/tools/whois", {"target": target})
            if result.get("success"):
                await ctx.info(f"✅ WHOIS completed for {target}")
            else:
                await ctx.error(f"❌ WHOIS failed for {target}")
            return result
        except Exception as e:
            await ctx.error(f"❌ WHOIS lookup failed: {e}")
            return {"error": str(e), "success": False}
