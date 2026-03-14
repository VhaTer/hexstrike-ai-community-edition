# mcp_tools/recon/theharvester.py

from typing import Dict, Any
from fastmcp import Context

def register_theharvester_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def theharvester_scan(
        ctx: Context,
        domain: str,
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        OSINT gathering using theHarvester — emails, hosts, IPs from public sources.

        Workflow position: passive OSINT alongside subfinder/amass.
        Uniquely valuable for email harvesting and LinkedIn/social media intel
        that subdomain tools don't cover.

        Parameters:
        - domain: target domain (e.g. 'example.com')
        - additional_args: extra theHarvester flags
            '-b google,bing,linkedin'  — specific sources (default: all)
            '-l 500'                   — limit results per source (default 100)
            '-f <file>'                — save results to HTML/XML file
            '-s'                       — use Shodan (requires API key)
            '-v'                       — verify host names via DNS resolution

        Prerequisites: none — passive queries to public OSINT sources.
        Some sources require API keys (Shodan, Hunter.io, etc.) configured
        in ~/.theHarvester/api-keys.yaml.

        Output:
        - Email addresses (valuable for phishing/password spray)
        - Hostnames and subdomains
        - IP addresses and ASN info
        - LinkedIn profiles (if -b linkedin used)
        - Virtual hosts

        Typical recon sequence:
            1. subfinder_scan(domain='example.com')
            2. theharvester_scan(domain='example.com')    — emails + OSINT
            3. amass_scan(domain='example.com')
            4. httpx probe on all discovered hosts
        """
        data = {
            "domain": domain,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting theHarvester: {domain}")
        result = hexstrike_client.safe_post("api/tools/recon/theharvester", data)
        if result.get("success"):
            await ctx.info(f"✅ theHarvester completed for {domain}")
        else:
            await ctx.error(f"❌ theHarvester failed for {domain}")
        return result
