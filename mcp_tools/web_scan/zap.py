# mcp_tools/web_scan/zap.py

from typing import Dict, Any
from fastmcp import Context

def register_zap_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def zap_scan(
        ctx: Context,
        target: str = "",
        scan_type: str = "baseline",
        api_key: str = "",
        daemon: bool = False,
        port: str = "8090",
        host: str = "0.0.0.0",
        format_type: str = "xml",
        output_file: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        OWASP ZAP automated web application security scanner.

        Workflow position: comprehensive automated scanning — good alternative
        to Burp Suite for automated OWASP Top 10 scanning.
        ZAP is open source and free — no license required.

        Parameters:
        - target: target URL (e.g. 'https://example.com')
        - scan_type: ZAP scan mode:
            'baseline' — passive scan only (fast, no active attacks, low noise)
            'full'     — full active scan (slower, finds more vulns)
            'api'      — API scan using OpenAPI/Swagger definition
        - api_key: ZAP API key (required if ZAP daemon is running)
        - daemon: start ZAP in daemon mode (background, API access)
        - port: ZAP daemon port (default 8090)
        - host: ZAP daemon host (default 0.0.0.0)
        - format_type: report format ('xml', 'json', 'html')
        - output_file: path to save ZAP report
        - additional_args: extra ZAP arguments

        Prerequisites: ZAP installed (zaproxy package on Kali).

        scan_type guide:
        - baseline — CI/CD integration, quick passive check, low false positives
        - full     — thorough assessment, active attacks, may be noisy
        - api      — REST/SOAP API testing with definition file

        zap vs burpsuite:
        - zap       — open source, free, good for automated pipelines
        - burpsuite — Pro license required for scanner, more powerful manual testing
        """
        data = {
            "target": target,
            "scan_type": scan_type,
            "api_key": api_key,
            "daemon": daemon,
            "port": port,
            "host": host,
            "format": format_type,
            "output_file": output_file,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting ZAP {scan_type} scan: {target}")
        result = hexstrike_client.safe_post("api/tools/zap", data)
        if result.get("success"):
            await ctx.info(f"✅ ZAP scan completed for {target}")
        else:
            await ctx.error(f"❌ ZAP scan failed for {target}")
        return result
