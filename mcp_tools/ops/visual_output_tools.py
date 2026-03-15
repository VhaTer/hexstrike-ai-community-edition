# mcp_tools/ops/visual_output_tools.py

from typing import Dict, Any
from fastmcp import Context

def register_visual_output_tools(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def get_live_dashboard(ctx: Context) -> Dict[str, Any]:
        """Get live dashboard showing active processes and system metrics."""
        await ctx.info("📊 Fetching live process dashboard")
        result = hexstrike_client.safe_get("api/processes/dashboard")
        if result.get("success", True):
            await ctx.info("✅ Live dashboard retrieved")
        else:
            await ctx.error("❌ Failed to retrieve dashboard")
        return result

    @mcp.tool()
    async def create_vulnerability_report(
        ctx: Context,
        vulnerabilities: str,
        target: str = "",
        scan_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Create a formatted vulnerability report with severity styling."""
        await ctx.info(f"📋 Creating vulnerability report for {target or 'target'}")
        data = {"vulnerabilities": vulnerabilities, "target": target, "scan_type": scan_type}
        result = hexstrike_client.safe_post("api/visual/vuln-report", data)
        if result.get("success"):
            await ctx.info("✅ Report created")
        else:
            await ctx.error("❌ Failed to create report")
        return result

    @mcp.tool()
    async def format_tool_output(
        ctx: Context,
        tool_name: str,
        output: str,
        success: bool = True
    ) -> Dict[str, Any]:
        """Format raw tool output with visual enhancements."""
        await ctx.info(f"🎨 Formatting output for {tool_name}")
        data = {"tool": tool_name, "output": output, "success": success}
        result = hexstrike_client.safe_post("api/visual/tool-output", data)
        if result.get("success"):
            await ctx.info(f"✅ Output formatted")
        else:
            await ctx.error(f"❌ Failed to format output")
        return result

    @mcp.tool()
    async def create_scan_summary(
        ctx: Context,
        target: str,
        tools_used: str,
        vulnerabilities_found: int = 0,
        execution_time: float = 0.0,
        findings: str = ""
    ) -> Dict[str, Any]:
        """Create a comprehensive scan summary report."""
        await ctx.info(f"📊 Creating scan summary for {target}")
        data = {
            "target": target,
            "tools_used": [t.strip() for t in tools_used.split(",")],
            "execution_time": execution_time,
            "vulnerabilities": [{"severity": "info"}] * vulnerabilities_found,
            "findings": findings
        }
        result = hexstrike_client.safe_post("api/visual/summary-report", data)
        if result.get("success"):
            await ctx.info("✅ Scan summary created")
        else:
            await ctx.error("❌ Failed to create summary")
        return result

    @mcp.tool()
    async def display_system_metrics(ctx: Context) -> Dict[str, Any]:
        """Display current system metrics and performance indicators."""
        await ctx.info("📈 Fetching system metrics")
        result = hexstrike_client.safe_get("api/visual/system-metrics")
        if result.get("success"):
            await ctx.info("✅ System metrics retrieved")
        else:
            await ctx.error("❌ Failed to retrieve metrics")
        return result
