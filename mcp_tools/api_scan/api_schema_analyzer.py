# mcp_tools/api_scan/api_schema_analyzer.py

from typing import Dict, Any
import asyncio
from fastmcp import Context

def register_api_schema_analyzer(mcp, hexstrike_client, logger):

    @mcp.tool()
    async def api_schema_analyzer(ctx: Context, schema_url: str, schema_type: str = "openapi") -> Dict[str, Any]:
        """
        Analyze API schemas and identify potential security issues.

        Args:
            schema_url: URL to the API schema (OpenAPI/Swagger/GraphQL)
            schema_type: Type of schema (openapi, swagger, graphql)

        Returns:
            Schema analysis results with security issues and recommendations
        """
        data = {
            "schema_url": schema_url,
            "schema_type": schema_type
        }

        await ctx.info(f"🔍 Starting API schema analysis: {schema_url}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: hexstrike_client.safe_post("api/tools/api_schema_analyzer", data)
        )
        done, _ = await asyncio.wait([future], timeout=30)
        if not done:
            await ctx.report_progress(50, 100)
            await ctx.info("⏳ Still running...")
        result = await future
        await ctx.report_progress(100, 100)

        if result.get("success"):
            analysis = result.get("schema_analysis_results", {})
            endpoint_count = len(analysis.get("endpoints_found", []))
            issue_count = len(analysis.get("security_issues", []))

            await ctx.info(f"✅ Schema analysis completed: {endpoint_count} endpoints, {issue_count} issues")

            if issue_count > 0:
                logger.warning(f"⚠️  Found {issue_count} security issues in schema!")
                for issue in analysis.get("security_issues", [])[:3]:  # Show first 3
                    severity = issue.get("severity", "UNKNOWN")
                    issue_type = issue.get("issue", "unknown")
                    logger.warning(f"   ├─ [{severity}] {issue_type}")

            if endpoint_count > 0:
                await ctx.info(f"📊 Discovered endpoints:")
                for endpoint in analysis.get("endpoints_found", [])[:5]:  # Show first 5
                    method = endpoint.get("method", "GET")
                    path = endpoint.get("path", "/")
                    await ctx.info(f"   ├─ {method} {path}")
        else:
            await ctx.error("❌ Schema analysis failed")

        return result
