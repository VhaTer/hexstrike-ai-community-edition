# mcp_tools/error_handling/test_error_recovery.py

from typing import Dict, Any
from fastmcp import Context

def register_test_error_recovery_tool(mcp, hexstrike_client, logger=None, HexStrikeColors=None):

    @mcp.tool()
    async def test_error_recovery(
        ctx: Context,
        tool_name: str,
        error_type: str = "timeout",
        target: str = "example.com"
    ) -> Dict[str, Any]:
        """
        Test the intelligent error recovery system with simulated failures.

        Use to verify the recovery system is working correctly or to
        understand what recovery strategy would be applied for a given error.

        Parameters:
        - tool_name: tool to simulate error for (e.g. 'nmap', 'sqlmap')
        - error_type: error type to simulate:
            'timeout'             — connection timeout
            'permission_denied'   — insufficient permissions
            'network_unreachable' — target unreachable
            'tool_not_found'      — tool not installed
            'rate_limited'        — target rate limiting
        - target: target for the simulated test

        Output: recovery strategy, action, success probability, and
        suggested alternative tools.
        """
        data = {
            "tool_name": tool_name,
            "error_type": error_type,
            "target": target
        }
        await ctx.info(f"🧪 Testing error recovery: {tool_name} / {error_type}")
        result = hexstrike_client.safe_post("api/error-handling/test-recovery", data)
        if result.get("success"):
            strategy = result.get("recovery_strategy", {})
            action = strategy.get("action", "unknown")
            prob = strategy.get("success_probability", 0)
            await ctx.info(f"✅ Recovery test completed — action: {action}, probability: {prob:.0%}")
            alternatives = result.get("alternative_tools", [])
            if alternatives:
                await ctx.info(f"💡 Alternatives: {', '.join(alternatives)}")
        else:
            await ctx.error("❌ Error recovery test failed")
        return result
