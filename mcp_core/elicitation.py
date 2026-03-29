"""
mcp_core/elicitation.py

Shared User Elicitation helper for destructive tool confirmation.

Usage:
    from mcp_core.elicitation import confirm_destructive_action

    confirmed = await confirm_destructive_action(ctx, "Deauth attack on AA:BB:CC", detail="wlan0mon")
    if not confirmed:
        return {"success": False, "error": "Cancelled by user"}
    # proceed with execution
"""

from fastmcp import Context


async def confirm_destructive_action(
    ctx: Context,
    action: str,
    detail: str = "",
    warning: str = "",
) -> bool:
    """
    Ask user to confirm a destructive action via ctx.elicit().

    Falls back gracefully if the MCP client does not support elicitation
    (e.g. Cursor, VS Code MCP) — in that case the action is BLOCKED and
    the user is instructed to confirm explicitly in their next message.

    Args:
        ctx:     FastMCP Context
        action:  Short description of the action (e.g. "Deauth attack on AA:BB:CC")
        detail:  Optional extra context shown in the message (e.g. interface name)
        warning: Optional extra warning line

    Returns:
        True if user confirmed, False if declined, cancelled, or not supported.
    """
    message = f"⚠️ DESTRUCTIVE ACTION — {action}"
    if detail:
        message += f"\n{detail}"
    if warning:
        message += f"\n⚠️ {warning}"
    message += "\n\nConfirm execution?"

    try:
        result = await ctx.elicit(message, response_type=bool)

        # FastMCP elicitation result has .action and .data
        if hasattr(result, "action"):
            if result.action == "accept" and result.data is True:
                await ctx.info("✅ Action confirmed by user")
                return True
            else:
                await ctx.info("🚫 Action cancelled by user")
                return False

        # Fallback: some versions return bool directly
        if isinstance(result, bool):
            if result:
                await ctx.info("✅ Action confirmed by user")
                return True
            await ctx.info("🚫 Action cancelled by user")
            return False

        await ctx.info("🚫 Action cancelled — unexpected elicitation response")
        return False

    except NotImplementedError:
        await ctx.error(
            "⛔ This MCP client does not support interactive confirmation.\n"
            f"To execute '{action}', explicitly confirm in your next message."
        )
        return False
    except Exception as e:
        await ctx.error(f"⛔ Elicitation failed: {e} — action blocked for safety")
        return False
