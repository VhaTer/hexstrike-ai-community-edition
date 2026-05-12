import pytest
from unittest.mock import AsyncMock, MagicMock
from mcp_core.elicitation import confirm_destructive_action


class MockResult:
    def __init__(self, action, data):
        self.action = action
        self.data = data


@pytest.fixture
def ctx():
    c = MagicMock()
    c.elicit = AsyncMock()
    c.info = AsyncMock()
    c.error = AsyncMock()
    return c


@pytest.mark.asyncio
class TestConfirmDestructiveAction:
    async def test_user_confirms_with_result_object(self, ctx):
        ctx.elicit.return_value = MockResult(action="accept", data=True)
        result = await confirm_destructive_action(ctx, "Test action")
        assert result is True
        ctx.info.assert_awaited_with("✅ Action confirmed by user")

    async def test_user_declines_with_result_object(self, ctx):
        ctx.elicit.return_value = MockResult(action="reject", data=False)
        result = await confirm_destructive_action(ctx, "Test action")
        assert result is False
        ctx.info.assert_awaited_with("🚫 Action cancelled by user")

    async def test_user_confirms_with_bool(self, ctx):
        ctx.elicit.return_value = True
        result = await confirm_destructive_action(ctx, "Test action")
        assert result is True
        ctx.info.assert_awaited_with("✅ Action confirmed by user")

    async def test_user_declines_with_bool(self, ctx):
        ctx.elicit.return_value = False
        result = await confirm_destructive_action(ctx, "Test action")
        assert result is False
        ctx.info.assert_awaited_with("🚫 Action cancelled by user")

    async def test_detail_and_warning_in_message(self, ctx):
        ctx.elicit.return_value = True
        await confirm_destructive_action(ctx, "Deauth", detail="wlan0mon", warning="Network impact")
        call_msg = ctx.elicit.call_args[0][0]
        assert "Deauth" in call_msg
        assert "wlan0mon" in call_msg
        assert "Network impact" in call_msg

    async def test_not_implemented_error(self, ctx):
        ctx.elicit.side_effect = NotImplementedError()
        result = await confirm_destructive_action(ctx, "Test action")
        assert result is False
        ctx.error.assert_awaited()

    async def test_generic_exception(self, ctx):
        ctx.elicit.side_effect = RuntimeError("connection lost")
        result = await confirm_destructive_action(ctx, "Test action")
        assert result is False
        ctx.error.assert_awaited()

    async def test_unexpected_response_type(self, ctx):
        ctx.elicit.return_value = "unexpected_string"
        result = await confirm_destructive_action(ctx, "Test action")
        assert result is False
        ctx.info.assert_awaited_with("🚫 Action cancelled — unexpected elicitation response")
