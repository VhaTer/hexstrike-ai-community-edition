import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace
from server_core.hexstrike_middleware import HexStrikeLoggingMiddleware, HexStrikeSessionMiddleware


def _make_context(name="test_tool", uri="test://uri", session_id="session-123", client_name=None):
    msg = SimpleNamespace()
    msg.name = name
    msg.uri = uri
    if client_name is not None:
        client_info = SimpleNamespace(name=client_name)
        msg.params = SimpleNamespace(clientInfo=client_info)
    else:
        msg.params = None

    fctx = SimpleNamespace(session_id=session_id)
    ctx = SimpleNamespace(message=msg, fastmcp_context=fctx)
    return ctx


@pytest.fixture
def mock_call_next():
    async def fn(_):
        return {"result": "ok"}
    return AsyncMock(wraps=fn)


@pytest.mark.asyncio
class TestHexStrikeLoggingMiddleware:
    async def test_on_call_tool_success(self, mock_call_next):
        ctx = _make_context()
        mw = HexStrikeLoggingMiddleware()
        result = await mw.on_call_tool(ctx, mock_call_next)
        assert result == {"result": "ok"}

    async def test_on_call_tool_error(self, mock_call_next):
        ctx = _make_context()
        mw = HexStrikeLoggingMiddleware()
        async def failing_call_next(_):
            raise ValueError("tool failed")
        with pytest.raises(ValueError):
            await mw.on_call_tool(ctx, failing_call_next)

    async def test_on_read_resource_disabled(self, mock_call_next):
        ctx = _make_context()
        mw = HexStrikeLoggingMiddleware(log_resources=False)
        result = await mw.on_read_resource(ctx, mock_call_next)
        assert result == {"result": "ok"}

    async def test_on_read_resource_enabled_success(self, mock_call_next):
        ctx = _make_context()
        mw = HexStrikeLoggingMiddleware(log_resources=True)
        result = await mw.on_read_resource(ctx, mock_call_next)
        assert result == {"result": "ok"}

    async def test_on_read_resource_enabled_error(self, mock_call_next):
        ctx = _make_context()
        mw = HexStrikeLoggingMiddleware(log_resources=True)
        async def failing_call_next(_):
            raise RuntimeError("resource error")
        with pytest.raises(RuntimeError):
            await mw.on_read_resource(ctx, failing_call_next)

    async def test_on_get_prompt_disabled(self, mock_call_next):
        ctx = _make_context()
        mw = HexStrikeLoggingMiddleware(log_prompts=False)
        result = await mw.on_get_prompt(ctx, mock_call_next)
        assert result == {"result": "ok"}

    async def test_on_get_prompt_enabled_success(self, mock_call_next):
        ctx = _make_context()
        mw = HexStrikeLoggingMiddleware(log_prompts=True)
        result = await mw.on_get_prompt(ctx, mock_call_next)
        assert result == {"result": "ok"}

    async def test_on_get_prompt_enabled_error(self, mock_call_next):
        ctx = _make_context()
        mw = HexStrikeLoggingMiddleware(log_prompts=True)
        async def failing_call_next(_):
            raise PermissionError("prompt denied")
        with pytest.raises(PermissionError):
            await mw.on_get_prompt(ctx, failing_call_next)


@pytest.mark.asyncio
class TestHexStrikeSessionMiddleware:
    async def test_on_initialize(self, mock_call_next):
        ctx = _make_context(client_name="test-client")
        mw = HexStrikeSessionMiddleware()
        result = await mw.on_initialize(ctx, mock_call_next)
        assert result == {"result": "ok"}

    async def test_on_initialize_no_client_info(self, mock_call_next):
        ctx = _make_context()
        mw = HexStrikeSessionMiddleware()
        result = await mw.on_initialize(ctx, mock_call_next)
        assert result == {"result": "ok"}

    async def test_on_initialize_params_without_client_info(self, mock_call_next):
        msg = SimpleNamespace(name="test", params=SimpleNamespace(clientInfo=None))
        fctx = SimpleNamespace(session_id="sid")
        ctx = SimpleNamespace(message=msg, fastmcp_context=fctx)
        mw = HexStrikeSessionMiddleware()
        result = await mw.on_initialize(ctx, mock_call_next)
        assert result == {"result": "ok"}

    async def test_session_id_unknown_when_no_fastmcp_context(self):
        from server_core.hexstrike_middleware import _get_session_id
        ctx = SimpleNamespace(message=None, fastmcp_context=None)
        sid = _get_session_id(ctx)
        assert sid == "unknown"

    async def test_session_id_unknown_on_runtime_error(self):
        from server_core.hexstrike_middleware import _get_session_id
        class RaisesFctx:
            @property
            def session_id(self):
                raise RuntimeError("no session")
        ctx = SimpleNamespace(
            message=SimpleNamespace(name="test"),
            fastmcp_context=RaisesFctx()
        )
        sid = _get_session_id(ctx)
        assert sid == "unknown"
