import logging
from types import SimpleNamespace
import asyncio
from unittest.mock import AsyncMock
import pytest


def pytest_configure(config):
    # basic logging config for tests
    logging.basicConfig(level=logging.DEBUG)


def make_logger():
    calls = {
        "info": [],
        "error": []
    }

    class Logger:
        def info(self, msg):
            calls["info"].append(msg)

        def error(self, msg):
            calls["error"].append(msg)

    return SimpleNamespace(logger=Logger(), calls=calls)


@pytest.fixture
def logger():
    return make_logger().logger


def make_mock_context():
    ctx = SimpleNamespace()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.debug = AsyncMock()
    ctx.report_progress = AsyncMock()
    ctx.read_resource = AsyncMock(return_value=SimpleNamespace(contents=[]))
    ctx.get_state = AsyncMock(return_value=None)
    ctx.set_state = AsyncMock()
    ctx.get_prompt = AsyncMock(return_value=SimpleNamespace(messages=[]))
    ctx.session_id = "test-session-fixed"
    return ctx


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def make_mcp():
    from mcp_core.server_setup import setup_mcp_server_standalone
    return setup_mcp_server_standalone()


async def call_run_security_tool(mcp, tool_name, parameters):
    """Shared helper to call the registered `run_security_tool` MCP tool."""
    import json
    tool = await mcp.get_tool("run_security_tool")
    assert tool is not None
    ctx = make_mock_context()
    payload = parameters if isinstance(parameters, str) else json.dumps(parameters)
    return await tool.fn(ctx, tool_name=tool_name, parameters=payload)


@pytest.fixture(autouse=True)
def clear_scan_cache():
    """Clear _scan_cache before every test to prevent cross-test cache hits."""
    try:
        from mcp_core.server_setup import _scan_cache
        _scan_cache.cache.clear()
    except Exception:
        pass
    yield
    try:
        from mcp_core.server_setup import _scan_cache
        _scan_cache.cache.clear()
    except Exception:
        pass
