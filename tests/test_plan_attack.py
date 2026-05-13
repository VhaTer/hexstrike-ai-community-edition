import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clear_scan_cache():
    """Clear global scan cache before every test (prevents state leak across tests)."""
    from mcp_core.server_setup import _scan_cache
    _scan_cache.cache.clear()
    _scan_cache.ttl_times.clear()


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


def test_plan_attack_triggers_analysis_and_returns_chain():
    # Prepare fake profile and chain
    fake_profile = SimpleNamespace(
        target_type=SimpleNamespace(value="web"),
        risk_level="low",
        confidence_score=0.9,
        to_dict=lambda: {"target_type": "web", "risk_level": "low"},
    )

    fake_chain = SimpleNamespace(
        steps=[{"tool": "nmap"}],
        risk_level="low",
        to_dict=lambda: {"steps": [{"tool": "nmap"}], "risk_level": "low"},
    )

    with patch(
        "server_core.intelligence.intelligent_decision_engine.IntelligentDecisionEngine.analyze_target",
        return_value=fake_profile,
    ) as mock_analyze, patch(
        "server_core.intelligence.intelligent_decision_engine.IntelligentDecisionEngine.create_attack_chain",
        return_value=fake_chain,
    ) as mock_chain:
        mcp = make_mcp()

        async def call_plan():
            tool = await mcp.get_tool("plan_attack")
            assert tool is not None
            ctx = make_mock_context()
            return await tool.fn(ctx, target="example.com", objective="comprehensive")

        result = run(call_plan())

    mock_analyze.assert_called_once()
    mock_chain.assert_called_once()
    assert isinstance(result, dict)
    assert result.get("risk_level") == "low"
    assert isinstance(result.get("steps"), list)


def test_plan_attack_uses_session_profile_and_suggests_prompt():
    # saved profile dict that will be returned from ctx.get_state
    saved_profile = {"target_type": "web", "risk_level": "low", "confidence_score": 0.9}

    fake_profile = SimpleNamespace(
        target_type=SimpleNamespace(value="web"),
        risk_level="low",
        confidence_score=0.9,
        to_dict=lambda: saved_profile,
        summary=lambda: "web profile",
    )

    fake_chain = SimpleNamespace(steps=[{"tool": "nmap"}], risk_level="low", to_dict=lambda: {"steps": [{"tool": "nmap"}], "risk_level": "low"})

    # Patch TargetProfile.from_dict to return our fake_profile and IDE.create_attack_chain
    with patch("shared.target_profile.TargetProfile.from_dict", return_value=fake_profile) as mock_from_dict, \
         patch("server_core.intelligence.intelligent_decision_engine.IntelligentDecisionEngine.create_attack_chain", return_value=fake_chain) as mock_chain:
        mcp = make_mcp()

        async def call_plan():
            tool = await mcp.get_tool("plan_attack")
            assert tool is not None
            # ctx.get_state returns a dict to simulate session-restored profile
            ctx = make_mock_context()
            ctx.get_state = AsyncMock(return_value=saved_profile)
            # prepare a prompt result with a hint message
            msg = SimpleNamespace(content=SimpleNamespace(text="This is a hint"))
            ctx.get_prompt = AsyncMock(return_value=SimpleNamespace(messages=[msg]))
            return await tool.fn(ctx, target="example.com", objective="comprehensive")

        result = run(call_plan())

    mock_from_dict.assert_called_once()
    mock_chain.assert_called_once()
    assert isinstance(result, dict)
    assert result.get("risk_level") == "low"
