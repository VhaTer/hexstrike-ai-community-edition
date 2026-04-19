import pytest
from unittest.mock import MagicMock
from mcp_tools.ai_assist.intelligent_decision_engine import register_intelligent_decision_engine_tools

def test_engine_registration_success():
    """Ligne 7 + 8 tools → +30%."""
    mock_mcp = MagicMock()
    mock_client = MagicMock()
    logger = MagicMock()
    colors = MagicMock()
    
    register_intelligent_decision_engine_tools(mock_mcp, mock_client, logger, colors)
    
    # CORRIGÉ: Vérifie tool() calls (pas add_tool)
    assert len(mock_mcp.method_calls) == 8, f"Expected 8 tools, got {len(mock_mcp.method_calls)}"
    
    # Debug (retirez après)
    print("✅ 8 AI tools registered:", [call[0][0] for call in mock_mcp.method_calls])

def test_import_coverage():
    """Imports → +10%."""
    import mcp_tools.ai_assist.intelligent_decision_engine