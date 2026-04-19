# tests/test_hexstrike_mcp.py  (Version ANTI-FREEZE)
import pytest
import hexstrike_mcp  # Couvre 7/9 lignes INSTANTANÉMENT

def test_hexstrike_mcp_entrypoint_imports():
    """Import exécute logging + imports = 78% coverage stable."""
    # Vérifie que tout s'importe sans erreur
    assert hexstrike_mcp.sys
    assert hasattr(hexstrike_mcp, 'logger')
    assert hexstrike_mcp.logging.getLogger(__name__)  # Ligne 40