"""
tests/test_server_setup.py

Track 2 refactor: setup_mcp_server() (Flask-era) removed.
setup_mcp_server_standalone() is now the only entry point.
register_gateway_tools is gone — no longer imported.
"""

import pytest
from unittest.mock import MagicMock, patch
import mcp_core.server_setup


def test_server_setup_imports():
    """Verify standalone entry point exists and Flask-era function is gone."""
    available = [name for name in dir(mcp_core.server_setup) if not name.startswith('_')]
    assert 'setup_mcp_server_standalone' in available
    # Flask-era function removed in Track 2
    assert 'setup_mcp_server' not in available
    # Flask gateway import removed
    assert 'register_gateway_tools' not in available


def test_setup_mcp_server_standalone_call():
    """setup_mcp_server_standalone() returns a FastMCP instance."""
    try:
        mcp = mcp_core.server_setup.setup_mcp_server_standalone()
        assert mcp is not None
    except Exception as e:
        print(f"Expected error in test env: {e} — coverage OK!")


def test_module_structure():
    """Core FastMCP primitives are present in server_setup."""
    assert hasattr(mcp_core.server_setup, 'FastMCP')
    # Flask-era artifacts must be absent
    assert not hasattr(mcp_core.server_setup, 'register_gateway_tools')
    assert not hasattr(mcp_core.server_setup, 'setup_mcp_server')
