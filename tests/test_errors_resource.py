"""Tests for errors MCP resource: errors://statistics.

Validates that IntelligentErrorHandler.get_error_statistics() returns
the expected structure.
"""

from datetime import datetime
import pytest
from server_core.error_handling import IntelligentErrorHandler, ErrorContext, ErrorType


@pytest.fixture
def handler():
    h = IntelligentErrorHandler()
    h.error_history = []
    return h


def _add_error(handler, tool, error_type, message="test error"):
    ctx = ErrorContext(
        tool_name=tool,
        target="test.target",
        parameters={},
        error_type=ErrorType(error_type) if isinstance(error_type, str) else error_type,
        error_message=message,
        attempt_count=1,
        timestamp=datetime.now(),
        stack_trace="",
        system_resources={},
    )
    handler.error_history.append(ctx)


class TestErrorsResource:
    """errors://statistics — error classification shape and content."""

    def test_get_error_statistics_empty(self, handler):
        s = handler.get_error_statistics()
        assert s == {"total_errors": 0}

    def test_after_one_error(self, handler):
        _add_error(handler, "nmap", "timeout")
        s = handler.get_error_statistics()
        assert s["total_errors"] == 1
        assert s["error_counts_by_type"]["timeout"] == 1
        assert s["error_counts_by_tool"]["nmap"] == 1
        assert s["recent_errors_count"] >= 1

    def test_multiple_tools_tracked(self, handler):
        _add_error(handler, "nmap", "timeout")
        _add_error(handler, "whatweb", "parsing_error")
        s = handler.get_error_statistics()
        assert s["total_errors"] == 2
        assert set(s["error_counts_by_tool"].keys()) == {"nmap", "whatweb"}

    def test_errors_by_type_aggregates(self, handler):
        _add_error(handler, "nmap", "timeout")
        _add_error(handler, "whatweb", "timeout")
        s = handler.get_error_statistics()
        assert s["error_counts_by_type"]["timeout"] == 2

    def test_recent_errors_contains_details(self, handler):
        _add_error(handler, "nmap", "timeout")
        s = handler.get_error_statistics()
        assert len(s["recent_errors"]) >= 1
        assert s["recent_errors"][0]["tool"] == "nmap"
        assert s["recent_errors"][0]["error_type"] == "timeout"

    def test_multiple_error_types(self, handler):
        _add_error(handler, "nmap", "timeout")
        _add_error(handler, "sqlmap", "network_unreachable")
        s = handler.get_error_statistics()
        assert "timeout" in s["error_counts_by_type"]
        assert "network_unreachable" in s["error_counts_by_type"]
