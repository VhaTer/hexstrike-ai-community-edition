import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, PropertyMock
from server_core.error_handling import (
    ErrorType, RecoveryAction, ErrorContext, RecoveryStrategy,
    IntelligentErrorHandler, GracefulDegradation
)


# ---------------------------------------------------------------------------
# IntelligentErrorHandler
# ---------------------------------------------------------------------------

class TestClassifyError:
    def setup_method(self):
        self.handler = IntelligentErrorHandler()

    def test_timeout_exception(self):
        assert self.handler.classify_error("msg", TimeoutError()) == ErrorType.TIMEOUT

    def test_permission_exception(self):
        assert self.handler.classify_error("msg", PermissionError()) == ErrorType.PERMISSION_DENIED

    def test_connection_exception(self):
        assert self.handler.classify_error("msg", ConnectionError()) == ErrorType.NETWORK_UNREACHABLE

    def test_filenotfound_exception(self):
        assert self.handler.classify_error("msg", FileNotFoundError()) == ErrorType.TOOL_NOT_FOUND

    def test_other_exception_falls_through_to_patterns(self):
        result = self.handler.classify_error("timeout occurred", ValueError())
        assert result == ErrorType.TIMEOUT

    def test_timeout_pattern(self):
        assert self.handler.classify_error("connection timeout") == ErrorType.TIMEOUT

    def test_permission_pattern(self):
        assert self.handler.classify_error("permission denied") == ErrorType.PERMISSION_DENIED

    def test_network_pattern(self):
        assert self.handler.classify_error("network unreachable") == ErrorType.NETWORK_UNREACHABLE

    def test_rate_limit_pattern(self):
        assert self.handler.classify_error("rate limit exceeded") == ErrorType.RATE_LIMITED

    def test_tool_not_found_pattern(self):
        assert self.handler.classify_error("command not found") == ErrorType.TOOL_NOT_FOUND

    def test_invalid_parameters_pattern(self):
        assert self.handler.classify_error("invalid argument") == ErrorType.INVALID_PARAMETERS

    def test_resource_exhausted_pattern(self):
        assert self.handler.classify_error("out of memory") == ErrorType.RESOURCE_EXHAUSTED

    def test_authentication_pattern(self):
        assert self.handler.classify_error("authentication failed") == ErrorType.AUTHENTICATION_FAILED

    def test_target_unreachable_pattern(self):
        assert self.handler.classify_error("target unreachable") == ErrorType.TARGET_UNREACHABLE

    def test_parsing_error_pattern(self):
        assert self.handler.classify_error("parse error") == ErrorType.PARSING_ERROR

    def test_unknown_pattern(self):
        assert self.handler.classify_error("some random error") == ErrorType.UNKNOWN

    def test_case_insensitive(self):
        assert self.handler.classify_error("TIMEOUT") == ErrorType.TIMEOUT


class TestHandleToolFailure:
    def setup_method(self):
        self.handler = IntelligentErrorHandler()

    def test_returns_recovery_strategy(self):
        strategy = self.handler.handle_tool_failure("nmap", TimeoutError("timed out"), {"target": "10.0.0.1"})
        assert isinstance(strategy, RecoveryStrategy)
        assert strategy.action is not None

    def test_strategy_for_known_error(self):
        strategy = self.handler.handle_tool_failure("nmap", PermissionError("denied"), {"target": "x", "attempt_count": 1})
        assert strategy.action in [RecoveryAction.ESCALATE_TO_HUMAN, RecoveryAction.SWITCH_TO_ALTERNATIVE_TOOL]

    def test_strategy_for_unknown_error(self):
        strategy = self.handler.handle_tool_failure("nmap", Exception("weird"), {"target": "x"})
        assert strategy.action in [RecoveryAction.RETRY_WITH_BACKOFF, RecoveryAction.ESCALATE_TO_HUMAN]

    def test_adds_to_error_history(self):
        self.handler.handle_tool_failure("nmap", Exception("err"), {"target": "x"})
        assert len(self.handler.error_history) == 1


class TestSelectBestStrategy:
    def setup_method(self):
        self.handler = IntelligentErrorHandler()

    def test_selects_highest_scored(self):
        strategies = [
            RecoveryStrategy(RecoveryAction.RETRY_WITH_BACKOFF, {}, 5, 2.0, 0.9, 10),
            RecoveryStrategy(RecoveryAction.ABORT_OPERATION, {}, 5, 1.0, 0.1, 100),
        ]
        ctx = ErrorContext("nmap", "x", {}, ErrorType.TIMEOUT, "msg", 1, datetime.now(), "", {}, [])
        result = self.handler._select_best_strategy(strategies, ctx)
        assert result.action == RecoveryAction.RETRY_WITH_BACKOFF

    def test_exhausted_strategies_return_escalate(self):
        strategies = [
            RecoveryStrategy(RecoveryAction.RETRY_WITH_BACKOFF, {}, 1, 2.0, 0.9, 10),
        ]
        ctx = ErrorContext("nmap", "x", {}, ErrorType.TIMEOUT, "msg", 5, datetime.now(), "", {}, [])
        result = self.handler._select_best_strategy(strategies, ctx)
        assert result.action == RecoveryAction.ESCALATE_TO_HUMAN

    def test_multiple_attempts_adjusts_probability(self):
        strategies = [
            RecoveryStrategy(RecoveryAction.RETRY_WITH_BACKOFF, {}, 5, 2.0, 0.9, 10),
        ]
        ctx1 = ErrorContext("nmap", "x", {}, ErrorType.TIMEOUT, "msg", 1, datetime.now(), "", {}, [])
        ctx3 = ErrorContext("nmap", "x", {}, ErrorType.TIMEOUT, "msg", 3, datetime.now(), "", {}, [])
        s1 = self.handler._select_best_strategy(strategies, ctx1)
        s3 = self.handler._select_best_strategy(strategies, ctx3)
        assert s1.action == s3.action  # Same strategy, just lower probability


class TestAutoAdjustParameters:
    def setup_method(self):
        self.handler = IntelligentErrorHandler()

    def test_known_tool_and_error(self):
        result = self.handler.auto_adjust_parameters("nmap", ErrorType.TIMEOUT, {"timing": "-T4"})
        assert "timing" in result

    def test_unknown_tool_uses_generic(self):
        result = self.handler.auto_adjust_parameters("unknown_tool", ErrorType.TIMEOUT, {})
        assert "timeout" in result

    def test_unknown_error_type_no_generic(self):
        result = self.handler.auto_adjust_parameters("nmap", ErrorType.PARSING_ERROR, {"a": "b"})
        # No specific or generic adjustments for parsing_error
        assert result == {"a": "b"}

    def test_rate_limited_generic(self):
        result = self.handler.auto_adjust_parameters("unknown", ErrorType.RATE_LIMITED, {})
        assert "delay" in result

    def test_resource_exhausted_generic(self):
        result = self.handler.auto_adjust_parameters("unknown", ErrorType.RESOURCE_EXHAUSTED, {})
        assert "threads" in result


class TestGetAlternativeTool:
    def setup_method(self):
        self.handler = IntelligentErrorHandler()

    def test_known_tool_returns_alternatives(self):
        result = self.handler.get_alternative_tool("nmap", {})
        assert result is not None
        assert result in ["rustscan", "masscan", "zmap"]

    def test_unknown_tool_returns_none(self):
        assert self.handler.get_alternative_tool("imaginary_tool", {}) is None

    def test_filter_prefer_faster_excludes_slow_tools(self):
        alt = self.handler.get_alternative_tool("nuclei", {"prefer_faster_tools": True})
        assert alt is not None
        assert alt != "w3af"  # w3af is excluded when prefer_faster

    def test_filter_require_no_privileges_excludes_privileged(self):
        alt = self.handler.get_alternative_tool("nmap", {"require_no_privileges": True})
        assert alt is not None
        assert alt != "nmap"
        # masscan may also be excluded

    def test_all_filtered_returns_unfiltered(self):
        # If all alternatives filtered out, return original list
        result = self.handler.get_alternative_tool("nmap", {"require_no_privileges": True, "prefer_faster_tools": True})
        assert result is not None  # Should return from unfiltered alternatives

    def test_all_alternatives_filtered_fallback(self):
        """When all alternatives are filtered, fallback to unfiltered list."""
        with patch.object(self.handler, "tool_alternatives", {"custom": ["nmap", "masscan"]}):
            result = self.handler.get_alternative_tool("custom", {"require_no_privileges": True})
            assert result is not None
            assert result == "nmap"


class TestEscalateToHuman:
    def setup_method(self):
        self.handler = IntelligentErrorHandler()

    def test_returns_escalation_data(self):
        ctx = ErrorContext("nmap", "10.0.0.1", {"p": 1}, ErrorType.TIMEOUT, "timeout", 2, datetime.now(), "", {}, [])
        result = self.handler.escalate_to_human(ctx, "high")
        assert result["tool"] == "nmap"
        assert result["target"] == "10.0.0.1"
        assert result["error_type"] == "timeout"
        assert result["urgency"] == "high"

    def test_default_urgency(self):
        ctx = ErrorContext("nmap", "x", {}, ErrorType.PERMISSION_DENIED, "denied", 1, datetime.now(), "", {}, [])
        result = self.handler.escalate_to_human(ctx)
        assert result["urgency"] == "medium"


class TestGetHumanSuggestions:
    def setup_method(self):
        self.handler = IntelligentErrorHandler()

    def _ctx(self, error_type):
        return ErrorContext("tool", "x", {}, error_type, "msg", 1, datetime.now(), "", {}, [])

    def test_permission_denied(self):
        s = self.handler._get_human_suggestions(self._ctx(ErrorType.PERMISSION_DENIED))
        assert any("sudo" in x for x in s)

    def test_tool_not_found(self):
        s = self.handler._get_human_suggestions(self._ctx(ErrorType.TOOL_NOT_FOUND))
        assert any("install" in x.lower() for x in s) or any("PATH" in x for x in s)

    def test_network_unreachable(self):
        s = self.handler._get_human_suggestions(self._ctx(ErrorType.NETWORK_UNREACHABLE))
        assert any("network" in x.lower() for x in s)

    def test_rate_limited(self):
        s = self.handler._get_human_suggestions(self._ctx(ErrorType.RATE_LIMITED))
        assert any("rate" in x.lower() for x in s) or any("wait" in x.lower() for x in s)

    def test_else_branch(self):
        s = self.handler._get_human_suggestions(self._ctx(ErrorType.PARSING_ERROR))
        assert any("Review" in x for x in s)


class TestGetSystemResources:
    def setup_method(self):
        self.handler = IntelligentErrorHandler()

    @patch("psutil.cpu_percent", return_value=50.0)
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    @patch("psutil.pids", return_value=list(range(100)))
    def test_returns_resources(self, mock_pids, mock_disk, mock_mem, mock_cpu):
        mock_mem.return_value.percent = 60.0
        mock_disk.return_value.percent = 70.0
        with patch("os.getloadavg", return_value=(1.0, 2.0, 3.0)):
            result = self.handler._get_system_resources()
        assert result["cpu_percent"] == 50.0
        assert result["memory_percent"] == 60.0
        assert result["disk_percent"] == 70.0

    @patch("psutil.cpu_percent", side_effect=Exception("boom"))
    def test_exception_returns_error_dict(self, mock_cpu):
        result = self.handler._get_system_resources()
        assert "error" in result


class TestAddToHistory:
    def setup_method(self):
        self.handler = IntelligentErrorHandler()
        self.ctx = ErrorContext("t", "x", {}, ErrorType.TIMEOUT, "m", 1, datetime.now(), "", {}, [])

    def test_adds_to_list(self):
        self.handler._add_to_history(self.ctx)
        assert len(self.handler.error_history) == 1

    def test_maintains_max_size(self):
        self.handler.max_history_size = 3
        for _ in range(5):
            self.handler._add_to_history(self.ctx)
        assert len(self.handler.error_history) == 3


class TestGetErrorStatistics:
    def setup_method(self):
        self.handler = IntelligentErrorHandler()

    def test_empty_history(self):
        stats = self.handler.get_error_statistics()
        assert stats["total_errors"] == 0

    def test_with_errors(self):
        now = datetime.now()
        ctx1 = ErrorContext("nmap", "x", {}, ErrorType.TIMEOUT, "t", 1, now, "", {}, [])
        ctx2 = ErrorContext("sqlmap", "x", {}, ErrorType.PERMISSION_DENIED, "p", 1, now, "", {}, [])
        self.handler._add_to_history(ctx1)
        self.handler._add_to_history(ctx2)
        stats = self.handler.get_error_statistics()
        assert stats["total_errors"] == 2
        assert stats["error_counts_by_type"]["timeout"] == 1
        assert stats["error_counts_by_type"]["permission_denied"] == 1
        assert stats["error_counts_by_tool"]["nmap"] == 1
        assert stats["recent_errors_count"] >= 2

    def test_old_error_not_recent(self):
        old = datetime.now() - timedelta(hours=2)
        ctx = ErrorContext("nmap", "x", {}, ErrorType.TIMEOUT, "t", 1, old, "", {}, [])
        self.handler._add_to_history(ctx)
        stats = self.handler.get_error_statistics()
        assert stats["recent_errors_count"] == 0


# ---------------------------------------------------------------------------
# GracefulDegradation
# ---------------------------------------------------------------------------

class TestCreateFallbackChain:
    def setup_method(self):
        self.gd = GracefulDegradation()

    def test_known_operation(self):
        chain = self.gd.create_fallback_chain("network_discovery")
        assert len(chain) > 0
        assert chain[0] in ["nmap", "rustscan", "masscan", "ping"]

    def test_excludes_failed_tools(self):
        chain = self.gd.create_fallback_chain("network_discovery", failed_tools=["nmap", "rustscan"])
        assert "nmap" not in chain
        assert "rustscan" not in chain

    def test_unknown_operation_returns_basic_fallback(self):
        chain = self.gd.create_fallback_chain("unknown_operation")
        assert chain == ["manual_testing"]

    def test_all_failed_returns_basic_fallback(self):
        chain = self.gd.create_fallback_chain("network_discovery",
            failed_tools=["nmap", "rustscan", "masscan", "zmap", "ping"])
        assert chain is not None
        assert len(chain) > 0


class TestHandlePartialFailure:
    def setup_method(self):
        self.gd = GracefulDegradation()

    def test_adds_degradation_info(self):
        result = self.gd.handle_partial_failure("network_discovery", {"target": "10.0.0.1"}, ["nmap"])
        assert result["degradation_info"]["operation"] == "network_discovery"
        assert result["degradation_info"]["failed_components"] == ["nmap"]
        assert result["degradation_info"]["partial_success"] is True

    def test_network_missing_ports_adds_basic_check(self):
        result = self.gd.handle_partial_failure("network_discovery", {}, ["nmap"])
        assert "open_ports" in result

    def test_network_port_check_empty_target(self):
        result = self.gd.handle_partial_failure("network_discovery", {}, ["nmap"])
        assert result["open_ports"] == []

    def test_web_missing_dirs_adds_basic_check(self):
        result = self.gd.handle_partial_failure("web_discovery", {}, ["gobuster"])
        assert "directories" in result

    def test_vuln_missing_adds_basic_check(self):
        result = self.gd.handle_partial_failure("vulnerability_scanning", {}, ["nuclei"])
        assert "vulnerabilities" in result

    def test_other_operation_no_gap_filling(self):
        result = self.gd.handle_partial_failure("parameter_discovery", {}, ["arjun"])
        assert "open_ports" not in result

    def test_manual_recommendations_present(self):
        result = self.gd.handle_partial_failure("network_discovery", {}, ["nmap"])
        assert "manual_recommendations" in result
        assert len(result["manual_recommendations"]) > 0


class TestBasicPortCheck:
    def setup_method(self):
        self.gd = GracefulDegradation()

    def test_empty_target(self):
        assert self.gd._basic_port_check("") == []

    @patch("socket.socket")
    def test_checks_common_ports(self, mock_sock):
        mock_instance = MagicMock()
        mock_instance.connect_ex.return_value = 0
        mock_sock.return_value.__enter__.return_value = mock_instance
        mock_sock.return_value = mock_instance
        result = self.gd._basic_port_check("10.0.0.1")
        assert isinstance(result, list)

    @patch("socket.socket")
    def test_exception_in_check(self, mock_sock):
        mock_sock.side_effect = Exception("fail")
        result = self.gd._basic_port_check("10.0.0.1")
        assert isinstance(result, list)


class TestBasicDirectoryCheck:
    def setup_method(self):
        self.gd = GracefulDegradation()

    def test_empty_target(self):
        assert self.gd._basic_directory_check("") == []

    @patch("requests.head")
    def test_finds_directories(self, mock_head):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_head.return_value = mock_response
        result = self.gd._basic_directory_check("http://example.com")
        assert len(result) > 0

    @patch("requests.head")
    def test_exception_skips_directory(self, mock_head):
        mock_head.side_effect = Exception("fail")
        result = self.gd._basic_directory_check("http://example.com")
        assert result == []

    @patch("requests.head")
    def test_not_found_skips_directory(self, mock_head):
        """Response status not in success list — not appended, covering the loop-back arc."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_head.return_value = mock_response
        result = self.gd._basic_directory_check("http://example.com")
        assert result == []


class TestBasicSecurityCheck:
    def setup_method(self):
        self.gd = GracefulDegradation()

    def test_empty_target(self):
        assert self.gd._basic_security_check("") == []

    @patch("requests.get")
    def test_missing_security_headers(self, mock_get):
        mock_response = MagicMock()
        mock_response.headers = {}
        mock_get.return_value = mock_response
        result = self.gd._basic_security_check("http://example.com")
        assert len(result) == 5  # 5 missing security headers

    @patch("requests.get")
    def test_present_security_headers(self, mock_get):
        mock_response = MagicMock()
        mock_response.headers = {
            "X-Frame-Options": "DENY",
            "X-Content-Type-Options": "nosniff",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000",
            "Content-Security-Policy": "default-src 'self'",
        }
        mock_get.return_value = mock_response
        result = self.gd._basic_security_check("http://example.com")
        assert len(result) == 0

    @patch("requests.get")
    def test_connection_error(self, mock_get):
        mock_get.side_effect = Exception("connection failed")
        result = self.gd._basic_security_check("http://example.com")
        assert len(result) == 1
        assert result[0]["type"] == "connection_error"


class TestGetManualRecommendations:
    def setup_method(self):
        self.gd = GracefulDegradation()

    def test_known_operation_returns_recommendations(self):
        recs = self.gd._get_manual_recommendations("network_discovery", [])
        assert len(recs) > 0

    def test_unknown_operation_returns_empty(self):
        recs = self.gd._get_manual_recommendations("unknown_op", [])
        assert recs == []

    def test_nmap_failed_adds_recommendation(self):
        recs = self.gd._get_manual_recommendations("network_discovery", ["nmap"])
        assert any("online port" in r.lower() for r in recs)

    def test_gobuster_failed_adds_recommendation(self):
        recs = self.gd._get_manual_recommendations("web_discovery", ["gobuster"])
        assert any("manual" in r.lower() for r in recs)

    def test_nuclei_failed_adds_recommendation(self):
        recs = self.gd._get_manual_recommendations("vulnerability_scanning", ["nuclei"])
        assert any("manual" in r.lower() for r in recs)


class TestIsCriticalOperation:
    def setup_method(self):
        self.gd = GracefulDegradation()

    def test_critical_operation(self):
        assert self.gd.is_critical_operation("network_discovery") is True

    def test_non_critical_operation(self):
        assert self.gd.is_critical_operation("parameter_discovery") is False
