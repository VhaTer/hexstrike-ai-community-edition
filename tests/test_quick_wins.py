"""Coverage wins: attack_chain, rate_limit_detector, technology_detector,
setup_logging, recovery_executor."""

import logging
import os
from unittest.mock import Mock, patch

import pytest

from shared.attack_chain import AttackChain
from shared.attack_step import AttackStep
from shared.target_profile import TargetProfile

from server_core.rate_limit_detector import RateLimitDetector
from server_core.technology_detector import TechnologyDetector


# =========================================================================
# attack_chain.py — cover empty steps branch (lines 25-26)
# =========================================================================

class TestAttackChain:
    def test_empty_steps_probability(self):
        profile = TargetProfile(target="10.0.0.1")
        chain = AttackChain(profile)
        chain.calculate_success_probability()
        assert chain.success_probability == 0.0

    def test_with_steps(self):
        profile = TargetProfile(target="10.0.0.1")
        chain = AttackChain(profile)
        step1 = AttackStep(
            tool="nmap", parameters={"target": "10.0.0.1"},
            expected_outcome="Port scan", success_probability=0.8,
            execution_time_estimate=60,
        )
        step2 = AttackStep(
            tool="sqlmap", parameters={"target": "10.0.0.1"},
            expected_outcome="SQL injection", success_probability=0.5,
            execution_time_estimate=120,
        )
        chain.add_step(step1)
        chain.add_step(step2)
        assert len(chain.steps) == 2
        assert "nmap" in chain.required_tools
        assert chain.estimated_time == 180

        chain.calculate_success_probability()
        assert chain.success_probability == pytest.approx(0.4)

    def test_to_dict(self):
        profile = TargetProfile(target="10.0.0.1")
        chain = AttackChain(profile)
        step = AttackStep(
            tool="nmap", parameters={"target": "10.0.0.1"},
            expected_outcome="Port scan", success_probability=0.9,
            execution_time_estimate=30, dependencies=["target-up"],
        )
        chain.add_step(step)
        chain.risk_level = "medium"
        d = chain.to_dict()
        assert d["target"] == "10.0.0.1"
        assert len(d["steps"]) == 1
        assert d["steps"][0]["tool"] == "nmap"
        assert d["steps"][0]["dependencies"] == ["target-up"]
        assert d["risk_level"] == "medium"


# =========================================================================
# rate_limit_detector.py — cover delay <= 0 branch (line 102 BrPart 102->105)
# =========================================================================

class TestRateLimitDetectorDelayBranch:
    def test_adjust_timing_no_delay(self):
        detector = RateLimitDetector()
        # Patch timing_profiles to have a profile with delay=0
        original = detector.timing_profiles.copy()
        detector.timing_profiles["no_delay"] = {"delay": 0, "threads": 5, "timeout": 10}
        result = detector.adjust_timing(
            {"additional_args": ""}, "no_delay"
        )
        detector.timing_profiles = original
        assert "delay" not in result["additional_args"]
        assert "-t 5" in result["additional_args"]


# =========================================================================
# technology_detector.py — cover port-not-found and duplicate-service branches
# =========================================================================

# =========================================================================
# setup_logging.py — remaining branches: StripCLFNoise, resolve_log_level,
# JSON log handler, PermissionError, SuppressWerkzeugBanner
# =========================================================================

def _make_record(msg, name="test"):
    return logging.LogRecord(name, logging.INFO, __file__, 1, msg, (), None)


class TestSetupLoggingBranches:
    def test_strip_clf_noise(self):
        from server_core.setup_logging import _StripCLFNoise
        filt = _StripCLFNoise()
        record = _make_record('10.0.0.1 - - [11/May/2026 12:00:00] "GET /" 200 -')
        assert filt.filter(record) is True
        assert "[" not in record.msg

    def test_strip_clf_noise_short_line(self):
        from server_core.setup_logging import _StripCLFNoise
        filt = _StripCLFNoise()
        record = _make_record('10.0.0.1 - ')
        assert filt.filter(record) is True

    def test_resolve_log_level_valid(self):
        with patch.dict(os.environ, {"HEXSTRIKE_LOG_LEVEL": "debug"}, clear=False):
            from server_core.setup_logging import _resolve_log_level
            assert _resolve_log_level() == logging.DEBUG

    def test_resolve_log_level_invalid(self):
        with patch.dict(os.environ, {"HEXSTRIKE_LOG_LEVEL": "bogus"}, clear=False):
            from server_core.setup_logging import _resolve_log_level
            assert _resolve_log_level() == logging.INFO

    def test_setup_logging_json_env(self, tmp_path):
        json_path = str(tmp_path / "test.json")
        with patch.dict(os.environ, {"HEXSTRIKE_JSON_LOG": json_path}, clear=False):
            from server_core.setup_logging import setup_logging
            root = setup_logging()
            assert root is not None

    def test_setup_logging_permission_error(self):
        with patch("server_core.setup_logging.RotatingFileHandler", side_effect=PermissionError):
            from server_core.setup_logging import setup_logging
            root = setup_logging()
            assert root is not None

    def test_suppress_werkzeug_banner(self):
        from server_core.setup_logging import setup_logging
        root = setup_logging()
        werkzeug_logger = logging.getLogger('werkzeug')
        banner_filters = [
            f for f in werkzeug_logger.filters
            if hasattr(f, '_BANNER_PREFIXES')
        ]
        assert len(banner_filters) >= 1

    def test_resolve_log_level_empty(self):
        with patch.dict(os.environ, {"HEXSTRIKE_LOG_LEVEL": ""}, clear=False):
            from server_core.setup_logging import _resolve_log_level
            assert _resolve_log_level() == logging.INFO


# =========================================================================
# recovery_executor.py — SWITCH_TO_ALTERNATIVE_TOOL + ADJUST_PARAMETERS
# =========================================================================

class _MockAction:
    """Simple action enum for tests, comparable by value."""
    def __init__(self, value):
        self.value = value
    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        if hasattr(other, 'value'):
            return self.value == other.value
        return NotImplemented
    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result


class _MockRecoveryActions:
    RETRY_WITH_BACKOFF = _MockAction("retry_with_backoff")
    RETRY_WITH_REDUCED_SCOPE = _MockAction("retry_with_reduced_scope")
    SWITCH_TO_ALTERNATIVE_TOOL = _MockAction("switch_to_alternative_tool")
    ADJUST_PARAMETERS = _MockAction("adjust_parameters")
    ESCALATE_TO_HUMAN = _MockAction("escalate_to_human")
    GRACEFUL_DEGRADATION = _MockAction("graceful_degradation")
    ABORT_OPERATION = _MockAction("abort_operation")


class _RecoveryStrategy:
    def __init__(self, action, **kwargs):
        self.action = action
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestRecoveryExecutorBranches:
    def test_switch_to_alternative_tool(self):
        from server_core.recovery_executor import execute_command_with_recovery
        exec_fn = Mock(side_effect=[
            {"success": False, "stderr": "tool failed"},
            {"success": True, "result": "alt tool worked"},
        ])
        err_handler = Mock()
        err_handler.handle_tool_failure.return_value = _RecoveryStrategy(
            _MockAction("switch_to_alternative_tool"),
            alternative_tool="nmap",
        )
        deg_mgr = Mock()
        deg_mgr.create_fallback_chain.return_value = ["fallback_tool"]
        logger = Mock()

        result = execute_command_with_recovery(
            tool_name="masscan",
            command="masscan 10.0.0.0/24",
            parameters={},
            execute_command_fn=exec_fn,
            error_handler=err_handler,
            degradation_manager=deg_mgr,
            rebuild_command_with_params_fn=Mock(return_value="rebuilt"),
            determine_operation_type_fn=Mock(return_value="scan"),
            recovery_action_enum=_MockRecoveryActions,
            logger=logger,
        )
        assert result["success"] is True
        logger.warning.assert_called()

    def test_switch_to_alternative_no_fallback(self):
        from server_core.recovery_executor import execute_command_with_recovery
        exec_fn = Mock(side_effect=[
            {"success": False, "stderr": "failed"},
            {"success": False, "stderr": "still failed"},
        ])
        err_handler = Mock()
        err_handler.handle_tool_failure.return_value = _RecoveryStrategy(
            _MockAction("switch_to_alternative_tool"),
            alternative_tool="nmap",
        )
        deg_mgr = Mock()
        deg_mgr.create_fallback_chain.return_value = []
        logger = Mock()

        result = execute_command_with_recovery(
            tool_name="masscan", command="masscan scan",
            parameters={}, max_attempts=1,
            execute_command_fn=exec_fn, error_handler=err_handler,
            degradation_manager=deg_mgr,
            rebuild_command_with_params_fn=Mock(return_value="rebuilt"),
            determine_operation_type_fn=Mock(return_value="scan"),
            recovery_action_enum=_MockRecoveryActions,
            logger=logger,
        )
        assert result["success"] is False

    def test_switch_to_alternative_no_alt_tool(self):
        from server_core.recovery_executor import execute_command_with_recovery
        exec_fn = Mock(side_effect=[
            {"success": False, "stderr": "failed"},
            {"success": False, "stderr": "still failed"},
        ])
        err_handler = Mock()
        err_handler.handle_tool_failure.return_value = _RecoveryStrategy(
            _MockAction("switch_to_alternative_tool"),
        )
        deg_mgr = Mock()
        logger = Mock()

        result = execute_command_with_recovery(
            tool_name="masscan", command="masscan scan",
            parameters={}, max_attempts=1,
            execute_command_fn=exec_fn, error_handler=err_handler,
            degradation_manager=deg_mgr,
            rebuild_command_with_params_fn=Mock(return_value="rebuilt"),
            determine_operation_type_fn=Mock(return_value="scan"),
            recovery_action_enum=_MockRecoveryActions,
            logger=logger,
        )
        assert result["success"] is False

    def test_adjust_parameters(self):
        from server_core.recovery_executor import execute_command_with_recovery
        exec_fn = Mock(side_effect=[
            {"success": False, "stderr": "bad params"},
            {"success": True, "result": "adjusted worked"},
        ])
        err_handler = Mock()
        err_handler.handle_tool_failure.return_value = _RecoveryStrategy(
            _MockAction("adjust_parameters"),
            adjusted_parameters={"timeout": "30"},
        )
        logger = Mock()

        result = execute_command_with_recovery(
            tool_name="curl", command="curl http://target",
            parameters={},
            execute_command_fn=exec_fn, error_handler=err_handler,
            degradation_manager=Mock(),
            rebuild_command_with_params_fn=Mock(return_value="rebuilt"),
            determine_operation_type_fn=Mock(return_value="web"),
            recovery_action_enum=_MockRecoveryActions,
            logger=logger,
        )
        assert result["success"] is True
        logger.warning.assert_called()


class TestTechnologyDetectorBranches:
    def test_port_not_in_services(self):
        detector = TechnologyDetector()
        result = detector.detect_technologies("example.com", ports=[9999, 80])
        assert "http" in result["services"]
        assert "9999" not in result["services"]

    def test_duplicate_ports_dedup(self):
        detector = TechnologyDetector()
        result = detector.detect_technologies("example.com", ports=[80, 80])
        assert result["services"].count("http") == 1
        assert len(result["services"]) == 1
