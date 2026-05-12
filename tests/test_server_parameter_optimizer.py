import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from server_core.parameter_optimizer import ParameterOptimizer
from shared.target_profile import TargetProfile


@pytest.fixture
def optimizer():
    return ParameterOptimizer()


@pytest.fixture
def target():
    return TargetProfile(target="10.0.0.1", open_ports=[80, 443])


@pytest.fixture
def mock_tech(tmp_path):
    """Mock TechnologyDetector to return specific technologies."""
    def _make(tech_dict):
        with patch.object(ParameterOptimizer, "_apply_technology_optimizations") as mock:
            mock.side_effect = lambda t, p, d: {**p, "_tech_detected": d}
            yield mock
    return _make


class TestBaseParameters:
    def test_nmap_base(self, optimizer, target):
        params = optimizer._get_base_parameters("nmap", target)
        assert params["target"] == "10.0.0.1"
        assert "-sS" in params["scan_type"]
        assert "ports" in params

    def test_gobuster_base(self, optimizer, target):
        params = optimizer._get_base_parameters("gobuster", target)
        assert params["mode"] == "dir"
        assert params["threads"] == 20

    def test_sqlmap_base(self, optimizer, target):
        params = optimizer._get_base_parameters("sqlmap", target)
        assert params["batch"] is True
        assert params["level"] == 1

    def test_nuclei_base(self, optimizer, target):
        params = optimizer._get_base_parameters("nuclei", target)
        assert "critical" in params["severity"]
        assert "threads" in params

    def test_unknown_tool(self, optimizer, target):
        params = optimizer._get_base_parameters("unknown_tool", target)
        assert params == {"target": "10.0.0.1"}


class TestApplyTechnologyOptimizations:
    def test_apache_gobuster(self, optimizer):
        params = {"target": "x"}
        result = optimizer._apply_technology_optimizations("gobuster", params, {"web_servers": ["apache"]})
        assert result.get("extensions") is not None

    def test_apache_nuclei(self, optimizer):
        params = {"target": "x"}
        result = optimizer._apply_technology_optimizations("nuclei", params, {"web_servers": ["apache"]})
        assert "apache" in result.get("tags", "")

    def test_nginx_gobuster(self, optimizer):
        params = {"target": "x"}
        result = optimizer._apply_technology_optimizations("gobuster", params, {"web_servers": ["nginx"]})
        assert "json" in result.get("extensions", "")

    def test_nginx_nuclei(self, optimizer):
        params = {"target": "x"}
        result = optimizer._apply_technology_optimizations("nuclei", params, {"web_servers": ["nginx"]})
        assert "nginx" in result.get("tags", "")

    def test_neither_web_server(self, optimizer):
        params = {"target": "x"}
        result = optimizer._apply_technology_optimizations("gobuster", params, {"web_servers": []})
        assert result == params

    def test_wordpress_gobuster(self, optimizer):
        params = {"target": "x"}
        result = optimizer._apply_technology_optimizations("gobuster", params, {"cms": ["wordpress"]})
        assert result.get("extensions") is not None
        assert result.get("additional_paths") is not None

    def test_wordpress_nuclei(self, optimizer):
        params = {"target": "x", "tags": ""}
        result = optimizer._apply_technology_optimizations("nuclei", params, {"cms": ["wordpress"]})
        assert "wordpress" in result.get("tags", "")

    def test_wordpress_wpscan(self, optimizer):
        params = {"target": "x"}
        result = optimizer._apply_technology_optimizations("wpscan", params, {"cms": ["wordpress"]})
        assert "ap,at,cb,dbe" in result.get("enumerate", "")

    def test_php_gobuster(self, optimizer):
        params = {"target": "x"}
        result = optimizer._apply_technology_optimizations("gobuster", params, {"languages": ["php"]})
        assert "php" in result.get("extensions", "")

    def test_php_sqlmap(self, optimizer):
        params = {"target": "x"}
        result = optimizer._apply_technology_optimizations("sqlmap", params, {"languages": ["php"]})
        assert result.get("dbms") == "mysql"

    def test_dotnet_gobuster(self, optimizer):
        params = {"target": "x"}
        result = optimizer._apply_technology_optimizations("gobuster", params, {"languages": ["dotnet"]})
        assert "aspx" in result.get("extensions", "")

    def test_dotnet_sqlmap(self, optimizer):
        params = {"target": "x"}
        result = optimizer._apply_technology_optimizations("sqlmap", params, {"languages": ["dotnet"]})
        assert result.get("dbms") == "mssql"

    def test_cloudflare_waf_gobuster(self, optimizer):
        params = {"threads": 20}
        result = optimizer._apply_technology_optimizations("gobuster", params, {"security": ["cloudflare"]})
        assert result.get("_stealth_mode") is True
        assert result["threads"] <= 5

    def test_incapsula_waf_gobuster(self, optimizer):
        params = {"threads": 20}
        result = optimizer._apply_technology_optimizations("gobuster", params, {"security": ["incapsula"]})
        assert result.get("_stealth_mode") is True

    def test_sucuri_waf_sqlmap(self, optimizer):
        params = {"threads": 10}
        result = optimizer._apply_technology_optimizations("sqlmap", params, {"security": ["sucuri"]})
        assert result.get("_stealth_mode") is True
        assert result.get("delay") == 2
        assert result.get("randomize") is True

    def test_no_security_no_stealth(self, optimizer):
        params = {"threads": 20}
        result = optimizer._apply_technology_optimizations("gobuster", params, {"security": []})
        assert result.get("_stealth_mode") is None


class TestApplyProfileOptimizations:
    def test_unknown_tool_returns_params(self, optimizer):
        result = optimizer._apply_profile_optimizations("unknown", {"a": 1}, "stealth")
        assert result == {"a": 1}

    def test_nmap_stealth(self, optimizer):
        result = optimizer._apply_profile_optimizations("nmap", {"target": "x"}, "stealth")
        assert result.get("scan_type") is not None
        assert "-T2" in result.get("timing", "")

    def test_nmap_aggressive(self, optimizer):
        result = optimizer._apply_profile_optimizations("nmap", {"target": "x"}, "aggressive")
        assert "-T5" in result.get("timing", "")

    def test_gobuster_aggressive(self, optimizer):
        result = optimizer._apply_profile_optimizations("gobuster", {"target": "x"}, "aggressive")
        assert result.get("threads") == 50

    def test_unknown_profile_copies_params(self, optimizer):
        result = optimizer._apply_profile_optimizations("nmap", {"target": "x"}, "nonexistent")
        assert result == {"target": "x"}

    def test_stealth_mode_forces_stealth_profile(self, optimizer):
        params = {"_stealth_mode": True, "threads": 20}
        result = optimizer._apply_profile_optimizations("gobuster", params, "aggressive")
        # Should override to stealth even though aggressive was requested
        assert result.get("threads") <= 5  # stealth has threads=5

    def test_stealth_mode_not_set_no_override(self, optimizer):
        params = {"threads": 20}
        result = optimizer._apply_profile_optimizations("gobuster", params, "aggressive")
        assert result.get("threads") == 50  # aggressive threads


class TestHandleToolFailure:
    def test_timeout_increases_timeout_and_reduces_threads(self, optimizer):
        with patch.object(optimizer.failure_recovery, "analyze_failure") as mock_af:
            mock_af.return_value = {
                "failure_type": "timeout",
                "alternative_tools": ["tool2"],
                "severity": "high"
            }
            result = optimizer.handle_tool_failure("nmap", "timeout error", 1, {"timeout": 30, "threads": 20})
        assert result["adjusted_parameters"]["timeout"] == 60
        assert result["adjusted_parameters"]["threads"] == 10
        assert "Increased timeout and reduced threads" in result["recovery_actions"]

    def test_timeout_no_threads_key(self, optimizer):
        with patch.object(optimizer.failure_recovery, "analyze_failure") as mock_af:
            mock_af.return_value = {
                "failure_type": "timeout",
                "alternative_tools": [],
                "severity": "high"
            }
            result = optimizer.handle_tool_failure("nmap", "timeout error", 1, {"timeout": 30})
        assert result["adjusted_parameters"]["timeout"] == 60
        assert "threads" not in result["adjusted_parameters"]

    def test_rate_limited_applies_stealth(self, optimizer):
        with patch.object(optimizer.failure_recovery, "analyze_failure") as mock_af:
            mock_af.return_value = {
                "failure_type": "rate_limited",
                "alternative_tools": [],
                "severity": "medium"
            }
            with patch.object(optimizer.rate_limiter, "adjust_timing") as mock_timing:
                mock_timing.return_value = {"threads": 3, "delay": "2s"}
                result = optimizer.handle_tool_failure("gobuster", "rate limited", 1, {"threads": 20})
        assert "Applied stealth timing profile" in result["recovery_actions"]
        assert result["adjusted_parameters"]["threads"] == 3

    def test_unknown_failure_no_crash(self, optimizer):
        with patch.object(optimizer.failure_recovery, "analyze_failure") as mock_af:
            mock_af.return_value = {
                "failure_type": "unknown",
                "alternative_tools": [],
                "severity": "low"
            }
            result = optimizer.handle_tool_failure("nmap", "weird error", 1, {"target": "x"})
        assert result["original_tool"] == "nmap"
        assert result["adjusted_parameters"]["target"] == "x"


class TestOptimizeParametersAdvanced:
    def test_default_context(self, optimizer, target):
        with patch.object(optimizer.tech_detector, "detect_technologies") as mock_detect:
            mock_detect.return_value = {}
            with patch.object(optimizer.performance_monitor, "monitor_system_resources") as mock_res:
                mock_res.return_value = {"cpu": 30}
                with patch.object(optimizer.performance_monitor, "optimize_based_on_resources") as mock_opt:
                    mock_opt.side_effect = lambda p, r: p
                    result = optimizer.optimize_parameters_advanced("nmap", target)
        assert "_optimization_metadata" in result
        assert result["_optimization_metadata"]["optimization_profile"] == "normal"

    def test_with_context(self, optimizer, target):
        with patch.object(optimizer.tech_detector, "detect_technologies") as mock_detect:
            mock_detect.return_value = {"web_servers": ["nginx"]}
            with patch.object(optimizer.performance_monitor, "monitor_system_resources") as mock_res:
                mock_res.return_value = {"cpu": 80}
                with patch.object(optimizer.performance_monitor, "optimize_based_on_resources") as mock_opt:
                    mock_opt.side_effect = lambda p, r: {**p, "_optimizations_applied": ["cpu_throttle"]}
                    result = optimizer.optimize_parameters_advanced(
                        "nmap", target, {"optimization_profile": "stealth", "headers": {}}
                    )
        assert result["_optimization_metadata"]["optimization_profile"] == "stealth"
        assert "cpu_throttle" in result["_optimization_metadata"]["optimizations_applied"]


class TestOptimizationProfiles:
    def test_sqlmap_stealth(self, optimizer):
        result = optimizer._apply_profile_optimizations("sqlmap", {"target": "x"}, "stealth")
        assert result.get("level") == 1
        assert result.get("risk") == 1
        assert result.get("threads") == 1

    def test_sqlmap_aggressive(self, optimizer):
        result = optimizer._apply_profile_optimizations("sqlmap", {"target": "x"}, "aggressive")
        assert result.get("level") == 3
        assert result.get("risk") == 3
        assert result.get("threads") == 10
