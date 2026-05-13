"""
tests/test_parameter_optimizer.py

Unit tests for mcp_core/parameter_optimizer.py
Covers: ParameterOptimizer.optimize(), handle_failure()
"""

import builtins
import sys
import pytest
from unittest.mock import patch, MagicMock

from mcp_core.technology_detector import TechProfile
from mcp_core.parameter_optimizer import ParameterOptimizer


@pytest.fixture(autouse=True)
def mock_psutil_low_load():
    """Default: psutil reports low CPU/memory so resource tuning never fires.
    Tests that explicitly need high load override this via their own patch."""
    mock_ps = MagicMock()
    mock_ps.cpu_percent.return_value = 30
    mock_ps.virtual_memory.return_value.percent = 40
    with patch("mcp_core.parameter_optimizer._PSUTIL_AVAILABLE", True), \
         patch("mcp_core.parameter_optimizer.psutil", mock_ps):
        yield mock_ps


@pytest.fixture
def optimizer():
    return ParameterOptimizer()


@pytest.fixture
def waf_profile():
    return TechProfile(security=["cloudflare"], web_servers=["nginx"])


@pytest.fixture
def wordpress_profile():
    return TechProfile(
        cms=["wordpress"],
        languages=["php"],
        web_servers=["apache"],
    )


@pytest.fixture
def clean_profile():
    return TechProfile(web_servers=["apache"])


# ---------------------------------------------------------------------------
# Base parameter injection
# ---------------------------------------------------------------------------

class TestBaseParams:
    def test_gobuster_gets_base_threads(self, optimizer):
        result = optimizer.optimize("gobuster", {"url": "http://example.com"})
        assert "threads" in result
        assert result["threads"] == 20

    def test_caller_params_not_overridden(self, optimizer):
        result = optimizer.optimize("gobuster", {"url": "http://example.com", "threads": 5})
        assert result["threads"] == 5

    def test_unknown_tool_no_base(self, optimizer):
        result = optimizer.optimize("unknowntool", {"target": "1.2.3.4"})
        assert result["target"] == "1.2.3.4"


# ---------------------------------------------------------------------------
# WAF detection → forced stealth
# ---------------------------------------------------------------------------

class TestWafStealth:
    def test_waf_forces_stealth(self, optimizer, waf_profile):
        result = optimizer.optimize("gobuster", {"url": "http://example.com"}, waf_profile)
        assert result["_optimizer"]["forced_stealth"] is True
        assert result["_optimizer"]["profile"] == "stealth"

    def test_waf_reduces_gobuster_threads(self, optimizer, waf_profile):
        result = optimizer.optimize("gobuster", {"url": "http://example.com"}, waf_profile)
        assert result["threads"] <= 5

    def test_no_waf_normal_profile(self, optimizer, clean_profile):
        result = optimizer.optimize("gobuster", {"url": "http://example.com"}, clean_profile, profile="normal")
        assert result["_optimizer"]["forced_stealth"] is False
        assert result["_optimizer"]["profile"] == "normal"


# ---------------------------------------------------------------------------
# WordPress optimizations
# ---------------------------------------------------------------------------

class TestWordpressOptimizations:
    def test_wordpress_adds_php_extensions(self, optimizer, wordpress_profile):
        result = optimizer.optimize("gobuster", {"url": "http://wp.example.com"}, wordpress_profile)
        assert "php" in result.get("additional_args", "")

    def test_wordpress_adds_wp_paths(self, optimizer, wordpress_profile):
        result = optimizer.optimize("gobuster", {"url": "http://wp.example.com"}, wordpress_profile)
        assert "_wp_paths" in result
        assert any("wp-admin" in p for p in result["_wp_paths"])

    def test_wordpress_nuclei_tags(self, optimizer, wordpress_profile):
        result = optimizer.optimize("nuclei", {"target": "http://wp.example.com"}, wordpress_profile)
        assert "wordpress" in result.get("additional_args", "")

    def test_wordpress_wpscan_full_enum(self, optimizer, wordpress_profile):
        result = optimizer.optimize("wpscan", {"url": "http://wp.example.com"}, wordpress_profile)
        assert "ap,at,cb,dbe" in result.get("additional_args", "")


# ---------------------------------------------------------------------------
# PHP optimizations
# ---------------------------------------------------------------------------

class TestPhpOptimizations:
    def test_php_adds_extensions(self, optimizer):
        php_profile = TechProfile(languages=["php"])
        result = optimizer.optimize("gobuster", {"url": "http://example.com"}, php_profile)
        assert "php" in result.get("additional_args", "")


# ---------------------------------------------------------------------------
# Profile templates
# ---------------------------------------------------------------------------

class TestProfiles:
    def test_stealth_profile_nmap(self, optimizer):
        result = optimizer.optimize("nmap", {"target": "1.2.3.4"}, profile="stealth")
        assert "-T2" in result.get("additional_args", "")

    def test_aggressive_profile_nmap(self, optimizer):
        result = optimizer.optimize("nmap", {"target": "1.2.3.4"}, profile="aggressive")
        assert "-T5" in result.get("additional_args", "")

    def test_stealth_sqlmap(self, optimizer):
        result = optimizer.optimize("sqlmap", {"url": "http://example.com"}, profile="stealth")
        assert "--delay=2" in result.get("additional_args", "")

    def test_aggressive_nuclei(self, optimizer):
        result = optimizer.optimize("nuclei", {"target": "http://example.com"}, profile="aggressive")
        assert "low" in result.get("severity", "")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_optimizer_metadata_present(self, optimizer):
        result = optimizer.optimize("gobuster", {"url": "http://example.com"})
        assert "_optimizer" in result
        assert "profile" in result["_optimizer"]

    def test_tech_summary_in_metadata(self, optimizer, wordpress_profile):
        result = optimizer.optimize("gobuster", {"url": "http://example.com"}, wordpress_profile)
        assert "wordpress" in result["_optimizer"]["tech_summary"]

    def test_no_profile_defaults_to_not_detected(self, optimizer):
        result = optimizer.optimize("gobuster", {"url": "http://example.com"})
        assert result["_optimizer"]["tech_summary"] == "not detected"


# ---------------------------------------------------------------------------
# Failure recovery
# ---------------------------------------------------------------------------

class TestFailureRecovery:
    def test_timeout_halves_threads(self, optimizer):
        params = {"threads": 20, "timeout": 30}
        result = optimizer.handle_failure("gobuster", params, "timeout")
        assert result["threads"] == 10
        assert result["timeout"] == 60

    def test_rate_limited_applies_stealth(self, optimizer):
        params = {"url": "http://example.com", "threads": 20}
        result = optimizer.handle_failure("gobuster", params, "rate_limited")
        assert result["threads"] <= 5

    def test_unknown_failure_no_crash(self, optimizer):
        params = {"url": "http://example.com"}
        result = optimizer.handle_failure("gobuster", params, "unknown_error")
        assert result["url"] == "http://example.com"


# ---------------------------------------------------------------------------
# Resource tuning (mocked psutil)
# ---------------------------------------------------------------------------

class TestResourceTuning:
    def test_high_cpu_reduces_threads(self, optimizer):
        with patch("mcp_core.parameter_optimizer._PSUTIL_AVAILABLE", True):
            with patch("mcp_core.parameter_optimizer.psutil") as mock_psutil:
                mock_psutil.cpu_percent.return_value = 90
                mock_psutil.virtual_memory.return_value.percent = 60
                params = {"threads": 20}
                result = optimizer._apply_resource_tuning("gobuster", params)
                assert result["threads"] == 10

    def test_low_cpu_no_reduction(self, optimizer):
        with patch("mcp_core.parameter_optimizer._PSUTIL_AVAILABLE", True):
            with patch("mcp_core.parameter_optimizer.psutil") as mock_psutil:
                mock_psutil.cpu_percent.return_value = 30
                mock_psutil.virtual_memory.return_value.percent = 40
                params = {"threads": 20}
                result = optimizer._apply_resource_tuning("gobuster", params)
                assert result["threads"] == 20


# ---------------------------------------------------------------------------
# Web server optimizations (apache / nginx)
# ---------------------------------------------------------------------------

class TestWebServerOptimizations:
    def test_apache_gobuster_extensions(self, optimizer):
        profile = TechProfile(web_servers=["apache"])
        result = optimizer.optimize("gobuster", {"url": "http://example.com"}, profile)
        additional = result.get("additional_args", "")
        assert "conf" in additional
        assert "htaccess" in additional

    def test_nginx_gobuster_extensions(self, optimizer):
        profile = TechProfile(web_servers=["nginx"])
        result = optimizer.optimize("gobuster", {"url": "http://example.com"}, profile)
        additional = result.get("additional_args", "")
        assert "json" in additional
        assert "conf" in additional

    def test_apache_does_not_affect_ffuf(self, optimizer):
        profile = TechProfile(web_servers=["apache"])
        result = optimizer.optimize("ffuf", {"url": "http://example.com/FUZZ"}, profile)
        additional = result.get("additional_args", "")
        assert "htaccess" not in additional

    def test_nginx_does_not_affect_nmap(self, optimizer):
        profile = TechProfile(web_servers=["nginx"])
        result = optimizer.optimize("nmap", {"target": "1.2.3.4"}, profile)
        additional = result.get("additional_args", "")
        assert "conf" not in additional


# ---------------------------------------------------------------------------
# Framework optimizations (django / rails)
# ---------------------------------------------------------------------------

class TestFrameworkOptimizations:
    def test_django_gobuster_paths(self, optimizer):
        profile = TechProfile(frameworks=["django"])
        result = optimizer.optimize("gobuster", {"url": "http://example.com"}, profile)
        assert "_django_paths" in result
        assert "/admin/" in result["_django_paths"]

    def test_rails_gobuster_paths(self, optimizer):
        profile = TechProfile(frameworks=["rails"])
        result = optimizer.optimize("gobuster", {"url": "http://example.com"}, profile)
        assert "_rails_paths" in result
        assert "/admin" in result["_rails_paths"]

    def test_django_not_applied_to_nmap(self, optimizer):
        profile = TechProfile(frameworks=["django"])
        result = optimizer.optimize("nmap", {"target": "1.2.3.4"}, profile)
        assert "_django_paths" not in result


# ---------------------------------------------------------------------------
# .NET language optimizations
# ---------------------------------------------------------------------------

class TestDotNetOptimizations:
    def test_dotnet_gobuster_extensions(self, optimizer):
        profile = TechProfile(languages=["dotnet"])
        result = optimizer.optimize("gobuster", {"url": "http://example.com"}, profile)
        additional = result.get("additional_args", "")
        assert "aspx" in additional

    def test_dotnet_ffuf_extensions(self, optimizer):
        profile = TechProfile(languages=["dotnet"])
        result = optimizer.optimize("ffuf", {"url": "http://example.com/FUZZ"}, profile)
        additional = result.get("additional_args", "")
        assert "aspx" in additional

    def test_dotnet_no_dup_when_x_already_set(self, optimizer):
        profile = TechProfile(languages=["dotnet"])
        result = optimizer.optimize("gobuster", {"url": "http://example.com", "additional_args": "-x asp"}, profile)
        additional = result.get("additional_args", "")
        assert additional.count("-x") == 1


# ---------------------------------------------------------------------------
# PHP optimizations across all affected tools + dedup
# ---------------------------------------------------------------------------

class TestPhpExtended:
    def test_php_ffuf_extensions(self, optimizer):
        profile = TechProfile(languages=["php"])
        result = optimizer.optimize("ffuf", {"url": "http://example.com/FUZZ"}, profile)
        additional = result.get("additional_args", "")
        assert "phtml" in additional

    def test_php_feroxbuster_extensions(self, optimizer):
        profile = TechProfile(languages=["php"])
        result = optimizer.optimize("feroxbuster", {"url": "http://example.com"}, profile)
        additional = result.get("additional_args", "")
        assert "php3" in additional

    def test_php_no_dup_when_x_already_set(self, optimizer):
        profile = TechProfile(languages=["php"])
        result = optimizer.optimize("gobuster", {"url": "http://example.com", "additional_args": "-x custom"}, profile)
        additional = result.get("additional_args", "")
        assert additional.count("-x") == 1

    def test_php_no_dup_when_php_in_args(self, optimizer):
        profile = TechProfile(languages=["php"])
        result = optimizer.optimize("gobuster", {"url": "http://example.com", "additional_args": ".php"}, profile)
        additional = result.get("additional_args", "")
        assert additional.count("-x") == 0

    def test_php_not_applied_to_sqlmap(self, optimizer):
        profile = TechProfile(languages=["php"])
        result = optimizer.optimize("sqlmap", {"url": "http://example.com"}, profile)
        additional = result.get("additional_args", "")
        assert "phtml" not in additional


# ---------------------------------------------------------------------------
# _apply_resource_tuning — edge cases
# ---------------------------------------------------------------------------

class TestResourceTuningExtended:
    def test_high_memory_reduces_threads(self, optimizer):
        with patch("mcp_core.parameter_optimizer._PSUTIL_AVAILABLE", True):
            with patch("mcp_core.parameter_optimizer.psutil") as mock_ps:
                mock_ps.cpu_percent.return_value = 30
                mock_ps.virtual_memory.return_value.percent = 95
                params = {"threads": 20}
                result = optimizer._apply_resource_tuning("gobuster", params)
                assert result["threads"] == 10

    def test_psutil_unavailable_no_reduction(self, optimizer):
        with patch("mcp_core.parameter_optimizer._PSUTIL_AVAILABLE", False):
            params = {"threads": 20}
            result = optimizer._apply_resource_tuning("gobuster", params)
            assert result["threads"] == 20

    def test_psutil_error_no_crash(self, optimizer):
        with patch("mcp_core.parameter_optimizer._PSUTIL_AVAILABLE", True):
            with patch("mcp_core.parameter_optimizer.psutil") as mock_ps:
                mock_ps.cpu_percent.side_effect = Exception("psutil error")
                params = {"threads": 20}
                result = optimizer._apply_resource_tuning("gobuster", params)
                assert result["threads"] == 20

    def test_concurrency_key_reduced(self, optimizer):
        with patch("mcp_core.parameter_optimizer._PSUTIL_AVAILABLE", True):
            with patch("mcp_core.parameter_optimizer.psutil") as mock_ps:
                mock_ps.cpu_percent.return_value = 90
                mock_ps.virtual_memory.return_value.percent = 60
                params = {"concurrency": 10}
                result = optimizer._apply_resource_tuning("gobuster", params)
                assert result["concurrency"] == 5

    def test_workers_key_reduced(self, optimizer):
        with patch("mcp_core.parameter_optimizer._PSUTIL_AVAILABLE", True):
            with patch("mcp_core.parameter_optimizer.psutil") as mock_ps:
                mock_ps.cpu_percent.return_value = 90
                mock_ps.virtual_memory.return_value.percent = 60
                params = {"workers": 8}
                result = optimizer._apply_resource_tuning("gobuster", params)
                assert result["workers"] == 4

    def test_multiple_thread_keys_all_reduced(self, optimizer):
        with patch("mcp_core.parameter_optimizer._PSUTIL_AVAILABLE", True):
            with patch("mcp_core.parameter_optimizer.psutil") as mock_ps:
                mock_ps.cpu_percent.return_value = 90
                mock_ps.virtual_memory.return_value.percent = 60
                params = {"threads": 20, "concurrency": 10, "workers": 8}
                result = optimizer._apply_resource_tuning("gobuster", params)
                assert result["threads"] == 10
                assert result["concurrency"] == 5
                assert result["workers"] == 4

    def test_caller_protected_threads_not_reduced(self, optimizer):
        with patch("mcp_core.parameter_optimizer._PSUTIL_AVAILABLE", True):
            with patch("mcp_core.parameter_optimizer.psutil") as mock_ps:
                mock_ps.cpu_percent.return_value = 90
                mock_ps.virtual_memory.return_value.percent = 60
                params = {"threads": 20}
                caller_keys = {"threads"}
                result = optimizer._apply_resource_tuning("gobuster", params, caller_keys)
                assert result["threads"] == 20

    def test_caller_protected_concurrency_still_reduced(self, optimizer):
        with patch("mcp_core.parameter_optimizer._PSUTIL_AVAILABLE", True):
            with patch("mcp_core.parameter_optimizer.psutil") as mock_ps:
                mock_ps.cpu_percent.return_value = 90
                mock_ps.virtual_memory.return_value.percent = 60
                params = {"threads": 20, "concurrency": 10}
                caller_keys = {"threads"}
                result = optimizer._apply_resource_tuning("gobuster", params, caller_keys)
                assert result["threads"] == 20
                assert result["concurrency"] == 5

    def test_reduce_to_minimum_one(self, optimizer):
        with patch("mcp_core.parameter_optimizer._PSUTIL_AVAILABLE", True):
            with patch("mcp_core.parameter_optimizer.psutil") as mock_ps:
                mock_ps.cpu_percent.return_value = 90
                mock_ps.virtual_memory.return_value.percent = 60
                params = {"threads": 1}
                result = optimizer._apply_resource_tuning("gobuster", params)
                assert result["threads"] == 1


# ---------------------------------------------------------------------------
# _apply_profile — edge cases
# ---------------------------------------------------------------------------

class TestProfileExtended:
    def test_numeric_override_from_base(self, optimizer):
        params = {"threads": 20, "additional_args": ""}
        result = optimizer._apply_profile("gobuster", params, "aggressive")
        assert result["threads"] == 50

    def test_numeric_caller_set_preserved(self, optimizer):
        params = {"threads": 10, "additional_args": ""}
        result = optimizer._apply_profile("gobuster", params, "aggressive")
        assert result["threads"] == 10

    def test_additional_args_appended(self, optimizer):
        params = {"additional_args": "-x php"}
        result = optimizer._apply_profile("gobuster", params, "stealth")
        assert "-t 5" in result["additional_args"]
        assert "to 30s" in result["additional_args"]

    def test_additional_args_empty_profile_kept(self, optimizer):
        params = {"additional_args": "-x php"}
        result = optimizer._apply_profile("gobuster", params, "normal")
        assert result["additional_args"] == "-x php"

    def test_additional_args_dedup_exact_match(self, optimizer):
        params = {"additional_args": "-p 1"}
        result = optimizer._apply_profile("ffuf", params, "stealth")
        assert result["additional_args"] == "-p 1"

    def test_severity_overridden_by_profile(self, optimizer):
        params = {"severity": "critical,high,medium"}
        result = optimizer._apply_profile("nuclei", params, "stealth")
        assert result["severity"] == "critical,high"

    def test_hydra_stealth_profile(self, optimizer):
        result = optimizer.optimize("hydra", {"target": "1.2.3.4"}, profile="stealth")
        assert "-W 3" in result.get("additional_args", "")

    def test_hydra_aggressive_profile(self, optimizer):
        result = optimizer.optimize("hydra", {"target": "1.2.3.4"}, profile="aggressive")
        assert "-f" in result.get("additional_args", "")

    def test_unknown_tool_profile_no_crash(self, optimizer):
        result = optimizer._apply_profile("unknowntool", {"target": "1.2.3.4"}, "stealth")
        assert result["target"] == "1.2.3.4"


# ---------------------------------------------------------------------------
# handle_failure — edge cases
# ---------------------------------------------------------------------------

class TestFailureExtended:
    def test_connection_refused_passthrough(self, optimizer):
        params = {"url": "http://example.com", "threads": 20}
        result = optimizer.handle_failure("gobuster", params, "connection_refused")
        assert result["threads"] == 20
        assert result["url"] == "http://example.com"

    def test_timeout_no_thread_keys(self, optimizer):
        params = {"timeout": 30}
        result = optimizer.handle_failure("gobuster", params, "timeout")
        assert result["timeout"] == 60

    def test_timeout_partial_thread_keys(self, optimizer):
        params = {"threads": 20, "timeout": 30}
        result = optimizer.handle_failure("gobuster", params, "timeout")
        assert result["threads"] == 10
        assert result["timeout"] == 60

    def test_rate_limited_ffuf_stealth_threads(self, optimizer):
        params = {"url": "http://example.com", "threads": 40}
        result = optimizer.handle_failure("ffuf", params, "rate_limited")
        assert result["threads"] == 5

    def test_rate_limited_nmap_stealth_args(self, optimizer):
        params = {"target": "1.2.3.4"}
        result = optimizer.handle_failure("nmap", params, "rate_limited")
        additional = result.get("additional_args", "")
        assert "-T2" in additional

    def test_rate_limited_nuclei_severity(self, optimizer):
        params = {"target": "http://example.com", "severity": "critical,high,medium"}
        result = optimizer.handle_failure("nuclei", params, "rate_limited")
        assert result["severity"] == "critical,high"

    def test_timeout_concurrency_key(self, optimizer):
        params = {"concurrency": 10, "timeout": 30}
        result = optimizer.handle_failure("gobuster", params, "timeout")
        assert result["concurrency"] == 5
        assert result["timeout"] == 60


# ---------------------------------------------------------------------------
# Integration: full optimize() flow with combined profiles
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_gobuster_waf_wordpress_full(self, optimizer):
        profile = TechProfile(
            cms=["wordpress"], languages=["php"],
            web_servers=["apache"], security=["cloudflare"],
        )
        result = optimizer.optimize("gobuster", {"url": "http://example.com"}, profile)
        assert result["_optimizer"]["forced_stealth"] is True
        assert result["_optimizer"]["profile"] == "stealth"
        assert "wp-admin" in str(result.get("_wp_paths", ""))
        assert result["threads"] <= 5

    def test_multi_tech_no_wp(self, optimizer):
        profile = TechProfile(
            languages=["php", "dotnet"],
            frameworks=["django", "rails"],
        )
        result = optimizer.optimize("gobuster", {"url": "http://example.com"}, profile)
        additional = result.get("additional_args", "")
        assert "php" in additional
        assert "_django_paths" in result
        assert "_rails_paths" in result

    def test_nuclei_waf_stealth_severity(self, optimizer):
        profile = TechProfile(security=["mod_security"])
        result = optimizer.optimize("nuclei", {"target": "http://example.com"}, profile)
        assert result["_optimizer"]["forced_stealth"] is True
        assert result["severity"] == "critical,high"

    def test_ffuf_apache_dotnet(self, optimizer):
        profile = TechProfile(
            languages=["dotnet"], web_servers=["apache"],
        )
        result = optimizer.optimize("ffuf", {"url": "http://example.com/FUZZ"}, profile)
        additional = result.get("additional_args", "")
        assert "aspx" in additional

    def test_sqlmap_aggressive_profile(self, optimizer):
        result = optimizer.optimize("sqlmap", {"url": "http://example.com"}, profile="aggressive")
        assert "--threads=10" in result.get("additional_args", "")

    def test_wpscan_normal_without_tech(self, optimizer):
        result = optimizer.optimize("wpscan", {"url": "http://example.com"})
        assert result.get("additional_args", "") == "--enumerate ap,at"

    def test_optimizer_preserves_caller_keys_full_flow(self, optimizer, waf_profile):
        result = optimizer.optimize(
            "gobuster", {"url": "http://example.com", "threads": 3}, waf_profile
        )
        assert result["threads"] == 3

    def test_tech_summary_nmap(self, optimizer):
        profile = TechProfile(web_servers=["apache"], languages=["php"])
        result = optimizer.optimize("nmap", {"target": "1.2.3.4"}, profile)
        summary = result["_optimizer"]["tech_summary"]
        assert "apache" in summary
        assert "php" in summary


# ---------------------------------------------------------------------------
# Test _check_psutil helper
# ---------------------------------------------------------------------------

class TestCheckPsutil:

    def test_check_psutil_available_returns_true(self):
        from mcp_core.parameter_optimizer import _check_psutil
        assert _check_psutil() is True

    def test_check_psutil_import_error_returns_false(self):
        from mcp_core.parameter_optimizer import _check_psutil
        saved = sys.modules.pop("psutil", None)
        saved_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("No module named psutil")
            return saved_import(name, *args, **kwargs)
        try:
            with patch("builtins.__import__", mock_import):
                assert _check_psutil() is False
        finally:
            if saved is not None:
                sys.modules["psutil"] = saved
