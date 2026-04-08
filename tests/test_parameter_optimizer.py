"""
tests/test_parameter_optimizer.py

Unit tests for mcp_core/parameter_optimizer.py
Covers: ParameterOptimizer.optimize(), handle_failure()
"""

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
