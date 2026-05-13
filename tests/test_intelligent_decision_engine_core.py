import pytest
from unittest.mock import patch, MagicMock
import socket
import os

from server_core.intelligence.intelligent_decision_engine import (
    IntelligentDecisionEngine,
    _tool_stats,
    parameter_optimizer,
)
from shared.target_types import TargetType, TechnologyStack
from shared.target_profile import TargetProfile
from shared.attack_chain import AttackChain


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    ide = IntelligentDecisionEngine()
    ide.disable_advanced_optimization()
    return ide


def make_profile(target: str = "http://example.com",
                 target_type: TargetType = TargetType.WEB_APPLICATION,
                 technologies=None,
                 open_ports=None,
                 cms_type: str = None,
                 ip_addresses=None,
                 subdomains=None,
                 attack_surface_score: float = 0.0,
                 confidence_score: float = 0.0) -> TargetProfile:
    p = TargetProfile(target=target)
    p.target_type = target_type
    p.technologies = technologies or []
    p.open_ports = open_ports or []
    p.cms_type = cms_type
    p.ip_addresses = ip_addresses or []
    p.subdomains = subdomains or []
    p.attack_surface_score = attack_surface_score
    p.confidence_score = confidence_score
    return p


# ===================================================================
# Zone A — _determine_target_type, _resolve_domain, _detect_technologies,
#           _detect_cms, _calculate_confidence
# ===================================================================

class TestResolveDomain:
    def test_resolves_http_url(self, engine):
        with patch("socket.gethostbyname", return_value="93.184.216.34"):
            result = engine._resolve_domain("http://example.com")
        assert result == ["93.184.216.34"]

    def test_resolves_hostname_str(self, engine):
        with patch("socket.gethostbyname", return_value="10.0.0.1"):
            result = engine._resolve_domain("example.com")
        assert result == ["10.0.0.1"]

    def test_socket_error_returns_empty(self, engine):
        with patch("socket.gethostbyname", side_effect=socket.gaierror("no addr")):
            result = engine._resolve_domain("http://example.com")
        assert result == []

    def test_empty_hostname_returns_empty(self, engine):
        result = engine._resolve_domain("")
        assert result == []


class TestDetermineTargetType:
    def test_http_url_web_app(self, engine):
        assert engine._determine_target_type("http://example.com") == TargetType.WEB_APPLICATION

    def test_api_path(self, engine):
        assert engine._determine_target_type("http://example.com/api/v1") == TargetType.API_ENDPOINT

    def test_ip_address(self, engine):
        assert engine._determine_target_type("192.168.1.1") == TargetType.NETWORK_HOST

    def test_domain_name(self, engine):
        assert engine._determine_target_type("example.com") == TargetType.WEB_APPLICATION

    def test_binary_file(self, engine):
        # Must not match domain pattern first (starts with /)
        assert engine._determine_target_type("/path/to/exploit.exe") == TargetType.BINARY_FILE

    def test_cloud_service(self, engine):
        # Must not match URL/IP/domain patterns before cloud check
        assert engine._determine_target_type("azure_blob_storage") == TargetType.CLOUD_SERVICE

    def test_unknown(self, engine):
        assert engine._determine_target_type("some_random_input") == TargetType.UNKNOWN


class TestDetectTechnologies:
    def test_wordpress_detected(self, engine):
        result = engine._detect_technologies("http://example.com/wp-admin")
        assert TechnologyStack.WORDPRESS in result

    def test_php_detected(self, engine):
        result = engine._detect_technologies("http://example.com/index.php")
        assert TechnologyStack.PHP in result

    def test_dotnet_detected(self, engine):
        result = engine._detect_technologies("http://example.com/default.aspx")
        assert TechnologyStack.DOTNET in result

    def test_unknown_fallback(self, engine):
        result = engine._detect_technologies("http://example.com")
        assert result == [TechnologyStack.UNKNOWN]


class TestDetectCms:
    def test_wordpress(self, engine):
        assert engine._detect_cms("http://example.com/wp-login") == "WordPress"

    def test_drupal(self, engine):
        assert engine._detect_cms("http://drupal.example.com") == "Drupal"

    def test_joomla(self, engine):
        assert engine._detect_cms("http://joomla.example.com") == "Joomla"

    def test_unknown(self, engine):
        assert engine._detect_cms("http://example.com") is None


class TestCalculateConfidence:
    def test_baseline_only(self, engine):
        p = make_profile(target_type=TargetType.UNKNOWN)
        assert engine._calculate_confidence(p) == 0.5

    def test_with_ip_addresses(self, engine):
        p = make_profile(target_type=TargetType.UNKNOWN, ip_addresses=["10.0.0.1"])
        assert engine._calculate_confidence(p) == 0.6

    def test_with_known_tech(self, engine):
        p = make_profile(target_type=TargetType.UNKNOWN, technologies=[TechnologyStack.PHP])
        assert engine._calculate_confidence(p) == 0.7

    def test_with_cms(self, engine):
        p = make_profile(target_type=TargetType.UNKNOWN,
                         technologies=[TechnologyStack.WORDPRESS],
                         cms_type="WordPress")
        assert engine._calculate_confidence(p) == pytest.approx(0.8)

    def test_with_known_target_type(self, engine):
        p = make_profile(target_type=TargetType.NETWORK_HOST)
        assert engine._calculate_confidence(p) == pytest.approx(0.6)

    def test_capped_at_one(self, engine):
        p = make_profile(
            target_type=TargetType.UNKNOWN,
            ip_addresses=["10.0.0.1"],
            technologies=[TechnologyStack.PHP],
            cms_type="WordPress",
        )
        p.target_type = TargetType.NETWORK_HOST
        # 0.5 + 0.1 (ip) + 0.2 (tech) + 0.1 (cms) + 0.1 (known_type) = 1.0
        assert engine._calculate_confidence(p) == pytest.approx(1.0)


# ===================================================================
# Zone B — select_optimal_tools
# ===================================================================

class TestSelectOptimalTools:
    def test_quick_selects_top_3(self, engine):
        p = make_profile(technologies=[])
        tools = engine.select_optimal_tools(p, objective="quick")
        assert len(tools) <= 3

    def test_comprehensive_filters_by_effectiveness(self, engine):
        p = make_profile(technologies=[])
        tools = engine.select_optimal_tools(p, objective="comprehensive")
        assert len(tools) > 0

    def test_stealth_filters_passive_tools(self, engine):
        p = make_profile(technologies=[])
        tools = engine.select_optimal_tools(p, objective="stealth")
        for t in tools:
            assert t in ("amass", "subfinder", "httpx", "nuclei")

    def test_unknown_objective_returns_all_base(self, engine):
        p = make_profile(technologies=[])
        tools = engine.select_optimal_tools(p, objective="bogus")
        # Should return all keys for WEB_APPLICATION (we used default target_type)
        assert len(tools) > 3

    def test_technology_wordpress_adds_wpscan(self, engine):
        # Use stealth so wpscan is NOT already in the selected list
        p = make_profile(technologies=[TechnologyStack.WORDPRESS])
        tools = engine.select_optimal_tools(p, objective="stealth")
        assert "wpscan" in tools

    def test_technology_php_adds_nikto(self, engine):
        p = make_profile(technologies=[TechnologyStack.PHP])
        tools = engine.select_optimal_tools(p, objective="stealth")
        assert "nikto" in tools


# ===================================================================
# Zone C — optimize_parameters (legacy path via disable_advanced_optimization)
#          All 20 _optimize_* methods
# ===================================================================

class TestOptimizeNmapParams:
    def test_web_app_scan(self, engine):
        p = make_profile()
        result = engine._optimize_nmap_params(p, {})
        assert result["ports"] == "80,443,8080,8443,8000,9000"

    def test_network_host_scan(self, engine):
        p = make_profile(target="192.168.1.1", target_type=TargetType.NETWORK_HOST)
        result = engine._optimize_nmap_params(p, {})
        assert result["scan_type"] == "-sS -O"

    def test_stealth_timing(self, engine):
        p = make_profile()
        result = engine._optimize_nmap_params(p, {"stealth": True})
        assert "T2" in result["additional_args"]

    def test_default_timing(self, engine):
        p = make_profile()
        result = engine._optimize_nmap_params(p, {})
        assert "T4" in result["additional_args"]


class TestOptimizeGobusterParams:
    def test_php_tech(self, engine):
        p = make_profile(technologies=[TechnologyStack.PHP])
        result = engine._optimize_gobuster_params(p, {})
        assert "-x php,html,txt,xml" in result["additional_args"]

    def test_dotnet_tech(self, engine):
        p = make_profile(technologies=[TechnologyStack.DOTNET])
        result = engine._optimize_gobuster_params(p, {})
        assert "-x asp,aspx,html,txt" in result["additional_args"]

    def test_java_tech(self, engine):
        p = make_profile(technologies=[TechnologyStack.JAVA])
        result = engine._optimize_gobuster_params(p, {})
        assert "-x jsp,html,txt,xml" in result["additional_args"]

    def test_other_tech(self, engine):
        p = make_profile(technologies=[TechnologyStack.NGINX])
        result = engine._optimize_gobuster_params(p, {})
        assert "-x html,php,txt,js" in result["additional_args"]

    def test_aggressive_mode(self, engine):
        p = make_profile(technologies=[TechnologyStack.PHP])
        result = engine._optimize_gobuster_params(p, {"aggressive": True})
        assert "-t 50" in result["additional_args"]

    def test_default_threads(self, engine):
        p = make_profile(technologies=[TechnologyStack.PHP])
        result = engine._optimize_gobuster_params(p, {})
        assert "-t 20" in result["additional_args"]


class TestOptimizeNucleiParams:
    def test_quick_severity(self, engine):
        p = make_profile()
        result = engine._optimize_nuclei_params(p, {"quick": True})
        assert result["severity"] == "critical,high"

    def test_default_severity(self, engine):
        p = make_profile()
        result = engine._optimize_nuclei_params(p, {})
        assert result["severity"] == "critical,high,medium"

    def test_wordpress_tag(self, engine):
        p = make_profile(technologies=[TechnologyStack.WORDPRESS])
        result = engine._optimize_nuclei_params(p, {})
        assert result["tags"] == "wordpress"

    def test_drupal_tag(self, engine):
        p = make_profile(technologies=[TechnologyStack.DRUPAL])
        result = engine._optimize_nuclei_params(p, {})
        assert result["tags"] == "drupal"

    def test_joomla_tag(self, engine):
        p = make_profile(technologies=[TechnologyStack.JOOMLA])
        result = engine._optimize_nuclei_params(p, {})
        assert result["tags"] == "joomla"

    def test_no_tech_tags(self, engine):
        p = make_profile(technologies=[TechnologyStack.APACHE])
        result = engine._optimize_nuclei_params(p, {})
        assert "tags" not in result


class TestOptimizeSqlmapParams:
    def test_php_dbms(self, engine):
        p = make_profile(technologies=[TechnologyStack.PHP])
        result = engine._optimize_sqlmap_params(p, {})
        assert "--dbms=mysql" in result["additional_args"]

    def test_dotnet_dbms(self, engine):
        p = make_profile(technologies=[TechnologyStack.DOTNET])
        result = engine._optimize_sqlmap_params(p, {})
        assert "--dbms=mssql" in result["additional_args"]

    def test_other_dbms(self, engine):
        p = make_profile(technologies=[TechnologyStack.NGINX])
        result = engine._optimize_sqlmap_params(p, {})
        assert result["additional_args"] == "--batch"

    def test_aggressive_mode(self, engine):
        p = make_profile(technologies=[TechnologyStack.PHP])
        result = engine._optimize_sqlmap_params(p, {"aggressive": True})
        assert "--level=3" in result["additional_args"]


class TestOptimizeFfufParams:
    def test_api_endpoint_codes(self, engine):
        p = make_profile(target="http://api.example.com", target_type=TargetType.API_ENDPOINT)
        result = engine._optimize_ffuf_params(p, {})
        assert result["match_codes"] == "200,201,202,204,301,302,401,403"

    def test_web_app_codes(self, engine):
        p = make_profile()
        result = engine._optimize_ffuf_params(p, {})
        assert result["match_codes"] == "200,204,301,302,307,401,403"

    def test_stealth_threads(self, engine):
        p = make_profile()
        result = engine._optimize_ffuf_params(p, {"stealth": True})
        assert "-t 10" in result["additional_args"]

    def test_default_threads(self, engine):
        p = make_profile()
        result = engine._optimize_ffuf_params(p, {})
        assert "-t 40" in result["additional_args"]


class TestOptimizeHydraParams:
    def test_ssh_service(self, engine):
        p = make_profile(open_ports=[22])
        result = engine._optimize_hydra_params(p, {})
        assert result["service"] == "ssh"

    def test_ftp_service(self, engine):
        p = make_profile(open_ports=[21])
        result = engine._optimize_hydra_params(p, {})
        assert result["service"] == "ftp"

    def test_http_service(self, engine):
        p = make_profile(open_ports=[80, 443])
        result = engine._optimize_hydra_params(p, {})
        assert result["service"] == "http-get"

    def test_default_service(self, engine):
        p = make_profile(open_ports=[8080])
        result = engine._optimize_hydra_params(p, {})
        assert result["service"] == "ssh"

    def test_conservative_params(self, engine):
        p = make_profile(open_ports=[22])
        result = engine._optimize_hydra_params(p, {})
        assert "-t 4 -w 30" in result["additional_args"]


class TestOptimizeRustscanParams:
    def test_stealth(self, engine):
        p = make_profile()
        result = engine._optimize_rustscan_params(p, {"stealth": True})
        assert result["ulimit"] == 1000
        assert result["batch_size"] == 500

    def test_aggressive(self, engine):
        p = make_profile()
        result = engine._optimize_rustscan_params(p, {"aggressive": True})
        assert result["ulimit"] == 10000

    def test_default(self, engine):
        p = make_profile()
        result = engine._optimize_rustscan_params(p, {})
        assert result["ulimit"] == 5000

    def test_comprehensive_scripts(self, engine):
        p = make_profile()
        result = engine._optimize_rustscan_params(p, {"objective": "comprehensive"})
        assert result["scripts"] is True

    def test_no_scripts_without_comprehensive(self, engine):
        p = make_profile()
        result = engine._optimize_rustscan_params(p, {})
        assert "scripts" not in result


class TestOptimizeMasscanParams:
    def test_stealth_rate(self, engine):
        p = make_profile()
        result = engine._optimize_masscan_params(p, {"stealth": True})
        assert result["rate"] == 100

    def test_aggressive_rate(self, engine):
        p = make_profile()
        result = engine._optimize_masscan_params(p, {"aggressive": True})
        assert result["rate"] == 10000

    def test_default_rate(self, engine):
        p = make_profile()
        result = engine._optimize_masscan_params(p, {})
        assert result["rate"] == 1000

    def test_banners_enabled(self, engine):
        p = make_profile()
        result = engine._optimize_masscan_params(p, {"service_detection": True})
        assert result["banners"] is True

    def test_banners_disabled(self, engine):
        p = make_profile()
        result = engine._optimize_masscan_params(p, {"service_detection": False})
        assert "banners" not in result or not result.get("banners")


class TestOptimizeNmapAdvancedParams:
    def test_stealth(self, engine):
        p = make_profile()
        result = engine._optimize_nmap_advanced_params(p, {"stealth": True})
        assert result["timing"] == "T2"
        assert result["stealth"] is True

    def test_aggressive(self, engine):
        p = make_profile()
        result = engine._optimize_nmap_advanced_params(p, {"aggressive": True})
        assert result["aggressive"] is True

    def test_default(self, engine):
        p = make_profile()
        result = engine._optimize_nmap_advanced_params(p, {})
        assert result["os_detection"] is True
        assert result["version_detection"] is True

    def test_web_app_nse(self, engine):
        p = make_profile(target_type=TargetType.WEB_APPLICATION)
        result = engine._optimize_nmap_advanced_params(p, {})
        assert result["nse_scripts"] == "http-*,ssl-*"

    def test_network_nse(self, engine):
        p = make_profile(target="192.168.1.1", target_type=TargetType.NETWORK_HOST)
        result = engine._optimize_nmap_advanced_params(p, {})
        assert result["nse_scripts"] == "default,discovery,safe"

    def test_other_target_type_no_nse(self, engine):
        p = make_profile(target_type=TargetType.API_ENDPOINT)
        result = engine._optimize_nmap_advanced_params(p, {})
        assert "nse_scripts" not in result


class TestOptimizeEnum4linuxNgParams:
    def test_basic_enum(self, engine):
        p = make_profile(target="192.168.1.1")
        result = engine._optimize_enum4linux_ng_params(p, {})
        assert result["shares"] is True
        assert result["users"] is True

    def test_with_auth(self, engine):
        p = make_profile(target="192.168.1.1")
        result = engine._optimize_enum4linux_ng_params(
            p, {"username": "admin", "password": "pass", "domain": "WORKGROUP"}
        )
        assert result["username"] == "admin"
        assert result["password"] == "pass"
        assert result["domain"] == "WORKGROUP"


class TestOptimizeAutoreconParams:
    def test_quick_scan(self, engine):
        p = make_profile()
        result = engine._optimize_autorecon_params(p, {"quick": True})
        assert result["port_scans"] == "top-100-ports"
        assert result["timeout"] == 180

    def test_comprehensive_scan(self, engine):
        p = make_profile()
        result = engine._optimize_autorecon_params(p, {"comprehensive": True})
        assert result["port_scans"] == "top-1000-ports"
        assert result["timeout"] == 600

    def test_output_dir(self, engine):
        p = make_profile(target="10.0.0.1")
        result = engine._optimize_autorecon_params(p, {})
        assert "autorecon_10_0_0_1" in result["output_dir"]

    def test_no_quick_no_comprehensive(self, engine):
        p = make_profile(target="10.0.0.1")
        # When both quick and comprehensive are False, defaults to comprehensive
        result = engine._optimize_autorecon_params(p, {"quick": False, "comprehensive": False})
        # The code doesn't set port_scans or timeout in this path
        # But still sets output_dir
        assert "output_dir" in result


class TestOptimizeGhidraParams:
    def test_quick(self, engine):
        p = make_profile(target="/bin/ls")
        result = engine._optimize_ghidra_params(p, {"quick": True})
        assert result["analysis_timeout"] == 120

    def test_comprehensive(self, engine):
        p = make_profile(target="/bin/ls")
        result = engine._optimize_ghidra_params(p, {"comprehensive": True})
        assert result["analysis_timeout"] == 600

    def test_project_name_from_binary(self, engine):
        p = make_profile(target="/usr/bin/ls")
        result = engine._optimize_ghidra_params(p, {})
        assert "ls" in result["project_name"]

    def test_no_quick_no_comprehensive(self, engine):
        p = make_profile(target="/bin/ls")
        # Both quick=False and comprehensive=False → hits else branch
        result = engine._optimize_ghidra_params(p, {"quick": False, "comprehensive": False})
        assert result["analysis_timeout"] == 300


class TestOptimizePwntoolsParams:
    def test_remote_exploit(self, engine):
        p = make_profile()
        result = engine._optimize_pwntools_params(p, {"remote_host": "10.0.0.1", "remote_port": 1337})
        assert result["exploit_type"] == "remote"
        assert result["target_host"] == "10.0.0.1"

    def test_local_exploit(self, engine):
        p = make_profile()
        result = engine._optimize_pwntools_params(p, {})
        assert result["exploit_type"] == "local"


class TestOptimizeRopperParams:
    def test_rop_gadgets(self, engine):
        p = make_profile()
        result = engine._optimize_ropper_params(p, {"exploit_type": "rop"})
        assert result["gadget_type"] == "rop"
        assert result["quality"] == 3

    def test_jop_gadgets(self, engine):
        p = make_profile()
        result = engine._optimize_ropper_params(p, {"exploit_type": "jop"})
        assert result["gadget_type"] == "jop"

    def test_all_gadgets(self, engine):
        p = make_profile()
        result = engine._optimize_ropper_params(p, {})
        assert result["gadget_type"] == "all"

    def test_with_arch(self, engine):
        p = make_profile()
        result = engine._optimize_ropper_params(p, {"arch": "x86_64"})
        assert result["arch"] == "x86_64"


class TestOptimizeAngrParams:
    def test_symbolic_execution(self, engine):
        p = make_profile()
        result = engine._optimize_angr_params(p, {"symbolic_execution": True})
        assert result["analysis_type"] == "symbolic"

    def test_cfg_analysis(self, engine):
        p = make_profile()
        result = engine._optimize_angr_params(p, {"symbolic_execution": False, "cfg_analysis": True})
        assert result["analysis_type"] == "cfg"

    def test_static_analysis(self, engine):
        p = make_profile()
        result = engine._optimize_angr_params(p, {"symbolic_execution": False, "cfg_analysis": False})
        assert result["analysis_type"] == "static"

    def test_find_address(self, engine):
        p = make_profile()
        result = engine._optimize_angr_params(p, {"find_address": "0x400000"})
        assert result["find_address"] == "0x400000"

    def test_avoid_addresses(self, engine):
        p = make_profile()
        result = engine._optimize_angr_params(p, {"avoid_addresses": "0x400000,0x400010"})
        assert result["avoid_addresses"] == "0x400000,0x400010"


class TestOptimizeProwlerParams:
    def test_default_provider(self, engine):
        p = make_profile()
        result = engine._optimize_prowler_params(p, {})
        assert result["provider"] == "aws"

    def test_custom_provider(self, engine):
        p = make_profile()
        result = engine._optimize_prowler_params(p, {"cloud_provider": "gcp"})
        assert result["provider"] == "gcp"

    def test_aws_profile_and_region(self, engine):
        p = make_profile()
        result = engine._optimize_prowler_params(
            p, {"aws_profile": "myprofile", "aws_region": "us-east-1"}
        )
        assert result["profile"] == "myprofile"
        assert result["region"] == "us-east-1"

    def test_output_format(self, engine):
        p = make_profile()
        result = engine._optimize_prowler_params(p, {})
        assert result["output_format"] == "json"


class TestOptimizeScoutSuiteParams:
    def test_default_provider(self, engine):
        p = make_profile()
        result = engine._optimize_scout_suite_params(p, {})
        assert result["provider"] == "aws"

    def test_custom_provider(self, engine):
        p = make_profile()
        result = engine._optimize_scout_suite_params(p, {"cloud_provider": "azure"})
        assert result["provider"] == "azure"

    def test_aws_profile(self, engine):
        p = make_profile()
        result = engine._optimize_scout_suite_params(p, {"aws_profile": "myprofile"})
        assert result["profile"] == "myprofile"

    def test_report_dir(self, engine):
        p = make_profile()
        result = engine._optimize_scout_suite_params(p, {})
        assert "scout-suite" in result["report_dir"]


class TestOptimizeKubeHunterParams:
    def test_kubernetes_target(self, engine):
        p = make_profile()
        result = engine._optimize_kube_hunter_params(p, {"kubernetes_target": "10.0.0.1"})
        assert result["target"] == "10.0.0.1"

    def test_cidr(self, engine):
        p = make_profile()
        result = engine._optimize_kube_hunter_params(p, {"cidr": "10.0.0.0/24"})
        assert result["cidr"] == "10.0.0.0/24"

    def test_interface(self, engine):
        p = make_profile()
        result = engine._optimize_kube_hunter_params(p, {"interface": "eth0"})
        assert result["interface"] == "eth0"

    def test_active_hunting(self, engine):
        p = make_profile()
        result = engine._optimize_kube_hunter_params(p, {"active_hunting": True})
        assert result["active"] == "true"

    def test_default_no_active(self, engine):
        p = make_profile()
        result = engine._optimize_kube_hunter_params(p, {})
        assert "active" not in result


class TestOptimizeTrivyParams:
    def test_registry_image(self, engine):
        p = make_profile(target="docker.io/nginx:latest")
        result = engine._optimize_trivy_params(p, {})
        assert result["scan_type"] == "image"

    def test_filesystem_with_isdir_true(self, engine):
        p = make_profile(target="/some/dir")
        with patch("os.path.isdir", return_value=True):
            result = engine._optimize_trivy_params(p, {})
        assert result["scan_type"] == "fs"

    def test_short_name_default_image(self, engine):
        p = make_profile(target="nginx:latest")
        result = engine._optimize_trivy_params(p, {})
        assert result["scan_type"] == "image"

    def test_severity_from_context(self, engine):
        p = make_profile(target="nginx:latest")
        result = engine._optimize_trivy_params(p, {"severity": "LOW,MEDIUM"})
        assert result["severity"] == "LOW,MEDIUM"

    def test_default_severity(self, engine):
        p = make_profile(target="nginx:latest")
        result = engine._optimize_trivy_params(p, {})
        assert result["severity"] == "HIGH,CRITICAL"


class TestOptimizeCheckovParams:
    def test_framework_from_context(self, engine):
        p = make_profile(target="/infra")
        result = engine._optimize_checkov_params(p, {"framework": "terraform"})
        assert result["framework"] == "terraform"

    def test_autodetect_terraform(self, engine):
        p = make_profile(target="/infra")
        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=["main.tf", "vars.tf"]):
                with patch("os.path.isfile", return_value=True):
                    result = engine._optimize_checkov_params(p, {})
        assert result["framework"] == "terraform"

    def test_autodetect_kubernetes(self, engine):
        p = make_profile(target="/infra")
        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=["deployment.yaml", "service.yml"]):
                with patch("os.path.isfile", return_value=True):
                    result = engine._optimize_checkov_params(p, {})
        assert result["framework"] == "kubernetes"

    def test_no_framework_auto(self, engine):
        p = make_profile(target="/infra")
        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=["readme.md"]):
                with patch("os.path.isfile", return_value=True):
                    result = engine._optimize_checkov_params(p, {})
        assert "framework" not in result

    def test_not_a_directory(self, engine):
        p = make_profile(target="/infra")
        with patch("os.path.isdir", return_value=False):
            result = engine._optimize_checkov_params(p, {})
        assert "framework" not in result

    def test_output_format(self, engine):
        p = make_profile(target="/infra")
        result = engine._optimize_checkov_params(p, {})
        assert result["output_format"] == "json"


class TestOptimizeParametersDispatch:
    def test_dispatches_nmap(self, engine):
        p = make_profile()
        result = engine.optimize_parameters("nmap", p)
        assert "target" in result

    def test_dispatches_gobuster(self, engine):
        p = make_profile()
        result = engine.optimize_parameters("gobuster", p)
        assert "url" in result

    def test_dispatches_nuclei(self, engine):
        p = make_profile()
        result = engine.optimize_parameters("nuclei", p)
        assert "target" in result

    def test_dispatches_sqlmap(self, engine):
        p = make_profile()
        result = engine.optimize_parameters("sqlmap", p)
        assert "url" in result

    def test_dispatches_ffuf(self, engine):
        p = make_profile()
        result = engine.optimize_parameters("ffuf", p)
        assert "url" in result

    def test_dispatches_hydra(self, engine):
        p = make_profile()
        result = engine.optimize_parameters("hydra", p)
        assert "target" in result

    def test_dispatches_rustscan(self, engine):
        p = make_profile()
        result = engine.optimize_parameters("rustscan", p)
        assert "target" in result

    def test_dispatches_masscan(self, engine):
        p = make_profile()
        result = engine.optimize_parameters("masscan", p)
        assert "target" in result

    def test_dispatches_nmap_advanced(self, engine):
        p = make_profile()
        result = engine.optimize_parameters("nmap-advanced", p)
        assert "target" in result

    def test_dispatches_enum4linux_ng(self, engine):
        p = make_profile(target="10.0.0.1")
        result = engine.optimize_parameters("enum4linux-ng", p)
        assert "target" in result

    def test_dispatches_autorecon(self, engine):
        p = make_profile()
        result = engine.optimize_parameters("autorecon", p)
        assert "target" in result

    def test_dispatches_ghidra(self, engine):
        p = make_profile(target="/bin/ls")
        result = engine.optimize_parameters("ghidra", p)
        assert "binary" in result

    def test_dispatches_pwntools(self, engine):
        p = make_profile(target="/bin/ls")
        result = engine.optimize_parameters("pwntools", p)
        assert "target_binary" in result

    def test_dispatches_ropper(self, engine):
        p = make_profile(target="/bin/ls")
        result = engine.optimize_parameters("ropper", p)
        assert "binary" in result

    def test_dispatches_angr(self, engine):
        p = make_profile(target="/bin/ls")
        result = engine.optimize_parameters("angr", p)
        assert "binary" in result

    def test_dispatches_prowler(self, engine):
        p = make_profile()
        result = engine.optimize_parameters("prowler", p)
        assert "provider" in result

    def test_dispatches_scout_suite(self, engine):
        p = make_profile()
        result = engine.optimize_parameters("scout-suite", p)
        assert "provider" in result

    def test_dispatches_kube_hunter(self, engine):
        p = make_profile()
        result = engine.optimize_parameters("kube-hunter", p)
        assert "report" in result

    def test_dispatches_trivy(self, engine):
        p = make_profile(target="nginx:latest")
        result = engine.optimize_parameters("trivy", p)
        assert "scan_type" in result

    def test_dispatches_checkov(self, engine):
        p = make_profile(target="/infra")
        with patch("os.path.isdir", return_value=False):
            result = engine.optimize_parameters("checkov", p)
        assert "directory" in result

    def test_unknown_tool_falls_back_to_advanced(self, engine):
        p = make_profile()
        result = engine.optimize_parameters("nonexistent-tool", p)
        # Should return result from parameter_optimizer.optimize_parameters_advanced
        assert isinstance(result, dict)

    def test_unknown_tool_with_mocked_advanced(self, engine):
        p = make_profile(target="http://example.com")
        with patch.object(parameter_optimizer, "optimize_parameters_advanced", return_value={"mocked": True}):
            result = engine.optimize_parameters("nonexistent-tool", p)
        assert result == {"mocked": True}


# ===================================================================
# Zone D — _effective_score
# ===================================================================

class TestEffectiveScore:
    def test_baseline_score(self, engine):
        score = engine._effective_score("nmap", TargetType.WEB_APPLICATION.value)
        assert score == pytest.approx(0.8, abs=0.01)

    def test_blended_with_stats(self, engine):
        with patch.object(_tool_stats, "blended_effectiveness", return_value=0.95):
            score = engine._effective_score("nmap", TargetType.WEB_APPLICATION.value)
        assert score == 0.95

    def test_unknown_tool_default(self, engine):
        score = engine._effective_score("no-such-tool", TargetType.WEB_APPLICATION.value)
        assert score == 0.5


# ===================================================================
# Zone E — create_attack_chain
# ===================================================================

class TestCreateAttackChain:
    def test_web_application_quick(self, engine):
        p = make_profile(confidence_score=0.8)
        chain = engine.create_attack_chain(p, objective="quick")
        assert isinstance(chain, AttackChain)
        assert len(chain.steps) > 0

    def test_web_application_comprehensive(self, engine):
        p = make_profile(confidence_score=0.8)
        chain = engine.create_attack_chain(p, objective="comprehensive")
        assert len(chain.steps) > 0

    def test_api_endpoint(self, engine):
        p = make_profile(target="http://api.example.com/v1",
                         target_type=TargetType.API_ENDPOINT,
                         confidence_score=0.8)
        chain = engine.create_attack_chain(p)
        assert len(chain.steps) > 0

    def test_network_host_comprehensive(self, engine):
        p = make_profile(target="10.0.0.1",
                         target_type=TargetType.NETWORK_HOST,
                         confidence_score=0.8)
        chain = engine.create_attack_chain(p, objective="comprehensive")
        assert len(chain.steps) > 0

    def test_network_host_quick(self, engine):
        p = make_profile(target="10.0.0.1",
                         target_type=TargetType.NETWORK_HOST,
                         confidence_score=0.8)
        chain = engine.create_attack_chain(p, objective="quick")
        assert len(chain.steps) > 0

    def test_binary_file_ctf(self, engine):
        p = make_profile(target="challenge.bin",
                         target_type=TargetType.BINARY_FILE,
                         confidence_score=0.8)
        chain = engine.create_attack_chain(p, objective="ctf")
        assert len(chain.steps) > 0

    def test_binary_file_default(self, engine):
        p = make_profile(target="challenge.bin",
                         target_type=TargetType.BINARY_FILE,
                         confidence_score=0.8)
        chain = engine.create_attack_chain(p, objective="comprehensive")
        assert len(chain.steps) > 0

    def test_cloud_service_aws(self, engine):
        p = make_profile(target="mybucket.amazonaws.com",
                         target_type=TargetType.CLOUD_SERVICE,
                         confidence_score=0.8)
        chain = engine.create_attack_chain(p, objective="aws")
        assert len(chain.steps) > 0

    def test_cloud_service_kubernetes(self, engine):
        p = make_profile(target="mybucket.amazonaws.com",
                         target_type=TargetType.CLOUD_SERVICE,
                         confidence_score=0.8)
        chain = engine.create_attack_chain(p, objective="kubernetes")
        assert len(chain.steps) > 0

    def test_cloud_service_containers(self, engine):
        p = make_profile(target="mybucket.amazonaws.com",
                         target_type=TargetType.CLOUD_SERVICE,
                         confidence_score=0.8)
        chain = engine.create_attack_chain(p, objective="containers")
        assert len(chain.steps) > 0

    def test_cloud_service_iac(self, engine):
        p = make_profile(target="mybucket.amazonaws.com",
                         target_type=TargetType.CLOUD_SERVICE,
                         confidence_score=0.8)
        chain = engine.create_attack_chain(p, objective="iac")
        assert len(chain.steps) > 0

    def test_cloud_service_multi_cloud(self, engine):
        p = make_profile(target="mybucket.amazonaws.com",
                         target_type=TargetType.CLOUD_SERVICE,
                         confidence_score=0.8)
        chain = engine.create_attack_chain(p, objective="other")
        assert len(chain.steps) > 0

    def test_unknown_bug_bounty_recon(self, engine):
        p = make_profile(target="example.com",
                         target_type=TargetType.UNKNOWN,
                         confidence_score=0.8)
        chain = engine.create_attack_chain(p, objective="bug_bounty_recon")
        assert len(chain.steps) > 0

    def test_unknown_bug_bounty_hunting(self, engine):
        p = make_profile(target="example.com",
                         target_type=TargetType.UNKNOWN,
                         confidence_score=0.8)
        chain = engine.create_attack_chain(p, objective="bug_bounty_hunting")
        assert len(chain.steps) > 0

    def test_unknown_bug_bounty_high_impact(self, engine):
        p = make_profile(target="example.com",
                         target_type=TargetType.UNKNOWN,
                         confidence_score=0.8)
        chain = engine.create_attack_chain(p, objective="bug_bounty_high_impact")
        assert len(chain.steps) > 0

    def test_unknown_default_web_recon(self, engine):
        p = make_profile(target="example.com",
                         target_type=TargetType.UNKNOWN,
                         confidence_score=0.8)
        chain = engine.create_attack_chain(p, objective="some_rando")
        assert len(chain.steps) > 0

    def test_chain_metrics_calculated(self, engine):
        p = make_profile(target="http://example.com",
                         target_type=TargetType.WEB_APPLICATION,
                         confidence_score=0.9,
                         attack_surface_score=8.0)
        chain = engine.create_attack_chain(p, objective="quick")
        assert chain.success_probability > 0
        assert chain.risk_level == "unknown"
