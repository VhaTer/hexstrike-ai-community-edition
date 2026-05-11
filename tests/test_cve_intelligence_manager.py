import json
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from server_core.intelligence.cve_intelligence_manager import CVEIntelligenceManager


@pytest.fixture
def mgr():
    return CVEIntelligenceManager()


# ---------------------------------------------------------------------------
# Static rendering methods
# ---------------------------------------------------------------------------

class TestRenderProgressBar:
    def test_style_cyber(self):
        result = CVEIntelligenceManager.render_progress_bar(0.5, style='cyber')
        assert '50.0%' in result

    def test_style_matrix(self):
        result = CVEIntelligenceManager.render_progress_bar(0.3, style='matrix')
        assert '30.0%' in result

    def test_style_neon(self):
        result = CVEIntelligenceManager.render_progress_bar(0.8, style='neon')
        assert '80.0%' in result

    def test_style_unknown(self):
        result = CVEIntelligenceManager.render_progress_bar(0.5, style='unknown')
        assert '50.0%' in result

    def test_clamp_zero(self):
        result = CVEIntelligenceManager.render_progress_bar(-0.5)
        assert '0.0%' in result

    def test_clamp_one(self):
        result = CVEIntelligenceManager.render_progress_bar(1.5)
        assert '100.0%' in result

    def test_with_eta_and_speed(self):
        result = CVEIntelligenceManager.render_progress_bar(0.5, eta=120, speed="100 req/s")
        assert 'ETA: 120' in result
        assert '100 req/s' in result

    def test_with_label(self):
        result = CVEIntelligenceManager.render_progress_bar(0.5, label="Scanning")
        assert 'Scanning' in result


class TestRenderVulnerabilityCard:
    def test_severity_critical(self):
        vuln = {"severity": "critical", "title": "RCE in kernel", "url": "http://example.com",
                "description": "Remote code execution vulnerability", "cvss_score": 9.5}
        result = CVEIntelligenceManager.render_vulnerability_card(vuln)
        assert 'CRITICAL' in result
        assert 'RCE in kernel' in result
        assert '9.5' in result

    def test_severity_high(self):
        vuln = {"severity": "high", "title": "Privilege escalation", "url": "http://ex.com",
                "description": "Local privilege escalation", "cvss_score": 7.5}
        result = CVEIntelligenceManager.render_vulnerability_card(vuln)
        assert 'HIGH' in result

    def test_severity_medium(self):
        vuln = {"severity": "medium", "title": "XSS in plugin", "url": "http://ex.com",
                "description": "Cross-site scripting", "cvss_score": 5.0}
        result = CVEIntelligenceManager.render_vulnerability_card(vuln)
        assert 'MEDIUM' in result

    def test_severity_low(self):
        vuln = {"severity": "low", "title": "Info leak", "url": "http://ex.com",
                "description": "Minor information disclosure", "cvss_score": 2.5}
        result = CVEIntelligenceManager.render_vulnerability_card(vuln)
        assert 'LOW' in result

    def test_severity_info(self):
        vuln = {"severity": "info", "title": "Info", "url": "http://ex.com",
                "description": "Informational", "cvss_score": 0.0}
        result = CVEIntelligenceManager.render_vulnerability_card(vuln)
        assert 'INFO' in result

    def test_severity_unknown(self):
        vuln = {"severity": "unknown", "title": "Test", "url": "http://ex.com",
                "description": "Unknown", "cvss_score": 0.0}
        result = CVEIntelligenceManager.render_vulnerability_card(vuln)
        assert 'UNKNOWN' in result


class TestCreateLiveDashboard:
    def test_empty(self):
        result = CVEIntelligenceManager.create_live_dashboard({})
        assert 'No active processes' in result

    def test_with_running_process(self):
        procs = {1234: {"command": "nmap -sV target", "status": "running",
                        "progress": 0.6, "runtime": 30, "eta": 20}}
        result = CVEIntelligenceManager.create_live_dashboard(procs)
        assert 'PID 1234' in result
        assert 'RUNNING' in result

    def test_with_paused_process(self):
        procs = {5678: {"command": "sqlmap", "status": "paused",
                        "progress": 0.5, "runtime": 10, "eta": 0}}
        result = CVEIntelligenceManager.create_live_dashboard(procs)
        assert 'PID 5678' in result
        assert 'PAUSED' in result

    def test_multiple_processes(self):
        procs = {
            1: {"command": "nmap", "status": "running", "progress": 0.8, "runtime": 5, "eta": 2},
            2: {"command": "gobuster", "status": "completed", "progress": 1.0, "runtime": 60, "eta": 0},
            3: {"command": "hydra", "status": "terminated", "progress": 0.3, "runtime": 120, "eta": 0},
        }
        result = CVEIntelligenceManager.create_live_dashboard(procs)
        assert 'PID 1' in result
        assert 'PID 2' in result
        assert 'PID 3' in result
        assert 'RUNNING' in result
        assert 'COMPLETED' in result
        assert 'TERMINATED' in result


# ---------------------------------------------------------------------------
# fetch_latest_cves — mocked network
# ---------------------------------------------------------------------------

def make_nvd_response(cve_id="CVE-2024-0001", severity="CRITICAL",
                      cvss_score=9.5, cvss_version="V31", description="Test vuln"):
    vuln = {
        "cve": {
            "id": cve_id,
            "published": "2024-01-01T00:00:00.000",
            "lastModified": "2024-01-02T00:00:00.000",
            "descriptions": [{"lang": "en", "value": description}],
            "references": [{"url": "https://example.com/ref"}],
            "metrics": {},
            "configurations": [
                {"nodes": [{"cpeMatch": [{"criteria": "cpe:2.3:a:vendor:product:1.0:*:*:*:*:*:*:*"}]}]}
            ],
        }
    }
    if cvss_version == "V31":
        vuln["cve"]["metrics"]["cvssMetricV31"] = [{
            "cvssData": {
                "baseScore": cvss_score,
                "baseSeverity": severity,
            }
        }]
    elif cvss_version == "V30":
        vuln["cve"]["metrics"]["cvssMetricV30"] = [{
            "cvssData": {
                "baseScore": cvss_score,
                "baseSeverity": severity,
            }
        }]
    elif cvss_version == "V2":
        vuln["cve"]["metrics"]["cvssMetricV2"] = [{
            "cvssData": {"baseScore": cvss_score}
        }]
    return {"vulnerabilities": [{"cve": vuln["cve"]}]}


def make_cve_detail(cve_id="CVE-2024-0001", severity="CRITICAL", cvss_score=9.5,
                     cvss_version="V31", attack_vector="NETWORK", attack_complexity="LOW",
                     privileges_required="NONE", user_interaction="NONE",
                     exploitability_subscore=3.9, description="Remote code execution vulnerability",
                     references=None):
    refs = references or [{"url": "https://example.com"}]
    vuln = {
        "cve": {
            "id": cve_id,
            "published": "2024-01-01T00:00:00.000",
            "lastModified": "2024-01-02T00:00:00.000",
            "descriptions": [{"lang": "en", "value": description}],
            "references": refs,
            "metrics": {},
        }
    }
    if cvss_version == "V31":
        vuln["cve"]["metrics"]["cvssMetricV31"] = [{
            "cvssData": {
                "baseScore": cvss_score,
                "baseSeverity": severity,
                "attackVector": attack_vector,
                "attackComplexity": attack_complexity,
                "privilegesRequired": privileges_required,
                "userInteraction": user_interaction,
                "exploitabilityScore": exploitability_subscore,
            }
        }]
    elif cvss_version == "V30":
        vuln["cve"]["metrics"]["cvssMetricV30"] = [{
            "cvssData": {
                "baseScore": cvss_score,
                "baseSeverity": severity,
                "attackVector": attack_vector,
                "attackComplexity": attack_complexity,
                "privilegesRequired": privileges_required,
                "userInteraction": user_interaction,
                "exploitabilityScore": exploitability_subscore,
            }
        }]
    elif cvss_version == "V2":
        vuln["cve"]["metrics"]["cvssMetricV2"] = [{
            "cvssData": {
                "baseScore": cvss_score,
                "attackVector": attack_vector,
                "attackComplexity": attack_complexity,
            }
        }]
    return {"vulnerabilities": [vuln]}


class TestFetchLatestCVEs:
    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    @patch("server_core.intelligence.cve_intelligence_manager.time.sleep")
    def test_fetch_success(self, mock_sleep, mock_get, mgr):
        nvd_data = make_nvd_response("CVE-2024-0001", "CRITICAL", 9.5, "V31")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = nvd_data
        mock_get.return_value = mock_response

        result = mgr.fetch_latest_cves(hours=24, severity_filter="CRITICAL")

        assert result["success"] is True
        assert result["total_found"] == 1
        assert result["cves"][0]["cve_id"] == "CVE-2024-0001"
        assert result["cves"][0]["severity"] == "CRITICAL"

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    @patch("server_core.intelligence.cve_intelligence_manager.time.sleep")
    def test_fetch_cvss_v30(self, mock_sleep, mock_get, mgr):
        nvd_data = make_nvd_response("CVE-2024-0002", "HIGH", 7.5, "V30")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = nvd_data
        mock_get.return_value = mock_response

        result = mgr.fetch_latest_cves(hours=24, severity_filter="HIGH")

        assert result["success"] is True
        assert result["cves"][0]["severity"] == "HIGH"

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    @patch("server_core.intelligence.cve_intelligence_manager.time.sleep")
    def test_fetch_cvss_v2(self, mock_sleep, mock_get, mgr):
        nvd_data = make_nvd_response("CVE-2024-0003", "CRITICAL", 9.0, "V2")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = nvd_data
        mock_get.return_value = mock_response

        result = mgr.fetch_latest_cves(hours=24, severity_filter="CRITICAL")

        assert result["success"] is True
        assert result["cves"][0]["severity"] == "CRITICAL"

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    @patch("server_core.intelligence.cve_intelligence_manager.time.sleep")
    def test_fetch_v2_severity_high(self, mock_sleep, mock_get, mgr):
        nvd_data = make_nvd_response("CVE-2024-0011", "HIGH", 7.5, "V2")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = nvd_data
        mock_get.return_value = mock_resp
        result = mgr.fetch_latest_cves(hours=24, severity_filter="HIGH")
        assert result["cves"][0]["severity"] == "HIGH"

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    @patch("server_core.intelligence.cve_intelligence_manager.time.sleep")
    def test_fetch_v2_severity_medium(self, mock_sleep, mock_get, mgr):
        nvd_data = make_nvd_response("CVE-2024-0012", "MEDIUM", 5.0, "V2")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = nvd_data
        mock_get.return_value = mock_resp
        result = mgr.fetch_latest_cves(hours=24, severity_filter="MEDIUM")
        assert result["cves"][0]["severity"] == "MEDIUM"

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    @patch("server_core.intelligence.cve_intelligence_manager.time.sleep")
    def test_fetch_v2_severity_low(self, mock_sleep, mock_get, mgr):
        nvd_data = make_nvd_response("CVE-2024-0013", "LOW", 2.0, "V2")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = nvd_data
        mock_get.return_value = mock_resp
        result = mgr.fetch_latest_cves(hours=24, severity_filter="LOW")
        assert result["cves"][0]["severity"] == "LOW"

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    @patch("server_core.intelligence.cve_intelligence_manager.time.sleep")
    def test_fetch_severity_mismatch_filtered_out(self, mock_sleep, mock_get, mgr):
        nvd_data = make_nvd_response("CVE-2024-0004", "LOW", 3.0, "V31")
        mock_first = MagicMock()
        mock_first.status_code = 200
        mock_first.json.return_value = nvd_data

        broad_empty = MagicMock()
        broad_empty.status_code = 200
        broad_empty.json.return_value = {"vulnerabilities": []}

        mock_get.side_effect = [mock_first, broad_empty]

        result = mgr.fetch_latest_cves(hours=24, severity_filter="CRITICAL")

        assert result["total_found"] == 0

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    @patch("server_core.intelligence.cve_intelligence_manager.time.sleep")
    def test_fetch_all_severity(self, mock_sleep, mock_get, mgr):
        nvd_data = make_nvd_response("CVE-2024-0005", "LOW", 3.0, "V31")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = nvd_data
        mock_get.return_value = mock_response

        result = mgr.fetch_latest_cves(hours=24, severity_filter="ALL")

        assert result["total_found"] == 1

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    @patch("server_core.intelligence.cve_intelligence_manager.time.sleep")
    def test_fetch_empty_then_broader_search(self, mock_sleep, mock_get, mgr):
        mock_empty = MagicMock()
        mock_empty.status_code = 200
        mock_empty.json.return_value = {"vulnerabilities": []}

        broad_data = make_nvd_response("CVE-2024-0999", "CRITICAL", 9.5, "V31")
        mock_broad = MagicMock()
        mock_broad.status_code = 200
        mock_broad.json.return_value = broad_data

        mock_get.side_effect = [mock_empty, mock_broad]

        result = mgr.fetch_latest_cves(hours=24, severity_filter="CRITICAL")

        assert result["success"] is True
        assert result["total_found"] >= 1

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    @patch("server_core.intelligence.cve_intelligence_manager.time.sleep")
    def test_fetch_broader_search_fails(self, mock_sleep, mock_get, mgr):
        mock_empty = MagicMock()
        mock_empty.status_code = 200
        mock_empty.json.return_value = {"vulnerabilities": []}

        mock_broad = MagicMock()
        mock_broad.status_code = 500

        mock_get.side_effect = [mock_empty, mock_broad]

        result = mgr.fetch_latest_cves(hours=24, severity_filter="CRITICAL")

        assert result["success"] is True
        assert result["total_found"] == 0

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    @patch("server_core.intelligence.cve_intelligence_manager.time.sleep")
    def test_fetch_api_error(self, mock_sleep, mock_get, mgr):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response

        result = mgr.fetch_latest_cves(hours=24)

        assert result["success"] is True
        assert result["total_found"] == 0

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    @patch("server_core.intelligence.cve_intelligence_manager.time.sleep")
    def test_fetch_network_error(self, mock_sleep, mock_get, mgr):
        from requests.exceptions import ConnectionError
        mock_get.side_effect = ConnectionError("Connection refused")

        result = mgr.fetch_latest_cves(hours=24)

        assert result["success"] is True
        assert result["total_found"] == 0

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    @patch("server_core.intelligence.cve_intelligence_manager.time.sleep")
    def test_fetch_top_level_exception(self, mock_sleep, mock_get, mgr):
        mock_get.side_effect = RuntimeError("Unexpected crash")

        result = mgr.fetch_latest_cves(hours=24)

        assert result["success"] is False
        assert "Unexpected crash" in result["error"]


# ---------------------------------------------------------------------------
# analyze_cve_exploitability — mocked network
# ---------------------------------------------------------------------------

class TestAnalyzeCVEExploitability:
    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    def test_analyze_success_v31(self, mock_get, mgr):
        data = make_cve_detail("CVE-2024-0001", "CRITICAL", 9.5, "V31")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = data
        mock_get.return_value = mock_response

        result = mgr.analyze_cve_exploitability("CVE-2024-0001")

        assert result["success"] is True
        assert result["severity"] == "CRITICAL"
        assert result["cvss_score"] == 9.5
        assert result["exploitability_level"] == "HIGH"
        assert result["attack_vector"] == "NETWORK"

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    def test_analyze_success_v30(self, mock_get, mgr):
        data = make_cve_detail("CVE-2024-0002", "HIGH", 7.5, "V30",
                                     attack_complexity="HIGH", privileges_required="LOW",
                                     user_interaction="REQUIRED", exploitability_subscore=2.0)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = data
        mock_get.return_value = mock_response

        result = mgr.analyze_cve_exploitability("CVE-2024-0002")

        assert result["success"] is True
        assert result["severity"] == "HIGH"
        assert result["cvss_score"] == 7.5

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    def test_analyze_success_v2(self, mock_get, mgr):
        data = make_cve_detail("CVE-2024-0003", "CRITICAL", 8.0, "V2",
                                     attack_vector="ADJACENT_NETWORK")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = data
        mock_get.return_value = mock_response

        result = mgr.analyze_cve_exploitability("CVE-2024-0003")

        assert result["success"] is True
        assert result["cvss_score"] == 0.0
        assert result["attack_vector"] == "UNKNOWN"

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    def test_analyze_no_metrics_uses_calculated_score(self, mock_get, mgr):
        vuln = {
            "cve": {
                "id": "CVE-2024-0004",
                "published": "",
                "lastModified": "",
                "descriptions": [{"lang": "en", "value": "Generic vulnerability"}],
                "references": [],
            }
        }
        response = {"vulnerabilities": [vuln]}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = response
        mock_get.return_value = mock_resp

        result = mgr.analyze_cve_exploitability("CVE-2024-0004")

        assert result["success"] is True
        assert result["cvss_score"] == 0.0
        assert result["attack_vector"] == "UNKNOWN"
        assert result["exploitability_level"] == "VERY_LOW"

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    def test_analyze_non_200(self, mock_get, mgr):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        result = mgr.analyze_cve_exploitability("CVE-2024-9999")

        assert result["success"] is False
        assert "404" in result["error"]

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    def test_analyze_no_vulnerabilities(self, mock_get, mgr):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_get.return_value = mock_resp

        result = mgr.analyze_cve_exploitability("CVE-2024-9999")

        assert result["success"] is False
        assert "not found" in result["error"]

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    def test_analyze_network_error(self, mock_get, mgr):
        from requests.exceptions import Timeout
        mock_get.side_effect = Timeout("Request timed out")

        result = mgr.analyze_cve_exploitability("CVE-2024-0001")

        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    def test_analyze_exploit_keywords_rce(self, mock_get, mgr):
        data = make_cve_detail("CVE-2024-0005", "CRITICAL", 9.5, "V31",
                                     description="Remote code execution in network service")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = data
        mock_get.return_value = mock_resp

        result = mgr.analyze_cve_exploitability("CVE-2024-0005")

        assert result["exploitability_score"] > 0.8

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    def test_analyze_exploit_keywords_auth_bypass(self, mock_get, mgr):
        data = make_cve_detail("CVE-2024-0006", "HIGH", 7.5, "V31",
                                     description="Authentication bypass in admin panel")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = data
        mock_get.return_value = mock_resp

        result = mgr.analyze_cve_exploitability("CVE-2024-0006")

        assert result["exploitability_score"] > 0.7

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    def test_analyze_public_exploit_in_references(self, mock_get, mgr):
        refs = [{"url": "https://www.exploit-db.com/exploits/12345"}]
        data = make_cve_detail("CVE-2024-0007", "CRITICAL", 9.5, "V31",
                                     references=refs)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = data
        mock_get.return_value = mock_resp

        result = mgr.analyze_cve_exploitability("CVE-2024-0007")

        assert result["exploit_availability"]["public_exploits"] is True
        assert result["exploit_availability"]["exploit_maturity"] == "PROOF_OF_CONCEPT"

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    def test_analyze_weaponization_high(self, mock_get, mgr):
        refs = [{"url": "https://github.com/test/exploit"}]
        data = make_cve_detail("CVE-2024-0008", "CRITICAL", 9.5, "V31",
                                     attack_vector="NETWORK", attack_complexity="LOW",
                                     exploitability_subscore=3.9, references=refs)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = data
        mock_get.return_value = mock_resp

        result = mgr.analyze_cve_exploitability("CVE-2024-0008")

        assert result["exploit_availability"]["weaponization_level"] == "HIGH"

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    def test_analyze_priority_immediate(self, mock_get, mgr):
        data = make_cve_detail("CVE-2024-0009", "CRITICAL", 9.5, "V31",
                                     exploitability_subscore=3.9)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = data
        mock_get.return_value = mock_resp

        result = mgr.analyze_cve_exploitability("CVE-2024-0009")

        assert result["threat_intelligence"]["recommended_priority"] == "IMMEDIATE"

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    def test_analyze_priority_low(self, mock_get, mgr):
        data = make_cve_detail("CVE-2024-0010", "LOW", 2.5, "V31",
                                     attack_vector="PHYSICAL", attack_complexity="HIGH",
                                     privileges_required="HIGH", user_interaction="REQUIRED",
                                     exploitability_subscore=0.5)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = data
        mock_get.return_value = mock_resp

        result = mgr.analyze_cve_exploitability("CVE-2024-0010")

        assert result["threat_intelligence"]["recommended_priority"] == "LOW"

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    def test_analyze_top_level_exception(self, mock_get, mgr):
        mock_get.side_effect = ValueError("Something broke")

        result = mgr.analyze_cve_exploitability("CVE-2024-0001")

        assert result["success"] is False


# ---------------------------------------------------------------------------
# search_existing_exploits — mocked network
# ---------------------------------------------------------------------------

class TestSearchExistingExploits:

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    @patch("server_core.intelligence.cve_intelligence_manager.time.sleep")
    def test_search_github_success(self, mock_sleep, mock_get, mgr):
        responses = []

        gh_resp = MagicMock()
        gh_resp.status_code = 200
        gh_resp.json.return_value = {
            "items": [{
                "id": 1,
                "name": "CVE-2024-0001-exploit",
                "description": "PoC for CVE-2024-0001",
                "owner": {"login": "testuser"},
                "created_at": "2024-01-01",
                "updated_at": "2024-01-02",
                "html_url": "https://github.com/test/CVE-2024-0001-exploit",
                "stargazers_count": 60,
                "forks_count": 15,
            }]
        }
        responses.append(gh_resp)

        nvd_resp = MagicMock()
        nvd_resp.status_code = 200
        nvd_resp.json.return_value = {"vulnerabilities": []}
        responses.append(nvd_resp)

        msf_resp = MagicMock()
        msf_resp.status_code = 200
        msf_resp.json.return_value = {"items": []}
        responses.append(msf_resp)

        mock_get.side_effect = responses

        result = mgr.search_existing_exploits("CVE-2024-0001")

        assert result["success"] is True
        assert result["exploits_found"] >= 1
        assert "github" in result["sources_searched"]

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    @patch("server_core.intelligence.cve_intelligence_manager.time.sleep")
    def test_search_github_non_matching(self, mock_sleep, mock_get, mgr):
        responses = []

        gh_resp = MagicMock()
        gh_resp.status_code = 200
        gh_resp.json.return_value = {
            "items": [{
                "id": 2,
                "name": "unrelated-repo",
                "description": "Some other project",
                "owner": {"login": "other"},
                "html_url": "https://github.com/other/repo",
                "stargazers_count": 5,
                "forks_count": 1,
            }]
        }
        responses.append(gh_resp)

        nvd_resp = MagicMock()
        nvd_resp.status_code = 200
        nvd_resp.json.return_value = {"vulnerabilities": []}
        responses.append(nvd_resp)

        msf_resp = MagicMock()
        msf_resp.status_code = 200
        msf_resp.json.return_value = {"items": []}
        responses.append(msf_resp)

        mock_get.side_effect = responses

        result = mgr.search_existing_exploits("CVE-2024-0001")

        assert result["success"] is True

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    @patch("server_core.intelligence.cve_intelligence_manager.time.sleep")
    def test_search_nvd_references_exploit_db(self, mock_sleep, mock_get, mgr):
        responses = []

        gh_resp = MagicMock()
        gh_resp.status_code = 200
        gh_resp.json.return_value = {"items": []}
        responses.append(gh_resp)

        nvd_resp = MagicMock()
        nvd_resp.status_code = 200
        nvd_resp.json.return_value = {
            "vulnerabilities": [{
                "cve": {
                    "id": "CVE-2024-0001",
                    "published": "2024-01-01",
                    "references": [
                        {"url": "https://www.exploit-db.com/exploits/12345"},
                        {"url": "https://packetstormsecurity.com/files/123"},
                        {"url": "https://www.rapid7.com/db/modules/exploit/test"},
                    ],
                }
            }]
        }
        responses.append(nvd_resp)

        msf_resp = MagicMock()
        msf_resp.status_code = 200
        msf_resp.json.return_value = {"items": []}
        responses.append(msf_resp)

        mock_get.side_effect = responses

        result = mgr.search_existing_exploits("CVE-2024-0001")

        assert result["success"] is True
        assert result["exploits_found"] >= 1
        exploit_sources = {e["source"] for e in result["exploits"]}
        assert "exploit-db" in exploit_sources
        assert "packetstorm" in exploit_sources

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    @patch("server_core.intelligence.cve_intelligence_manager.time.sleep")
    def test_search_metasploit_module(self, mock_sleep, mock_get, mgr):
        responses = []

        gh_resp = MagicMock()
        gh_resp.status_code = 200
        gh_resp.json.return_value = {"items": []}
        responses.append(gh_resp)

        nvd_resp = MagicMock()
        nvd_resp.status_code = 200
        nvd_resp.json.return_value = {"vulnerabilities": []}
        responses.append(nvd_resp)

        msf_resp = MagicMock()
        msf_resp.status_code = 200
        msf_resp.json.return_value = {
            "items": [{
                "path": "exploits/multi/http/vuln.rb",
                "name": "vuln.rb",
                "sha": "abc123",
                "html_url": "https://github.com/rapid7/metasploit-framework/blob/main/modules/exploits/multi/http/vuln.rb",
            }]
        }
        responses.append(msf_resp)

        mock_get.side_effect = responses

        result = mgr.search_existing_exploits("CVE-2024-0001")

        assert result["success"] is True
        assert "metasploit" in result["sources_searched"]

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    @patch("server_core.intelligence.cve_intelligence_manager.time.sleep")
    def test_search_github_rate_limited(self, mock_sleep, mock_get, mgr):
        responses = []

        gh_resp = MagicMock()
        gh_resp.status_code = 403
        responses.append(gh_resp)

        nvd_resp = MagicMock()
        nvd_resp.status_code = 200
        nvd_resp.json.return_value = {"vulnerabilities": []}
        responses.append(nvd_resp)

        msf_resp = MagicMock()
        msf_resp.status_code = 403
        responses.append(msf_resp)

        mock_get.side_effect = responses

        result = mgr.search_existing_exploits("CVE-2024-0001")

        assert result["success"] is True

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    @patch("server_core.intelligence.cve_intelligence_manager.time.sleep")
    def test_search_network_errors(self, mock_sleep, mock_get, mgr):
        from requests.exceptions import ConnectionError
        mock_get.side_effect = ConnectionError("Network error")

        result = mgr.search_existing_exploits("CVE-2024-0001")

        assert result["success"] is True

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    @patch("server_core.intelligence.cve_intelligence_manager.time.sleep")
    def test_search_top_level_exception(self, mock_sleep, mock_get, mgr):
        mock_get.side_effect = RuntimeError("Unexpected")

        result = mgr.search_existing_exploits("CVE-2024-0001")

        assert result["success"] is False


class TestAnalyzeEdgeCases:

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    def test_weaponization_medium_no_public_exploits(self, mock_get, mgr):
        data = make_cve_detail(
            "CVE-2024-0020", "HIGH", 9.0, "V31",
            attack_vector="NETWORK", attack_complexity="LOW",
            privileges_required="NONE", user_interaction="NONE",
            exploitability_subscore=3.9, description="Vulnerability without PoC",
            references=[{"url": "https://example.com"}])
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = data
        mock_get.return_value = mock_resp

        result = mgr.analyze_cve_exploitability("CVE-2024-0020")
        assert result["exploit_availability"]["weaponization_level"] == "MEDIUM"

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    def test_priority_medium_severity_high(self, mock_get, mgr):
        data = make_cve_detail(
            "CVE-2024-0021", "HIGH", 7.5, "V31",
            attack_vector="NETWORK", attack_complexity="LOW",
            privileges_required="NONE", user_interaction="NONE",
            exploitability_subscore=1.5, description="Medium severity vuln",
            references=[])
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = data
        mock_get.return_value = mock_resp

        result = mgr.analyze_cve_exploitability("CVE-2024-0021")
        assert result["threat_intelligence"]["recommended_priority"] in ("MEDIUM", "HIGH")

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    def test_attack_vector_adjacent_network(self, mock_get, mgr):
        data = make_cve_detail(
            "CVE-2024-0022", "HIGH", 6.0, "V31",
            attack_vector="ADJACENT_NETWORK", attack_complexity="HIGH",
            privileges_required="LOW", user_interaction="REQUIRED",
            exploitability_subscore=0.0, description="Adjacent network vuln",
            references=[])
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = data
        mock_get.return_value = mock_resp

        result = mgr.analyze_cve_exploitability("CVE-2024-0022")
        assert result["success"] is True

    @patch("server_core.intelligence.cve_intelligence_manager.requests.get")
    @patch("server_core.intelligence.cve_intelligence_manager.time.sleep")
    def test_github_fair_reliability(self, mock_sleep, mock_get, mgr):
        responses = []
        gh_resp = MagicMock()
        gh_resp.status_code = 200
        gh_resp.json.return_value = {
            "items": [{
                "id": 100,
                "name": "CVE-2024-0023-poc",
                "description": "PoC for CVE-2024-0023",
                "owner": {"login": "tester"},
                "created_at": "2024-01-01",
                "updated_at": "2024-01-02",
                "html_url": "https://github.com/test/poc",
                "stargazers_count": 25,
                "forks_count": 3,
            }]
        }
        responses.append(gh_resp)

        nvd_resp = MagicMock()
        nvd_resp.status_code = 200
        nvd_resp.json.return_value = {"vulnerabilities": []}
        responses.append(nvd_resp)

        msf_resp = MagicMock()
        msf_resp.status_code = 200
        msf_resp.json.return_value = {"items": []}
        responses.append(msf_resp)

        mock_get.side_effect = responses
        result = mgr.search_existing_exploits("CVE-2024-0023")
        assert result["success"] is True


class TestRenderBanner:
    def test_banner(self):
        result = CVEIntelligenceManager.create_banner()
        assert result is not None
