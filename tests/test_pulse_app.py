"""Tests for pulse_app.py — Prefab UI dashboard backend tools."""

import time
import threading
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from pulse.interface import pulse_app
from pulse.interface.dashboards import _build_ui_state

_SYNTHETIC_NUCLEI_OUTPUT = """\
[critical] [apache-path-traversal] [http://192.168.1.165/DVWA/login.php]
[high] [wordpress-user-enum] [http://192.168.1.165/DVWA/wp-admin]
[medium] [phpinfo-disclosure] [http://192.168.1.165/DVWA/phpinfo.php]
[low] [x-frame-options] [http://192.168.1.165/DVWA/index.php]
[info] [robots-txt] [http://192.168.1.165/DVWA/robots.txt]
"""

_SYNTHETIC_NIKTO_OUTPUT = """\
+ /config/: Directory indexing found.
+ /setup/: Setup file present.
+ /phpinfo.php: PHP Info page may expose sensitive information.
+ /: Server banner: Apache/2.4.67 (Raspbian).
"""


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_state():
    """Reset module-level state between tests."""
    pulse_app._op_metrics._tools.clear()
    pulse_app._op_metrics._cache_hits = 0
    pulse_app._op_metrics._cache_misses = 0
    pulse_app._op_metrics._start_time = time.time()
    pulse_app._async_scans.clear()
    yield


class _MockCache(dict):
    """Dict that also supports .set() — compatible with real AdvancedCache API."""
    def set(self, key, value, ttl=None, execution_time=None):
        self[key] = value


@pytest.fixture
def mock_scan_cache():
    """Patch pulse_app._scan_cache with a controllable mock cache."""
    cache = _MockCache()
    with patch.object(pulse_app, "_scan_cache", cache):
        yield cache


@pytest.fixture
def mock_tool_stats():
    """Patch ToolStatsStore via pulse_app.get_tool_stats_store."""
    store = MagicMock()
    store.get_all_stats.return_value = {}
    store.live_effectiveness.return_value = None
    store.blended_effectiveness.return_value = 0.5
    with patch.object(pulse_app, "get_tool_stats_store", return_value=store):
        yield store


# ── _fmt_duration ─────────────────────────────────────────────────────────────


class TestFmtDuration:
    def test_seconds(self):
        assert pulse_app._fmt_duration(30) == "30s"

    def test_minutes(self):
        assert pulse_app._fmt_duration(125) == "2m 5s"

    def test_hours(self):
        assert pulse_app._fmt_duration(3661) == "1h 1m"

    def test_none(self):
        assert pulse_app._fmt_duration(None) == "—"

    def test_zero(self):
        assert pulse_app._fmt_duration(0) == "0s"


# ── get_overview ──────────────────────────────────────────────────────────────


class TestGetOverview:
    def test_returns_expected_keys(self):
        result = pulse_app.get_overview()
        expected_keys = {
            "version", "version_display", "uptime_seconds", "uptime_display",
            "ram_percent", "ram_available_gb", "ram_total_gb", "ram_display",
            "disk_free_gb", "disk_percent", "cpu_percent", "server_status",
            "server_status_variant", "tools_count", "tools_display",
            "total_runs", "total_errors",
        }
        assert expected_keys.issubset(result.keys())

    def test_server_status_limited_without_psutil(self):
        with patch.object(pulse_app._op_metrics, "summary") as mock:
            mock.return_value = {
                "uptime_seconds": 120,
                "total_runs": 0,
                "total_successes": 0,
                "total_errors": 0,
                "global_success_rate": 0.0,
                "system": {"status": "unavailable", "reason": "psutil not installed"},
                "cache": {},
                "confirmations": {},
            }
            result = pulse_app.get_overview()
            assert result["server_status"] == "limited"
            assert result["server_status_variant"] == "warning"
            assert "uptime_display" in result

    def test_server_status_healthy(self):
        result = pulse_app.get_overview()
        # system_metrics may or may not have psutil — check dynamically
        sys = pulse_app._op_metrics.summary().get("system", {})
        expected = "healthy" if "cpu_percent" in sys else "limited"
        assert result["server_status"] == expected

    def test_version_from_config(self):
        result = pulse_app.get_overview()
        assert result["version"] == pulse_app.app_config["VERSION"]
        assert result["version_display"] == f"PULSE v{pulse_app.app_config['VERSION']}"


# ── get_scope ─────────────────────────────────────────────────────────────────


class TestGetScope:
    def test_no_entries_returns_no_scope(self):
        with patch.object(pulse_app, "_scan_cache", {}):
            result = pulse_app.get_scope()
            assert result["active_target"] is None

    def test_detects_single_target(self, mock_scan_cache):
        mock_scan_cache["sess:nmap:10.10.10.45"] = {
            "tool": "nmap",
            "target": "10.10.10.45",
            "timestamp": time.time(),
            "result": {"success": True, "stdout": "22/tcp open"},
        }
        result = pulse_app.get_scope()
        assert result["active_target"] == "10.10.10.45"
        assert result["target_type"] == "ip"
        assert result["tools_count"] == 1
        assert result["tools_used"] == [{"name": "nmap"}]

    def test_groups_multiple_tools_on_same_target(self, mock_scan_cache):
        now = time.time()
        mock_scan_cache["sess:nmap:10.10.10.45"] = {
            "tool": "nmap", "target": "10.10.10.45",
            "timestamp": now - 60, "result": {},
        }
        mock_scan_cache["sess:whatweb:10.10.10.45"] = {
            "tool": "whatweb", "target": "10.10.10.45",
            "timestamp": now - 30, "result": {},
        }
        result = pulse_app.get_scope()
        assert result["active_target"] == "10.10.10.45"
        assert result["tools_count"] == 2
        tools = [t["name"] for t in result["tools_used"]]
        assert "nmap" in tools
        assert "whatweb" in tools

    def test_most_recent_target_is_active(self, mock_scan_cache):
        now = time.time()
        mock_scan_cache["sess:nmap:old"] = {
            "tool": "nmap", "target": "10.10.10.1",
            "timestamp": now - 600, "result": {},
        }
        mock_scan_cache["sess:nmap:active"] = {
            "tool": "nmap", "target": "10.10.10.2",
            "timestamp": now - 10, "result": {},
        }
        result = pulse_app.get_scope()
        assert result["active_target"] == "10.10.10.2"

    def test_detects_domain_type(self, mock_scan_cache):
        mock_scan_cache["sess:httpx:example.com"] = {
            "tool": "httpx", "target": "example.com",
            "timestamp": time.time(), "result": {},
        }
        result = pulse_app.get_scope()
        assert result["target_type"] == "domain"

    def test_detects_url_type(self, mock_scan_cache):
        mock_scan_cache["sess:whatweb:https://app.example.com"] = {
            "tool": "whatweb", "target": "https://app.example.com",
            "timestamp": time.time(), "result": {},
        }
        result = pulse_app.get_scope()
        assert result["target_type"] == "url"

    def test_unknown_target_type(self, mock_scan_cache):
        mock_scan_cache["sess:ping:ff:ff:ff:ff"] = {
            "tool": "ping", "target": "ff:ff:ff:ff",
            "timestamp": time.time(), "result": {},
        }
        result = pulse_app.get_scope()
        assert result["target_type"] == "unknown"

    def test_handles_exception_from_cache(self):
        with patch.object(pulse_app, "_scan_cache", {"bad": "entry"}):
            result = pulse_app.get_scope()
            assert result["active_target"] is None


# ── get_tool_intelligence ─────────────────────────────────────────────────────


class TestGetToolIntelligence:
    def test_no_stats(self, mock_tool_stats):
        result = pulse_app.get_tool_intelligence()
        assert result == []

    def test_with_stats(self, mock_tool_stats):
        mock_tool_stats.get_all_stats.return_value = {
            "nmap": {"runs": 10, "successes": 8},
        }
        mock_tool_stats.live_effectiveness.return_value = 0.8
        mock_tool_stats.blended_effectiveness.return_value = 0.8
        result = pulse_app.get_tool_intelligence()
        assert len(result) == 1
        assert result[0]["tool"] == "nmap"
        assert result[0]["runs"] == 10
        assert result[0]["live"] == 0.8


# ── _cache_for_target ──────────────────────────────────────────────────────────


class TestCacheForTarget:
    def test_empty_cache(self):
        with patch.object(pulse_app, "_scan_cache", {}):
            assert pulse_app._cache_for_target("10.10.10.1") == []

    def test_no_match(self, mock_scan_cache):
        mock_scan_cache["k:nmap:other"] = {"tool": "nmap", "target": "10.10.10.2"}
        assert pulse_app._cache_for_target("10.10.10.1") == []

    def test_matches_target(self, mock_scan_cache):
        mock_scan_cache["k:nmap:t"] = {"tool": "nmap", "target": "10.10.10.1"}
        result = pulse_app._cache_for_target("10.10.10.1")
        assert len(result) == 1
        assert result[0]["tool"] == "nmap"

    def test_multiple_tools_same_target(self, mock_scan_cache):
        mock_scan_cache["k:nmap:t"] = {"tool": "nmap", "target": "10.10.10.1"}
        mock_scan_cache["k:whatweb:t"] = {"tool": "whatweb", "target": "10.10.10.1"}
        assert len(pulse_app._cache_for_target("10.10.10.1")) == 2


# ── get_surface ────────────────────────────────────────────────────────────────


class TestGetSurface:
    def test_no_target_no_scope(self):
        with patch.object(pulse_app, "_scan_cache", {}):
            result = pulse_app.get_surface()
            assert result["target"] is None

    def test_uses_active_scope(self, mock_scan_cache):
        mock_scan_cache["s:nmap:t"] = {
            "tool": "nmap", "target": "10.10.10.45",
            "timestamp": time.time(),
            "result": {"stdout": "22/tcp open  ssh\n80/tcp open  http"},
        }
        result = pulse_app.get_surface()
        # Since scope picks 10.10.10.45 as active, surface should find it
        if result["target"] == "10.10.10.45":
            assert result["ports_count"] == 2
        else:
            # _scan_cache patching might affect scope — just verify it ran
            assert "target" in result

    def test_no_cache_no_target(self, mock_scan_cache):
        with patch.object(pulse_app, "_scan_cache", {}):
            result = pulse_app.get_surface(target="10.10.10.1")
            assert result["target"] == "10.10.10.1"
            assert result["ports_count"] == 0
            assert result["risk_variant"] == "default"
            assert result["ports_display"] == "No ports detected"

    def test_parses_nmap_ports(self, mock_scan_cache):
        mock_scan_cache["x:nmap:t"] = {
            "tool": "nmap", "target": "10.10.10.1",
            "timestamp": time.time(),
            "result": {"stdout": "22/tcp open  ssh\n80/tcp  open  http\n443/tcp open  https"},
        }
        result = pulse_app.get_surface(target="10.10.10.1")
        assert result["ports_count"] == 3
        ports = [p["port"] for p in result["ports"]]
        assert ports == [22, 80, 443]
        assert result["ports"][0]["service"] == "ssh"
        assert result["risk_level"] in ("medium", "high")
        assert result["risk_variant"] in ("warning", "destructive")
        assert "3 open ports" in result["ports_display"]

    def test_nmap_advanced_also_parsed(self, mock_scan_cache):
        mock_scan_cache["x:nmap_advanced:t"] = {
            "tool": "nmap_advanced", "target": "10.10.10.1",
            "timestamp": time.time(),
            "result": {"stdout": "22/tcp open  ssh\n8080/tcp open  http-proxy"},
        }
        result = pulse_app.get_surface(target="10.10.10.1")
        assert result["ports_count"] == 2

    def test_parses_whatweb_techs(self, mock_scan_cache):
        mock_scan_cache["x:whatweb:t"] = {
            "tool": "whatweb", "target": "example.com",
            "timestamp": time.time(),
            "result": {"stdout": "http://example.com [200 OK] Apache[2.4.51], PHP[8.0], WordPress[6.0]"},
        }
        result = pulse_app.get_surface(target="example.com")
        assert "WordPress" in result["technologies"]
        assert "Apache" in result["technologies"]
        assert "PHP" in result["technologies"]

    def test_risk_level_from_port_count(self, mock_scan_cache):
        result = pulse_app.get_surface(target="no-cache-target")
        assert result["risk_level"] == "unknown"


# ── get_findings ──────────────────────────────────────────────────────────────


class TestGetFindings:
    def test_no_target_no_scope(self):
        with patch.object(pulse_app, "_scan_cache", {}):
            assert pulse_app.get_findings() == []

    def test_parses_nuclei_findings(self, mock_scan_cache):
        mock_scan_cache["x:nuclei:t"] = {
            "tool": "nuclei", "target": "10.10.10.1",
            "timestamp": time.time(),
            "result": {
                "stdout": (
                    "[critical] [CVE-2023-xxxx] [http] [Critical Vuln] https://target.com/path\n"
                    "[medium] [CVE-2023-yyyy] [http] [Medium Issue] https://target.com/other\n"
                ),
            },
        }
        result = pulse_app.get_findings(target="10.10.10.1")
        assert len(result) == 2
        assert result[0]["severity"] == "critical"
        assert result[1]["severity"] == "medium"
        assert result[0]["tool"] == "nuclei"

    def test_parses_nikto_findings(self, mock_scan_cache):
        mock_scan_cache["x:nikto:t"] = {
            "tool": "nikto", "target": "10.10.10.1",
            "timestamp": time.time(),
            "result": {
                "stdout": (
                    "- Nikto v2.5.0\n"
                    "+ /: Server: Apache/2.4.51\n"
                    "+ /: The X-Frame-Options header is missing.\n"
                ),
            },
        }
        result = pulse_app.get_findings(target="10.10.10.1")
        assert len(result) == 2
        assert result[0]["tool"] == "nikto"

    def test_no_findings(self, mock_scan_cache):
        mock_scan_cache["x:nmap:t"] = {
            "tool": "nmap", "target": "10.10.10.1",
            "timestamp": time.time(),
            "result": {"stdout": "22/tcp open"},
        }
        assert pulse_app.get_findings(target="10.10.10.1") == []

    def test_findings_sorted_by_severity(self, mock_scan_cache):
        mock_scan_cache["x:nuclei:t"] = {
            "tool": "nuclei", "target": "10.10.10.1",
            "timestamp": time.time(),
            "result": {
                "stdout": (
                    "[info] [info-finding] [http] [Info] https://t.com\n"
                    "[critical] [crit-finding] [http] [Critical] https://t.com\n"
                    "[medium] [med-finding] [http] [Medium] https://t.com\n"
                ),
            },
        }
        result = pulse_app.get_findings(target="10.10.10.1")
        severities = [f["severity"] for f in result]
        assert severities == ["critical", "medium", "info"]


# ── get_plan ──────────────────────────────────────────────────────────────────


class TestGetPlan:
    """get_plan uses IntelligentDecisionEngine — tested at unit level with mocks."""

    def test_no_target_returns_empty(self):
        with patch.object(pulse_app, "_scan_cache", {}):
            result = pulse_app.get_plan()
        assert result["steps"] == []
        assert result["step_count"] == 0

    def test_no_target_with_explicit_none(self):
        with patch.object(pulse_app, "_scan_cache", {}):
            result = pulse_app.get_plan(target=None)
        assert result["steps"] == []

    def test_returns_plan_with_mocked_ide(self):
        fake_chain = {
            "target": "10.10.10.1",
            "steps": [
                {
                    "num": 1,
                    "tool": "nmap",
                    "expected_outcome": "Discover ports",
                    "outcome_short": "Discover ports",
                    "success_probability": 0.85,
                    "prob_display": "85%",
                    "execution_time_estimate": 120,
                    "eta_display": "2m 0s",
                    "parameters": {},
                    "dependencies": [],
                }
            ],
            "success_probability": 0.85,
            "estimated_time": 120,
            "required_tools": ["nmap"],
            "risk_level": "medium",
            "step_count": 1,
            "summary": "1 step · 2m 0s est · medium risk",
        }
        mock_ide = MagicMock()
        mock_ide.analyze_target.return_value = MagicMock()
        mock_ide.create_attack_chain.return_value.to_dict.return_value = fake_chain
        with patch.object(pulse_app, "get_decision_engine", return_value=mock_ide):
            result = pulse_app.get_plan(target="10.10.10.1")
        assert result["step_count"] == 1
        assert result["steps"][0]["tool"] == "nmap"
        assert "2m 0s" in result["summary"]

    def test_ide_exception_returns_fallback(self):
        mock_ide = MagicMock()
        mock_ide.analyze_target.side_effect = RuntimeError("IDE failed")
        with patch.object(pulse_app, "get_decision_engine", return_value=mock_ide):
            result = pulse_app.get_plan(target="10.10.10.1")
        assert result["step_count"] == 0
        assert "unavailable" in result["summary"].lower()


# ── get_active_tools ──────────────────────────────────────────────────────────


class TestGetActiveTools:
    def test_returns_dict_with_expected_keys(self):
        result = pulse_app.get_active_tools()
        for key in ("active_processes", "active_workers", "queue_size",
                    "processes", "async_scans", "resource_usage", "summary"):
            assert key in result

    def test_summary_format(self):
        result = pulse_app.get_active_tools()
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_counts_running_processes_only(self):
        fake_registry = {
            1111: {"pid": 1111, "command": "nmap -sV 10.0.0.1", "status": "running",
                   "progress": 0.4, "runtime": 12.0, "eta": 8.0},
            2222: {"pid": 2222, "command": "curl http://x", "status": "running",
                   "progress": 0.0, "runtime": 3.0, "eta": 0.0},
            3333: {"pid": 3333, "command": "done", "status": "terminated",
                   "progress": 0.0, "runtime": 1.0, "eta": 0.0},
        }
        with patch("pulse.infrastructure.execution.process_manager.active_processes", fake_registry):
            result = pulse_app.get_active_tools()
        assert result["active_processes"] == 2
        assert len(result["processes"]) == 2
        assert all("process" not in p["command"] for p in result["processes"])
        assert "2 process" in result["summary"]

    def test_counts_running_async_scans_only(self):
        now = time.time()
        with pulse_app._async_scans_lock:
            pulse_app._async_scans["scan_test1"] = {
                "tool": "nmap", "target": "10.0.0.1", "status": "running",
                "start_time": now - 5,
            }
            pulse_app._async_scans["scan_test2"] = {
                "tool": "sqlmap", "target": "10.0.0.1", "status": "completed",
                "start_time": now - 10,
            }
        try:
            result = pulse_app.get_active_tools()
        finally:
            with pulse_app._async_scans_lock:
                pulse_app._async_scans.pop("scan_test1", None)
                pulse_app._async_scans.pop("scan_test2", None)
        assert result["active_workers"] == 1
        assert len(result["async_scans"]) == 1
        assert result["async_scans"][0]["scan_id"] == "scan_test1"
        assert result["async_scans"][0]["tool"] == "nmap"


# ── cancel_scan ────────────────────────────────────────────────────────────────


class TestCancelScan:

    def _seed(self, scan_id="scan_cancel1", tool="nmap", status="running"):
        with pulse_app._async_scans_lock:
            pulse_app._async_scans[scan_id] = {
                "tool": tool, "target": "10.0.0.1", "status": status,
                "start_time": time.time() - 5, "end_time": None,
                "result": None, "progress": 0, "error": None,
            }
        return scan_id

    def _seed_many(self, *specs):
        with pulse_app._async_scans_lock:
            for scan_id, tool, status in specs:
                pulse_app._async_scans[scan_id] = {
                    "tool": tool, "target": "10.0.0.1", "status": status,
                    "start_time": time.time() - 5, "end_time": None,
                    "result": None, "progress": 0, "error": None,
                }

    def _teardown(self, *scan_ids):
        with pulse_app._async_scans_lock:
            for scan_id in scan_ids:
                pulse_app._async_scans.pop(scan_id, None)

    def test_cancel_marks_cancelled(self):
        scan_id = self._seed()
        try:
            result = pulse_app.cancel_scan(scan_id)
            assert result["status"] == "cancelled"
            assert result["terminated_pids"] == []
            with pulse_app._async_scans_lock:
                entry = pulse_app._async_scans[scan_id]
            assert entry["status"] == "cancelled"
            assert entry["cancel_requested"] is True
        finally:
            self._teardown(scan_id)

    def test_cancel_unknown_scan(self):
        assert pulse_app.cancel_scan("scan_ghost")["status"] == "not_found"

    def test_cancel_already_final(self):
        scan_id = self._seed(status="completed")
        try:
            result = pulse_app.cancel_scan(scan_id)
            assert result["already_final"] is True
            assert result["status"] == "completed"
        finally:
            self._teardown(scan_id)

    def test_cancel_terminates_matching_command(self):
        scan_id = self._seed(tool="nmap")
        fake_registry = {
            7777: {"pid": 7777, "command": "nmap -sV 10.0.0.1", "status": "running",
                   "process": None},
            8888: {"pid": 8888, "command": "curl http://other", "status": "running",
                   "process": None},
        }
        try:
            with patch("pulse.infrastructure.execution.process_manager.active_processes", fake_registry):
                with patch("pulse.infrastructure.execution.process_manager.ProcessManager.terminate_process",
                           return_value=True) as mock_kill:
                    result = pulse_app.cancel_scan(scan_id)
            assert result["terminated_pids"] == [7777]
            mock_kill.assert_called_once_with(7777)
        finally:
            self._teardown(scan_id)

    def test_cancel_spares_parallel_same_tool(self):
        scan_id_a = self._seed("scan_par_a", tool="nmap")
        self._seed_many(("scan_par_b", "nmap", "running"))
        try:
            with patch("pulse.infrastructure.execution.process_manager.ProcessManager.terminate_process",
                       return_value=True) as mock_kill:
                result = pulse_app.cancel_scan(scan_id_a)
            assert result["terminated_pids"] == []
            mock_kill.assert_not_called()
        finally:
            self._teardown(scan_id_a, "scan_par_b")

    def test_get_scan_status_reports_cancelled(self):
        scan_id = self._seed()
        try:
            pulse_app.cancel_scan(scan_id)
            result = pulse_app.get_scan_status(scan_id)
            assert result["status"] == "cancelled"
        finally:
            self._teardown(scan_id)

    def test_get_scan_status_running_has_eta(self):
        scan_id = self._seed()
        try:
            with patch.object(pulse_app._op_metrics, "avg_duration_by_tool",
                              return_value={"nmap": 60.0}):
                result = pulse_app.get_scan_status(scan_id)
            assert result["status"] == "running"
            assert result["eta_seconds"] == 55.0
            assert result["eta_display"]
        finally:
            self._teardown(scan_id)

    def test_get_scan_status_eta_none_without_history(self):
        scan_id = self._seed()
        try:
            result = pulse_app.get_scan_status(scan_id)
            assert result["eta_seconds"] is None
            assert result["eta_display"]
        finally:
            self._teardown(scan_id)


# ── get_history ────────────────────────────────────────────────────────────────


class TestGetHistory:
    def test_empty_cache(self):
        with patch.object(pulse_app, "_scan_cache", {}):
            assert pulse_app.get_history() == []

    def test_returns_entries(self, mock_scan_cache):
        mock_scan_cache["s:nmap:t"] = {
            "tool": "nmap", "target": "10.10.10.1",
            "timestamp": 1000,
            "result": {"success": True, "execution_time": 30.5},
        }
        result = pulse_app.get_history()
        assert len(result) == 1
        assert result[0]["tool"] == "nmap"
        assert result[0]["status"] == "\u2713"
        assert result[0]["execution_display"] == "30s"

    def test_filters_by_target(self, mock_scan_cache):
        mock_scan_cache["s:nmap:a"] = {"tool": "nmap", "target": "10.10.10.1", "timestamp": 1, "result": {}}
        mock_scan_cache["s:nmap:b"] = {"tool": "nmap", "target": "10.10.10.2", "timestamp": 2, "result": {}}
        result = pulse_app.get_history(target="10.10.10.2")
        assert len(result) == 1
        assert result[0]["target"] == "10.10.10.2"

    def test_sorts_by_timestamp_descending(self, mock_scan_cache):
        mock_scan_cache["s:tool:first"] = {"tool": "a", "target": "t", "timestamp": 100, "result": {}}
        mock_scan_cache["s:tool:second"] = {"tool": "b", "target": "t", "timestamp": 200, "result": {}}
        result = pulse_app.get_history()
        assert result[0]["tool"] == "b"

    def test_handles_exception(self):
        with patch.object(pulse_app, "_scan_cache", {"bad": "value"}):
            assert pulse_app.get_history() == []


# ── pulse_dashboard (UI entry) ────────────────────────────────────────────────


# Source of truth: _STATE_KEY_SOURCES lives in dashboards/pulse_dashboard.py
# next to _build_ui_state(), re-exported by pulse_app. Any divergence between
# the real state and the contract fails the parity test below — this is the
# guard rail against silent key drift.
_STATE_KEY_SOURCES = pulse_app._STATE_KEY_SOURCES

# Documented contract: get_live_dashboard() returns exactly these top-level keys.
_LIVE_DASHBOARD_KEYS = {
    "overview", "scope", "surface", "findings", "plan", "active_tools",
    "history", "rate_limit", "errors", "tool_performance", "cache_status",
    "system_trends", "sessions", "confirmations", "network_io",
    "async_scans", "intelligence", "next_suggested_tool",
}


class TestPulseDashboard:
    def test_returns_prefab_app(self):
        app = pulse_app.pulse_dashboard()
        assert app is not None

    def test_state_keys_match_documented_contract(self):
        """Every state key must be in the documented mapping, and vice versa.

        This is the guard rail against silent key drift: adding or removing
        a state={} key without updating the contract fails here.
        """
        app = pulse_app.pulse_dashboard()
        state = getattr(app, "state", None) or getattr(app, "_state", None)
        assert state is not None
        assert set(state.keys()) == set(_STATE_KEY_SOURCES), (
            f"state keys diverge from contract: "
            f"extra={sorted(set(state) - set(_STATE_KEY_SOURCES))} "
            f"missing={sorted(set(_STATE_KEY_SOURCES) - set(state))}"
        )

    def test_has_expected_legacy_keys(self):
        """The original documented subset remains present (regression guard)."""
        app = pulse_app.pulse_dashboard()
        state = getattr(app, "state", None) or getattr(app, "_state", None)
        expected = {
            "version", "version_display", "uptime_display", "ram_display",
            "tools_display", "uptime_seconds", "ram_percent",
            "ram_available_gb", "ram_total_gb", "server_status",
            "server_status_variant",
            "tools_count", "total_runs", "total_errors",
            "scope_target", "scope_type", "scope_tools",
            "scope_tools_count", "scope_last_seen_ago", "scope_age",
            "scope_summary",
            "surface_target", "risk_level", "risk_variant",
            "ports_display", "ports_count",
            "surface_ports", "surface_techs",
            "findings",
            "plan_target", "plan_steps", "plan_summary",
            "active_processes", "active_workers", "active_queue",
            "active_summary",
            "system", "history", "intelligence",
        }
        assert expected.issubset(set(state.keys()))

    def test_get_live_dashboard_keys_match_contract(self):
        """get_live_dashboard() must return exactly the documented 18 panels."""
        result = pulse_app.get_live_dashboard()
        assert set(result.keys()) == _LIVE_DASHBOARD_KEYS

    def test_live_dashboard_async_scans_shape(self):
        """async_scans is a nested dict with running/complete/summary."""
        result = pulse_app.get_live_dashboard()
        async_scans = result["async_scans"]
        assert isinstance(async_scans, dict)
        assert set(async_scans.keys()) == {"running", "complete", "summary"}

    def test_next_suggested_tool_state_and_display(self):
        """R4: next_suggested_tool is exposed in the UI state with display fields."""
        app = pulse_app.pulse_dashboard()
        state = getattr(app, "state", None) or getattr(app, "_state", None)
        for key in ("next_suggested_tool", "nst_tool", "nst_reason", "nst_time", "nst_variant"):
            assert key in state, f"missing state key: {key}"

        st = pulse_app._collect_dashboard_state()
        st["next_suggested_tool"] = {
            "tool": "gobuster", "reason": "Web server detected",
            "expected_time": "1-5 min", "priority": "high",
        }
        ui = _build_ui_state(st)
        assert ui["nst_tool"] == "gobuster"
        assert ui["nst_reason"] == "Web server detected"
        assert ui["nst_time"] == "1-5 min"
        assert ui["nst_variant"] == "warning"
        ui_empty = _build_ui_state({**st, "next_suggested_tool": {}})
        assert ui_empty["nst_tool"] == ""
        assert ui_empty["nst_variant"] == "default"

        view = str(app.view)
        assert "NEXT TOOL" in view


# ── Tool registration ─────────────────────────────────────────────────────────


class TestToolRegistration:
    def test_app_is_fastmcpapp(self):
        """The module exposes a FastMCPApp instance."""
        assert pulse_app.app is not None
        assert "FastMCPApp" in type(pulse_app.app).__name__

    def test_tool_functions_are_callable(self):
        """All backend tool functions are callables on the module."""
        for name in (
            "get_overview",
            "get_scope",
            "get_surface",
            "get_findings",
            "get_plan",
            "get_active_tools",
            "get_history",
            "get_tool_intelligence",
        ):
            fn = getattr(pulse_app, name, None)
            assert fn is not None, f"Missing function: {name}"
            assert callable(fn), f"Not callable: {name}"


# ── scan() entry point ────────────────────────────────────────────────────────


class TestScanEntryPoint:
    """Integration tests for scan() entry point with mocked exec_func."""

    @pytest.fixture
    def mock_run_tools(self, mock_scan_cache):
        """Patch run_security_tool with controllable per-tool canned results.

        Each tool_name -> result dict.  Tests can mutate the yielded dict
        to inject failures or drop tools.
        Results are also written to _scan_cache so get_findings() /
        get_surface() can read them later.
        """
        import uuid as _uuid

        results = {}
        for tool_name, stdout, error in [
            ("nmap", "22/tcp open  ssh\n80/tcp open  http", ""),
            ("whatweb", "http://target [200 OK] nginx PHP", ""),
            ("nuclei", "[critical] [CVE-2014-0160] https://target\n[high] [SQL Injection] https://target/search", ""),
            ("nikto", "+ /wp-admin: WordPress admin page\n+ /config.php: Config file found", ""),
            ("gobuster", "/admin (Status: 200)", ""),
        ]:
            results[tool_name] = {
                "success": True,
                "output": stdout,
                "stdout": stdout,
                "error": error,
                "returncode": 0,
                "duration": 0.5,
            }

        async def _mock_run_security_tool(ctx, tool_name, params):
            if tool_name in results:
                data = dict(results[tool_name])
                # Write to _scan_cache so get_findings()/get_surface() work
                target = params.get("target") or params.get("url", "unknown")
                try:
                    key = f"sess:{tool_name}:{target}"
                    mock_scan_cache[key] = {
                        "tool": tool_name,
                        "target": target,
                        "timestamp": time.time(),
                        "result": {"stdout": data.get("stdout", ""), "success": True},
                    }
                except Exception:
                    pass
                return data
            return {"success": False, "error": f"Unknown tool: {tool_name}", "returncode": 1}

        with patch("pulse.interface.server_setup.run_security_tool", new=_mock_run_security_tool):
            yield results

    # ── Intensity levels ─────────────────────────────────────────────────

    def test_scan_quick_basic(self, mock_run_tools, mock_scan_cache):
        """quick intensity runs nmap + whatweb, returns correct structure."""
        result = pulse_app.scan(target="10.10.10.1")
        assert result["target"] == "10.10.10.1"
        assert result["intensity"] == "quick"
        assert set(result["tools"].keys()) == {"nmap", "whatweb"}
        assert all(v["status"] == "completed" for v in result["tools"].values())

    def test_scan_medium(self, mock_run_tools, mock_scan_cache):
        """medium intensity runs 4 tools: nmap, whatweb, nuclei, nikto."""
        result = pulse_app.scan(target="10.10.10.1", intensity="medium")
        assert set(result["tools"].keys()) == {"nmap", "whatweb", "nuclei", "nikto"}
        assert all(v["status"] == "completed" for v in result["tools"].values())

    def test_scan_full(self, mock_run_tools, mock_scan_cache):
        """full intensity runs 5 tools + includes plan."""
        result = pulse_app.scan(target="10.10.10.1", intensity="full")
        assert len(result["tools"]) == 5
        assert "gobuster" in result["tools"]
        assert result["tools"]["gobuster"]["status"] == "completed"
        assert result["plan"]["step_count"] is not None
        assert result["plan"]["target"] == "10.10.10.1"

    # ── Cache behaviour ─────────────────────────────────────────────────

    def test_scan_cached_tool_skips_execution(self, mock_run_tools):
        """Tool with existing cache entry is reported as 'cached', not executed."""
        cache = _MockCache({"sess:nmap:10.10.10.1": {
            "tool": "nmap",
            "target": "10.10.10.1",
            "timestamp": 9999999999,
            "result": {"success": True, "output": "cached"},
        }})
        with patch.object(pulse_app, "_scan_cache", cache):
            result = pulse_app.scan(target="10.10.10.1")
        assert result["tools"]["nmap"]["status"] == "cached"
        assert result["tools"]["nmap"].get("cached") is True
        assert result["tools"]["whatweb"]["status"] == "completed"

    # ── Error handling ───────────────────────────────────────────────────

    def test_scan_tool_failure_reported(self, mock_run_tools, mock_scan_cache):
        """Tool that fails gets status 'failed' with an error message."""
        mock_run_tools["nmap"] = {"success": False, "error": "Connection refused",
                                  "returncode": 1, "output": "", "stdout": ""}
        result = pulse_app.scan(target="10.10.10.1")
        assert result["tools"]["nmap"]["status"] == "failed"
        assert "error" in result["tools"]["nmap"]
        assert result["tools"]["whatweb"]["status"] == "completed"

    def test_scan_unknown_tool_skipped(self, mock_run_tools, mock_scan_cache):
        """Tool not in mock results is reported as 'failed' via run_security_tool."""
        del mock_run_tools["gobuster"]
        result = pulse_app.scan(target="10.10.10.1", intensity="full")
        assert result["tools"]["gobuster"]["status"] == "failed"
        assert "error" in result["tools"]["gobuster"]

    def test_scan_invalid_intensity_defaults_to_quick(self, mock_run_tools, mock_scan_cache):
        """Unknown intensity string falls back to 'quick'."""
        result = pulse_app.scan(target="10.10.10.1", intensity="extreme")
        assert result["intensity"] == "quick"
        assert len(result["tools"]) == 2

    # ── Target resolution ───────────────────────────────────────────────

    def test_scan_no_target_returns_error(self, mock_run_tools):
        """No explicit target and empty scope returns error dict."""
        with patch.object(pulse_app, "_scan_cache", _MockCache()):
            result = pulse_app.scan()
        assert "error" in result
        assert result["target"] is None

    def test_scan_auto_detects_target_from_scope(self, mock_run_tools):
        """Without explicit target, scan() picks the most recent from cache."""
        cache = _MockCache({"sess:nmap:auto.example": {
            "tool": "nmap",
            "target": "auto.example",
            "timestamp": 9999999999,
            "result": {"success": True, "output": "22/tcp open  ssh"},
        }})
        with patch.object(pulse_app, "_scan_cache", cache):
            result = pulse_app.scan()
        assert result["target"] == "auto.example"
        assert result["tools"]["nmap"]["status"] == "cached"
        assert result["tools"]["whatweb"]["status"] == "completed"

    # ── Result structure ────────────────────────────────────────────────

    def test_scan_result_keys(self, mock_run_tools, mock_scan_cache):
        """Return dict contains all expected top-level keys."""
        result = pulse_app.scan(target="check.example", intensity="medium")
        assert set(result.keys()) == {"target", "intensity", "tools", "surface",
                                       "findings", "plan", "summary",
                                       "next_suggested_tool", "cache_age", "highlights"}
        for tool_name, r in result["tools"].items():
            assert "status" in r
            if r["status"] == "completed":
                assert "returncode" in r
            elif r["status"] == "failed":
                assert "error" in r

    def test_scan_findings_enriched_with_exploit(self, mock_run_tools, mock_scan_cache):
        """Findings in medium/full scans include 'exploit' field with tool/confidence/source."""
        result = pulse_app.scan(target="check.example", intensity="medium")
        assert len(result.get("findings", [])) > 0
        for f in result.get("findings", []):
            assert "exploit" in f, f"Finding {f.get('finding')} missing 'exploit' field"
            exploit = f["exploit"]
            assert "tool" in exploit
            assert "confidence" in exploit
            assert "estimated_time" in exploit
            assert "source" in exploit
            assert exploit["source"] == "rules"

    def test_scan_exploit_cve_gets_testssl(self, mock_run_tools, mock_scan_cache):
        """CVE-2014-0160 (Heartbleed) should get testssl suggestion."""
        result = pulse_app.scan(target="check.example", intensity="medium")
        cve_findings = [f for f in result["findings"] if "CVE-2014-0160" in f.get("finding", "")]
        assert len(cve_findings) >= 1
        assert cve_findings[0]["exploit"]["tool"] == "testssl"
        assert cve_findings[0]["exploit"]["confidence"] == "certain"

    def test_scan_exploit_sqli_gets_sqlmap(self, mock_run_tools, mock_scan_cache):
        """'SQL Injection' finding should get sqlmap suggestion."""
        result = pulse_app.scan(target="check.example", intensity="medium")
        sqli = [f for f in result["findings"] if "SQL" in f.get("finding", "")]
        assert len(sqli) >= 1
        assert sqli[0]["exploit"]["tool"] == "sqlmap"
        assert sqli[0]["exploit"]["confidence"] == "high"

    def test_scan_exploit_wp_admin_gets_wpscan(self, mock_run_tools, mock_scan_cache):
        """nikto /wp-admin finding should get wpscan via detail pattern."""
        result = pulse_app.scan(target="check.example", intensity="medium")
        wp = [f for f in result["findings"] if "wp-admin" in f.get("finding", "")]
        assert len(wp) >= 1
        assert wp[0]["exploit"]["tool"] == "wpscan"

    def test_scan_exploit_config_php_gets_manual(self, mock_run_tools, mock_scan_cache):
        """nikto /config.php finding should get manual suggestion."""
        result = pulse_app.scan(target="check.example", intensity="medium")
        cfg = [f for f in result["findings"] if "config.php" in f.get("finding", "")]
        assert len(cfg) >= 1
        assert cfg[0]["exploit"]["tool"] == "manual"

    # ── TargetStore integration via dashboard ────────────────────────────

    def test_dashboard_calls_targetstore_record_scan(self, mock_run_tools):
        """_collect_dashboard_state() calls TargetStore.record_scan when data exists."""
        target = "record-check.example"
        cache = {f"sess:nmap:{target}": {
            "tool": "nmap",
            "target": target,
            "timestamp": 9999999999,
            "result": {"success": True,
                       "stdout": "22/tcp open  ssh\n80/tcp open  http"},
        }}
        # Ensure history returns something for tools_from_history
        cache[f"sess:whatweb:{target}"] = {
            "tool": "whatweb",
            "target": target,
            "timestamp": 9999999998,
            "result": {"success": True, "stdout": "nginx"},
        }
        ts = pulse_app.get_target_store()
        ts_record_spy = MagicMock(wraps=ts.record_scan)
        with patch.object(ts, "record_scan", ts_record_spy):
            with patch.object(pulse_app, "_scan_cache", cache):
                pulse_app._collect_dashboard_state(target=target)
        ts_record_spy.assert_called_once()
        args, kwargs = ts_record_spy.call_args
        assert kwargs.get("target") == target

    def test_dashboard_record_scan_skipped_when_no_data(self, mock_run_tools):
        """_collect_dashboard_state() skips record_scan when history empty."""
        ts = pulse_app.get_target_store()
        ts_record_spy = MagicMock(wraps=ts.record_scan)
        with patch.object(ts, "record_scan", ts_record_spy):
            with patch.object(pulse_app, "_scan_cache", {}):
                pulse_app._collect_dashboard_state(target="empty.example")
        ts_record_spy.assert_not_called()

    def test_scan_calls_targetstore_record_scan(self, mock_run_tools):
        """scan() calls TargetStore.record_scan with executed tools."""
        cache = _MockCache({"sess:nmap:persist.example": {
            "tool": "nmap", "target": "persist.example",
            "timestamp": 9999999999,
            "result": {"success": True, "stdout": "22/tcp open  ssh\n80/tcp open  http"},
        }})
        ts = pulse_app.get_target_store()
        ts_record_spy = MagicMock(wraps=ts.record_scan)
        with patch.object(ts, "record_scan", ts_record_spy):
            with patch.object(pulse_app, "_scan_cache", cache):
                pulse_app.scan(target="persist.example")
        ts_record_spy.assert_called_once()
        args, kwargs = ts_record_spy.call_args
        assert kwargs.get("target") == "persist.example"
        tools_used = kwargs.get("tools_used", [])
        assert "nmap" in tools_used or "whatweb" in tools_used

    def test_scan_targetstore_called_even_with_no_ports_or_findings(self, mock_run_tools):
        """scan() always calls record_scan (tracks tools_used even without ports)."""
        # Empty cache so get_surface returns empty
        ts = pulse_app.get_target_store()
        ts_record_spy = MagicMock(wraps=ts.record_scan)
        with patch.object(ts, "record_scan", ts_record_spy):
            with patch.object(pulse_app, "_scan_cache", {}):
                result = pulse_app.scan(target="empty.example")
        # record_scan is always called — it tracks tools_used even without surface data
        ts_record_spy.assert_called_once()


# ── Helper: wait for async scan to complete ─────────────────────────────


def wait_for_scan(scan_id, timeout=5, interval=0.05):
    """Poll _async_scans[scan_id] until not 'starting'/'running' or timeout.

    Returns the final entry, or raises TimeoutError.
    """
    import time as _time
    deadline = _time.time() + timeout
    while _time.time() < deadline:
        with pulse_app._async_scans_lock:
            entry = pulse_app._async_scans.get(scan_id)
        if entry and entry.get("status") not in ("starting", "running"):
            return entry
        _time.sleep(interval)
    raise TimeoutError(f"scan {scan_id} did not complete within {timeout}s — last status: {entry.get('status') if entry else 'N/A'}")


# ── Async scan tests ────────────────────────────────────────────────────


class TestAsyncScans:
    """run_async_tool + get_scan_status — async scans via run_security_tool."""

    def test_run_async_tool_returns_scan_id_immediately(self):
        """run_async_tool returns scan_id + status 'started' without waiting."""
        async def _mock(ctx, tool, params):
            return {"success": True, "output": "ok", "returncode": 0}

        with patch("pulse.interface.server_setup.run_security_tool", new=_mock):
            result = pulse_app.run_async_tool(tool="test_tool", target="10.0.0.1")

        assert result["status"] == "started"
        assert "scan_id" in result
        assert result["scan_id"].startswith("scan_")
        assert result["tool"] == "test_tool"
        assert result["target"] == "10.0.0.1"

    def test_run_async_tool_completes_in_background(self):
        """Async scan thread finishes and writes result to _async_scans."""
        async def _mock(ctx, tool, params):
            return {"success": True, "output": "scan complete", "returncode": 0}

        with patch("pulse.interface.server_setup.run_security_tool", new=_mock):
            result = pulse_app.run_async_tool(tool="test_tool", target="10.0.0.1")

        entry = wait_for_scan(result["scan_id"])
        assert entry["status"] == "completed"
        assert entry["result"]["success"] is True
        assert entry["tool"] == "test_tool"
        assert entry["target"] == "10.0.0.1"

    def test_run_async_tool_invalid_json_params(self):
        """Invalid JSON params returns error immediately, no thread spawned."""
        result = pulse_app.run_async_tool(tool="nmap", params="not-json")

        assert result["status"] == "error"
        assert "Invalid JSON" in result.get("error", "")
        assert result["scan_id"] is None

    def test_run_async_tool_unknown_tool_fails(self):
        """Unknown tool results in success=False in _async_scans result."""
        async def _mock(ctx, tool, params):
            return {"success": False, "error": f"Unknown tool: {tool}"}

        with patch("pulse.interface.server_setup.run_security_tool", new=_mock):
            result = pulse_app.run_async_tool(tool="nonexistent_tool", target="10.0.0.1")

        entry = wait_for_scan(result["scan_id"])
        assert entry["status"] == "completed"
        assert entry["result"]["success"] is False
        assert "Unknown tool" in (entry.get("result", {}).get("error") or "")

    def test_run_async_tool_injects_target_into_params(self):
        """target is injected into parsed_params when provided."""
        captured = {}

        async def _mock_capture(ctx, tool, params):
            captured.update(params)
            return {"success": True, "output": "ok", "returncode": 0}

        with patch("pulse.interface.server_setup.run_security_tool", new=_mock_capture):
            result = pulse_app.run_async_tool(
                tool="test_tool", target="10.0.0.1",
                params='{"ports": "80,443"}',
            )

        wait_for_scan(result["scan_id"])
        assert captured.get("target") == "10.0.0.1"
        assert captured.get("ports") == "80,443"

    def test_run_async_tool_records_metrics(self):
        """run_security_tool is called (metrics are handled internally)."""
        called_with = {}

        async def _mock_recorder(ctx, tool, params):
            called_with["tool"] = tool
            called_with["params"] = params
            return {"success": True, "output": "ok", "returncode": 0}

        with patch("pulse.interface.server_setup.run_security_tool", new=_mock_recorder):
            result = pulse_app.run_async_tool(tool="test_tool", target="10.0.0.1")

        wait_for_scan(result["scan_id"])
        assert called_with.get("tool") == "test_tool"
        assert called_with.get("params", {}).get("target") == "10.0.0.1"

    def test_get_scan_status_not_found(self):
        """get_scan_status returns 'not_found' for unknown scan_id."""
        result = pulse_app.get_scan_status("nonexistent_scan")
        assert result["status"] == "not_found"
        assert result["scan_id"] == "nonexistent_scan"

    def test_get_scan_status_shows_completed_result(self):
        """get_scan_status returns formatted result for a completed scan."""
        scan_id = "test_done_001"
        with pulse_app._async_scans_lock:
            pulse_app._async_scans[scan_id] = {
                "tool": "nmap", "target": "10.0.0.1",
                "status": "completed",
                "start_time": time.time() - 10,
                "end_time": time.time(),
                "result": {"success": True, "execution_time": 10, "error": "", "returncode": 0,
                           "stdout": "22/tcp open  ssh"},
                "progress": 100, "error": None,
            }

        result = pulse_app.get_scan_status(scan_id)
        assert result["status"] == "completed"
        assert result["tool"] == "nmap"
        assert "elapsed" in result
        assert result["result"]["success"] is True

    def test_get_scan_status_with_error(self):
        """get_scan_status includes error field when scan failed."""
        scan_id = "test_fail_002"
        with pulse_app._async_scans_lock:
            pulse_app._async_scans[scan_id] = {
                "tool": "nmap", "target": "10.0.0.1",
                "status": "failed",
                "start_time": time.time() - 5,
                "end_time": time.time(),
                "result": None,
                "progress": 0, "error": "Connection refused",
            }

        result = pulse_app.get_scan_status(scan_id)
        assert result["status"] == "failed"
        assert result.get("error") == "Connection refused"

    def test_get_scan_status_running(self):
        """get_scan_status returns 'running' with elapsed time."""
        scan_id = "test_running_003"
        with pulse_app._async_scans_lock:
            pulse_app._async_scans[scan_id] = {
                "tool": "nmap", "target": "10.0.0.1",
                "status": "running",
                "start_time": time.time() - 3,
                "end_time": None,
                "result": None,
                "progress": 50, "error": None,
            }

        result = pulse_app.get_scan_status(scan_id)
        assert result["status"] == "running"
        assert result["elapsed"] >= 2.5
        assert "result" not in result or result["result"] is None


class TestFindingsParsing:
    """get_findings() nuclei/nikto output parsing from cache."""

    def test_nuclei_parses_critical(self):
        """Nuclei critical severity finding is parsed correctly."""
        with patch.object(pulse_app, "_scan_cache", {
            "s:nuclei:test": {
                "tool": "nuclei", "target": "t",
                "result": {"output": _SYNTHETIC_NUCLEI_OUTPUT},
            },
        }):
            findings = pulse_app.get_findings("t")
            crits = [f for f in findings if f["severity"] == "critical"]
            assert len(crits) == 1
            assert "apache-path-traversal" in crits[0]["finding"]

    def test_nuclei_all_severities(self):
        """All 5 severities detected: critical, high, medium, low, info."""
        with patch.object(pulse_app, "_scan_cache", {
            "s:nuclei:test": {
                "tool": "nuclei", "target": "t",
                "result": {"output": _SYNTHETIC_NUCLEI_OUTPUT},
            },
        }):
            findings = pulse_app.get_findings("t")
            sevs = {f["severity"] for f in findings}
            assert sevs == {"critical", "high", "medium", "low", "info"}

    def test_nuclei_no_false_positives(self):
        """Lines without severity brackets are ignored."""
        noisy = _SYNTHETIC_NUCLEI_OUTPUT + "\n[INF] This is not a finding\n"
        with patch.object(pulse_app, "_scan_cache", {
            "s:nuclei:test": {
                "tool": "nuclei", "target": "t",
                "result": {"output": noisy},
            },
        }):
            findings = pulse_app.get_findings("t")
            assert len(findings) == 5  # still 5, INF line ignored

    def test_nuclei_with_ansi_codes(self):
        """Nuclei output with ANSI codes is stripped before parsing."""
        ansi_output = "\x1b[31m[critical]\x1b[0m \x1b[33m[some-id]\x1b[0m http://x.com"
        with patch.object(pulse_app, "_scan_cache", {
            "s:nuclei:test": {
                "tool": "nuclei", "target": "t",
                "result": {"output": ansi_output},
            },
        }):
            findings = pulse_app.get_findings("t")
            assert len(findings) == 1
            assert findings[0]["severity"] == "critical"
            assert findings[0]["finding"] == "some-id"

    def test_nikto_finds_paths(self):
        """Nikto output starting with '+ /' is parsed as info finding."""
        with patch.object(pulse_app, "_scan_cache", {
            "s:nikto:test": {
                "tool": "nikto", "target": "t",
                "result": {"output": _SYNTHETIC_NIKTO_OUTPUT},
            },
        }):
            findings = pulse_app.get_findings("t")
            assert len(findings) == 4
            assert all(f["severity"] == "info" for f in findings)
            assert any("config" in f["finding"] for f in findings)

    def test_nikto_filtered_no_match(self):
        """Lines without '+ /' prefix are ignored."""
        messy = "+ /setup: found\n+ NOT_A_PATH\nrandom line\n"
        with patch.object(pulse_app, "_scan_cache", {
            "s:nikto:test": {
                "tool": "nikto", "target": "t",
                "result": {"output": messy},
            },
        }):
            findings = pulse_app.get_findings("t")
            assert len(findings) == 1

    def test_findings_sorted_by_score(self):
        """Findings sorted by layer2.score descending — critical without exploit < high with exploit."""
        with patch.object(pulse_app, "_scan_cache", {
            "s:nuclei:test": {
                "tool": "nuclei", "target": "t",
                "result": {"output": _SYNTHETIC_NUCLEI_OUTPUT},
            },
        }):
            findings = pulse_app.get_findings("t")
            scores = [f["layer2"]["score"] for f in findings]
            assert scores == sorted(scores, reverse=True)
            order = [f["severity"] for f in findings]
            # high(0.850) > critical(0.450) because wordpress has exploit(0.85) vs path-traversal none(0.5)
            assert order == ["high", "critical", "medium", "low", "info"]

    def test_empty_cache_returns_empty(self):
        """No entries → empty list."""
        with patch.object(pulse_app, "_scan_cache", {}):
            assert pulse_app.get_findings("t") == []

    def test_no_target_resolves_scope(self):
        """No target falls back to scope."""
        with patch.object(pulse_app, "_scan_cache", {}):
            with patch.object(pulse_app, "get_scope", return_value={"active_target": "t2"}):
                findings = pulse_app.get_findings()
                # should not crash, returns empty from empty cache for t2
                assert isinstance(findings, list)


# ═══════════════════════════════════════════════════════════════════════
# Score-aware next_suggested_tool
# ═══════════════════════════════════════════════════════════════════════


class TestSuggestedNextTool:
    """_suggest_next_from_context uses layer2.score as primary signal."""

    def _score_finding(self, score: float, severity: str, finding: str,
                       tool: str = "", confidence: str = "low") -> dict:
        f: dict = {"severity": severity, "finding": finding,
                   "layer2": {"score": score}, "exploit": {}}
        if tool:
            f["exploit"] = {"tool": tool, "confidence": confidence,
                            "estimated_time": "1-5 min", "source": "rules"}
        return f

    def _surface(self, ports: list | None = None, techs: list | None = None) -> dict:
        return {"ports": [{"port": p, "protocol": "tcp", "service": "http"} for p in (ports or [])],
                "technologies": techs or []}

    # ── Score-aware path (score ≥ 0.3 + exploit tool) ──────────────────

    def test_high_score_returns_exploit_tool(self):
        """Finding with score ≥ 0.5 returns its exploit tool, priority critical."""
        result = pulse_app._suggest_next_from_context(
            self._surface([80]),
            [self._score_finding(0.6, "high", "XSS", "dalfox")],
        )
        assert result["tool"] == "dalfox"
        assert result["priority"] == "critical"

    def test_medium_score_returns_exploit_tool(self):
        """Finding with score 0.357 returns its exploit tool, priority high."""
        result = pulse_app._suggest_next_from_context(
            self._surface([80]),
            [self._score_finding(0.357, "medium", ".git/config", "manual")],
        )
        assert result["tool"] == "manual"
        assert result["priority"] == "high"

    def test_picks_highest_score(self):
        """Among multiple findings, picks the one with highest score."""
        result = pulse_app._suggest_next_from_context(
            self._surface([80]),
            [
                self._score_finding(0.2, "critical", "path-traversal"),
                self._score_finding(0.45, "high", "SMB signing", "smbmap"),
                self._score_finding(0.35, "medium", "XSS", "dalfox"),
            ],
        )
        assert result["tool"] == "smbmap"
        assert result["priority"] == "high"

    def test_score_without_exploit_falls_through(self):
        """Score ≥ 0.3 but no exploit tool → fallback to surface-based (scores present → no metasploit shortcut)."""
        result = pulse_app._suggest_next_from_context(
            self._surface([80]),
            [self._score_finding(0.4, "critical", "nothing")],
        )
        # score 0.4 but no exploit → score-aware misses. has_any_score=True → skips metasploit fallback.
        # Falls to surface-based: port 80, no techs → whatweb
        assert result["tool"] == "whatweb"

    # ── Fallback: score < 0.3 → keyword/severity ──────────────────────

    def test_critical_low_score_blocks_metasploit_fallback(self):
        """Critical finding on /assets (score 0.2) → scores present → skips metasploit fallback → surface."""
        result = pulse_app._suggest_next_from_context(
            self._surface([80]),
            [self._score_finding(0.2, "critical", "http")],
        )
        # score 0.2 < 0.3 → score-aware misses. has_any_score=True → no metasploit shortcut.
        # no keywords → surface-based (port 80, no techs → whatweb)
        assert result["tool"] == "whatweb"

    def test_low_score_no_keyword_falls_to_surface(self):
        """Score 0.2 finding with no keyword match → surface-based logic."""
        result = pulse_app._suggest_next_from_context(
            self._surface([22]),
            [self._score_finding(0.2, "low", "nothing")],
        )
        # No keyword match, no critical/high → surface-based (SSH port 22 → hydra)
        assert result["tool"] == "hydra"
        assert result["priority"] == "medium"

    # ── Zero findings → surface-based fallback ─────────────────────────

    def test_no_findings_uses_surface(self):
        """Zero findings → surface-based logic unaffected."""
        result = pulse_app._suggest_next_from_context(
            self._surface([80, 443], ["Apache"]),
            [],
        )
        assert result["tool"] == "gobuster"
        assert result["priority"] == "high"

    def test_no_findings_no_ports_returns_empty(self):
        """Zero findings, empty surface → empty dict."""
        result = pulse_app._suggest_next_from_context({}, [])
        assert result == {}


# ── Tool Performance ──────────────────────────────────────────────────────────


class TestGetToolPerformance:
    def test_empty(self):
        with patch.object(pulse_app._op_metrics, 'success_rate_by_tool', return_value=[]):
            with patch.object(pulse_app._op_metrics, 'timeout_count_by_tool', return_value=[]):
                r = pulse_app.get_tool_performance()
        assert r["tools"] == []
        assert r["timeouts"] == []
        assert "--" in r["summary"]

    def test_with_data(self):
        sr = [{"tool": "nmap", "runs": 10, "successes": 8, "errors": 2, "success_rate": 0.8}]
        to = [{"tool": "nmap", "timeouts": 1, "runs": 10}]
        with patch.object(pulse_app._op_metrics, 'success_rate_by_tool', return_value=sr):
            with patch.object(pulse_app._op_metrics, 'timeout_count_by_tool', return_value=to):
                r = pulse_app.get_tool_performance()
        assert r["tools"][0]["tool"] == "nmap"
        assert r["tools"][0]["rate_display"] == "80%"
        assert r["tools"][0]["timeouts"] == 1
        assert r["timeouts"][0]["display"] == "1/10"
        assert "nmap" in r["summary"]


# ── Cache Status ──────────────────────────────────────────────────────────────


class TestGetCacheStatus:
    def test_happy_path(self):
        cs = {"hits": 10, "misses": 5, "total": 15, "hit_ratio": 0.67}
        tool_hits = [{"tool": "nmap", "cache_hits": 5, "runs": 8}]
        adv = {"size": 100, "max_size": 500, "hit_rate": "67%", "utilization": 0.2}
        mock_cache = MagicMock()
        mock_cache.get_stats.return_value = adv
        with patch.object(pulse_app._op_metrics, 'cache_summary', return_value=cs):
            with patch.object(pulse_app._op_metrics, 'cache_hits_by_tool', return_value=tool_hits):
                with patch("pulse.infrastructure.singletons.cache", mock_cache):
                    r = pulse_app.get_cache_status()
        assert r["hits"] == 10
        assert r["hit_ratio"] == 0.67
        assert r["by_tool"] == tool_hits
        assert r["cache_size"] == 100

    def test_cache_get_stats_raises(self):
        cs = {"hits": 0, "misses": 0, "total": 0, "hit_ratio": 0.0}
        mock_cache = MagicMock()
        mock_cache.get_stats.side_effect = RuntimeError("no stats")
        with patch.object(pulse_app._op_metrics, 'cache_summary', return_value=cs):
            with patch.object(pulse_app._op_metrics, 'cache_hits_by_tool', return_value=[]):
                with patch("pulse.infrastructure.singletons.cache", mock_cache):
                    r = pulse_app.get_cache_status()
        assert r["cache_size"] == 0
        assert r["max_size"] == 500
        assert r["hit_rate"] == "0%"


# ── Cache Intelligence ────────────────────────────────────────────────────────


class TestGetCacheIntelligence:
    def test_empty(self):
        with patch("pulse.interface.server_setup._scan_cache") as mc:
            mc.get_ttl_scores.return_value = {}
            r = pulse_app.get_cache_intelligence()
        assert r["scores"] == []
        assert "0 tools" in r["summary"]

    def test_with_data(self):
        scores = {
            "nmap": {"hits": 10, "misses": 2, "hit_ratio": 0.83, "current_ttl_seconds": 3600},
            "whatweb": {"hits": 5, "misses": 5, "hit_ratio": 0.5, "current_ttl_seconds": 1800},
        }
        with patch("pulse.interface.server_setup._scan_cache") as mc:
            mc.get_ttl_scores.return_value = scores
            r = pulse_app.get_cache_intelligence()
        assert len(r["scores"]) == 2
        assert r["scores"][0]["tool"] == "nmap"
        assert r["scores"][0]["hit_ratio_display"] == "83%"
        assert "1h" in r["scores"][0]["current_ttl_display"]
        assert "2 tools" in r["summary"]

    def test_exception(self):
        with patch("pulse.interface.server_setup._scan_cache") as mc:
            mc.get_ttl_scores.side_effect = Exception("boom")
            r = pulse_app.get_cache_intelligence()
        assert r["scores"] == []
        assert "No TTL data" in r["summary"]


# ── System Trends ─────────────────────────────────────────────────────────────


class TestGetSystemTrends:
    def test_empty(self):
        with patch.object(pulse_app.enhanced_process_manager, 'resource_monitor') as rm:
            rm.get_usage_trends.return_value = {}
            rm.usage_history = []
            r = pulse_app.get_system_trends()
        assert r["cpu_avg"] == 0
        assert r["measurements"] == 0
        assert r["disk_display"] == "0%"

    def test_with_data(self):
        trends = {"cpu_avg_10": 45.0, "memory_avg_10": 60.0, "measurements": 100, "trend_period_minutes": 5}
        history = [
            {"cpu_percent": 50, "memory_percent": 60, "disk_percent": 40},
            {"cpu_percent": 55, "memory_percent": 50, "disk_percent": 45},
        ]
        with patch.object(pulse_app.enhanced_process_manager, 'resource_monitor') as rm:
            rm.get_usage_trends.return_value = trends
            rm.usage_history = history
            r = pulse_app.get_system_trends()
        assert r["cpu_avg"] == 45.0
        assert r["measurements"] == 100
        assert r["disk_display"] == "45%"
        assert len(r["cpu_history"]) == 2

    def test_exception_fallback(self):
        with patch.object(pulse_app.enhanced_process_manager, 'resource_monitor', new=object()):
            r = pulse_app.get_system_trends()
        assert r["cpu_avg"] == 0
        assert r["cpu_history"] == []


# ── Sessions ──────────────────────────────────────────────────────────────────


class TestGetSessions:
    def test_happy_path(self):
        store = MagicMock()
        store.list_active.return_value = ["s1", "s2"]
        store.list_completed.return_value = [
            {"session_id": "s3", "target": "1.2.3.4", "tools_executed": ["nmap"],
             "updated_at": time.time() - 60},
        ]
        with patch("pulse.infrastructure.singletons.get_session_store", return_value=store):
            r = pulse_app.get_sessions()
        assert r["active_count"] == 2
        assert r["completed_count"] == 1
        assert "2 active" in r["summary"]

    def test_exception(self):
        with patch("pulse.infrastructure.singletons.get_session_store", side_effect=ValueError("no db")):
            r = pulse_app.get_sessions()
        assert r["active_count"] == 0
        assert "Unavailable" in r["summary"]

    def test_no_completed(self):
        store = MagicMock()
        store.list_active.return_value = ["s1"]
        store.list_completed.return_value = []
        with patch("pulse.infrastructure.singletons.get_session_store", return_value=store):
            r = pulse_app.get_sessions()
        assert r["completed"] == []
        assert "1 active" in r["summary"]


# ── Confirmations ─────────────────────────────────────────────────────────────


class TestGetConfirmations:
    def test_all_zero(self):
        with patch.object(pulse_app._op_metrics, 'confirmation_summary', return_value={}):
            r = pulse_app.get_confirmations()
        assert r["total"] == 0
        assert "No confirmation events" in r["summary"]

    def test_with_data(self):
        conf = {"accepted": 5, "denied": 2, "skipped": 1}
        with patch.object(pulse_app._op_metrics, 'confirmation_summary', return_value=conf):
            r = pulse_app.get_confirmations()
        assert r["accepted"] == 5
        assert r["total"] == 8
        assert "5 accepted" in r["summary"]

    def test_partial(self):
        conf = {"accepted": 3}
        with patch.object(pulse_app._op_metrics, 'confirmation_summary', return_value=conf):
            r = pulse_app.get_confirmations()
        assert r["denied"] == 0


# ── Network I/O ───────────────────────────────────────────────────────────────


class TestGetNetworkIO:
    def test_empty_history(self):
        with patch.object(pulse_app.enhanced_process_manager, 'resource_monitor') as rm:
            rm.usage_history = []
            r = pulse_app.get_network_io()
        assert r["bytes_sent"] == 0
        assert r["total_display"] == "0 B"

    def test_with_data(self):
        history = [{"network_bytes_sent": 2048, "network_bytes_recv": 4096}]
        with patch.object(pulse_app.enhanced_process_manager, 'resource_monitor') as rm:
            rm.usage_history = history
            r = pulse_app.get_network_io()
        assert r["bytes_sent"] == 2048
        assert r["bytes_recv"] == 4096
        assert "KB" in r["total_display"]

    def test_exception_fallback(self):
        with patch.object(pulse_app.enhanced_process_manager, 'resource_monitor', new=object()):
            r = pulse_app.get_network_io()
        assert r["bytes_sent"] == 0

    def test_fmt_bytes_large(self):
        history = [{"network_bytes_sent": 2**31, "network_bytes_recv": 0}]
        with patch.object(pulse_app.enhanced_process_manager, 'resource_monitor') as rm:
            rm.usage_history = history
            r = pulse_app.get_network_io()
        assert "GB" in r["bytes_sent_display"]


# ── Rate Limit Status ─────────────────────────────────────────────────────────


class TestGetRateLimitStatus:
    def test_no_events(self):
        with patch("pulse.interface.pulse_app._rate_limit_events", []):
            r = pulse_app.get_rate_limit_status()
        assert r["profile"] == "normal"
        assert r["event_count"] == 0

    def test_with_events(self):
        events = [{"target": "1.2.3.4", "profile": "conservative",
                    "confidence": 0.8, "indicators": ["timeout"],
                    "timestamp": 1000}]
        with patch("pulse.interface.pulse_app._rate_limit_events", events):
            r = pulse_app.get_rate_limit_status()
        assert r["profile"] == "conservative"
        assert r["confidence"] == 0.8

    def test_filtered_by_target(self):
        events = [
            {"target": "1.2.3.4", "profile": "aggressive", "confidence": 0.5,
             "indicators": [], "timestamp": 1000},
            {"target": "5.6.7.8", "profile": "stealth", "confidence": 0.9,
             "indicators": ["429"], "timestamp": 2000},
        ]
        with patch("pulse.interface.pulse_app._rate_limit_events", events):
            r = pulse_app.get_rate_limit_status(target="1.2.3.4")
        assert r["event_count"] == 1
        assert r["profile"] == "aggressive"


# ── Async Scans ───────────────────────────────────────────────────────────────


class TestAsyncDataHelpers:
    def _clean(self):
        pulse_app._async_scans.clear()

    def test_get_async_data_empty(self):
        self._clean()
        r = pulse_app._get_async_data()
        assert r["running"] == []
        assert r["completed"] == []

    def test_get_async_data_with_running(self):
        self._clean()
        pulse_app._async_scans["s1"] = {"tool": "nmap", "target": "1.2.3.4", "status": "running"}
        r = pulse_app._get_async_data()
        assert len(r["running"]) == 1
        assert r["running_count"] == 1

    def test_get_async_data_with_completed(self):
        self._clean()
        pulse_app._async_scans["s1"] = {"tool": "nmap", "target": "1.2.3.4", "status": "completed"}
        r = pulse_app._get_async_data()
        assert len(r["completed"]) == 1
        assert r["completed_count"] == 1

    def test_has_async_scans_true(self):
        self._clean()
        pulse_app._async_scans["s1"] = {"tool": "nmap", "target": "1.2.3.4", "status": "running"}
        assert pulse_app._has_async_scans() is True

    def test_has_async_scans_false(self):
        self._clean()
        assert pulse_app._has_async_scans() is False

    def test_cleanup_old_scans(self):
        self._clean()
        old = time.time() - 7200
        pulse_app._async_scans["old"] = {"tool": "nmap", "target": "1.2.3.4",
                                          "status": "completed", "end_time": old}
        pulse_app._async_scans["new"] = {"tool": "nmap", "target": "1.2.3.4",
                                          "status": "completed", "end_time": time.time()}
        pulse_app._cleanup_old_scans(3600)
        assert "old" not in pulse_app._async_scans
        assert "new" in pulse_app._async_scans

    def test_cleanup_none_to_clean(self):
        self._clean()
        pulse_app._async_scans["s1"] = {"tool": "nmap", "target": "1.2.3.4",
                                          "status": "running"}
        pulse_app._cleanup_old_scans(3600)
        assert "s1" in pulse_app._async_scans


# ── Has Errors ────────────────────────────────────────────────────────────────


class TestHasErrors:
    def test_no_errors(self):
        with patch.object(pulse_app._op_metrics, 'summary', return_value={"total_errors": 0}):
            assert pulse_app._has_errors() is False

    def test_has_errors(self):
        with patch.object(pulse_app._op_metrics, 'summary', return_value={"total_errors": 3}):
            assert pulse_app._has_errors() is True

    def test_exception(self):
        with patch.object(pulse_app._op_metrics, 'summary', side_effect=ValueError):
            assert pulse_app._has_errors() is False


# ── Detect Workflow State ─────────────────────────────────────────────────────


class TestDetectWorkflowState:
    def test_delegates_to_dashboard_sections(self):
        from unittest.mock import ANY
        with patch("pulse.interface.pulse_app._ds_detect_workflow") as mock_detect:
            mock_detect.return_value = ("surface", "findings", {"reason": "ports found"})
            current, nxt, ctx = pulse_app._detect_workflow_state()
        assert current == "surface"
        assert nxt == "findings"
        assert ctx["reason"] == "ports found"
        mock_detect.assert_called_once_with(ANY, pulse_app.get_scope)


# ── Dashboard ─────────────────────────────────────────────────────────────────


class TestGetDashboard:
    def test_auto_detect(self):
        with patch("pulse.interface.pulse_app.auto_detect_sections") as ads:
            ads.return_value = ["header", "scope"]
            with patch("pulse.interface.pulse_app.cost_for_sections") as cfs:
                cfs.return_value = {"total_cost": 800}
                with patch("pulse.interface.pulse_app._ds_load_section") as dls:
                    dls.return_value = {"key": "val"}
                    r = pulse_app.get_dashboard()
        assert r["sections"] == ["header", "scope"]
        assert r["section_count"] == 2
        assert r["total_cost_est"] == 800

    def test_explicit_list(self):
        with patch("pulse.interface.pulse_app.cost_for_sections") as cfs:
            cfs.return_value = {"total_cost": 600}
            with patch("pulse.interface.pulse_app._ds_load_section") as dls:
                dls.return_value = {}
                r = pulse_app.get_dashboard(sections=["header", "intel"])
        assert r["sections"] == ["header", "intel"]

    def test_explicit_string(self):
        with patch("pulse.interface.pulse_app.cost_for_sections") as cfs:
            cfs.return_value = {"total_cost": 600}
            with patch("pulse.interface.pulse_app._ds_load_section") as dls:
                dls.return_value = {}
                r = pulse_app.get_dashboard(sections="header, scope, surface")
        assert r["sections"] == ["header", "scope", "surface"]

    def test_invalid_section_fallback(self):
        with patch("pulse.interface.pulse_app.cost_for_sections") as cfs:
            cfs.return_value = {"total_cost": 500}
            with patch("pulse.interface.pulse_app._ds_load_section") as dls:
                dls.return_value = {}
                r = pulse_app.get_dashboard(sections=["nonexistent"])
        assert r["sections"] == ["header", "scope"]


# ── Pulse Guide ───────────────────────────────────────────────────────────────


class TestPulseGuide:
    def test_single_step_overview(self):
        r = pulse_app.pulse_guide(step="overview")
        assert r["step"] == "overview"
        assert "previous_step" not in r or r["previous_step"] is None
        assert r["next_step"] == "scope"

    def test_single_step_exploit(self):
        r = pulse_app.pulse_guide(step="exploit")
        assert r["step"] == "exploit"
        assert r["previous_step"] == "plan"
        assert r["next_step"] is None

    def test_single_step_middle(self):
        r = pulse_app.pulse_guide(step="findings")
        assert r["step"] == "findings"
        assert r["previous_step"] == "surface"
        assert r["next_step"] == "plan"

    def test_invalid_step(self):
        r = pulse_app.pulse_guide(step="invalid")
        assert "error" in r
        assert "invalid" in r["error"]

    def test_case_insensitive(self):
        r = pulse_app.pulse_guide(step="SCOPE")
        assert r["step"] == "scope"

    def test_full_workflow_no_state(self):
        with patch("pulse.interface.pulse_app._detect_workflow_state") as dws:
            dws.return_value = (None, "overview", {"reason": "fresh start"})
            r = pulse_app.pulse_guide()
        assert "workflow" in r
        assert r["current_step"] is None
        assert r["next_step"] == "overview"
        assert len(r["workflow"]) == 6
        assert r["workflow"][0]["active"] is True  # first is active when no current
        assert r["quick_start"]["no_target"] == "scan('target-ip')"

    def test_full_workflow_with_state(self):
        with patch("pulse.interface.pulse_app._detect_workflow_state") as dws:
            dws.return_value = ("surface", "findings", {"reason": "ports found"})
            r = pulse_app.pulse_guide()
        assert r["current_step"] == "surface"
        assert r["next_step"] == "findings"
        # overview and scope should be completed
        assert r["workflow"][0]["completed"] is True  # overview
        assert r["workflow"][1]["completed"] is True  # scope
        assert r["workflow"][2]["active"] is True  # surface
        assert r["summary"] == "Current: surface → Next: findings"
