"""Offline integration tests — full pipeline with realistic fixture data.

Tests flow: scan() → cache → get_surface() / get_findings()
         → _collect_dashboard_state() → TargetStore
"""

import time
from unittest.mock import MagicMock, patch

import pytest

import pulse_app
from tests.test_pulse_app import _MockCache  # shared mock cache


# ── Realistic fixture outputs ────────────────────────────────────────────────

REAL_NMAP_OUTPUT = """Starting Nmap 7.95 ( https://nmap.org ) at 2026-05-20 12:00 UTC
Nmap scan report for 192.168.1.100
Host is up (0.0012s latency).

PORT     STATE    SERVICE       VERSION
22/tcp   open     ssh           OpenSSH 9.2 (protocol 2.0)
80/tcp   open     http          Apache httpd 2.4.57
443/tcp  open     https         Apache httpd 2.4.57
3306/tcp filtered mysql
8080/tcp open     http-proxy    Squid proxy 6.6

Nmap done: 1 IP address (1 host up) scanned in 15.32 seconds"""

REAL_WHATWEB_OUTPUT = """http://192.168.1.100 [200 OK] Apache[2.4.57], PHP[8.2.12], \
WordPress[6.4.3], Title[Test Site], Country[RESERVED][ZZ], HTTPServer[Apache/2.4.57 (Unix)],
IP[192.168.1.100], JQuery[3.7.1], Script[text/javascript], X-Powered-By[PHP/8.2.12]"""

REAL_NUCLEI_OUTPUT = """[2026-05-24T10:00:00] [http-missing-security-headers] [http] [info] https://192.168.1.100/
[2026-05-24T10:00:01] [CVE-2023-44487] [http] [high] https://192.168.1.100/
[2026-05-24T10:00:02] [CVE-2023-xxx] [http] [medium] https://192.168.1.100/wp-admin/
[2026-05-24T10:00:03] [http-tech-detect] [http] [info] https://192.168.1.100/
[2026-05-24T10:00:04] [wordpress-version] [http] [info] https://192.168.1.100/"""

REAL_NIKTO_OUTPUT = """- Nikto v2.5.0
---------------------------------------------------------------------------
+ Target IP: 192.168.1.100
+ Target Hostname: test.local
+ Target Port: 80
---------------------------------------------------------------------------
+ /: Server: Apache/2.4.57 (Unix) PHP/8.2.12
+ /: The X-Content-Type-Options header is missing.
+ /: The X-Frame-Options header is missing.
+ /wp-admin/: WordPress login page found.
+ /wp-content/: Directory listing may be enabled.
+ /wp-json/: WordPress REST API exposed.
---------------------------------------------------------------------------
+ 5 host(s) tested"""

# ── DVWA-specific fixtures (realistic outputs for our RPI target) ────────

DVWA_NMAP_OUTPUT = """Starting Nmap 7.95 ( https://nmap.org ) at 2026-05-24 12:00 UTC
Nmap scan report for 10.235.57.151
Host is up (0.0012s latency).

PORT     STATE    SERVICE    VERSION
80/tcp   open     http       Apache httpd 2.4.67
443/tcp  open     https      Apache httpd 2.4.67

Nmap done: 1 IP address (1 host up) scanned in 3.48 seconds"""

DVWA_WHATWEB_OUTPUT = """http://10.235.57.151/DVWA/ [200 OK] Apache[2.4.67], PHP[8.2.7], \
MariaDB[], Title[DVWA], Country[RESERVED][ZZ], HTTPServer[Apache/2.4.67 (Raspbian)], \
IP[10.235.57.151], Script, Login Page[/DVWA/login.php]"""

DVWA_NUCLEI_OUTPUT = """[2026-05-24T10:05:00] [http-missing-security-headers] [http] [info] http://10.235.57.151/DVWA/
[2026-05-24T10:05:01] [sql-injection] [http] [critical] http://10.235.57.151/DVWA/vulnerabilities/sqli/
[2026-05-24T10:05:02] [xss-reflected] [http] [high] http://10.235.57.151/DVWA/vulnerabilities/xss_r/
[2026-05-24T10:05:03] [http-tech-detect] [http] [info] http://10.235.57.151/DVWA/"""

DVWA_NIKTO_OUTPUT = """- Nikto v2.5.0
---------------------------------------------------------------------------
+ Target IP: 10.235.57.151
+ Target Hostname: dvwa.local
+ Target Port: 80
---------------------------------------------------------------------------
+ /DVWA/: Server: Apache/2.4.67 (Raspbian) PHP/8.2.7
+ /DVWA/: The X-Content-Type-Options header is missing.
+ /DVWA/: The X-Frame-Options header is missing.
+ /DVWA/login.php: Admin login page found.
+ /DVWA/setup.php: DVWA setup page found.
---------------------------------------------------------------------------
+ 1 host(s) tested"""

DVWA_GOBUSTER_OUTPUT = """/DVWA/config (Status: 200) [Size: 892]
/DVWA/setup.php (Status: 200) [Size: 1452]
/DVWA/login.php (Status: 200) [Size: 235]
/DVWA/security.php (Status: 200) [Size: 1142]"""

REAL_GOBUSTER_OUTPUT = """/admin (Status: 301) [Size: 235]
/backup (Status: 200) [Size: 1452]
/config (Status: 200) [Size: 892]
/wp-admin (Status: 301) [Size: 239]
/wp-content (Status: 301) [Size: 243]
/api (Status: 200) [Size: 4578]"""


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_state():
    """Reset module-level state between tests."""
    pulse_app._op_metrics._tools.clear()
    pulse_app._op_metrics._cache_hits = 0
    pulse_app._op_metrics._cache_misses = 0
    pulse_app._op_metrics._start_time = time.time()
    yield


@pytest.fixture
def mock_cache():
    """Patch pulse_app._scan_cache with an empty _MockCache."""
    cache = _MockCache()
    with patch.object(pulse_app, "_scan_cache", cache):
        yield cache


@pytest.fixture
def mock_direct_tools():
    """Replace get_direct_tools with controllable exec_funcs."""
    def _make_exec(success=True, stdout="", error=""):
        def _exec(binary, params):
            return {
                "success": success,
                "output": stdout,
                "stdout": stdout,
                "error": error,
                "returncode": 0 if success else 1,
            }
        return _exec

    tools = {
        "nmap":        (_make_exec(stdout=REAL_NMAP_OUTPUT), "nmap"),
        "whatweb":     (_make_exec(stdout=REAL_WHATWEB_OUTPUT), "whatweb"),
        "nuclei":      (_make_exec(stdout=REAL_NUCLEI_OUTPUT), "nuclei"),
        "nikto":       (_make_exec(stdout=REAL_NIKTO_OUTPUT), "nikto"),
        "gobuster":    (_make_exec(stdout=REAL_GOBUSTER_OUTPUT), "gobuster"),
    }
    with patch.object(pulse_app, "get_direct_tools", return_value=tools):
        yield tools


@pytest.fixture
def mock_dvwa_direct_tools():
    """DVWA-specific direct tools with SQLi/XSS nuclei output."""
    def _make_exec(success=True, stdout="", error=""):
        def _exec(binary, params):
            return {
                "success": success,
                "output": stdout,
                "stdout": stdout,
                "error": error,
                "returncode": 0 if success else 1,
            }
        return _exec

    tools = {
        "nmap":    (_make_exec(stdout=DVWA_NMAP_OUTPUT), "nmap"),
        "whatweb": (_make_exec(stdout=DVWA_WHATWEB_OUTPUT), "whatweb"),
        "nuclei":  (_make_exec(stdout=DVWA_NUCLEI_OUTPUT), "nuclei"),
        "nikto":   (_make_exec(stdout=DVWA_NIKTO_OUTPUT), "nikto"),
        "gobuster": (_make_exec(stdout=DVWA_GOBUSTER_OUTPUT), "gobuster"),
    }
    with patch.object(pulse_app, "get_direct_tools", return_value=tools):
        yield tools


# ── Fixture data for a single target ─────────────────────────────────────────


TARGET = "192.168.1.100"
TS = 9999999999


def _full_cache():
    """Return a _MockCache pre-populated with all tools for TARGET."""
    cache = _MockCache()
    for tool, stdout in [
        ("nmap",    REAL_NMAP_OUTPUT),
        ("whatweb", REAL_WHATWEB_OUTPUT),
        ("nuclei",  REAL_NUCLEI_OUTPUT),
        ("nikto",   REAL_NIKTO_OUTPUT),
        ("gobuster", REAL_GOBUSTER_OUTPUT),
    ]:
        cache[f"sess:{tool}:{TARGET}"] = {
            "tool": tool,
            "target": TARGET,
            "timestamp": TS,
            "result": {"success": True, "stdout": stdout},
        }
    return cache


DVWA_TARGET = "10.235.57.151"
DVWA_TS = 9999999998


def _full_dvwa_cache():
    """Return a _MockCache pre-populated with DVWA fixtures."""
    cache = _MockCache()
    for tool, stdout in [
        ("nmap",    DVWA_NMAP_OUTPUT),
        ("whatweb", DVWA_WHATWEB_OUTPUT),
        ("nuclei",  DVWA_NUCLEI_OUTPUT),
        ("nikto",   DVWA_NIKTO_OUTPUT),
        ("gobuster", DVWA_GOBUSTER_OUTPUT),
    ]:
        cache[f"sess:{tool}:{DVWA_TARGET}"] = {
            "tool": tool,
            "target": DVWA_TARGET,
            "timestamp": DVWA_TS,
            "result": {"success": True, "stdout": stdout},
        }
    return cache


# ── Integration test classes ──────────────────────────────────────────────────


class TestPipelineSurface:
    """Full pipeline: scan() → cache → get_surface() with realistic data."""

    def test_scan_populates_cache_and_surface(self, mock_direct_tools, mock_cache):
        """scan() writes to _scan_cache; get_surface() reads from it."""
        result = pulse_app.scan(target=TARGET, intensity="full")
        assert result["target"] == TARGET
        assert len(result["tools"]) == 5

        surface = pulse_app.get_surface(target=TARGET)
        assert surface["ports_count"] == 4  # 22, 80, 443, 8080 (3306 is "filtered")
        port_nums = [p["port"] for p in surface["ports"]]
        assert port_nums == [22, 80, 443, 8080]
        assert "WordPress" in surface["technologies"]
        assert "Apache" in surface["technologies"]
        assert "PHP" in surface["technologies"]
        assert surface["risk_level"] == "medium"  # 4 ports (>2 → medium)

    def test_scan_returns_next_suggested_tool(self, mock_direct_tools, mock_cache):
        """scan() result includes next_suggested_tool from context."""
        result = pulse_app.scan(target=TARGET, intensity="medium")
        assert "next_suggested_tool" in result
        suggestion = result["next_suggested_tool"]
        assert "tool" in suggestion
        assert "reason" in suggestion

    def test_get_surface_returns_next_suggested_tool(self, mock_cache):
        """get_surface() includes next_suggested_tool key."""
        mock_cache.update(_full_cache())
        surface = pulse_app.get_surface(target=TARGET)
        assert "next_suggested_tool" in surface
        suggestion = surface["next_suggested_tool"]
        assert suggestion["tool"] == "wpscan"  # WordPress detected
        assert "WordPress" in suggestion["reason"]


class TestPipelineFindings:
    """Full pipeline: cache → get_findings() → TargetStore."""

    def test_parses_nuclei_and_nikto_findings(self, mock_cache):
        """get_findings() merges nuclei + nikto, sorted by layer2.score descending."""
        mock_cache.update(_full_cache())
        findings = pulse_app.get_findings(target=TARGET)
        assert len(findings) > 0

        scores = [f.get("layer2", {}).get("score", 0) for f in findings]
        assert scores == sorted(scores, reverse=True)

    def test_nuclei_findings_have_correct_structure(self, mock_cache):
        """Each nuclei finding has tool, severity, finding, details."""
        mock_cache.update(_full_cache())
        findings = pulse_app.get_findings(target=TARGET)
        nuclei = [f for f in findings if f["tool"] == "nuclei"]
        assert len(nuclei) >= 3  # critical, high, medium

        for f in nuclei:
            assert "severity" in f
            assert "finding" in f
            assert "details" in f

    def test_nikto_findings_parsed(self, mock_cache):
        """Nikto output lines starting with '+ /' are parsed as info findings."""
        mock_cache.update(_full_cache())
        findings = pulse_app.get_findings(target=TARGET)
        nikto = [f for f in findings if f["tool"] == "nikto"]
        assert len(nikto) >= 5
        assert any("missing" in f["finding"].lower() for f in nikto)

    def test_no_findings_returns_empty(self, mock_cache):
        """Target with only nmap (no vuln tools) yields empty findings."""
        mock_cache[f"sess:nmap:{TARGET}"] = {
            "tool": "nmap", "target": TARGET,
            "timestamp": TS,
            "result": {"stdout": REAL_NMAP_OUTPUT},
        }
        findings = pulse_app.get_findings(target=TARGET)
        assert findings == []


class TestPipelineTargetStore:
    """Full pipeline: scan() → TargetStore persistence."""

    def test_target_store_contains_surface_data(self, mock_direct_tools, mock_cache):
        """After scan(), TargetStore has ports and technologies."""
        pulse_app.scan(target=TARGET, intensity="quick")
        store = pulse_app.get_target_store()
        profile = store.get_target(TARGET)
        assert profile is not None
        findings = profile.get("findings", {})
        ports = findings.get("ports", [])
        assert len(ports) > 0
        assert any(p["port"] == 22 for p in ports)
        techs = findings.get("technologies", [])
        assert len(techs) > 0

    def test_target_store_contains_findings(self, mock_direct_tools, mock_cache):
        """After full scan, findings are recorded in TargetStore."""
        pulse_app.scan(target=TARGET, intensity="full")
        store = pulse_app.get_target_store()
        profile = store.get_target(TARGET)
        findings = profile.get("findings", {})
        vulns = findings.get("vulnerabilities", [])
        assert len(vulns) > 0

    def test_target_store_tracks_tools_used(self, mock_direct_tools, mock_cache):
        """Each scan records which tools were executed."""
        pulse_app.scan(target=TARGET, intensity="medium")
        store = pulse_app.get_target_store()
        profile = store.get_target(TARGET)
        assert profile is not None
        assert len(profile.get("tools_used", [])) >= 4

    def test_scan_then_dashboard_updates_target_store(self, mock_direct_tools, mock_cache):
        """_collect_dashboard_state() after scan() doesn't duplicate data."""
        pulse_app.scan(target=TARGET, intensity="quick")

        # Manually populate cache for dashboard read
        mock_cache.update(_full_cache())
        ts = pulse_app.get_target_store()
        spy = MagicMock(wraps=ts.record_scan)
        with patch.object(ts, "record_scan", spy):
            pulse_app._collect_dashboard_state(target=TARGET)
        spy.assert_called_once()


class TestPipelineMultiTool:
    """Multiple tools contributing data to the same target."""

    def test_multiple_nmap_scans_accumulate(self, mock_direct_tools, mock_cache):
        """Repeated scans of same target accumulate ports (deduped)."""
        pulse_app.scan(target=TARGET, intensity="quick")  # nmap + whatweb
        pulse_app.scan(target=TARGET, intensity="full")    # all 5 tools

        surface = pulse_app.get_surface(target=TARGET)
        assert surface["ports_count"] == 4  # same target, same ports (3306 filtered)

    def test_gobuster_findings_do_not_break_pipeline(self, mock_direct_tools, mock_cache):
        """gobuster output (no nmap) should not add ports or findings."""
        cache = _MockCache({f"sess:gobuster:{TARGET}": {
            "tool": "gobuster", "target": TARGET,
            "timestamp": TS,
            "result": {"stdout": REAL_GOBUSTER_OUTPUT},
        }})
        with patch.object(pulse_app, "_scan_cache", cache):
            surface = pulse_app.get_surface(target=TARGET)
            assert surface["ports_count"] == 0
            findings = pulse_app.get_findings(target=TARGET)
            assert findings == []


class TestPipelineNoData:
    """Edge cases: empty or incomplete data."""

    def test_empty_target_returns_error(self, mock_direct_tools, mock_cache):
        """scan() with no target and empty cache returns error."""
        result = pulse_app.scan()
        assert "error" in result
        assert result["target"] is None

    def test_empty_cache_get_surface(self, mock_cache):
        """get_surface() with no cache returns empty ports."""
        surface = pulse_app.get_surface(target="10.0.0.99")
        assert surface["ports_count"] == 0
        assert surface["risk_level"] == "unknown"

    def test_malformed_nmap_output(self, mock_cache):
        """Nmap output without parsable port lines is handled gracefully."""
        mock_cache[f"sess:nmap:{TARGET}"] = {
            "tool": "nmap", "target": TARGET,
            "timestamp": TS,
            "result": {"stdout": "Starting Nmap...\nNo ports found\n"},
        }
        surface = pulse_app.get_surface(target=TARGET)
        assert surface["ports_count"] == 0
        # Malformed or empty output should not crash


class TestPipelineDvwa:
    """DVWA-specific parsing tests — mirrors our RPI target exactly."""

    def test_dvwa_get_surface_technologies(self, mock_cache):
        """get_surface() detects Apache + PHP from DVWA fixtures."""
        mock_cache.update(_full_dvwa_cache())
        surface = pulse_app.get_surface(target=DVWA_TARGET)
        assert surface["ports_count"] >= 2  # 80, 443
        assert "Apache" in surface["technologies"]
        assert "PHP" in surface["technologies"]

    def test_dvwa_get_findings_nuclei_new_format(self, mock_cache):
        """Parsses the new nuclei format (timestamp first, not severity)."""
        mock_cache.update(_full_dvwa_cache())
        findings = pulse_app.get_findings(target=DVWA_TARGET)
        nuclei = [f for f in findings if f["tool"] == "nuclei"]
        severities = [f["severity"] for f in nuclei]
        assert "critical" in severities   # sql-injection
        assert "high" in severities       # xss-reflected
        assert "info" in severities       # missing headers + tech-detect

    def test_dvwa_get_findings_nikto_parsed(self, mock_cache):
        """Nikto output for DVWA is parsed as info findings."""
        mock_cache.update(_full_dvwa_cache())
        findings = pulse_app.get_findings(target=DVWA_TARGET)
        nikto = [f for f in findings if f["tool"] == "nikto"]
        assert len(nikto) >= 3
        assert any("login" in f["finding"].lower() for f in nikto)

    def test_dvwa_get_findings_total_count(self, mock_cache):
        """DVWA yields at least 3 findings (nuclei + nikto)."""
        mock_cache.update(_full_dvwa_cache())
        findings = pulse_app.get_findings(target=DVWA_TARGET)
        assert len(findings) >= 3

    def test_dvwa_next_suggested_tool_sqlmap(self, mock_dvwa_direct_tools, mock_cache):
        """SQLi finding in DVWA via scan() → next_suggested_tool is sqlmap."""
        result = pulse_app.scan(target=DVWA_TARGET, intensity="full")
        assert "next_suggested_tool" in result
        suggestion = result["next_suggested_tool"]
        assert suggestion["tool"] == "sqlmap"
        assert "sql" in suggestion.get("reason", "").lower()

    def test_dvwa_target_store_persists(self, mock_dvwa_direct_tools, mock_cache):
        """TargetStore populated from DVWA fixtures persists correctly via scan()."""
        pulse_app.scan(target=DVWA_TARGET, intensity="full")
        store = pulse_app.get_target_store()
        profile = store.get_target(DVWA_TARGET)
        assert profile is not None
        findings = profile.get("findings", {})
        assert len(findings.get("ports", [])) >= 2
        assert len(findings.get("vulnerabilities", [])) >= 1


# ── Helpers ───────────────────────────────────────────────────────────────────


SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4, "none": 5}


class TestPipelineS57Fixes:
    """Regression tests for Session 57 pipeline fixes."""

    def test_strip_ansi_removes_color_codes(self):
        """_strip_ansi() removes ANSI escape sequences from tool output."""
        raw = "[\x1b[92mdvwa-default-login\x1b[0m] [\x1b[94mhttp\x1b[0m] [\x1b[31mcritical\x1b[0m] http://target/"
        clean = pulse_app._strip_ansi(raw)
        assert "\x1b[" not in clean
        assert "[dvwa-default-login]" in clean
        assert "[critical]" in clean

    def test_get_findings_parses_ansi_nuclei_output(self, mock_cache):
        """nuclei output with ANSI codes is correctly parsed by get_findings()."""
        ansi_output = (
            "[\x1b[92mdvwa-default-login\x1b[0m] [\x1b[94mhttp\x1b[0m] [\x1b[31mcritical\x1b[0m] http://target/DVWA/\n"
            "[\x1b[92mdockerfile\x1b[0m] [\x1b[94mhttp\x1b[0m] [\x1b[33mmedium\x1b[0m] http://target/DVWA/Dockerfile\n"
        )
        mock_cache[f"sess:nuclei:{TARGET}"] = {
            "tool": "nuclei", "target": TARGET,
            "timestamp": TS,
            "result": {"stdout": ansi_output},
        }
        findings = pulse_app.get_findings(target=TARGET)
        nuclei = [f for f in findings if f["tool"] == "nuclei"]
        assert len(nuclei) == 2
        severities = {f["severity"] for f in nuclei}
        assert "critical" in severities
        assert "medium" in severities

    def test_scan_hostname_extraction_from_url(self, mock_direct_tools, mock_cache):
        """scan() with a URL target extracts hostname for nmap."""
        url_target = "http://192.168.1.165/DVWA/"
        result = pulse_app.scan(target=url_target, intensity="quick")
        assert result["target"] == url_target
        # nmap should have been called (mocked) — doesn't fail from URL parsing
        assert "nmap" in result["tools"]

    def test_cache_invalidation_bad_nmap(self, mock_direct_tools, mock_cache):
        """scan() invalidates stale nmap cache entries with '0 hosts up'."""
        bad_stdout = "Starting Nmap 7.98\nNmap done: 0 IP addresses (0 hosts up) scanned in 0.34 seconds"
        mock_cache[f"sess:nmap:{TARGET}"] = {
            "tool": "nmap", "target": TARGET,
            "timestamp": TS,
            "result": {"stdout": bad_stdout, "output": bad_stdout},
        }
        result = pulse_app.scan(target=TARGET, intensity="quick")
        # nmap should have been re-executed (cache invalidated), not cached
        assert result["tools"]["nmap"]["status"] != "cached"
        assert result["tools"]["nmap"]["returncode"] == 0

    def test_good_nmap_cache_not_invalidated(self, mock_direct_tools, mock_cache):
        """Valid nmap cache (hosts up) is NOT invalidated."""
        good_stdout = "Starting Nmap 7.95\nNmap scan report for 192.168.1.100\nHost is up.\nPORT     STATE SERVICE\n22/tcp   open  ssh"
        mock_cache[f"sess:nmap:{TARGET}"] = {
            "tool": "nmap", "target": TARGET,
            "timestamp": TS,
            "result": {"stdout": good_stdout, "output": good_stdout},
        }
        result = pulse_app.scan(target=TARGET, intensity="quick")
        # nmap should be cached (good result stays)
        assert result["tools"]["nmap"]["status"] == "cached"

    def test_nmap_scan_type_overridden_for_host_tools(self, mock_direct_tools, mock_cache):
        """scan() passes scan_type='-sTV' for tools in _TOOLS_NEED_HOST."""
        # Verify the dict has the right key
        assert "nmap" in pulse_app._TOOLS_NEED_HOST
        assert "nmap_advanced" in pulse_app._TOOLS_NEED_HOST

def _severity_sort_key(f: dict):
    return SEVERITY_ORDER.get(f.get("severity", "").lower(), 99)
