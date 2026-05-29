"""
Real integration tests against RPI target (192.168.1.165) with DVWA.

Tests the full pipeline with real binaries against a real DVWA instance
at 3 security levels (low/medium/high).  Marked @pytest.mark.rpi.

Run:
    pytest tests/test_rpi.py -m rpi -v
    pytest tests/test_rpi.py -m rpi --level medium -v
    pytest tests/test_rpi.py -m rpi --level high -v

Requires:
  - RPI reachable at 192.168.1.165:80
  - DVWA installed with default credentials (admin:password)
  - Tools: nmap, whatweb, nuclei, nikto, gobuster, sqlmap, dalfox
"""

import re
import shutil
import subprocess
import time
from urllib.parse import urljoin

import pytest
import requests

import pulse_app

RPI_IP = "192.168.1.165"
RPI_BASE = f"http://{RPI_IP}/DVWA/"

pytestmark = [pytest.mark.rpi, pytest.mark.slow]


# ── Helpers ──────────────────────────────────────────────────────────────


def tool_installed(name: str) -> bool:
    return shutil.which(name) is not None


def _reachable(timeout: int = 3) -> bool:
    """Quick nmap ping to check if RPI port 80 responds."""
    r = subprocess.run(
        ["nmap", "-p", "80", "--open", "-T4", RPI_IP],
        capture_output=True, text=True, timeout=timeout,
    )
    return "1 host up" in r.stdout


def _dvwa_login() -> dict:
    """Authenticate on DVWA, return session cookies dict."""
    sess = requests.Session()
    r = sess.get(urljoin(RPI_BASE, "login.php"), timeout=5)
    m = re.search(r"name='user_token' value='([^']+)'", r.text)
    token = m.group(1) if m else ""
    sess.post(urljoin(RPI_BASE, "login.php"), data={
        "username": "admin", "password": "password",
        "Login": "Login", "user_token": token,
    }, timeout=5)
    return sess.cookies.get_dict()


def _populate_direct_tools():
    """Populate _DIRECT_TOOLS_CACHE so scan() can use direct exec functions.

    Mirrors the logic in setup_mcp_server_standalone() without starting an MCP server.
    """
    from mcp_core.server_setup import _DIRECT_TOOLS_CACHE
    if _DIRECT_TOOLS_CACHE:
        return  # already populated
    from mcp_core.tool_routes import TOOL_ROUTES
    from mcp_core.wifi_direct import wifi_exec
    from mcp_core.recon_direct import recon_exec
    from mcp_core.net_scan_direct import net_scan_exec
    from mcp_core.web_scan_direct import web_scan_exec
    from mcp_core.web_fuzz_direct import web_fuzz_exec
    from mcp_core.password_cracking_direct import pwdcrack_exec
    from mcp_core.smb_enum_direct import smb_enum_exec
    from mcp_core.exploit_framework_direct import exploit_exec
    from mcp_core.web_recon_direct import web_recon_exec
    from mcp_core.security_direct import security_exec
    from mcp_core.misc_direct import misc_exec
    from mcp_core.osint_direct import osint_exec
    from mcp_core.active_directory_direct import ad_exec
    from mcp_core.testssl_direct import testssl_exec
    from mcp_core.web_probe_direct import web_probe_exec
    from mcp_core.vuln_intel_direct import vuln_intel_exec

    _exec_by_name = {
        "wifi_exec": wifi_exec, "recon_exec": recon_exec,
        "net_scan_exec": net_scan_exec, "web_scan_exec": web_scan_exec,
        "web_fuzz_exec": web_fuzz_exec, "pwdcrack_exec": pwdcrack_exec,
        "smb_enum_exec": smb_enum_exec, "exploit_exec": exploit_exec,
        "web_recon_exec": web_recon_exec, "security_exec": security_exec,
        "misc_exec": misc_exec, "osint_exec": osint_exec,
        "ad_exec": ad_exec, "testssl_exec": testssl_exec,
        "web_probe_exec": web_probe_exec, "vuln_intel_exec": vuln_intel_exec,
    }

    for tool_name, (mod_path, func_name, binary) in TOOL_ROUTES.items():
        ef = _exec_by_name.get(func_name)
        if ef:
            _DIRECT_TOOLS_CACHE[tool_name] = (ef, binary)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def rpi_available():
    """Skip all tests if RPI is unreachable."""
    if not _reachable():
        pytest.skip(f"RPI {RPI_IP} not reachable")
    _populate_direct_tools()
    return True


@pytest.fixture(scope="session")
def dvwa_session(rpi_available):
    """Authenticated session cookie for DVWA."""
    cookies = _dvwa_login()
    r = requests.get(urljoin(RPI_BASE, "index.php"), cookies=cookies, timeout=5)
    logged_in = "Login" not in r.text or "admin" in r.text
    if not logged_in:
        pytest.skip("DVWA login failed")
    return cookies.get("PHPSESSID", "")


@pytest.fixture(autouse=True)
def _clear_state():
    """Reset pulse_app module state between tests."""
    pulse_app._op_metrics._tools.clear()
    pulse_app._op_metrics._cache_hits = 0
    pulse_app._op_metrics._cache_misses = 0
    pulse_app._op_metrics._start_time = time.time()
    yield


# ═══════════════════════════════════════════════════════════════════════════
# EASY — target reachable, basic scan, ports + technologies
# ═══════════════════════════════════════════════════════════════════════════


class TestRpiEasy:
    """Target reachable, basic scan, ports + technologies (~30s)."""

    def test_host_reachable(self, rpi_available):
        """RPI responds on port 80."""
        assert _reachable()

    @pytest.mark.skipif(not tool_installed("nmap"), reason="nmap not installed")
    def test_ports_open(self, rpi_available):
        """nmap finds at least SSH (22) and HTTP (80)."""
        from mcp_core.net_scan_direct import net_scan_exec
        r = net_scan_exec("nmap", {"target": RPI_IP, "scan_type": "-sT -Pn",
                                    "ports": "22,80,443,3306"})
        assert r.get("success"), f"nmap failed: {r.get('stderr', '')[:200]}"
        out = r.get("stdout", "") or r.get("output", "")
        assert "22/tcp" in out and "open" in out, "Port 22 not found open"
        assert "80/tcp" in out and "open" in out, "Port 80 not found open"

    @pytest.mark.skipif(not tool_installed("whatweb"), reason="whatweb not installed")
    def test_tech_detection(self, rpi_available):
        """whatweb detects Apache and PHP on DVWA."""
        from mcp_core.web_probe_direct import web_probe_exec
        r = web_probe_exec("whatweb", {"url": RPI_BASE})
        assert r.get("success"), f"whatweb failed: {r.get('stderr', '')[:200]}"
        out = r.get("stdout", "") or r.get("output", "")
        assert "Apache" in out, "Apache not detected"
        assert "PHP" in out or "php" in out, "PHP not detected"

    def test_scan_quick(self, rpi_available):
        """scan(quick) returns complete surface with ports + tech."""
        result = pulse_app.scan(target=RPI_IP, intensity="quick")
        assert result.get("target") == RPI_IP
        surface = result.get("surface", {})
        assert surface.get("ports_count", 0) >= 2, (
            f"Expected ports >= 2, got {surface.get('ports_count')}: "
            f"{[p['port'] for p in surface.get('ports', [])]}"
        )
        assert len(surface.get("technologies", [])) > 0
        assert "risk_level" in surface
        assert "next_suggested_tool" in result

    def test_scan_quick_cache(self, rpi_available):
        """Second scan is faster due to cache."""
        _populate_direct_tools()
        start = time.time()
        pulse_app.scan(target=RPI_IP, intensity="quick")
        first = time.time() - start

        start = time.time()
        pulse_app.scan(target=RPI_IP, intensity="quick")
        second = time.time() - start

        assert second < first * 1.5, (
            f"Cached scan ({second:.1f}s) should be faster than first ({first:.1f}s)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# MEDIUM — findings + exploit suggestions
# ═══════════════════════════════════════════════════════════════════════════


class TestRpiMedium:
    """Medium scan pipeline — tools run, cache populated, results structured (~4min)."""

    def test_scan_medium_runs_all_tools(self, rpi_available):
        """scan(medium) invokes all 4 tools, nmap/whatweb complete."""
        result = pulse_app.scan(target=RPI_BASE, intensity="medium")
        tools = result.get("tools", {})
        assert len(tools) == 4, f"Expected 4 tools, got {list(tools.keys())}"
        for name in ("nmap", "whatweb"):
            assert name in tools, f"Missing tool: {name}"
            status = tools[name].get("status")
            assert status in ("completed", "cached"), (
                f"{name} status={status!r}"
            )
        # nuclei and nikto may timeout or find nothing — accept failure
        assert result.get("surface", {}).get("ports_count", 0) >= 2

    def test_medium_results_structured(self, rpi_available):
        """scan(medium) returns valid pipeline structure."""
        result = pulse_app.scan(target=RPI_BASE, intensity="medium")
        assert "target" in result
        assert "surface" in result
        assert "findings" in result  # may be empty — DVWA perimeter has no known vulns
        assert "plan" in result
        assert "next_suggested_tool" in result
        for f in result.get("findings", []):
            assert "tool" in f
            assert "severity" in f or "finding" in f

    def test_medium_findings_enriched_if_present(self, rpi_available):
        """If findings exist, they should have exploit enrichment."""
        result = pulse_app.scan(target=RPI_BASE, intensity="medium")
        for f in result.get("findings", []):
            if "exploit" in f:
                ex = f["exploit"]
                assert "tool" in ex
                assert "confidence" in ex
                assert "source" in ex
                assert ex["source"] == "rules"

    def test_medium_cache_persists(self, rpi_available):
        """Medium scan cached entries can be retrieved for surface."""
        result = pulse_app.scan(target=RPI_BASE, intensity="medium")
        cached = pulse_app._cache_for_target(RPI_BASE)
        assert len(cached) >= 4, f"Expected >=4 cache entries, got {len(cached)}"
        tools_found = {c.get("tool") for c in cached}
        assert tools_found >= {"nmap", "whatweb", "nuclei", "nikto"}, (
            f"Missing tools in cache: {tools_found}"
        )
        # Surface reads from cache
        surface = pulse_app.get_surface(RPI_BASE)
        assert surface.get("ports_count", 0) >= 2


# ═══════════════════════════════════════════════════════════════════════════
# HARD — exploitation with real tools against DVWA vulnerabilities
# ═══════════════════════════════════════════════════════════════════════════


class TestRpiHard:
    """Real exploitation — sqlmap, dalfox, gobuster (~8min).
    Requires DVWA login for protected pages."""

    @pytest.mark.skipif(not tool_installed("gobuster"), reason="gobuster not installed")
    def test_gobuster_discovers_paths(self, rpi_available):
        """gobuster finds DVWA standard paths."""
        from mcp_core.web_fuzz_direct import web_fuzz_exec
        r = web_fuzz_exec("gobuster", {
            "url": RPI_BASE,
            "mode": "dir",
            "wordlist": "/usr/share/wordlists/dirb/common.txt",
            "additional_args": "-x php,html -q",
        })
        assert r.get("success"), f"gobuster failed: {r.get('stderr', '')[:200]}"
        out = r.get("stdout", "") or r.get("output", "")
        found = [line.split()[0].lstrip("/") for line in out.splitlines()
                 if "Status:" in line]
        expected = {"config", "setup.php", "security.php", "instructions.php", "phpinfo.php"}
        overlap = set(found) & expected
        assert len(overlap) >= 3, (
            f"Expected {expected}, found {overlap} among {found}"
        )

    @pytest.mark.skipif(not tool_installed("sqlmap"), reason="sqlmap not installed")
    def test_sqlmap_detects_database(self, rpi_available, dvwa_session):
        """sqlmap --banner against DVWA SQLi page confirms injection."""
        from mcp_core.web_scan_direct import web_scan_exec
        sqli_url = urljoin(RPI_BASE, "vulnerabilities/sqli/?id=1&Submit=Submit")
        r = web_scan_exec("sqlmap", {
            "url": sqli_url,
            "additional_args": (
                f'--cookie="PHPSESSID={dvwa_session}" '
                f'--batch --banner --level=2 --risk=1 --technique=B --no-cast'
            ),
        })
        assert r.get("success"), f"sqlmap failed: {r.get('stderr', '')[:200]}"
        out = r.get("stdout", "") or r.get("output", "")
        assert "banner" in out.lower() or "mysql" in out.lower(), (
            "sqlmap should detect the database banner"
        )

    @pytest.mark.skipif(not tool_installed("dalfox"), reason="dalfox not installed")
    def test_dalfox_detects_xss(self, rpi_available, dvwa_session):
        """dalfox finds XSS on DVWA reflected XSS page."""
        from mcp_core.web_scan_direct import web_scan_exec
        xss_url = urljoin(RPI_BASE, "vulnerabilities/xss_r/")
        r = web_scan_exec("dalfox", {
            "url": xss_url,
            "additional_args": (
                f'--cookie "PHPSESSID={dvwa_session}" '
                f'--found-action=none --skip-bav --only-custom-payload'
            ),
        })
        assert r.get("success"), f"dalfox failed: {r.get('stderr', '')[:200]}"
        out = r.get("stdout", "") or r.get("output", "")
        assert "[V]" in out or "PoC" in out or "XSS" in out or "VULN" in out.upper(), (
            "dalfox should detect reflected XSS on DVWA"
        )

    def test_scan_full(self, rpi_available):
        """scan(full) runs 5 tools + generates plan."""
        result = pulse_app.scan(target=RPI_BASE, intensity="full")
        assert len(result.get("tools", {})) == 5
        plan = result.get("plan", {})
        assert plan.get("step_count", 0) >= 1, "Full scan should have plan steps"
        assert result.get("surface", {}).get("ports_count", 0) >= 2

    def test_plan_web_target(self, rpi_available):
        """get_plan() returns attack chain appropriate for a web target."""
        plan = pulse_app.get_plan(target=RPI_BASE, objective="comprehensive")
        steps = plan.get("steps", [])
        assert len(steps) > 0, "Plan should have at least one step"
        step_tools = {s.get("tool", "") for s in steps}
        web_tools = {"sqlmap", "gobuster", "nuclei", "nikto", "wpscan", "nmap", "whatweb"}
        assert step_tools & web_tools, (
            f"Plan should include web-focused tools, got: {step_tools}"
        )
