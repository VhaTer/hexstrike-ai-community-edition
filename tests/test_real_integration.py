"""
Real integration tests — fires up hexstrike exec functions and CLI against
installed tools and live targets (scanme.nmap.org, localhost).

Skips any tool not found on PATH via shutil.which().
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HEXSTRIKE = ROOT / "hexstrike.py"
SCANME = "scanme.nmap.org"


def tool_installed(name: str) -> bool:
    return shutil.which(name) is not None


# ============================================================================
# Via exec functions directly (fastest path)
# ============================================================================

class TestExecDirect:
    def test_strings_binary(self):
        from mcp_core.misc_direct import misc_exec
        r = misc_exec("strings", {"file_path": "/usr/bin/env", "max_len": 4})
        assert r.get("success"), f"strings failed: {r.get('stderr', '')[:200]}"
        out = r.get("stdout", "") or r.get("output", "")
        assert len(out) > 0

    def test_xxd_hexdump(self):
        from mcp_core.misc_direct import misc_exec
        r = misc_exec("xxd", {"file_path": "/usr/bin/env", "max_bytes": 32})
        assert r.get("success"), f"xxd failed: {r.get('stderr', '')[:200]}"
        out = r.get("stdout", "") or r.get("output", "")
        assert len(out) > 0

    def test_hashid(self):
        from mcp_core.password_cracking_direct import pwdcrack_exec
        r = pwdcrack_exec("hashid", {"hash_value": "5f4dcc3b5aa765d61d8327deb882cf99"})
        assert r.get("success"), f"hashid failed: {r.get('stderr', '')[:200]}"
        out = r.get("stdout", "") or r.get("output", "")
        assert "MD5" in out or "md5" in out.lower()

    @pytest.mark.skipif(not tool_installed("nmap"), reason="nmap not installed")
    def test_nmap_ping_scan(self):
        from mcp_core.net_scan_direct import net_scan_exec
        r = net_scan_exec("nmap", {"target": SCANME, "scan_type": "-sn"})
        assert r.get("success"), f"nmap ping scan failed: {r.get('stderr', '')[:200]}"
        out = r.get("stdout", "") or r.get("output", "")
        assert "Nmap" in out

    @pytest.mark.skipif(not tool_installed("exiftool"), reason="exiftool not installed")
    def test_exiftool(self):
        from mcp_core.misc_direct import misc_exec
        r = misc_exec("exiftool", {"file_path": str(HEXSTRIKE)})
        assert r.get("success"), f"exiftool failed: {r.get('stderr', '')[:200]}"
        out = r.get("stdout", "") or r.get("output", "")
        assert len(out) > 0

    @pytest.mark.skipif(not tool_installed("objdump"), reason="objdump not installed")
    def test_objdump(self):
        from mcp_core.misc_direct import misc_exec
        r = misc_exec("objdump", {"binary": "/usr/bin/env", "flags": "-f"})
        assert r.get("success"), f"objdump failed: {r.get('stderr', '')[:200]}"
        out = r.get("stdout", "") or r.get("output", "")
        assert "file format" in out.lower()

    @pytest.mark.skipif(not tool_installed("sqlmap"), reason="sqlmap not installed")
    def test_sqlmap_version(self):
        from mcp_core.web_scan_direct import web_scan_exec
        r = web_scan_exec("sqlmap", {"url": "http://testphp.vulnweb.com", "scan_type": "--version"})
        assert r.get("success"), f"sqlmap failed: {r.get('stderr', '')[:200]}"
        out = r.get("stdout", "") or r.get("output", "")
        assert "sqlmap" in out.lower()


# ============================================================================
# Via CLI subprocess (tests DIRECT_ROUTES routing + JSON output)
# ============================================================================

class TestCLIScan:
    """Tests the CLI scan subcommand end-to-end."""

    def test_cli_unknown_tool(self):
        r = subprocess.run(
            [sys.executable, str(HEXSTRIKE), "scan", "nonexistent_tool_xyz"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode != 0
        assert "Unknown" in r.stderr or "Unknown" in r.stdout

    def test_cli_json_flag(self):
        r = subprocess.run(
            [sys.executable, str(HEXSTRIKE), "scan", "strings",
             "", "-p", "file_path=/usr/bin/env", "-p", "max_len=4", "--json"],
            capture_output=True, text=True, timeout=30,
        )
        assert r.returncode == 0, f"CLI failed: {r.stderr}"
        data = json.loads(r.stdout)
        assert data.get("success") is True
        assert len(data.get("stdout", "") or data.get("output", "")) > 0

    def test_cli_output_file(self, tmp_path):
        out_file = tmp_path / "result.json"
        r = subprocess.run(
            [sys.executable, str(HEXSTRIKE), "scan", "xxd",
             "", "-p", "file_path=/usr/bin/env", "-p", "max_bytes=16", "--json",
             "-o", str(out_file)],
            capture_output=True, text=True, timeout=30,
        )
        assert r.returncode == 0, f"CLI failed: {r.stderr}"
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert data.get("success") is True

    def test_cli_tools_list(self):
        r = subprocess.run(
            [sys.executable, str(HEXSTRIKE), "tools"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 0
        assert "nmap" in r.stdout

    @pytest.mark.skipif(not tool_installed("nmap"), reason="nmap not installed")
    def test_cli_nmap_json(self):
        r = subprocess.run(
            [sys.executable, str(HEXSTRIKE), "scan", "nmap", SCANME,
             "-p", "scan_type=-sn", "--json"],
            capture_output=True, text=True, timeout=60,
        )
        assert r.returncode == 0, f"nmap CLI failed: {r.stderr[:200]}"
        data = json.loads(r.stdout)
        assert data.get("success") is True
        out = data.get("stdout", "") or data.get("output", "")
        assert "Nmap" in out or "Host is up" in out

    def test_cli_version(self):
        r = subprocess.run(
            [sys.executable, str(HEXSTRIKE), "--version"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 0
        assert "hexstrike" in r.stdout.lower()

    @pytest.mark.skipif(not tool_installed("hashid"), reason="hashid not installed")
    def test_cli_hashid_json(self):
        r = subprocess.run(
            [sys.executable, str(HEXSTRIKE), "scan", "hashid",
             "", "-p", "hash_value=5f4dcc3b5aa765d61d8327deb882cf99", "--json"],
            capture_output=True, text=True, timeout=30,
        )
        assert r.returncode == 0, f"hashid CLI failed: {r.stderr[:200]}"
        data = json.loads(r.stdout)
        assert data.get("success") is True
