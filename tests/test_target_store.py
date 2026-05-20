"""Unit tests for server_core/target_store.py — TargetStore."""

import json
import os
import threading
import time

import pytest

from server_core.target_store import TargetStore


class TestTargetStore:
    """TargetStore unit tests with temp directory isolation."""

    @pytest.fixture
    def store(self, tmp_path):
        return TargetStore(data_dir=str(tmp_path))

    # ── Construction ────────────────────────────────────────────────────

    def test_construction_creates_file_on_write(self, tmp_path):
        ts = TargetStore(data_dir=str(tmp_path))
        store_file = tmp_path / "target_store.json"
        assert not store_file.exists()
        ts.record_scan("a.example")
        assert store_file.exists()

    def test_construction_empty_store(self, store):
        assert store.get_all_targets() == []
        assert store.get_target("nonexistent") is None

    # ── record_scan ─────────────────────────────────────────────────────

    def test_record_scan_creates_profile(self, store):
        store.record_scan("test.example")
        p = store.get_target("test.example")
        assert p is not None
        assert p["target"] == "test.example"
        assert p["scan_count"] == 1

    def test_record_scan_increments_count(self, store):
        store.record_scan("x.example")
        store.record_scan("x.example")
        p = store.get_target("x.example")
        assert p["scan_count"] == 2

    def test_record_scan_with_surface_ports(self, store):
        store.record_scan("host.a", surface_data={"ports": [{"port": 80, "service": "http", "product": "nginx"}]})
        p = store.get_target("host.a")
        assert len(p["findings"]["ports"]) == 1
        assert p["findings"]["ports"][0]["port"] == 80

    def test_record_scan_with_surface_technologies(self, store):
        store.record_scan("host.b", surface_data={"technologies": ["nginx", "python"]})
        p = store.get_target("host.b")
        assert "nginx" in p["findings"]["technologies"]
        assert "python" in p["findings"]["technologies"]

    def test_record_scan_with_surface_services(self, store):
        store.record_scan("host.c", surface_data={"services": ["http", "ssh"]})
        p = store.get_target("host.c")
        assert "http" in p["findings"]["services"]
        assert "ssh" in p["findings"]["services"]

    def test_record_scan_with_findings(self, store):
        findings = [
            {"id": "XSS-001", "severity": "high", "tool": "nuclei", "details": "Reflected XSS"},
            {"id": "SQL-001", "severity": "critical", "tool": "sqlmap", "details": "SQL injection"},
        ]
        store.record_scan("vuln.example", findings=findings)
        p = store.get_target("vuln.example")
        assert len(p["findings"]["vulnerabilities"]) == 2

    def test_record_scan_with_tools_used(self, store):
        store.record_scan("t.example", tools_used=["nmap", "whatweb"])
        p = store.get_target("t.example")
        assert "nmap" in p["tools_used"]
        assert "whatweb" in p["tools_used"]

    def test_record_scan_with_session_id(self, store):
        store.record_scan("s.example", session_id="sess-001", tools_used=["nmap"])
        p = store.get_target("s.example")
        assert len(p["sessions"]) == 1
        assert p["sessions"][0]["session_id"] == "sess-001"

    def test_record_scan_deduplicates_ports(self, store):
        surf = {"ports": [{"port": 80, "service": "http"}]}
        store.record_scan("dup.example", surface_data=surf)
        store.record_scan("dup.example", surface_data=surf)
        p = store.get_target("dup.example")
        assert len(p["findings"]["ports"]) == 1

    def test_record_scan_deduplicates_vulns(self, store):
        findings = [{"id": "XSS-001", "severity": "high"}]
        store.record_scan("d.example", findings=findings)
        store.record_scan("d.example", findings=findings)
        p = store.get_target("d.example")
        assert len(p["findings"]["vulnerabilities"]) == 1

    # ── add_finding ─────────────────────────────────────────────────────

    def test_add_finding_port(self, store):
        store.record_scan("p.example")
        store.add_finding("p.example", "port", {"port": 8080, "service": "http-proxy"}, tool="nmap")
        p = store.get_target("p.example")
        assert any(e["port"] == 8080 for e in p["findings"]["ports"])

    def test_add_finding_service(self, store):
        store.record_scan("s.example")
        store.add_finding("s.example", "service", "https", tool="whatweb")
        p = store.get_target("s.example")
        assert "https" in p["findings"]["services"]

    def test_add_finding_technology(self, store):
        store.record_scan("t.example")
        store.add_finding("t.example", "technology", "react", tool="whatweb")
        p = store.get_target("t.example")
        assert "react" in p["findings"]["technologies"]

    def test_add_finding_vulnerability(self, store):
        store.record_scan("v.example")
        store.add_finding("v.example", "vulnerability", {"id": "XSS-002", "severity": "medium"}, tool="nuclei")
        p = store.get_target("v.example")
        assert len(p["findings"]["vulnerabilities"]) == 1
        assert p["findings"]["vulnerabilities"][0]["id"] == "XSS-002"

    def test_add_finding_nonexistent_target(self, store):
        store.add_finding("ghost", "port", {"port": 22})
        assert store.get_target("ghost") is None

    def test_add_finding_unknown_type_debug(self, store):
        store.record_scan("u.example")
        store.add_finding("u.example", "unknown_type", {}, tool="test")
        p = store.get_target("u.example")
        assert p is not None

    # ── get_target ──────────────────────────────────────────────────────

    def test_get_target_deep_copy(self, store):
        store.record_scan("copy.example")
        p1 = store.get_target("copy.example")
        p1["scan_count"] = 999
        p2 = store.get_target("copy.example")
        assert p2["scan_count"] == 1

    # ── get_all_targets ─────────────────────────────────────────────────

    def test_get_all_targets_returns_sorted(self, store):
        store.record_scan("z.example")
        store.record_scan("a.example")
        names = [t["target"] for t in store.get_all_targets()]
        assert names == ["a.example", "z.example"]

    def test_get_all_targets_summary_fields(self, store):
        store.record_scan("s.example", surface_data={"ports": [{"port": 443}]},
                          findings=[{"id": "V-1", "severity": "info"}], tools_used=["nmap"])
        summary = store.get_all_targets()[0]
        assert summary["scan_count"] == 1
        assert summary["tools_count"] == 1
        assert summary["findings"]["ports"] == 1
        assert summary["findings"]["vulnerabilities"] == 1

    # ── Persistence ─────────────────────────────────────────────────────

    def test_persistence_across_instances(self, tmp_path):
        ts1 = TargetStore(data_dir=str(tmp_path))
        ts1.record_scan("persistent.example", surface_data={"ports": [{"port": 22}],
                                                             "technologies": ["openssh"]},
                         findings=[{"id": "WEAK-CIPHER", "severity": "medium", "tool": "nmap"}],
                         tools_used=["nmap"])
        ts2 = TargetStore(data_dir=str(tmp_path))
        p = ts2.get_target("persistent.example")
        assert p is not None
        assert p["scan_count"] == 1
        assert len(p["findings"]["ports"]) == 1
        assert len(p["findings"]["technologies"]) == 1

    def test_persistence_corrupted_file(self, tmp_path):
        store_file = tmp_path / "target_store.json"
        store_file.write_text("{corrupted json")
        ts = TargetStore(data_dir=str(tmp_path))
        assert ts.get_all_targets() == []

    def test_persistence_non_dict_json(self, tmp_path):
        store_file = tmp_path / "target_store.json"
        store_file.write_text(json.dumps(["not", "a", "dict"]))
        ts = TargetStore(data_dir=str(tmp_path))
        assert ts.get_all_targets() == []

    # ── Thread safety ───────────────────────────────────────────────────

    def test_concurrent_record_scan(self, store):
        n = 20
        barrier = threading.Barrier(n)
        results = []

        def writer(i):
            barrier.wait()
            store.record_scan(f"concurrent-{i % 5}.example")
            results.append(i)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == n
        for i in range(5):
            p = store.get_target(f"concurrent-{i}.example")
            assert p is not None
            assert p["scan_count"] == n // 5

    def test_concurrent_add_finding(self, store):
        store.record_scan("shared.example")
        n = 10
        barrier = threading.Barrier(n)

        def adder(i):
            barrier.wait()
            store.add_finding("shared.example", "vulnerability",
                              {"id": f"V-{i:04d}", "severity": "info"}, tool="test")

        threads = [threading.Thread(target=adder, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        p = store.get_target("shared.example")
        assert len(p["findings"]["vulnerabilities"]) == n

    # ── Save failure ───────────────────────────────────────────────────

    def test_save_oserror_does_not_crash(self, tmp_path):
        """_save_locked catches OSError gracefully, in-memory data preserved."""
        subdir = tmp_path / "readonly"
        subdir.mkdir()
        os.chmod(subdir, 0o555)
        ts = TargetStore(data_dir=str(subdir))
        ts.record_scan("survive.example")
        p = ts.get_target("survive.example")
        assert p is not None
        assert p["scan_count"] == 1
        os.chmod(subdir, 0o755)

    def test_save_oserror_missing_directory(self, tmp_path):
        """_save_locked handles missing directory gracefully."""
        ts = TargetStore(data_dir=str(tmp_path))
        # After construction, move the data dir away
        gone = tmp_path / "gone"
        gone.mkdir()
        ts2 = TargetStore(data_dir=str(gone))
        import shutil
        shutil.rmtree(str(gone))
        ts2.record_scan("ghost.example")  # no crash
        p = ts2.get_target("ghost.example")
        assert p is not None
        assert p["scan_count"] == 1
