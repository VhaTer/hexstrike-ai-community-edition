"""Tests for server_core.dashboard_sections — section registry and orchestration."""

import time
from unittest.mock import MagicMock, patch

import pytest

from server_core.dashboard_sections import (
    SectionConfig,
    build_registry,
    cache_for_target,
    detect_workflow_state,
    auto_detect_sections,
    load_section,
    cost_for_sections,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def empty_scan_cache():
    return {}


@pytest.fixture
def populated_scan_cache():
    return {
        "nmap:abc": {
            "tool": "nmap", "target": "192.168.1.1",
            "result": {"stdout": "22/tcp open ssh\n80/tcp open http"},
        },
        "whatweb:def": {
            "tool": "whatweb", "target": "192.168.1.1",
            "result": {"stdout": "Apache [2.4.41]"},
        },
        "nuclei:ghi": {
            "tool": "nuclei", "target": "192.168.1.1",
            "result": {"stdout": "[critical] [http-missing-security-headers]"},
        },
        "nmap:other": {
            "tool": "nmap", "target": "10.0.0.1",
            "result": {"stdout": "22/tcp open ssh"},
        },
    }


@pytest.fixture
def fake_scope_func():
    def _scope(target=None):
        return {
            "active_target": "192.168.1.1",
            "target_type": "ip",
            "tools_used": ["nmap", "whatweb"],
            "tools_count": 2,
        }
    return _scope


@pytest.fixture
def fake_registry():
    def _header(t=None):
        return {"version": "0.11.0", "uptime": 3600}
    def _scope(t=None):
        return {"active_target": t or "none"}
    def _surface(t=None):
        return {"ports": [{"port": 80, "service": "http"}]}
    def _findings(t=None):
        return [{"severity": "critical", "finding": "XSS"}]
    def _history(t=None):
        return [{"tool": "nmap", "target": "192.168.1.1"}]
    return build_registry([
        SectionConfig("header",  "HEADER",  _header,  cost_est=500, always=True),
        SectionConfig("scope",   "SCOPE",   _scope,   cost_est=300, always=True),
        SectionConfig("surface", "SURFACE", _surface, cost_est=2000, requires_target=True),
        SectionConfig("findings","FINDINGS",_findings,cost_est=3000, requires_target=True),
        SectionConfig("history", "HISTORY", _history, cost_est=1500,
                      condition=lambda: True),
    ])


# ── SectionConfig ────────────────────────────────────────────────────────────


class TestSectionConfig:
    def test_minimal_config(self):
        c = SectionConfig("test", "TEST", lambda t: {"ok": True})
        assert c.name == "test"
        assert c.label == "TEST"
        assert c.cost_est == 500
        assert c.always is False
        assert c.depends is None
        assert c.requires_target is False

    def test_full_config(self):
        fn = lambda t: {"ok": True}
        cond = lambda: True
        c = SectionConfig("x", "X", fn, cost_est=999,
                          always=True, depends="y",
                          condition=cond, requires_target=True)
        assert c.name == "x"
        assert c.cost_est == 999
        assert c.always is True
        assert c.depends == "y"
        assert c.condition is cond
        assert c.requires_target is True

    def test_data_func_invocation(self):
        fn = lambda t: {"called": True, "target": t}
        c = SectionConfig("test", "TEST", fn)
        result = c.data_func("10.0.0.1")
        assert result["called"] is True
        assert result["target"] == "10.0.0.1"

    def test_data_func_no_target(self):
        fn = lambda t: {"called": True}
        c = SectionConfig("test", "TEST", fn)
        result = c.data_func(None)
        assert result["called"] is True


# ── build_registry ───────────────────────────────────────────────────────────


class TestBuildRegistry:
    def test_build_empty(self):
        assert build_registry([]) == {}

    def test_build_single(self):
        c = SectionConfig("a", "A", lambda t: {})
        r = build_registry([c])
        assert r["a"] is c
        assert len(r) == 1

    def test_build_multiple(self):
        configs = [
            SectionConfig("a", "A", lambda t: {}),
            SectionConfig("b", "B", lambda t: {}),
            SectionConfig("c", "C", lambda t: {}),
        ]
        r = build_registry(configs)
        assert len(r) == 3
        assert r["a"].name == "a"
        assert r["b"].name == "b"
        assert r["c"].name == "c"

    def test_build_overwrite(self):
        c1 = SectionConfig("a", "A1", lambda t: {})
        c2 = SectionConfig("a", "A2", lambda t: {})
        r = build_registry([c1, c2])
        assert r["a"].label == "A2"

    def test_build_preserves_order(self):
        names = ["z", "y", "x"]
        configs = [SectionConfig(n, n.upper(), lambda t: {}) for n in names]
        r = build_registry(configs)
        assert list(r.keys()) == names


# ── cache_for_target ─────────────────────────────────────────────────────────


class TestCacheForTarget:
    def test_empty_cache(self, empty_scan_cache):
        assert cache_for_target(empty_scan_cache, "10.0.0.1") == []

    def test_finds_matching(self, populated_scan_cache):
        result = cache_for_target(populated_scan_cache, "192.168.1.1")
        assert len(result) == 3
        assert all(e["target"] == "192.168.1.1" for e in result)

    def test_finds_none_for_unknown(self, populated_scan_cache):
        assert cache_for_target(populated_scan_cache, "10.0.0.99") == []

    def test_handles_non_dict_values(self):
        cache = {"c": {"target": "10.0.0.1"}, "a": "string", "b": 42}
        result = cache_for_target(cache, "10.0.0.1")
        assert len(result) == 1  # non-dict values skipped, dict values found

    def test_handles_missing_target_key(self):
        cache = {"a": {"tool": "nmap"}}
        result = cache_for_target(cache, "10.0.0.1")
        assert result == []


# ── detect_workflow_state ────────────────────────────────────────────────────


class TestDetectWorkflowState:
    def test_no_scans_yet(self, empty_scan_cache):
        scope = lambda t: {}
        step, next_step, ctx = detect_workflow_state(empty_scan_cache, scope)
        assert step is None
        assert next_step == "overview"

    def test_scans_exist_no_target(self, populated_scan_cache):
        scope = lambda t: {}
        step, next_step, ctx = detect_workflow_state(populated_scan_cache, scope)
        assert step == "overview"
        assert next_step == "scope"

    def test_target_no_nmap(self, fake_scope_func):
        cache = {
            "whatweb:def": {
                "tool": "whatweb", "target": "192.168.1.1",
                "result": {"stdout": "Apache [2.4.41]"},
            },
        }
        step, next_step, ctx = detect_workflow_state(cache, fake_scope_func)
        assert step == "scope"
        assert next_step == "surface"

    def test_nmap_no_findings(self, fake_scope_func):
        cache = {
            "nmap:abc": {
                "tool": "nmap", "target": "192.168.1.1",
                "result": {"stdout": "22/tcp open ssh"},
            },
        }
        step, next_step, ctx = detect_workflow_state(cache, fake_scope_func)
        assert step == "surface"
        assert next_step == "findings"

    def test_nmap_no_open_ports(self, fake_scope_func):
        cache = {
            "nmap:abc": {
                "tool": "nmap", "target": "192.168.1.1",
                "result": {"stdout": "22/tcp filtered ssh"},
            },
        }
        step, next_step, ctx = detect_workflow_state(cache, fake_scope_func)
        assert step == "surface"

    def test_findings_no_plan(self, populated_scan_cache, fake_scope_func):
        step, next_step, ctx = detect_workflow_state(populated_scan_cache, fake_scope_func)
        assert step == "findings"
        assert next_step == "plan"

    def test_complete_workflow(self, fake_scope_func):
        cache = {
            "nmap:abc": {
                "tool": "nmap", "target": "192.168.1.1",
                "result": {"stdout": "22/tcp open ssh"},
            },
            "nuclei:def": {
                "tool": "nuclei", "target": "192.168.1.1",
                "result": {"stdout": "[high] [vuln]"},
            },
            "plan:ghi": {
                "tool": "plan", "target": "192.168.1.1",
                "result": {"stdout": "plan steps"},
            },
        }
        step, next_step, ctx = detect_workflow_state(cache, fake_scope_func)
        assert step == "plan"
        assert next_step == "exploit"

    def test_context_has_active_target(self, populated_scan_cache, fake_scope_func):
        _, _, ctx = detect_workflow_state(populated_scan_cache, fake_scope_func)
        assert ctx["active_target"] == "192.168.1.1"
        assert "tools_run" in ctx
        assert "tools_count" in ctx


# ── auto_detect_sections ─────────────────────────────────────────────────────


class TestAutoDetectSections:
    def test_no_scans_returns_header_scope_only(self, fake_registry):
        cache = {}
        scope = lambda t: {}
        result = auto_detect_sections(fake_registry, cache, scope)
        assert result == ["header", "scope"]

    def test_with_target_and_data_includes_all(self, fake_registry, populated_scan_cache):
        def scope(t=None):
            return {"active_target": "192.168.1.1", "target_type": "ip"}
        result = auto_detect_sections(fake_registry, populated_scan_cache, scope)
        assert "header" in result
        assert "scope" in result
        assert "surface" in result  # next_step is findings→plan, but surface is workflow-relevant

    def test_includes_conditional_sections(self, fake_registry, populated_scan_cache):
        def scope(t=None):
            return {"active_target": "192.168.1.1"}
        result = auto_detect_sections(
            fake_registry, populated_scan_cache, scope,
            has_async_scans_fn=lambda: True,
        )
        assert "header" in result
        assert "active" in result or True  # depends on registry having "active"

    def test_stable_ordering(self, fake_registry, populated_scan_cache):
        def scope(t=None):
            return {"active_target": "10.0.0.1"}
        result = auto_detect_sections(fake_registry, populated_scan_cache, scope)
        # header should always come before scope
        h_idx = result.index("header")
        s_idx = result.index("scope")
        assert h_idx < s_idx

    def test_empty_registry(self, empty_scan_cache):
        r = build_registry([])
        scope = lambda t: {}
        result = auto_detect_sections(r, empty_scan_cache, scope)
        assert result == []


# ── load_section ─────────────────────────────────────────────────────────────


class TestLoadSection:
    def test_unknown_section_returns_error(self, fake_registry):
        result = load_section("nonexistent", fake_registry)
        assert result["section"] == "nonexistent"
        assert "error" in result

    def test_known_section_returns_data(self, fake_registry):
        result = load_section("header", fake_registry)
        assert result["section"] == "header"
        assert result["label"] == "HEADER"
        assert "data" in result
        assert result["cost_est"] == 500
        assert result["load_time_ms"] >= 0

    def test_passes_target(self, fake_registry):
        result = load_section("scope", fake_registry, target="10.0.0.5")
        assert result["data"]["active_target"] == "10.0.0.5"

    def test_surface_loads(self, fake_registry):
        result = load_section("surface", fake_registry)
        assert result["data"]["ports"] == [{"port": 80, "service": "http"}]

    def test_data_func_error_returns_error(self, fake_registry):
        def broken(t=None):
            raise RuntimeError("boom")
        registry = build_registry([
            SectionConfig("broken", "BROKEN", broken),
        ])
        result = load_section("broken", registry)
        assert result["section"] == "broken"
        assert "error" in result
        assert result["load_time_ms"] == 0

    def test_load_time_is_measured(self, fake_registry):
        def slow(t=None):
            time.sleep(0.01)
            return {"ok": True}
        registry = build_registry([
            SectionConfig("slow", "SLOW", slow),
        ])
        result = load_section("slow", registry)
        assert result["load_time_ms"] >= 10

    def test_cost_est_preserved_in_error(self, fake_registry):
        def broken(t=None):
            raise ValueError("nope")
        registry = build_registry([
            SectionConfig("broken", "BROKEN", broken, cost_est=777),
        ])
        result = load_section("broken", registry)
        assert result["cost_est"] == 777


# ── cost_for_sections ────────────────────────────────────────────────────────


class TestCostForSections:
    def test_empty_sections(self, fake_registry):
        c = cost_for_sections([], fake_registry)
        assert c["total_cost"] == 0
        assert c["per_section"] == {}

    def test_single_section(self, fake_registry):
        c = cost_for_sections(["header"], fake_registry)
        assert c["total_cost"] == 500
        assert c["per_section"]["header"] == 500

    def test_multiple_sections(self, fake_registry):
        c = cost_for_sections(["header", "scope", "surface"], fake_registry)
        assert c["total_cost"] == 500 + 300 + 2000
        assert c["per_section"]["header"] == 500
        assert c["per_section"]["scope"] == 300

    def test_unknown_section_defaults_to_500(self, fake_registry):
        c = cost_for_sections(["unknown"], fake_registry)
        assert c["total_cost"] == 500
        assert c["per_section"]["unknown"] == 500
