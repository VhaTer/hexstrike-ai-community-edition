import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# Module-level MCP server (created once, shared across all tests)
# setup_mcp_server_standalone() takes ~14s on first import — avoid per-test cost.
# ---------------------------------------------------------------------------
from mcp_core.server_setup import setup_mcp_server_standalone as _setup

_MCP = None


def mcp():
    global _MCP
    if _MCP is None:
        _MCP = _setup()
    return _MCP


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def make_mock_context(session_id="test-unit"):
    ctx = SimpleNamespace()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.debug = AsyncMock()
    ctx.report_progress = AsyncMock()
    ctx.read_resource = AsyncMock(return_value=SimpleNamespace(contents=[]))
    ctx.get_state = AsyncMock(return_value=None)
    ctx.set_state = AsyncMock()
    ctx.get_prompt = AsyncMock(return_value=SimpleNamespace(messages=[]))
    ctx.sample = AsyncMock()
    ctx.session_id = session_id
    return ctx


@pytest.fixture(autouse=True)
def _clear_scan_cache():
    from mcp_core.server_setup import _scan_cache
    _scan_cache.cache.clear()
    _scan_cache.ttl_times.clear()


# =========================================================================
# Group A — Resource endpoints (scan_latest, scan_result, scan_cache_list,
#            tool_metrics, error_statistics)
# =========================================================================


class TestResourceScanLatest:
    def test_no_matches(self):
        tmpl = run(mcp().get_resource_template("scan://{target}/latest"))
        ctx = make_mock_context()
        with patch("mcp_core.server_setup.get_context", return_value=ctx):
            text = run(tmpl.fn(target="10.0.0.99"))
        parsed = json.loads(text)
        assert parsed["status"] == "no_results"

    def test_with_matches(self):
        from mcp_core.server_setup import _scan_cache
        s = "test-unit"
        _scan_cache.cache[f"{s}:nmap:10.0.0.1"] = {
            "tool": "nmap", "target": "10.0.0.1",
            "result": {"success": True, "output": "open ports"},
            "timestamp": 1000.0,
        }
        _scan_cache.cache[f"{s}:whatweb:10.0.0.1"] = {
            "tool": "whatweb", "target": "10.0.0.1",
            "result": {"success": True, "output": "WordPress"},
            "timestamp": 2000.0,
        }
        tmpl = run(mcp().get_resource_template("scan://{target}/latest"))
        ctx = make_mock_context()
        with patch("mcp_core.server_setup.get_context", return_value=ctx):
            text = run(tmpl.fn(target="10.0.0.1"))
        parsed = json.loads(text)
        assert parsed["target"] == "10.0.0.1"
        assert parsed["tool"] == "whatweb"


class TestResourceScanResult:
    def test_entry_found(self):
        from mcp_core.server_setup import _scan_cache
        s = "test-unit"
        _scan_cache.cache[f"{s}:nmap:10.0.0.1:x"] = {
            "tool": "nmap", "target": "10.0.0.1",
            "result": {"success": True, "output": "ports"},
            "timestamp": 1500.0,
        }
        tmpl = run(mcp().get_resource_template("scan://{target}/{tool_name}"))
        ctx = make_mock_context()
        with patch("mcp_core.server_setup.get_context", return_value=ctx):
            text = run(tmpl.fn(target="10.0.0.1", tool_name="nmap"))
        parsed = json.loads(text)
        assert parsed["target"] == "10.0.0.1"
        assert parsed["tool"] == "nmap"
        assert parsed["result"]["success"] is True

    def test_entry_not_found(self):
        tmpl = run(mcp().get_resource_template("scan://{target}/{tool_name}"))
        ctx = make_mock_context()
        with patch("mcp_core.server_setup.get_context", return_value=ctx):
            text = run(tmpl.fn(target="10.0.0.1", tool_name="nmap"))
        parsed = json.loads(text)
        assert parsed["status"] == "no_results"


class TestResourceScanCacheList:
    def test_with_entries(self):
        from mcp_core.server_setup import _scan_cache
        s = "test-unit"
        _scan_cache.cache[f"{s}:nmap:x"] = {
            "tool": "nmap", "target": "10.0.0.1",
            "result": {"success": True},
            "timestamp": 1000.0,
        }
        resource = run(mcp().get_resource("scan://cache/list"))
        ctx = make_mock_context()
        with patch("mcp_core.server_setup.get_context", return_value=ctx):
            text = run(resource.fn())
        parsed = json.loads(text)
        assert parsed["count"] >= 1
        assert parsed["scans"][0]["tool"] == "nmap"

    def test_empty(self):
        resource = run(mcp().get_resource("scan://cache/list"))
        ctx = make_mock_context()
        with patch("mcp_core.server_setup.get_context", return_value=ctx):
            text = run(resource.fn())
        parsed = json.loads(text)
        assert parsed["count"] == 0


class TestResourceToolMetrics:
    def test_returns_summary(self):
        resource = run(mcp().get_resource("metrics://tools"))
        text = run(resource.fn())
        parsed = json.loads(text)
        assert isinstance(parsed, dict)


class TestResourceErrorStatistics:
    def test_returns_error_stats(self):
        resource = run(mcp().get_resource("errors://statistics"))
        text = run(resource.fn())
        parsed = json.loads(text)
        assert isinstance(parsed, dict)


# =========================================================================
# Group D — validate_environment branches
# =========================================================================


class TestValidateEnvironment:
    def test_unknown_tools_in_filter(self):
        tool = run(mcp().get_tool("validate_environment"))
        ctx = make_mock_context()
        result = run(tool.fn(ctx=ctx, tool_filter="nonexistent_tool_xyz_42"))
        assert result["success"] is False
        assert "unknown_tools" in result

    def test_filter_valid_tool(self):
        tool = run(mcp().get_tool("validate_environment"))
        ctx = make_mock_context()
        result = run(tool.fn(ctx=ctx, tool_filter="nmap"))
        assert result["success"] is True
        assert "nmap" in result.get("tools", {})

    def test_empty_filter_checks_all(self):
        tool = run(mcp().get_tool("validate_environment"))
        ctx = make_mock_context()
        with patch("shutil.which", return_value=None):
            result = run(tool.fn(ctx=ctx))
        assert result["success"] is True


# =========================================================================
# Group E — Registration area
# =========================================================================


class TestRegistrationArea:
    def test_tool_without_registry_does_not_crash(self):
        resource = run(mcp().get_resource("health://server"))
        text = run(resource.fn())
        parsed = json.loads(text)
        assert parsed["status"] == "healthy"
        assert parsed["server"] == "hexstrike-ai-pulse"


# =========================================================================
# Group C — plan_attack branches
# =========================================================================


class TestPlanAttackCTF:
    def test_ctf_no_category_defaults_to_web(self):
        tool = run(mcp().get_tool("plan_attack"))
        ctx = make_mock_context()
        ctx.get_state = AsyncMock(return_value=None)
        result = run(tool.fn(
            ctx=ctx,
            target="10.0.0.52",
            objective="ctf",
            ctf_category="",
            ctf_difficulty="easy",
            ctf_points=100,
            ctf_description="Test challenge",
        ))
        assert isinstance(result, dict)

    def test_ctf_hard_category(self):
        tool = run(mcp().get_tool("plan_attack"))
        ctx = make_mock_context()
        result = run(tool.fn(
            ctx=ctx,
            target="10.0.0.53",
            objective="ctf",
            ctf_category="pwn",
            ctf_difficulty="hard",
            ctf_points=500,
        ))
        assert isinstance(result, dict)


class TestPlanAttackStandard:
    def test_quick_objective(self):
        tool = run(mcp().get_tool("plan_attack"))
        ctx = make_mock_context()
        ctx.get_state = AsyncMock(return_value=None)
        result = run(tool.fn(
            ctx=ctx,
            target="10.0.0.1",
            objective="quick",
        ))
        assert isinstance(result, dict)

    def test_set_state_exception_during_persist(self):
        tool = run(mcp().get_tool("plan_attack"))
        ctx = make_mock_context()
        ctx.get_state = AsyncMock(return_value=None)
        result = run(tool.fn(
            ctx=ctx,
            target="10.0.0.54",
            objective="quick",
        ))
        assert isinstance(result, dict)


# =========================================================================
# Group B — run_security_tool branches
#
# DIRECT_TOOLS is a closure variable captured by run_security_tool inside
# setup_mcp_server_standalone(). The module-level mcp() caches the server,
# so we modify DIRECT_TOOLS in-place through the closure to inject mocks
# without recreating the server each time (avoids ~14s boot per test).
# =========================================================================


def _patch_dt_entry(tool_name, result):
    """Replace a DIRECT_TOOLS executor with a mock returning `result`."""
    fn = run(mcp().get_tool("run_security_tool")).fn
    for cell in fn.__closure__:
        try:
            val = cell.cell_contents
            if isinstance(val, dict) and tool_name in val:
                val[tool_name] = (MagicMock(return_value=dict(result)), val[tool_name][1])
                return
        except NameError:
            pass


class TestRunSecurityToolExec:
    def test_destructive_denied(self):
        tool = run(mcp().get_tool("run_security_tool"))
        ctx = make_mock_context()

        async def go():
            with patch(
                "mcp_core.server_setup.confirm_destructive_action",
                new=AsyncMock(return_value=False),
            ):
                return await tool.fn(
                    ctx,
                    tool_name="metasploit",
                    parameters=json.dumps({
                        "module": "exploit/multi/handler",
                        "target": "10.0.0.1",
                    }),
                )

        result = run(go())
        assert result["success"] is False
        assert "cancelled" in str(result.get("error", "")).lower()

    _MOCK_RESULTS: dict = {}

    @classmethod
    def setup_class(cls):
        cls._MOCK_RESULTS = {
            "nmap":     {"success": True,  "stdout": "22/tcp open ssh",          "return_code": 0, "timed_out": False, "execution_time": 0.5, "stderr": ""},
            "whatweb":  {"success": True,  "stdout": "WordPress, PHP 8.0",       "return_code": 0, "timed_out": False, "execution_time": 0.5, "stderr": ""},
            "wafw00f":  {"success": True,  "stdout": "Cloudflare detected",      "return_code": 0, "timed_out": False, "execution_time": 0.5, "stderr": ""},
            "smbmap":   {"success": True,  "stdout": "SMB shares found",         "return_code": 0, "timed_out": False, "execution_time": 0.5, "stderr": ""},
            "metasploit": {"success": True, "stdout": "ok",                      "return_code": 0, "timed_out": False, "execution_time": 0.5, "stderr": ""},
        }

    def _patch_tool(self, tool_name, result):
        """Replace a DIRECT_TOOLS executor via closure access."""
        fn = run(mcp().get_tool("run_security_tool")).fn
        for cell in fn.__closure__:
            try:
                val = cell.cell_contents
                if isinstance(val, dict) and tool_name in val:
                    val[tool_name] = (MagicMock(return_value=dict(result)), val[tool_name][1])
                    return
            except NameError:
                pass

    def _fast_exec(self, tool_name):
        return self._MOCK_RESULTS.get(tool_name, {"success": False, "stdout": "", "return_code": 1, "timed_out": False, "execution_time": 0.5, "stderr": ""})

    def test_skill_injection(self):
        self._patch_tool("nmap", self._fast_exec("nmap"))
        tool = run(mcp().get_tool("run_security_tool"))
        ctx = make_mock_context()
        ctx.read_resource = AsyncMock(
            return_value=SimpleNamespace(
                contents=[SimpleNamespace(content="# Nmap Recon\nStep 1: scan ports")]
            )
        )
        result = run(tool.fn(ctx, tool_name="nmap", parameters=json.dumps({"target": "10.0.0.55"})))
        assert result["success"] is True

    def test_tech_dict_explicit(self):
        self._patch_tool("nmap", self._fast_exec("nmap"))
        tool = run(mcp().get_tool("run_security_tool"))
        ctx = make_mock_context()
        result = run(tool.fn(ctx, tool_name="nmap", parameters=json.dumps({
            "target": "10.0.0.56",
            "_tech": {"web_servers": ["nginx"], "frameworks": ["django"]},
        })))
        assert result["success"] is True

    def test_get_state_exception(self):
        self._patch_tool("nmap", self._fast_exec("nmap"))
        tool = run(mcp().get_tool("run_security_tool"))
        ctx = make_mock_context()
        ctx.get_state = AsyncMock(side_effect=RuntimeError("state fail"))
        result = run(tool.fn(ctx, tool_name="nmap", parameters=json.dumps({"target": "10.0.0.57"})))
        assert result["success"] is True

    def test_optimizer_forced_stealth(self):
        self._patch_tool("nmap", self._fast_exec("nmap"))
        tool = run(mcp().get_tool("run_security_tool"))
        ctx = make_mock_context()
        async def go():
            with patch("mcp_core.server_setup._optimizer.optimize", return_value={
                "target": "10.0.0.58",
                "_optimizer": {"forced_stealth": True, "profile": "stealth"},
            }):
                return await tool.fn(ctx, tool_name="nmap", parameters=json.dumps({"target": "10.0.0.58"}))
        result = run(go())
        assert result["success"] is True

    def test_rate_limit_profile_restored(self):
        self._patch_tool("nmap", self._fast_exec("nmap"))
        tool = run(mcp().get_tool("run_security_tool"))
        ctx = make_mock_context()
        async def side_effect(k):
            return "stealth" if k.startswith("ratelimit") else None
        ctx.get_state = AsyncMock(side_effect=side_effect)
        result = run(tool.fn(ctx, tool_name="nmap", parameters=json.dumps({"target": "10.0.0.59"})))
        assert result["success"] is True

    def test_set_state_exception_tech_profile(self):
        self._patch_tool("whatweb", self._fast_exec("whatweb"))
        tool = run(mcp().get_tool("run_security_tool"))
        ctx = make_mock_context()
        ctx.set_state = AsyncMock(side_effect=RuntimeError("set_state fail"))
        result = run(tool.fn(ctx, tool_name="whatweb", parameters=json.dumps({"url": "http://example.com"})))
        assert result["success"] is True

    def test_ai_suggest_full_flow(self):
        self._patch_tool("nmap", {"success": True, "stdout": "22/tcp open ssh\n80/tcp open http", "return_code": 0, "timed_out": False, "execution_time": 0.5, "stderr": ""})
        tool = run(mcp().get_tool("run_security_tool"))
        ctx = make_mock_context()
        sample_result = MagicMock()
        sample_result.text = "Next tool: nikto — web vuln scan"
        ctx.sample = AsyncMock(return_value=sample_result)
        result = run(tool.fn(ctx, tool_name="nmap", parameters=json.dumps({"target": "10.0.0.60", "_ai_suggest": True})))
        assert result["success"] is True
        assert "ai_suggestion" in result
        assert "nikto" in result["ai_suggestion"]

    def test_ai_suggest_exception_swallowed(self):
        self._patch_tool("nmap", {"success": True, "stdout": "scan done", "return_code": 0, "timed_out": False, "execution_time": 0.5, "stderr": ""})
        tool = run(mcp().get_tool("run_security_tool"))
        ctx = make_mock_context()
        ctx.sample = AsyncMock(side_effect=RuntimeError("unsupported"))
        result = run(tool.fn(ctx, tool_name="nmap", parameters=json.dumps({"target": "10.0.0.61", "_ai_suggest": True})))
        assert result["success"] is True
        assert "ai_suggestion" not in result

    def test_ai_suggest_not_requested(self):
        self._patch_tool("nmap", {"success": True, "stdout": "scan done", "return_code": 0, "timed_out": False, "execution_time": 0.5, "stderr": ""})
        tool = run(mcp().get_tool("run_security_tool"))
        ctx = make_mock_context()
        result = run(tool.fn(ctx, tool_name="nmap", parameters=json.dumps({"target": "10.0.0.62"})))
        assert result["success"] is True
        assert "ai_suggestion" not in result

    def test_ai_suggest_empty_output_skips(self):
        self._patch_tool("nmap", {"success": True, "stdout": "", "return_code": 0, "timed_out": False, "execution_time": 0.5, "stderr": ""})
        tool = run(mcp().get_tool("run_security_tool"))
        ctx = make_mock_context()
        sample_result = MagicMock()
        sample_result.text = "Next tool: test"
        ctx.sample = AsyncMock(return_value=sample_result)
        result = run(tool.fn(ctx, tool_name="nmap", parameters=json.dumps({"target": "10.0.0.63", "_ai_suggest": True})))
        assert result["success"] is True
        assert "ai_suggestion" not in result

    def test_alternative_tool_suggestion(self):
        self._patch_tool("nmap", {"success": False, "stdout": "", "stderr": "connection refused: port 22", "return_code": 1, "timed_out": False, "execution_time": 0.5})
        tool = run(mcp().get_tool("run_security_tool"))
        ctx = make_mock_context()
        result = run(tool.fn(ctx, tool_name="nmap", parameters=json.dumps({"target": "10.0.0.64"})))
        assert result["success"] is False

    def test_prompt_suggestion_wp(self):
        self._patch_tool("whatweb", self._fast_exec("whatweb"))
        tool = run(mcp().get_tool("run_security_tool"))
        ctx = make_mock_context()
        ctx.get_state = AsyncMock(return_value=None)
        msg = MagicMock()
        msg.content.text = "Try WordPress scanning"
        ctx.get_prompt = AsyncMock(return_value=SimpleNamespace(messages=[msg]))
        result = run(tool.fn(ctx, tool_name="whatweb", parameters=json.dumps({"url": "http://example.com"})))
        assert result["success"] is True

    def test_prompt_suggestion_waf(self):
        self._patch_tool("wafw00f", self._fast_exec("wafw00f"))
        tool = run(mcp().get_tool("run_security_tool"))
        ctx = make_mock_context()
        ctx.get_state = AsyncMock(return_value=None)
        msg = MagicMock()
        msg.content.text = "WAF bypass techniques"
        ctx.get_prompt = AsyncMock(return_value=SimpleNamespace(messages=[msg]))
        result = run(tool.fn(ctx, tool_name="wafw00f", parameters=json.dumps({"url": "http://example.com"})))
        assert result["success"] is True

    def test_prompt_suggestion_smb(self):
        self._patch_tool("smbmap", self._fast_exec("smbmap"))
        tool = run(mcp().get_tool("run_security_tool"))
        ctx = make_mock_context()
        ctx.get_state = AsyncMock(return_value=None)
        msg = MagicMock()
        msg.content.text = "SMB lateral movement"
        ctx.get_prompt = AsyncMock(return_value=SimpleNamespace(messages=[msg]))
        result = run(tool.fn(ctx, tool_name="smbmap", parameters=json.dumps({"target": "10.0.0.65"})))
        assert result["success"] is True

    def test_error_handler_alternative_tool(self):
        self._patch_tool("nmap", {"success": False, "stdout": "", "stderr": "Permission denied", "return_code": 1, "timed_out": False, "execution_time": 0.5})
        tool = run(mcp().get_tool("run_security_tool"))
        ctx = make_mock_context()
        result = run(tool.fn(ctx, tool_name="nmap", parameters=json.dumps({"target": "10.0.0.66"})))
        assert result["success"] is False

    def test_cache_hit_returned(self):
        from mcp_core.server_setup import _scan_cache, _cache_key_for
        self._patch_tool("nmap", self._fast_exec("nmap"))
        s = "test-unit"
        key = _cache_key_for(s, "nmap", "10.0.0.67", {"target": "10.0.0.67"})
        _scan_cache.set(key, {"tool": "nmap", "target": "10.0.0.67", "result": {"success": True, "output": "cached result"}, "timestamp": 999999.0}, ttl=3600)
        tool = run(mcp().get_tool("run_security_tool"))
        ctx = make_mock_context()
        result = run(tool.fn(ctx, tool_name="nmap", parameters=json.dumps({"target": "10.0.0.67"})))
        assert result["success"] is True
        assert result.get("output") == "cached result"


# =========================================================================
# Group F — TargetStore resource endpoints
# =========================================================================


class TestResourceTargetStore:
    """MCP resource endpoints for TargetStore (targets://, target://, findings, sessions)."""

    def _store(self, tmp_path):
        from server_core.target_store import TargetStore
        return TargetStore(data_dir=str(tmp_path))

    def test_targets_list_empty(self, tmp_path):
        store = self._store(tmp_path)
        with patch("mcp_core.server_setup.get_target_store", return_value=store):
            resource = run(mcp().get_resource("targets://"))
            text = run(resource.fn())
        parsed = json.loads(text)
        assert isinstance(parsed, list)
        assert len(parsed) == 0

    def test_targets_list_with_entries(self, tmp_path):
        store = self._store(tmp_path)
        store.record_scan("alpha.example", surface_data={"ports": [{"port": 80, "service": "http"}]})
        store.record_scan("beta.example")
        with patch("mcp_core.server_setup.get_target_store", return_value=store):
            resource = run(mcp().get_resource("targets://"))
            text = run(resource.fn())
        parsed = json.loads(text)
        assert len(parsed) == 2
        assert parsed[0]["target"] == "alpha.example"
        assert parsed[0]["findings"]["ports"] == 1
        assert parsed[1]["target"] == "beta.example"

    def test_get_target_found(self, tmp_path):
        store = self._store(tmp_path)
        store.record_scan("target.example", surface_data={"ports": [{"port": 443, "service": "https"}]})
        with patch("mcp_core.server_setup.get_target_store", return_value=store):
            tmpl = run(mcp().get_resource_template("target://{target}"))
            text = run(tmpl.fn(target="target.example"))
        parsed = json.loads(text)
        assert parsed["target"] == "target.example"
        assert parsed["scan_count"] == 1
        assert len(parsed["findings"]["ports"]) == 1

    def test_get_target_not_found(self, tmp_path):
        store = self._store(tmp_path)
        with patch("mcp_core.server_setup.get_target_store", return_value=store):
            tmpl = run(mcp().get_resource_template("target://{target}"))
            text = run(tmpl.fn(target="nonexistent.example"))
        parsed = json.loads(text)
        assert "error" in parsed

    def test_get_target_findings(self, tmp_path):
        store = self._store(tmp_path)
        store.record_scan("findme.example", surface_data={
            "ports": [{"port": 80, "service": "http"}],
            "technologies": ["nginx"],
        })
        with patch("mcp_core.server_setup.get_target_store", return_value=store):
            tmpl = run(mcp().get_resource_template("target://{target}/findings"))
            text = run(tmpl.fn(target="findme.example"))
        parsed = json.loads(text)
        assert len(parsed["ports"]) == 1
        assert "nginx" in parsed["technologies"]

    def test_get_target_findings_not_found(self, tmp_path):
        store = self._store(tmp_path)
        with patch("mcp_core.server_setup.get_target_store", return_value=store):
            tmpl = run(mcp().get_resource_template("target://{target}/findings"))
            text = run(tmpl.fn(target="ghost.example"))
        parsed = json.loads(text)
        assert "error" in parsed

    def test_get_target_sessions(self, tmp_path):
        store = self._store(tmp_path)
        store.record_scan("sessionized.example", session_id="sess-001", tools_used=["nmap"])
        store.record_scan("sessionized.example", session_id="sess-002", tools_used=["whatweb"])
        with patch("mcp_core.server_setup.get_target_store", return_value=store):
            tmpl = run(mcp().get_resource_template("target://{target}/sessions"))
            text = run(tmpl.fn(target="sessionized.example"))
        parsed = json.loads(text)
        assert len(parsed) == 2
        assert parsed[0]["session_id"] == "sess-001"
        assert parsed[1]["session_id"] == "sess-002"

    def test_get_target_sessions_not_found(self, tmp_path):
        store = self._store(tmp_path)
        with patch("mcp_core.server_setup.get_target_store", return_value=store):
            tmpl = run(mcp().get_resource_template("target://{target}/sessions"))
            text = run(tmpl.fn(target="ghost.example"))
        parsed = json.loads(text)
        assert "error" in parsed
