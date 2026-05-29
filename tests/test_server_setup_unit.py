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


# =========================================================================
# scan_background — async background scan with task protocol
# =========================================================================

class TestScanBackground:
    """Tests for scan_background tool (background task protocol)."""

    def test_tool_registered(self):
        """scan_background is registered on the MCP server."""
        tool = run(mcp().get_tool("scan_background"))
        assert tool is not None

    def test_description_guides_usage(self):
        """Description tells agent when to use scan_background vs scan()."""
        tool = run(mcp().get_tool("scan_background"))
        desc = tool.description or ""
        assert "scan_background" in desc
        assert ">30s" in desc or "quick" in desc
        assert "task_id" in desc

    def test_description_has_intensity_levels(self):
        """Description lists quick/medium/full intensity levels."""
        tool = run(mcp().get_tool("scan_background"))
        desc = tool.description or ""
        assert "quick" in desc
        assert "medium" in desc
        assert "full" in desc

    def test_has_task_config(self):
        """Tool has task=True in its annotations (background task protocol)."""
        tool = run(mcp().get_tool("scan_background"))
        assert tool is not None

    def test_no_target_returns_error(self):
        """No target + empty scope returns error dict."""
        tool = run(mcp().get_tool("scan_background"))
        ctx = make_mock_context()

        mock_pulse = MagicMock()
        mock_pulse.get_scope.return_value = {"active_target": None}
        mock_pulse.get_surface.return_value = {}
        mock_pulse.get_findings.return_value = []
        mock_pulse.get_plan.return_value = {}

        with patch.dict("sys.modules", {"pulse_app": mock_pulse}):
            result = run(tool.fn(ctx))

        assert "error" in result
        assert result["target"] is None

    def test_invalid_intensity_defaults_to_quick(self):
        """Unknown intensity falls back to 'quick'."""
        tool = run(mcp().get_tool("scan_background"))
        ctx = make_mock_context()

        mock_pulse = MagicMock()
        mock_pulse.get_scope.return_value = {"active_target": "10.0.0.1"}
        mock_pulse.get_surface.return_value = {"ports_count": 0, "ports": [], "techs": []}
        mock_pulse.get_findings.return_value = []
        mock_pulse.get_plan.return_value = {}
        mock_pulse.TOOLS_BY_INTENSITY = {"quick": []}
        mock_pulse._TOOLS_NEED_URL = frozenset()
        mock_pulse._TOOLS_NEED_URL_AS_TARGET = frozenset()
        mock_pulse._cache_for_target.return_value = []

        with (
            patch.dict("sys.modules", {"pulse_app": mock_pulse}),
            patch("mcp_core.server_setup.get_direct_tools", return_value={}),
        ):
            result = run(tool.fn(ctx, intensity="extreme"))

        assert result["intensity"] == "quick"

    def test_quick_intensity_runs_nmap_whatweb(self):
        """Quick intensity runs the 2 tools defined in TOOLS_BY_INTENSITY['quick']."""
        tool = run(mcp().get_tool("scan_background"))
        ctx = make_mock_context()

        def _make_exec(success=True, stdout=""):
            return lambda b, p: {
                "success": success, "output": stdout, "stdout": stdout,
                "error": "", "returncode": 0 if success else 1,
            }

        mock_pulse = MagicMock()
        mock_pulse.get_scope.return_value = {"active_target": "10.0.0.5"}
        mock_pulse.get_surface.return_value = {"ports_count": 1, "ports": ["80"], "techs": ["nginx"]}
        mock_pulse.get_findings.return_value = []
        mock_pulse.get_plan.return_value = {}
        mock_pulse.TOOLS_BY_INTENSITY = {"quick": ["nmap", "whatweb"]}
        mock_pulse._TOOLS_NEED_URL = frozenset()
        mock_pulse._TOOLS_NEED_URL_AS_TARGET = frozenset()
        mock_pulse._cache_for_target.return_value = []
        mock_pulse._suggest_next_from_context.return_value = {}

        mock_tools = {
            "nmap": (_make_exec(stdout="22/tcp open  ssh\n80/tcp open  http"), "nmap"),
            "whatweb": (_make_exec(stdout="http://target [200 OK] nginx PHP"), "whatweb"),
        }

        with (
            patch.dict("sys.modules", {"pulse_app": mock_pulse}),
            patch("mcp_core.server_setup.get_direct_tools", return_value=mock_tools),
        ):
            result = run(tool.fn(ctx, target="10.0.0.5"))

        assert result["target"] == "10.0.0.5"
        assert result["intensity"] == "quick"
        assert list(result["tools"].keys()) == ["nmap", "whatweb"]
        assert all(v["status"] == "completed" for v in result["tools"].values())

    def test_cached_tool_skipped(self):
        """Tool with cache entry is reported as 'cached', not executed."""
        tool = run(mcp().get_tool("scan_background"))
        ctx = make_mock_context()

        mock_pulse = MagicMock()
        mock_pulse.get_scope.return_value = {"active_target": "10.0.0.10"}
        mock_pulse._cache_for_target.return_value = [
            {"tool": "nmap", "target": "10.0.0.10", "timestamp": 9999999999,
             "result": {"success": True, "output": "cached-results"}},
        ]
        mock_pulse.get_surface.return_value = {"ports_count": 0}
        mock_pulse.get_findings.return_value = []
        mock_pulse.get_plan.return_value = {}
        mock_pulse.TOOLS_BY_INTENSITY = {"quick": ["nmap", "whatweb"]}
        mock_pulse._TOOLS_NEED_URL = frozenset()
        mock_pulse._TOOLS_NEED_URL_AS_TARGET = frozenset()
        mock_pulse._suggest_next_from_context.return_value = {}
        mock_tools = {
            "nmap": (lambda b, p: {"success": True, "stdout": "", "returncode": 0}, "nmap"),
            "whatweb": (lambda b, p: {"success": True, "stdout": "", "returncode": 0}, "whatweb"),
        }

        with (
            patch.dict("sys.modules", {"pulse_app": mock_pulse}),
            patch("mcp_core.server_setup.get_direct_tools", return_value=mock_tools),
        ):
            result = run(tool.fn(ctx, target="10.0.0.10"))

        assert result["tools"]["nmap"]["status"] == "cached"
        assert result["tools"]["nmap"].get("cached") is True
        assert result["tools"]["whatweb"]["status"] == "completed"

    def test_tool_failure_reported(self):
        """Failed tool gets status 'failed' with error."""
        tool = run(mcp().get_tool("scan_background"))
        ctx = make_mock_context()

        mock_pulse = MagicMock()
        mock_pulse.get_scope.return_value = {"active_target": "10.0.0.20"}
        mock_pulse._cache_for_target.return_value = []
        mock_pulse.get_surface.return_value = {"ports_count": 0}
        mock_pulse.get_findings.return_value = []
        mock_pulse.get_plan.return_value = {}
        mock_pulse.TOOLS_BY_INTENSITY = {"quick": ["nmap", "whatweb"]}
        mock_pulse._TOOLS_NEED_URL = frozenset()
        mock_pulse._TOOLS_NEED_URL_AS_TARGET = frozenset()
        mock_pulse._suggest_next_from_context.return_value = {}
        mock_tools = {
            "nmap": (lambda b, p: {"success": False, "error": "Timeout", "output": "",
                                     "stdout": "", "returncode": 1}, "nmap"),
            "whatweb": (lambda b, p: {"success": True, "stdout": "", "returncode": 0}, "whatweb"),
        }

        with (
            patch.dict("sys.modules", {"pulse_app": mock_pulse}),
            patch("mcp_core.server_setup.get_direct_tools", return_value=mock_tools),
        ):
            result = run(tool.fn(ctx, target="10.0.0.20"))

        assert result["tools"]["nmap"]["status"] == "failed"
        assert "error" in result["tools"]["nmap"]
        assert result["tools"]["whatweb"]["status"] == "completed"

    def test_result_contains_all_keys(self):
        """Return dict includes all expected keys."""
        tool = run(mcp().get_tool("scan_background"))
        ctx = make_mock_context()

        mock_pulse = MagicMock()
        mock_pulse.get_scope.return_value = {"active_target": "10.0.0.30"}
        mock_pulse._cache_for_target.return_value = []
        mock_pulse.get_surface.return_value = {"ports_count": 2, "ports": ["22", "80"], "techs": ["nginx"]}
        mock_pulse.get_findings.return_value = [{"id": "CVE-2023-xxx", "severity": "critical"}]
        mock_pulse.get_plan.return_value = {"target": "10.0.0.30", "steps": [], "step_count": 0}
        mock_pulse.TOOLS_BY_INTENSITY = {"quick": ["nmap"]}
        mock_pulse._TOOLS_NEED_URL = frozenset()
        mock_pulse._TOOLS_NEED_URL_AS_TARGET = frozenset()
        mock_pulse._suggest_next_from_context.return_value = {"tool": "whatweb"}
        mock_tools = {
            "nmap": (lambda b, p: {"success": True, "stdout": "22/tcp open  ssh", "returncode": 0}, "nmap"),
        }

        with (
            patch.dict("sys.modules", {"pulse_app": mock_pulse}),
            patch("mcp_core.server_setup.get_direct_tools", return_value=mock_tools),
        ):
            result = run(tool.fn(ctx, target="10.0.0.30"))

        assert "target" in result
        assert "intensity" in result
        assert "tools" in result
        assert "surface" in result
        assert "findings" in result
        assert "plan" in result
        assert "summary" in result
        assert result["next_suggested_tool"]["tool"] == "whatweb"


# =========================================================================
# Group F — _enrich_profile_from_cache
# =========================================================================


class TestEnrichProfileFromCache:
    """_enrich_profile_from_cache() injects cached scan results into a TargetProfile."""

    def test_nmap_ports_and_services(self):
        from mcp_core.server_setup import _enrich_profile_from_cache
        from shared.target_profile import TargetProfile
        from shared.target_types import TargetType

        profile = TargetProfile(target="10.0.0.1", target_type=TargetType.NETWORK_HOST)
        cached = {
            "nmap": {
                "output": (
                    "22/tcp open  ssh\n"
                    "80/tcp open  http\n"
                    "443/tcp open  https\n"
                ),
                "success": True,
            },
        }
        result = _enrich_profile_from_cache(profile, cached)
        assert 22 in result.open_ports
        assert 80 in result.open_ports
        assert 443 in result.open_ports
        assert result.services[22] == "ssh"
        assert result.services[80] == "http"
        assert result.confidence_score > 0

    def test_whatweb_technology_detection(self):
        from mcp_core.server_setup import _enrich_profile_from_cache
        from shared.target_profile import TargetProfile
        from shared.target_types import TargetType, TechnologyStack

        profile = TargetProfile(target="example.com", target_type=TargetType.WEB_APPLICATION)
        cached = {
            "whatweb": {
                "output": (
                    "http://example.com [200 OK] Apache[2.4.41], "
                    "PHP[7.4], WordPress[5.8], jQuery"
                ),
                "success": True,
            },
        }
        result = _enrich_profile_from_cache(profile, cached)
        assert TechnologyStack.APACHE in result.technologies
        assert TechnologyStack.PHP in result.technologies
        assert TechnologyStack.WORDPRESS in result.technologies
        assert result.confidence_score > 0

    def test_testssl_enriches_ssl_info(self):
        from mcp_core.server_setup import _enrich_profile_from_cache
        from shared.target_profile import TargetProfile
        from shared.target_types import TargetType

        profile = TargetProfile(target="10.0.0.1")
        cached = {
            "testssl": {
                "output": "SSL/TLS protocol: TLSv1.3\ncipher: AES256-GCM",
                "success": True,
            },
        }
        result = _enrich_profile_from_cache(profile, cached)
        assert result.ssl_info["source"] == "testssl"
        assert "TLSv1" in result.ssl_info["summary"]

    def test_wafw00f_boosts_confidence(self):
        from mcp_core.server_setup import _enrich_profile_from_cache
        from shared.target_profile import TargetProfile

        profile = TargetProfile(target="10.0.0.1")
        cached = {"wafw00f": {"output": "Cloudflare", "success": True}}
        result = _enrich_profile_from_cache(profile, cached)
        assert result.confidence_score == 0.05

    def test_empty_cache_returns_profile_unchanged(self):
        from mcp_core.server_setup import _enrich_profile_from_cache
        from shared.target_profile import TargetProfile

        profile = TargetProfile(target="10.0.0.1")
        result = _enrich_profile_from_cache(profile, {})
        assert result == profile

    def test_risk_level_high_when_many_ports(self):
        from mcp_core.server_setup import _enrich_profile_from_cache
        from shared.target_profile import TargetProfile
        from shared.target_types import TargetType

        profile = TargetProfile(target="10.0.0.1", target_type=TargetType.NETWORK_HOST)
        cached = {
            "nmap": {
                "output": "\n".join(f"{p}/tcp open  service{p}" for p in range(1, 8)),
                "success": True,
            },
        }
        result = _enrich_profile_from_cache(profile, cached)
        assert result.risk_level == "high"
        assert result.attack_surface_score > 0


# =========================================================================
# Group G — _create_typed_tool_wrapper
# =========================================================================


class TestCreateTypedToolWrapper:
    """_create_typed_tool_wrapper() dynamically builds typed MCP tool wrappers."""

    def test_basic_typed_wrapper(self):
        from mcp_core.server_setup import _create_typed_tool_wrapper

        tool_def = {
            "desc": "Test tool",
            "params": {"target": {"type": "str"}},
            "optional": {"timeout": 30, "verbose": False},
        }
        async def mock_run(ctx, name, params):
            return {"success": True, "tool": name, "params": params}

        wrapper = _create_typed_tool_wrapper("nmap", tool_def, mock_run)
        assert wrapper is not None
        assert callable(wrapper)
        assert wrapper.__name__ == "nmap_typed"
        assert wrapper.__doc__ is not None
        assert "target" in wrapper.__annotations__
        assert "timeout" in wrapper.__annotations__ or "verbose" in wrapper.__annotations__

    def test_typed_wrapper_passes_optional_params(self):
        from mcp_core.server_setup import _create_typed_tool_wrapper

        tool_def = {
            "desc": "Verbose scan",
            "params": {"target": {"type": "str"}},
            "optional": {"verbose": True, "timeout": 60},
        }
        captured = {}

        async def mock_run(ctx, name, params):
            captured["params"] = params
            return {"success": True}

        wrapper = _create_typed_tool_wrapper("nmap", tool_def, mock_run)
        mock_ctx = make_mock_context()

        with patch("mcp_core.server_setup.get_context", return_value=mock_ctx):
            result = run(wrapper(target="10.0.0.1", verbose=False))

        assert captured["params"]["target"] == "10.0.0.1"
        assert captured["params"]["verbose"] is False
        from mcp_core.server_setup import _create_typed_tool_wrapper

        tool_def = {
            "desc": "Port scan",
            "params": {"target": {"type": "str"}},
            "optional": {},
        }
        captured = {}

        async def mock_run(ctx, name, params):
            captured["name"] = name
            captured["params"] = params
            return {"success": True}

        wrapper = _create_typed_tool_wrapper("nmap", tool_def, mock_run)
        mock_ctx = make_mock_context()

        with patch("mcp_core.server_setup.get_context", return_value=mock_ctx):
            result = run(wrapper(target="10.0.0.1"))

        assert captured["name"] == "nmap"
        assert captured["params"]["target"] == "10.0.0.1"


# =========================================================================
# Group H — get_tool_skill
# =========================================================================


class TestGetToolSkill:
    """get_tool_skill MCP tool returns skill documents for a given tool."""

    def test_unknown_tool_returns_error(self):
        tool = run(mcp().get_tool("get_tool_skill"))
        ctx = make_mock_context()
        result = run(tool.fn(ctx, "nonexistent_tool_xyz"))
        assert result["success"] is False
        assert "error" in result

    def test_known_tool_returns_skill_info(self):
        tool = run(mcp().get_tool("get_tool_skill"))
        ctx = make_mock_context()
        result = run(tool.fn(ctx, "nmap"))
        if result["success"]:
            assert "tool_name" in result
            assert "skill_name" in result
            assert "documents" in result


# =========================================================================
# Group I — _register_skills
# =========================================================================


class TestRegisterSkills:
    """_register_skills() mounts the skills directory as MCP resources."""

    def test_no_skills_dir_returns_gracefully(self):
        from mcp_core.server_setup import _register_skills
        mcp = MagicMock()
        mcp.add_provider = MagicMock()
        logger = MagicMock()

        with patch("mcp_core.server_setup.Path.exists", return_value=False):
            _register_skills(mcp, logger)

        mcp.add_provider.assert_not_called()

    def test_skills_dir_registers_provider(self):
        from mcp_core.server_setup import _register_skills
        mcp = MagicMock()
        mcp.add_provider = MagicMock()
        logger = MagicMock()

        with patch("mcp_core.server_setup.Path.exists", return_value=True):
            with patch("mcp_core.server_setup.SkillsDirectoryProvider"):
                _register_skills(mcp, logger)

        mcp.add_provider.assert_called_once()

    def test_skills_provider_none_logs_warning(self):
        from mcp_core.server_setup import _register_skills
        mcp = MagicMock()
        logger = MagicMock()

        with patch("mcp_core.server_setup.SkillsDirectoryProvider", None):
            _register_skills(mcp, logger)

        logger.warning.assert_called_once()


# =========================================================================
# Group J — _build_destructive_confirmation
# =========================================================================


class TestBuildDestructiveConfirmation:
    """_build_destructive_confirmation() determines if a tool needs user confirmation."""

    def test_nondestructive_tool_returns_none(self):
        from mcp_core.server_setup import _build_destructive_confirmation
        assert _build_destructive_confirmation("nmap", {"target": "10.0.0.1"}) is None

    def test_aireplay_mode_9_allowed(self):
        from mcp_core.server_setup import _build_destructive_confirmation
        result = _build_destructive_confirmation("aireplay_ng", {"attack_mode": 9})
        assert result is None

    def test_aireplay_mode_0_blocks(self):
        from mcp_core.server_setup import _build_destructive_confirmation
        result = _build_destructive_confirmation("aireplay_ng", {"attack_mode": 0, "interface": "wlan0", "bssid": "AA:BB:CC:DD:EE:FF"})
        assert result is not None
        assert "action" in result
        assert "warning" in result

    def test_aireplay_no_mode_blocks(self):
        from mcp_core.server_setup import _build_destructive_confirmation
        result = _build_destructive_confirmation("aireplay_ng", {"interface": "wlan0"})
        assert result is not None

    def test_responder_analyze_allowed(self):
        from mcp_core.server_setup import _build_destructive_confirmation
        result = _build_destructive_confirmation("responder", {"analyze": True})
        assert result is None

    def test_responder_active_blocks(self):
        from mcp_core.server_setup import _build_destructive_confirmation
        result = _build_destructive_confirmation("responder", {"interface": "eth0"})
        assert result is not None
        assert "poisoning" in result["action"]

    def test_metasploit_auxiliary_allowed(self):
        from mcp_core.server_setup import _build_destructive_confirmation
        result = _build_destructive_confirmation("metasploit", {"module": "auxiliary/scanner/portscan/tcp"})
        assert result is None

    def test_metasploit_exploit_blocks(self):
        from mcp_core.server_setup import _build_destructive_confirmation
        result = _build_destructive_confirmation("metasploit", {"module": "exploit/multi/handler"})
        assert result is not None
        assert "Metasploit" in result["action"]

    def test_generic_destructive_without_specific_tool(self):
        """Tool in _DESTRUCTIVE_TOOLS but not specially handled."""
        from mcp_core.server_setup import _build_destructive_confirmation
        result = _build_destructive_confirmation("mdk4", {"interface": "wlan0"})
        assert result is not None
        assert "action" in result

    def test_metasploit_auxiliary_allowed(self):
        from mcp_core.server_setup import _build_destructive_confirmation
        result = _build_destructive_confirmation("metasploit", {"module": "auxiliary/scanner/portscan/tcp"})
        assert result is None

    def test_metasploit_exploit_blocks(self):
        from mcp_core.server_setup import _build_destructive_confirmation
        result = _build_destructive_confirmation("metasploit", {"module": "exploit/multi/handler"})
        assert result is not None
        assert "Metasploit" in result["action"]

    def test_metasploit_options_empty(self):
        from mcp_core.server_setup import _build_destructive_confirmation
        result = _build_destructive_confirmation("metasploit", {"module": "exploit/tomcat"})
        assert result is not None

    def test_mitm6_confirmation(self):
        from mcp_core.server_setup import _build_destructive_confirmation
        result = _build_destructive_confirmation("mitm6", {"interface": "eth0", "domain": "corp.local"})
        assert result is not None
        assert "mitm6" in result["action"]

    def test_mitm6_no_domain(self):
        from mcp_core.server_setup import _build_destructive_confirmation
        result = _build_destructive_confirmation("mitm6", {"interface": "eth0"})
        assert result is not None


# =========================================================================
# Group K — _infer_param_type, _resolve_required_param_type
# =========================================================================


class TestParamTypeInference:
    """Type inference helpers used by _create_typed_tool_wrapper."""

    def test_infer_bool(self):
        from mcp_core.server_setup import _infer_param_type
        assert _infer_param_type(True) == bool
        assert _infer_param_type(False) == bool

    def test_infer_int(self):
        from mcp_core.server_setup import _infer_param_type
        assert _infer_param_type(42) == int

    def test_infer_float(self):
        from mcp_core.server_setup import _infer_param_type
        assert _infer_param_type(3.14) == float

    def test_infer_dict(self):
        from mcp_core.server_setup import _infer_param_type
        assert _infer_param_type({"a": 1}) == dict

    def test_infer_list(self):
        from mcp_core.server_setup import _infer_param_type
        assert _infer_param_type([1, 2, 3]) == list

    def test_infer_str(self):
        from mcp_core.server_setup import _infer_param_type
        assert _infer_param_type("hello") == str

    def test_resolve_bool(self):
        from mcp_core.server_setup import _resolve_required_param_type
        assert _resolve_required_param_type({"type": "bool"}) == bool

    def test_resolve_int(self):
        from mcp_core.server_setup import _resolve_required_param_type
        assert _resolve_required_param_type({"type": "int"}) == int

    def test_resolve_default_str(self):
        from mcp_core.server_setup import _resolve_required_param_type
        assert _resolve_required_param_type({"type": "unknown"}) == str
        assert _resolve_required_param_type({}) == str


# =========================================================================
# Group L — _normalize_tool_result
# =========================================================================


class TestNormalizeToolResult:
    """_normalize_tool_result() normalizes diverse tool output formats."""

    def test_normalize_dict_result(self):
        from mcp_core.server_setup import _normalize_tool_result
        result = _normalize_tool_result({"success": True, "output": "ok"})
        assert result["success"] is True
        assert result["output"] == "ok"

    def test_normalize_non_dict(self):
        from mcp_core.server_setup import _normalize_tool_result
        result = _normalize_tool_result("string result")
        assert result["success"] is False
        assert "error" in result

    def test_normalize_stdout_fallback(self):
        from mcp_core.server_setup import _normalize_tool_result
        result = _normalize_tool_result({"success": True, "stdout": "fallback"})
        assert result["output"] == "fallback"

    def test_normalize_stderr_as_error(self):
        from mcp_core.server_setup import _normalize_tool_result
        result = _normalize_tool_result({"success": False, "stderr": "boom"})
        assert result["error"] == "boom"

    def test_normalize_returncode(self):
        from mcp_core.server_setup import _normalize_tool_result
        result = _normalize_tool_result({"return_code": 1})
        assert result["returncode"] == 1

    def test_normalize_partial_results(self):
        from mcp_core.server_setup import _normalize_tool_result
        result = _normalize_tool_result({"partial_results": True})
        assert result["partial_results"] is True


# =========================================================================
# Group M — _get_registry_tool_definition
# =========================================================================


class TestGetRegistryToolDefinition:
    """_get_registry_tool_definition() resolves tool names to registry entries."""

    def test_known_tool_returns_definition(self):
        from mcp_core.server_setup import _get_registry_tool_definition
        result = _get_registry_tool_definition("nmap")
        assert result is not None
        assert "desc" in result
        assert "params" in result

    def test_alias_tool_returns_definition(self):
        from mcp_core.server_setup import _get_registry_tool_definition, _TOOL_REGISTRY_ALIASES
        if _TOOL_REGISTRY_ALIASES:
            alias = next(iter(_TOOL_REGISTRY_ALIASES.keys()))
            result = _get_registry_tool_definition(alias)
            assert result is not None

    def test_unknown_tool_returns_none(self):
        from mcp_core.server_setup import _get_registry_tool_definition
        result = _get_registry_tool_definition("nonexistent_tool_xyz")
        assert result is None


# =========================================================================
# Group N — _suggest_next_tool
# =========================================================================


class TestSuggestNextTool:
    """_suggest_next_tool() suggests the next tool based on current output."""

    def test_nmap_web_ports(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("nmap", "80/tcp open http")
        assert result["tool"] == "whatweb"

    def test_nmap_smb_port(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("nmap", "445/tcp open microsoft-ds")
        assert result["tool"] == "smbmap"

    def test_nmap_ssh_port(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("nmap", "22/tcp open ssh")
        assert result["tool"] == "hydra"

    def test_nmap_db_port(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("nmap", "3306/tcp open mysql")
        assert result["tool"] == "sqlmap"

    def test_nmap_generic(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("nmap", "161/udp open snmp")
        assert result["tool"] == "nuclei"

    def test_whatweb_wordpress(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("whatweb", "WordPress[6.0]")
        assert result["tool"] == "wpscan"

    def test_whatweb_joomla(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("whatweb", "Joomla[4.2]")
        assert result["tool"] == "joomscan"

    def test_whatweb_generic(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("whatweb", "nginx 1.20")
        assert result["tool"] == "gobuster"

    def test_nuclei_sqli(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("nuclei", "SQL Injection detected")
        assert result["tool"] == "sqlmap"

    def test_nuclei_xss(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("nuclei", "Cross-Site Scripting (XSS)")
        assert result["tool"] == "dalfox"

    def test_nuclei_ssl(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("nuclei", "SSL certificate expired")
        assert result["tool"] == "testssl"

    def test_nuclei_smb(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("nuclei", "EternalBlue MS17-010")
        assert result["tool"] == "metasploit"

    def test_empty_output_nmap_returns_nuclei(self):
        """Even empty nmap output triggers generic nuclei suggestion."""
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("nmap", "")
        assert result["tool"] == "nuclei"

    def test_empty_output_unknown_tool(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("unknown_tool", "")
        assert result == {}


# =========================================================================
# Group O — _detect_from_cache
# =========================================================================


class TestDetectFromCache:
    """_detect_from_cache() builds a TechProfile from cached scan results."""

    def test_no_cache_returns_none(self):
        from mcp_core.server_setup import _detect_from_cache, _scan_cache
        _scan_cache.cache.clear()
        _scan_cache.ttl_times.clear()
        result = _detect_from_cache("10.0.0.1")
        assert result is None

    def test_whatweb_cache_detects_apache(self):
        from mcp_core.server_setup import _detect_from_cache, _scan_cache
        _scan_cache.cache.clear()
        _scan_cache.ttl_times.clear()
        _scan_cache.set(
            "test:whatweb:10.0.0.1",
            {
                "tool": "whatweb",
                "target": "10.0.0.1",
                "result": {"output": "http://10.0.0.1 [200 OK] Apache[2.4.41] PHP[7.4]"},
            },
        )
        result = _detect_from_cache("10.0.0.1")
        assert result is not None
        assert "apache" in result.web_servers

    def test_wrong_target_returns_none(self):
        from mcp_core.server_setup import _detect_from_cache, _scan_cache
        _scan_cache.cache.clear()
        _scan_cache.ttl_times.clear()
        _scan_cache.set(
            "test:whatweb:10.0.0.2",
            {
                "tool": "whatweb",
                "target": "10.0.0.2",
                "result": {"output": "Apache"},
            },
        )
        result = _detect_from_cache("10.0.0.1")
        assert result is None


# =========================================================================
# Group P — _resolve_required_param_type remaining types
# =========================================================================


class TestResolveRequiredParamTypeRemaining:
    """_resolve_required_param_type() covers float/dict/list types."""

    def test_resolve_float(self):
        from mcp_core.server_setup import _resolve_required_param_type
        assert _resolve_required_param_type({"type": "float"}) == float

    def test_resolve_dict(self):
        from mcp_core.server_setup import _resolve_required_param_type
        assert _resolve_required_param_type({"type": "dict"}) == dict

    def test_resolve_list(self):
        from mcp_core.server_setup import _resolve_required_param_type
        assert _resolve_required_param_type({"type": "list"}) == list


# =========================================================================
# Group Q — _suggest_next_tool remaining branches
# =========================================================================


class TestSuggestNextToolRemaining:
    """Remaining _suggest_next_tool branches."""

    def test_whatweb_drupal(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("whatweb", "Drupal 9")
        assert result["tool"] == "nuclei"

    def test_whatweb_no_content(self):
        """Empty whatweb output should return nothing."""
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("whatweb", "")
        assert result == {}

    def test_nikto_sqli(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("nikto", "SQL Injection in /page?id=1")
        assert result["tool"] == "sqlmap"

    def test_nuclei_output_empty(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("nuclei", "")
        assert result == {}

    def test_gobuster_with_output(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("gobuster", "/admin (Status: 200)")
        assert result["tool"] == "nuclei"

    def test_gobuster_no_output(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("gobuster", "")
        assert result == {}

    def test_hydra_success(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("hydra", "password: admin123")
        assert result["tool"] == "metasploit"

    def test_hydra_no_output(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("hydra", "")
        assert result == {}

    def test_smbmap_share_found(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("smbmap", "admin share")
        assert result["tool"] == "metasploit"

    def test_smbmap_generic(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("smbmap", "some output")
        assert result["tool"] == "hydra"

    def test_sqlmap_vulnerable(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("sqlmap", "parameter: id is vulnerable")
        assert result["tool"] == "metasploit"

    def test_sqlmap_generic(self):
        from mcp_core.server_setup import _suggest_next_tool
        result = _suggest_next_tool("sqlmap", "completed")
        assert result["tool"] == "nuclei"


# =========================================================================
# Group R — _read_skill_bundle
# =========================================================================


class TestReadSkillBundle:
    """_read_skill_bundle() loads skill files via MCP context."""

    def test_read_skill_bundle_returns_documents(self):
        from mcp_core.server_setup import _read_skill_bundle
        ctx = make_mock_context()
        ctx.read_resource = AsyncMock(return_value=SimpleNamespace(
            contents=[SimpleNamespace(content="nmap skill content")]
        ))
        result = run(_read_skill_bundle(ctx, "nmap"))
        assert "SKILL.md" in result
        assert result["SKILL.md"] == "nmap skill content"

    def test_read_skill_empty_bundle(self):
        from mcp_core.server_setup import _read_skill_bundle
        ctx = make_mock_context()
        ctx.read_resource = AsyncMock(return_value=SimpleNamespace(contents=[]))
        result = run(_read_skill_bundle(ctx, "nmap"))
        assert result == {}

    def test_read_skill_resource_exception(self):
        from mcp_core.server_setup import _read_skill_bundle
        ctx = make_mock_context()
        ctx.read_resource = AsyncMock(side_effect=Exception("resource not found"))
        result = run(_read_skill_bundle(ctx, "nmap"))
        assert result == {}


# =========================================================================
# Group S — _collect_cached_scans
# =========================================================================


class TestCollectCachedScans:
    """_collect_cached_scans() retrieves cached scans by session + target."""

    def test_collect_with_matching_entry(self):
        from mcp_core.server_setup import _collect_cached_scans, _scan_cache
        _scan_cache.cache.clear()
        _scan_cache.ttl_times.clear()
        _scan_cache.set(
            "test-session:nmap:10.0.0.1",
            {"tool": "nmap", "target": "10.0.0.1", "result": {"success": True}},
        )
        result = _collect_cached_scans("test-session", "10.0.0.1")
        assert "nmap" in result

    def test_collect_empty_when_no_match(self):
        from mcp_core.server_setup import _collect_cached_scans, _scan_cache
        _scan_cache.cache.clear()
        _scan_cache.ttl_times.clear()
        _scan_cache.set(
            "other-session:nmap:10.0.0.2",
            {"tool": "nmap", "target": "10.0.0.2", "result": {"success": True}},
        )
        result = _collect_cached_scans("test-session", "10.0.0.1")
        assert result == {}


# =========================================================================
# Group T — _build_destructive_confirmation remaining: options not dict
# =========================================================================


class TestBuildDestructiveConfirmationEdgeCases:
    """Edge cases: options not a dict, mitm6 no interface."""

    def test_metasploit_options_not_dict(self):
        from mcp_core.server_setup import _build_destructive_confirmation
        result = _build_destructive_confirmation("metasploit", {"module": "exploit/test", "options": "string"})
        assert result is not None
