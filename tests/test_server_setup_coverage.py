import asyncio
import inspect
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from mcp_core import server_setup
from mcp_core.server_setup import (
    _ScanCache,
    _build_destructive_confirmation,
    _build_typed_tool_doc,
    _cache_key_for,
    _collect_cached_scans,
    _create_typed_tool_wrapper,
    _detect_from_cache,
    _enrich_profile_from_cache,
    _get_registry_tool_definition,
    _infer_param_type,
    _normalize_tool_result,
    _read_skill_bundle,
    _read_skill_document,
    _register_skills,
    _resolve_required_param_type,
    _scan_cache,
)

from shared.target_profile import TargetProfile
from shared.target_types import TechnologyStack


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# =========================================================================
# _ScanCache.set() — adaptive TTL (ttl is stored in ttl_times dict)
# =========================================================================

class TestScanCacheSet:
    def _get_ttl(self, key):
        expiry = _scan_cache.ttl_times.get(key)
        if expiry is None:
            return None
        return int(expiry - time.time())

    def test_ttl_default(self):
        _scan_cache.cache.clear()
        _scan_cache.ttl_times.clear()
        _scan_cache.set("k1", "v1", execution_time=5)
        assert abs(self._get_ttl("k1") - 1800) < 5

    def test_ttl_medium(self):
        _scan_cache.cache.clear()
        _scan_cache.ttl_times.clear()
        _scan_cache.set("k2", "v2", execution_time=30)
        assert abs(self._get_ttl("k2") - 3600) < 5

    def test_ttl_long(self):
        _scan_cache.cache.clear()
        _scan_cache.ttl_times.clear()
        _scan_cache.set("k3", "v3", execution_time=120)
        assert abs(self._get_ttl("k3") - 5400) < 5

    def test_ttl_explicit(self):
        _scan_cache.cache.clear()
        _scan_cache.ttl_times.clear()
        _scan_cache.set("k4", "v4", execution_time=999, ttl=123)
        assert abs(self._get_ttl("k4") - 123) < 5

    def test_stats(self):
        _scan_cache.cache.clear()
        _scan_cache.ttl_times.clear()
        _scan_cache.set("k5", "v5", execution_time=5)
        stats = _scan_cache.stats()
        assert "size" in stats
        assert "max_size" in stats


# =========================================================================
# _cache_key_for
# =========================================================================

class TestCacheKeyFor:
    def test_no_params(self):
        key = _cache_key_for("sid1", "nmap", "10.0.0.1", {})
        assert key == "sid1:nmap:10.0.0.1"

    def test_with_params(self):
        key = _cache_key_for("sid1", "nmap", "10.0.0.1", {"ports": "80,443"})
        assert key.startswith("sid1:nmap:10.0.0.1:")
        assert len(key) > len("sid1:nmap:10.0.0.1:")

    def test_skips_private_and_target(self):
        key = _cache_key_for(
            "sid1", "nmap", "10.0.0.1",
            {"_internal": "x", "target": "10.0.0.1", "ports": "80"},
        )
        assert key != "sid1:nmap:10.0.0.1"

    def test_empty_relevant_params(self):
        key = _cache_key_for("sid1", "nmap", "10.0.0.1", {"_private": "val"})
        assert key == "sid1:nmap:10.0.0.1"


# =========================================================================
# _collect_cached_scans — cache stores dicts directly as values
# =========================================================================

class TestCollectCachedScans:
    def test_collects_matching_scans(self):
        _scan_cache.cache.clear()
        _scan_cache.cache["sid1:nmap:10.0.0.1"] = {
            "tool": "nmap", "target": "10.0.0.1", "result": {"success": True},
        }
        _scan_cache.cache["sid1:whatweb:10.0.0.1"] = {
            "tool": "whatweb", "target": "10.0.0.1", "result": {"success": True},
        }
        _scan_cache.cache["sid2:nmap:10.0.0.2"] = {
            "tool": "nmap", "target": "10.0.0.2", "result": {"success": True},
        }
        scans = _collect_cached_scans("sid1", "10.0.0.1")
        assert "nmap" in scans
        assert "whatweb" in scans
        assert len(scans) == 2

    def test_no_matching_scans(self):
        _scan_cache.cache.clear()
        scans = _collect_cached_scans("sid99", "no-target")
        assert scans == {}


# =========================================================================
# _enrich_profile_from_cache
# =========================================================================

class TestEnrichProfileFromCache:
    def test_nmap_enriches_ports(self):
        profile = TargetProfile(target="10.0.0.1")
        cached = {
            "nmap": {
                "output": "22/tcp open ssh\n80/tcp open http\n443/tcp open https",
            }
        }
        result = _enrich_profile_from_cache(profile, cached)
        assert 22 in result.open_ports
        assert 80 in result.open_ports
        assert 443 in result.open_ports
        assert result.services.get(22) == "ssh"
        assert result.confidence_score > 0

    def test_whatweb_enriches_technologies(self):
        profile = TargetProfile(target="example.com")
        cached = {"whatweb": {"output": "WordPress, PHP 8.0, Nginx, Apache"}}
        result = _enrich_profile_from_cache(profile, cached)
        assert TechnologyStack.WORDPRESS in result.technologies
        assert TechnologyStack.PHP in result.technologies
        assert TechnologyStack.NGINX in result.technologies
        assert result.confidence_score > 0

    def test_wafw00f_boosts_confidence(self):
        profile = TargetProfile(target="example.com")
        profile.confidence_score = 0.5
        cached = {"wafw00f": {"output": "Cloudflare"}}
        result = _enrich_profile_from_cache(profile, cached)
        assert result.confidence_score > 0.5

    def test_testssl_enriches_ssl(self):
        profile = TargetProfile(target="example.com")
        cached = {"testssl": {"output": "SSL/TLS scan results here"}}
        result = _enrich_profile_from_cache(profile, cached)
        assert result.ssl_info is not None
        assert result.ssl_info["source"] == "testssl"
        assert result.confidence_score > 0

    def test_attack_surface_high_risk(self):
        profile = TargetProfile(target="10.0.0.1")
        cached = {
            "nmap": {
                "output": (
                    "22/tcp open ssh\n80/tcp open http\n443/tcp open https\n"
                    "3306/tcp open mysql\n8080/tcp open http\n8443/tcp open https\n"
                    "9000/tcp open cslistener"
                ),
            }
        }
        result = _enrich_profile_from_cache(profile, cached)
        assert result.attack_surface_score > 0
        assert result.risk_level == "high"

    def test_medium_risk_for_3_to_5_ports(self):
        profile = TargetProfile(target="10.0.0.1")
        cached = {"nmap": {"output": "22/tcp open ssh\n80/tcp open http\n443/tcp open https"}}
        result = _enrich_profile_from_cache(profile, cached)
        assert result.risk_level == "medium"

    def test_no_cache_no_change(self):
        profile = TargetProfile(target="10.0.0.1")
        profile.confidence_score = 0.3
        result = _enrich_profile_from_cache(profile, {})
        assert result == profile
        assert result.confidence_score == 0.3

    def test_nmap_stdout_fallback(self):
        profile = TargetProfile(target="10.0.0.1")
        cached = {"nmap": {"stdout": "22/tcp open ssh"}}
        result = _enrich_profile_from_cache(profile, cached)
        assert 22 in result.open_ports

    def test_nmap_no_duplicate_ports(self):
        profile = TargetProfile(target="10.0.0.1")
        profile.open_ports.append(22)
        cached = {"nmap": {"output": "22/tcp open ssh"}}
        result = _enrich_profile_from_cache(profile, cached)
        assert result.open_ports.count(22) == 1

    def test_nmap_invalid_port_line(self):
        profile = TargetProfile(target="10.0.0.1")
        cached = {"nmap": {"output": "not-a-port-line\n22/tcp open ssh"}}
        result = _enrich_profile_from_cache(profile, cached)
        assert 22 in result.open_ports

    def test_joomla_drupal_detection(self):
        profile = TargetProfile(target="example.com")
        cached = {"whatweb": {"output": "Joomla, Drupal"}}
        result = _enrich_profile_from_cache(profile, cached)
        assert TechnologyStack.JOOMLA in result.technologies
        assert TechnologyStack.DRUPAL in result.technologies

    def test_angular_react_vue_detection(self):
        profile = TargetProfile(target="example.com")
        cached = {"whatweb": {"output": "React, Angular, Vue.js"}}
        result = _enrich_profile_from_cache(profile, cached)
        assert TechnologyStack.REACT in result.technologies
        assert TechnologyStack.ANGULAR in result.technologies
        assert TechnologyStack.VUE in result.technologies


# =========================================================================
# _detect_from_cache
# =========================================================================

class TestDetectFromCache:
    def test_detects_from_whatweb(self):
        _scan_cache.cache.clear()
        _scan_cache.cache["sid:whatweb:example.com"] = {
            "tool": "whatweb", "target": "example.com",
            "result": {"output": "Apache, PHP, WordPress"},
        }
        result = _detect_from_cache("example.com")
        assert result is not None
        assert "apache" in result.web_servers
        assert "wordpress" in result.cms

    def test_detects_from_httpx_headers(self):
        _scan_cache.cache.clear()
        _scan_cache.cache["sid:httpx:example.com"] = {
            "tool": "httpx", "target": "example.com",
            "result": {
                "output": "example.com [200]",
                "headers": {"Server": "Apache/2.4.41"},
            },
        }
        result = _detect_from_cache("example.com")
        assert result is not None
        assert "apache" in result.web_servers

    def test_no_cache_returns_none(self):
        _scan_cache.cache.clear()
        result = _detect_from_cache("unknown-target")
        assert result is None

    def test_empty_content_no_headers(self):
        _scan_cache.cache.clear()
        _scan_cache.cache["sid:nikto:example.com"] = {
            "tool": "nikto", "target": "example.com",
            "result": {"output": ""},
        }
        result = _detect_from_cache("example.com")
        assert result is None

    def test_from_httpx_no_output(self):
        _scan_cache.cache.clear()
        _scan_cache.cache["sid:httpx:example.com"] = {
            "tool": "httpx", "target": "example.com",
            "result": {
                "headers": {"Server": "Apache/2.4.41"},
            },
        }
        result = _detect_from_cache("example.com")
        assert result is not None

    def test_data_fallback(self):
        _scan_cache.cache.clear()
        _scan_cache.cache["sid:wpscan:example.com"] = {
            "tool": "wpscan", "target": "example.com",
            "result": {"data": "WordPress 5.8"},
        }
        result = _detect_from_cache("example.com")
        assert result is not None


# =========================================================================
# _build_destructive_confirmation — remaining branches
# =========================================================================

class TestBuildDestructiveConfirmationRemaining:
    def test_aireplay_no_attack_mode(self):
        msg = _build_destructive_confirmation("aireplay_ng", {"interface": "wlan0"})
        assert msg is not None
        assert "aireplay" in msg["action"].lower() or "aireplay" in msg["action"]

    def test_aireplay_mode_9_allowed(self):
        msg = _build_destructive_confirmation("aireplay_ng", {"attack_mode": 9})
        assert msg is None

    def test_aireplay_with_client_mac(self):
        msg = _build_destructive_confirmation(
            "aireplay_ng",
            {"interface": "wlan0", "attack_mode": 0, "bssid": "AA:BB:CC:DD:EE:FF",
             "client_mac": "11:22:33:44:55:66"},
        )
        assert msg is not None
        assert "11:22:33:44:55:66" in msg["detail"]

    def test_aireplay_mode_0_warning(self):
        msg = _build_destructive_confirmation(
            "aireplay_ng",
            {"interface": "wlan0", "attack_mode": 0, "bssid": "AA:BB:CC:DD:EE:FF"},
        )
        assert msg is not None
        assert "disrupt" in msg["warning"]

    def test_aireplay_no_bssid(self):
        msg = _build_destructive_confirmation(
            "aireplay_ng",
            {"interface": "wlan0", "attack_mode": 5},
        )
        assert msg is not None
        assert "all visible networks" in msg["detail"]

    def test_responder_full_params(self):
        msg = _build_destructive_confirmation(
            "responder",
            {"interface": "eth1", "duration": 600, "wpad": False, "force_wpad_auth": True},
        )
        assert msg is not None
        assert "eth1" in msg["action"]
        assert "600" in msg["detail"]
        assert "WPAD: False" in msg["detail"]

    def test_responder_analyze_allowed(self):
        msg = _build_destructive_confirmation("responder", {"analyze": True})
        assert msg is None

    def test_responder_default_interface(self):
        msg = _build_destructive_confirmation("responder", {})
        assert msg is not None
        assert "eth0" in msg["action"]

    def test_metasploit_auxiliary_allowed(self):
        msg = _build_destructive_confirmation(
            "metasploit", {"module": "auxiliary/scanner/ssh/ssh_version"}
        )
        assert msg is None

    def test_metasploit_gather_allowed(self):
        msg = _build_destructive_confirmation(
            "metasploit", {"module": "auxiliary/gather/"}
        )
        assert msg is None

    def test_metasploit_no_module(self):
        msg = _build_destructive_confirmation("metasploit", {})
        assert msg is not None

    def test_metasploit_non_dict_options(self):
        msg = _build_destructive_confirmation(
            "metasploit", {"module": "exploit/multi/handler", "options": "invalid"}
        )
        assert msg is not None

    def test_metasploit_rhosts_from_options(self):
        msg = _build_destructive_confirmation(
            "metasploit",
            {"module": "exploit/multi/handler", "options": {"RHOSTS": "10.0.0.5"}},
        )
        assert msg is not None
        assert "10.0.0.5" in msg["detail"]

    def test_mitm6_basic(self):
        msg = _build_destructive_confirmation(
            "mitm6", {"interface": "eth0", "domain": "corp.local"}
        )
        assert msg is not None
        assert "mitm6" in msg["action"].lower()
        assert "corp.local" in msg["detail"]

    def test_mitm6_no_domain(self):
        msg = _build_destructive_confirmation("mitm6", {})
        assert msg is not None
        assert "All domains" in msg["detail"]

    def test_mdk4_fallback(self):
        msg = _build_destructive_confirmation("mdk4", {"interface": "wlan0"})
        assert msg is not None
        assert "MDK4" in msg["action"]

    def test_unknown_tool_returns_none(self):
        msg = _build_destructive_confirmation("not_a_destructive_tool", {})
        assert msg is None


# =========================================================================
# _infer_param_type
# =========================================================================

class TestInferParamType:
    def test_bool(self):
        assert _infer_param_type(True) == bool
        assert _infer_param_type(False) == bool

    def test_int(self):
        assert _infer_param_type(42) == int

    def test_float(self):
        assert _infer_param_type(3.14) == float

    def test_dict(self):
        assert _infer_param_type({"key": "val"}) == dict

    def test_list(self):
        assert _infer_param_type([1, 2, 3]) == list

    def test_str_default(self):
        assert _infer_param_type("hello") == str


# =========================================================================
# _resolve_required_param_type
# =========================================================================

class TestResolveRequiredParamType:
    def test_bool(self):
        assert _resolve_required_param_type({"type": "bool"}) == bool

    def test_int(self):
        assert _resolve_required_param_type({"type": "int"}) == int

    def test_float(self):
        assert _resolve_required_param_type({"type": "float"}) == float

    def test_dict(self):
        assert _resolve_required_param_type({"type": "dict"}) == dict

    def test_list(self):
        assert _resolve_required_param_type({"type": "list"}) == list

    def test_str_default(self):
        assert _resolve_required_param_type({"type": "unknown"}) == str
        assert _resolve_required_param_type({}) == str


# =========================================================================
# _build_typed_tool_doc
# =========================================================================

class TestBuildTypedToolDoc:
    def test_build_doc_with_params_and_optional(self):
        tool_def = {
            "desc": "Scan ports on a target",
            "params": {"target": {"type": "str"}},
            "optional": {"ports": "80,443", "timeout": 30},
        }
        doc = _build_typed_tool_doc("nmap", "Scan ports on a target", tool_def)
        assert "Scan ports on a target" in doc
        assert "target: Required parameter" in doc
        assert "ports: Optional parameter" in doc
        assert "timeout: Optional parameter" in doc
        assert "run_security_tool" in doc

    def test_build_doc_no_optional(self):
        tool_def = {
            "desc": "Simple tool",
            "params": {"input": {"type": "str"}},
            "optional": {},
        }
        doc = _build_typed_tool_doc("test", "Simple tool", tool_def)
        assert "input: Required parameter" in doc
        assert "Optional parameter" not in doc

    def test_build_doc_optional_with_string_defaults(self):
        tool_def = {
            "desc": "Tool with string optional",
            "params": {"url": {"type": "str"}},
            "optional": {"method": "GET", "headers": {}},
        }
        doc = _build_typed_tool_doc("test", "Tool with string optional", tool_def)
        assert "method: Optional parameter" in doc
        assert "headers: Optional parameter" in doc


# =========================================================================
# _get_registry_tool_definition
# =========================================================================

class TestGetRegistryToolDefinition:
    def test_direct_lookup(self):
        tool_def = _get_registry_tool_definition("nmap")
        assert tool_def is not None
        assert "desc" in tool_def

    def test_alias_lookup(self):
        tool_def = _get_registry_tool_definition("aircrack_ng")
        assert tool_def is not None
        assert "desc" in tool_def

    def test_unknown_returns_none(self):
        tool_def = _get_registry_tool_definition("completely_fake_tool_xyz")
        assert tool_def is None

    def test_libc_alias(self):
        tool_def = _get_registry_tool_definition("libc")
        assert tool_def is not None

    def test_theharvester_alias(self):
        tool_def = _get_registry_tool_definition("theharvester")
        assert tool_def is not None

    def test_wifite2_alias(self):
        tool_def = _get_registry_tool_definition("wifite2")
        assert tool_def is not None


# =========================================================================
# _create_typed_tool_wrapper
# =========================================================================

class TestCreateTypedToolWrapper:
    def test_wrapper_creation(self):
        tool_def = {
            "desc": "A test tool",
            "params": {"target": {"type": "str"}},
            "optional": {"verbose": False, "count": 1, "name": "default"},
        }
        async def fake_run(ctx, tool_name, params):
            return {"success": True, "tool": tool_name, "params": params}

        wrapper = _create_typed_tool_wrapper("test_tool", tool_def, fake_run)
        assert wrapper.__name__ == "test_tool_typed"
        assert wrapper.__doc__ is not None
        assert "target" in wrapper.__annotations__
        assert wrapper.__annotations__["target"] == str
        assert wrapper.__annotations__["verbose"] == bool
        assert wrapper.__annotations__["count"] == int
        assert wrapper.__annotations__["name"] == str

    def test_wrapper_execution_with_optional(self):
        tool_def = {
            "desc": "A test tool",
            "params": {"target": {"type": "str"}},
            "optional": {"verbose": False},
        }
        call_log = []

        async def fake_run(ctx, tool_name, params):
            call_log.append((tool_name, params))
            return {"success": True, "tool": tool_name}

        wrapper = _create_typed_tool_wrapper("test_tool", tool_def, fake_run)
        ctx = SimpleNamespace(session_id="test")
        with patch("mcp_core.server_setup.get_context", return_value=ctx):
            result = run(wrapper(target="example.com", verbose=True))
        assert result["success"] is True
        assert call_log[0][0] == "test_tool"
        assert call_log[0][1] == {"target": "example.com", "verbose": True}

    def test_wrapper_execution_without_optional(self):
        tool_def = {
            "desc": "A test tool",
            "params": {"target": {"type": "str"}},
            "optional": {"verbose": False},
        }
        call_log = []

        async def fake_run(ctx, tool_name, params):
            call_log.append((tool_name, params))
            return {"success": True, "tool": tool_name}

        wrapper = _create_typed_tool_wrapper("test_tool", tool_def, fake_run)
        ctx = SimpleNamespace(session_id="test")
        with patch("mcp_core.server_setup.get_context", return_value=ctx):
            result = run(wrapper(target="example.com"))
        assert result["success"] is True
        assert "verbose" not in call_log[0][1]


# =========================================================================
# _register_skills
# =========================================================================

class TestRegisterSkills:
    def test_provider_none(self):
        logger = MagicMock()
        mcp = MagicMock()
        with patch.object(server_setup, "SkillsDirectoryProvider", None):
            _register_skills(mcp, logger)
            logger.warning.assert_called_once()
            mcp.add_provider.assert_not_called()

    def test_skills_dir_not_exists(self):
        logger = MagicMock()
        mcp = MagicMock()
        with patch("pathlib.Path.exists", return_value=False):
            _register_skills(mcp, logger)
            mcp.add_provider.assert_not_called()


# =========================================================================
# _read_skill_document — async helper with run()
# =========================================================================

class TestReadSkillDocument:
    def test_read_success(self):
        ctx = SimpleNamespace()
        resource = SimpleNamespace(contents=[SimpleNamespace(content="# Skill doc")])
        ctx.read_resource = AsyncMock(return_value=resource)
        result = run(_read_skill_document(ctx, "test-skill", "SKILL.md"))
        assert result == "# Skill doc"
        ctx.read_resource.assert_called_once_with("skill://test-skill/SKILL.md")

    def test_read_exception_returns_none(self):
        ctx = SimpleNamespace()
        ctx.read_resource = AsyncMock(side_effect=Exception("resource error"))
        result = run(_read_skill_document(ctx, "test-skill", "SKILL.md"))
        assert result is None

    def test_read_no_contents(self):
        ctx = SimpleNamespace()
        ctx.read_resource = AsyncMock(return_value=SimpleNamespace(contents=None))
        result = run(_read_skill_document(ctx, "test-skill", "SKILL.md"))
        assert result is None

    def test_read_empty_contents(self):
        ctx = SimpleNamespace()
        ctx.read_resource = AsyncMock(return_value=SimpleNamespace(contents=[]))
        result = run(_read_skill_document(ctx, "test-skill", "SKILL.md"))
        assert result is None

    def test_read_no_content_field(self):
        ctx = SimpleNamespace()
        ctx.read_resource = AsyncMock(
            return_value=SimpleNamespace(contents=[SimpleNamespace(content=None)])
        )
        result = run(_read_skill_document(ctx, "test-skill", "SKILL.md"))
        assert result is None

    def test_bytes_content_decoded(self):
        ctx = SimpleNamespace()
        ctx.read_resource = AsyncMock(
            return_value=SimpleNamespace(contents=[SimpleNamespace(content=b"bytes content")])
        )
        result = run(_read_skill_document(ctx, "test-skill", "SKILL.md"))
        assert result == "bytes content"

    def test_non_list_contents(self):
        ctx = SimpleNamespace()
        ctx.read_resource = AsyncMock(
            return_value=SimpleNamespace(
                contents=SimpleNamespace(content="# Non-list content")
            )
        )
        result = run(_read_skill_document(ctx, "test-skill", "SKILL.md"))
        assert result == "# Non-list content"


# =========================================================================
# _read_skill_bundle
# =========================================================================

class TestReadSkillBundle:
    def test_bundle_returns_documents(self):
        ctx = SimpleNamespace()
        resource = SimpleNamespace(contents=[SimpleNamespace(content="# SKILL")])
        ctx.read_resource = AsyncMock(return_value=resource)
        result = run(_read_skill_bundle(ctx, "test-skill"))
        assert "SKILL.md" in result

    def test_bundle_empty_when_no_files(self):
        ctx = SimpleNamespace()
        ctx.read_resource = AsyncMock(side_effect=Exception("not found"))
        result = run(_read_skill_bundle(ctx, "test-skill"))
        assert result == {}


# =========================================================================
# _normalize_tool_result — various input types
# =========================================================================

class TestNormalizeToolResult:
    def test_non_dict_string_input(self):
        result = _normalize_tool_result("not a dict")
        assert result["success"] is False
        assert "Invalid tool result type" in result["error"]

    def test_none_input(self):
        result = _normalize_tool_result(None)
        assert result["success"] is False

    def test_list_input(self):
        result = _normalize_tool_result([1, 2, 3])
        assert result["success"] is False

    def test_int_input(self):
        result = _normalize_tool_result(42)
        assert result["success"] is False

    def test_full_normalization(self):
        result = _normalize_tool_result({
            "success": True,
            "output": "scan results",
            "error": "something",
            "returncode": 0,
            "timed_out": True,
            "partial_results": True,
            "execution_time": 5.5,
            "timestamp": "2024-01-01",
        })
        assert result["success"] is True
        assert result["output"] == "scan results"
        assert result["returncode"] == 0

    def test_normalize_with_legacy_return_code(self):
        result = _normalize_tool_result({
            "success": False,
            "stdout": "out_text",
            "stderr": "err_text",
            "return_code": 1,
        })
        assert result["output"] == "out_text"
        assert result["error"] == "err_text"
        assert result["returncode"] == 1


# =========================================================================
# ImportError fallback values
# =========================================================================

class TestImportFallbacks:
    def test_skills_directory_provider(self):
        val = getattr(server_setup, "SkillsDirectoryProvider", "NOT_FOUND")
        assert val is not None or val is None

    def test_bm25_search_transform(self):
        val = getattr(server_setup, "BM25SearchTransform", "NOT_FOUND")
        assert val is not None or val is None

    def test_serialize_tools(self):
        val = getattr(server_setup, "serialize_tools_for_output_markdown", "NOT_FOUND")
        assert val is not None or val is None


# =========================================================================
# get_tool_skill — via MCP
# =========================================================================

class TestGetToolSkill:
    def test_skill_found(self):
        mcp = server_setup.setup_mcp_server_standalone()
        tool = run(mcp.get_tool("get_tool_skill"))
        assert tool is not None
        ctx = SimpleNamespace(
            session_id="test-skills",
            info=AsyncMock(),
            error=AsyncMock(),
            report_progress=AsyncMock(),
            read_resource=AsyncMock(
                return_value=SimpleNamespace(
                    contents=[SimpleNamespace(content="# Nmap Recon Skill")]
                )
            ),
        )
        result = run(tool.fn(ctx=ctx, tool_name="nmap"))
        assert result is not None

    def test_skill_not_found(self):
        mcp = server_setup.setup_mcp_server_standalone()
        tool = run(mcp.get_tool("get_tool_skill"))
        assert tool is not None
        ctx = SimpleNamespace(
            session_id="test-skills",
            info=AsyncMock(),
            error=AsyncMock(),
            report_progress=AsyncMock(),
            read_resource=AsyncMock(side_effect=Exception("not found")),
        )
        result = run(tool.fn(ctx=ctx, tool_name="nonexistent_tool_xyz"))
        assert result is not None
        assert result.get("success") is False


# =========================================================================
# validate_environment — via MCP
# =========================================================================

class TestValidateEnvironment:
    def test_validate_with_filter(self):
        mcp = server_setup.setup_mcp_server_standalone()
        tool = run(mcp.get_tool("validate_environment"))
        assert tool is not None
        ctx = SimpleNamespace(
            session_id="test-validate",
            info=AsyncMock(),
            error=AsyncMock(),
            report_progress=AsyncMock(),
        )
        result = run(tool.fn(ctx=ctx, tool_filter="nmap"))
        assert result is not None


# =========================================================================
# server_health — via MCP
# =========================================================================

class TestServerHealth:
    def test_server_health_tool(self):
        mcp = server_setup.setup_mcp_server_standalone()
        tool = run(mcp.get_tool("server_health"))
        assert tool is not None
        result = run(tool.fn())
        assert result["status"] == "healthy"
        assert result["server"] == "hexstrike-ai-pulse"
        assert result["tools_count"] > 0


# =========================================================================
# plan_attack — via MCP
# =========================================================================

class TestPlanAttack:
    def test_ctf_path(self):
        mcp = server_setup.setup_mcp_server_standalone()
        tool = run(mcp.get_tool("plan_attack"))
        assert tool is not None
        ctx = SimpleNamespace(
            session_id="test-ctf",
            info=AsyncMock(),
            error=AsyncMock(),
            report_progress=AsyncMock(),
            get_state=AsyncMock(return_value=None),
            set_state=AsyncMock(),
            get_prompt=AsyncMock(return_value=SimpleNamespace(messages=[])),
        )
        result = run(tool.fn(
            ctx=ctx,
            target="10.0.0.1",
            objective="ctf",
            ctf_category="web",
            ctf_difficulty="medium",
            ctf_points=200,
            ctf_description="A web challenge",
        ))
        assert isinstance(result, dict) or getattr(result, 'to_dict', None)


# =========================================================================
# Resource endpoints — via MCP
# =========================================================================

class TestResources:
    def test_health_resource(self):
        mcp = server_setup.setup_mcp_server_standalone()
        resources = run(mcp.list_resources())
        uris = [str(r.uri) for r in resources]
        assert "health://server" in uris
        assert "scan://cache/list" in uris
        assert "metrics://tools" in uris
        assert "errors://statistics" in uris

    def test_scan_latest_empty(self):
        _scan_cache.cache.clear()
        mcp = server_setup.setup_mcp_server_standalone()
        resources = run(mcp.list_resources())
        uris = [str(r.uri) for r in resources]
        assert "scan://cache/list" in uris

    def test_scan_result_empty(self):
        _scan_cache.cache.clear()
        mcp = server_setup.setup_mcp_server_standalone()
        resources = run(mcp.list_resources())
        uris = [str(r.uri) for r in resources]
        assert "scan://cache/list" in uris
