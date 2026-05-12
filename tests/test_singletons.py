"""Coverage for singletons.py — 41% → 100%. Tests lazy accessors + __getattr__."""

import pytest

from server_core import singletons


def test_group_a_imports():
    assert singletons.cache is not None
    assert singletons.enhanced_process_manager is not None
    assert singletons.error_handler is not None
    assert singletons.degradation_manager is not None
    assert singletons.ROCKYOU_PATH is not None
    assert singletons.COMMON_DIRB_PATH is not None
    assert singletons.COMMON_DIRSEARCH_PATH is not None


class TestGetSessionStore:
    def test_first_call_creates(self):
        s = singletons.get_session_store()
        assert s is not None

    def test_second_call_returns_same(self):
        a = singletons.get_session_store()
        b = singletons.get_session_store()
        assert a is b


class TestGetWordlistStore:
    def test_first_call_creates(self):
        w = singletons.get_wordlist_store()
        assert w is not None

    def test_second_call_returns_same(self):
        a = singletons.get_wordlist_store()
        b = singletons.get_wordlist_store()
        assert a is b


class TestGetCveIntelligence:
    def test_first_call_creates(self):
        c = singletons.get_cve_intelligence()
        assert c is not None

    def test_second_call_returns_same(self):
        a = singletons.get_cve_intelligence()
        b = singletons.get_cve_intelligence()
        assert a is b


class TestGetExploitGenerator:
    def test_first_call_creates(self):
        e = singletons.get_exploit_generator()
        assert e is not None

    def test_second_call_returns_same(self):
        a = singletons.get_exploit_generator()
        b = singletons.get_exploit_generator()
        assert a is b


class TestGetVulnerabilityCorrelator:
    def test_first_call_creates(self):
        v = singletons.get_vulnerability_correlator()
        assert v is not None

    def test_second_call_returns_same(self):
        a = singletons.get_vulnerability_correlator()
        b = singletons.get_vulnerability_correlator()
        assert a is b


class TestGetDecisionEngine:
    def test_first_call_creates(self):
        d = singletons.get_decision_engine()
        assert d is not None

    def test_second_call_returns_same(self):
        a = singletons.get_decision_engine()
        b = singletons.get_decision_engine()
        assert a is b


class TestGetBugbountyManager:
    def test_first_call_creates(self):
        b = singletons.get_bugbounty_manager()
        assert b is not None

    def test_second_call_returns_same(self):
        a = singletons.get_bugbounty_manager()
        b = singletons.get_bugbounty_manager()
        assert a is b


class TestGetFileuploadFramework:
    def test_first_call_creates(self):
        f = singletons.get_fileupload_framework()
        assert f is not None

    def test_second_call_returns_same(self):
        a = singletons.get_fileupload_framework()
        b = singletons.get_fileupload_framework()
        assert a is b


class TestGetCtfManager:
    def test_first_call_creates(self):
        c = singletons.get_ctf_manager()
        assert c is not None

    def test_second_call_returns_same(self):
        a = singletons.get_ctf_manager()
        b = singletons.get_ctf_manager()
        assert a is b


class TestGetCtfTools:
    def test_first_call_creates(self):
        t = singletons.get_ctf_tools()
        assert t is not None

    def test_second_call_returns_same(self):
        a = singletons.get_ctf_tools()
        b = singletons.get_ctf_tools()
        assert a is b


class TestGetCtfAutomator:
    def test_first_call_creates(self):
        a = singletons.get_ctf_automator()
        assert a is not None

    def test_second_call_returns_same(self):
        a = singletons.get_ctf_automator()
        b = singletons.get_ctf_automator()
        assert a is b


class TestGetCtfCoordinator:
    def test_first_call_creates(self):
        c = singletons.get_ctf_coordinator()
        assert c is not None

    def test_second_call_returns_same(self):
        a = singletons.get_ctf_coordinator()
        b = singletons.get_ctf_coordinator()
        assert a is b


class TestGetAttrBackwardCompat:
    """Test module-level __getattr__ for backward-compat aliases."""

    def test_session_store_alias(self):
        s = getattr(singletons, "session_store")
        assert s is not None

    def test_wordlist_store_alias(self):
        w = getattr(singletons, "wordlist_store")
        assert w is not None

    def test_cve_intelligence_alias(self):
        c = getattr(singletons, "cve_intelligence")
        assert c is not None

    def test_exploit_generator_alias(self):
        e = getattr(singletons, "exploit_generator")
        assert e is not None

    def test_vulnerability_correlator_alias(self):
        v = getattr(singletons, "vulnerability_correlator")
        assert v is not None

    def test_decision_engine_alias(self):
        d = getattr(singletons, "decision_engine")
        assert d is not None

    def test_bugbounty_manager_alias(self):
        b = getattr(singletons, "bugbounty_manager")
        assert b is not None

    def test_fileupload_framework_alias(self):
        f = getattr(singletons, "fileupload_framework")
        assert f is not None

    def test_ctf_manager_alias(self):
        c = getattr(singletons, "ctf_manager")
        assert c is not None

    def test_ctf_tools_alias(self):
        c = getattr(singletons, "ctf_tools")
        assert c is not None

    def test_ctf_automator_alias(self):
        c = getattr(singletons, "ctf_automator")
        assert c is not None

    def test_ctf_coordinator_alias(self):
        c = getattr(singletons, "ctf_coordinator")
        assert c is not None

    def test_unknown_attr_raises(self):
        with pytest.raises(AttributeError, match="nonexistent"):
            getattr(singletons, "nonexistent")
