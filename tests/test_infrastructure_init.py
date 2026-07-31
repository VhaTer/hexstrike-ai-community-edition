"""Tests for pulse.infrastructure lazy import mechanism."""

import pulse.infrastructure


class TestLazyImports:
    def test_cache_resolves(self):
        c = pulse.infrastructure.storage.cache
        assert c is not None

    def test_enhanced_process_manager_resolves(self):
        epm = pulse.infrastructure.execution.enhanced_process_manager
        assert epm is not None

    def test_telemetry_resolves(self):
        t = pulse.infrastructure.telemetry
        assert t is not None

    def test_error_handler_resolves(self):
        eh = pulse.infrastructure.error_handler
        assert eh is not None

    def test_rocyou_path_resolves(self):
        path = pulse.infrastructure.ROCKYOU_PATH
        assert isinstance(path, str)

    def test_hexstrikecache_class(self):
        cls = pulse.infrastructure.HexStrikeCache
        assert cls is not None

    def test_modern_visual_engine_class(self):
        cls = pulse.infrastructure.ModernVisualEngine
        assert cls is not None

    def test_execute_command_resolves(self):
        fn = pulse.infrastructure._execute_command
        assert callable(fn)

    def test_execute_command_with_recovery_resolves(self):
        fn = pulse.infrastructure._execute_command_with_recovery
        assert callable(fn)

    def test_unknown_attr_raises(self):
        try:
            pulse.infrastructure.nonexistent_attr
            assert False, "Should have raised"
        except AttributeError:
            pass

    def test_alias_spec(self):
        # _execute_command is aliased: spec[1] = "execute_command"
        fn = pulse.infrastructure._execute_command
        assert callable(fn)

    def test_intelligence_class_resolves(self):
        cls = pulse.infrastructure.IntelligentDecisionEngine
        assert cls is not None

    def test_ctf_challenge_resolves(self):
        cls = pulse.infrastructure.CTFChallenge
        assert cls is not None

    def test_getattr_is_idempotent(self):
        """Second access returns cached value (no import)."""
        c1 = pulse.infrastructure.storage.cache
        c2 = pulse.infrastructure.storage.cache
        assert c1 is c2
