from server_core.failure_recovery_system import FailureRecoverySystem


class TestFailureRecoverySystem:
    def setup_method(self):
        self.frs = FailureRecoverySystem()

    def test_timeout_pattern(self):
        r = self.frs.analyze_failure("connection timeout occurred", 0)
        assert r["failure_type"] == "timeout"

    def test_permission_denied(self):
        r = self.frs.analyze_failure("permission denied to file", 0)
        assert r["failure_type"] == "permission_denied"

    def test_not_found(self):
        r = self.frs.analyze_failure("command not found", 0)
        assert r["failure_type"] == "not_found"

    def test_network_error(self):
        r = self.frs.analyze_failure("connection refused on port 80", 0)
        assert r["failure_type"] == "network_error"

    def test_rate_limited(self):
        r = self.frs.analyze_failure("rate limit exceeded", 0)
        assert r["failure_type"] == "rate_limited"

    def test_authentication_required(self):
        r = self.frs.analyze_failure("unauthorized access", 0)
        assert r["failure_type"] == "authentication_required"

    def test_unknown_failure(self):
        r = self.frs.analyze_failure("some random error", 0)
        assert r["failure_type"] == "unknown"
        assert r["confidence"] == 0.0

    def test_exit_code_1_adds_confidence(self):
        r = self.frs.analyze_failure("some error", 1)
        assert r["confidence"] > 0

    def test_exit_code_124_timeout(self):
        r = self.frs.analyze_failure("some error", 124)
        assert r["failure_type"] == "timeout"
        assert r["confidence"] >= 0.5

    def test_exit_code_126_permission(self):
        r = self.frs.analyze_failure("some error", 126)
        assert r["failure_type"] == "permission_denied"
        assert r["confidence"] >= 0.5

    def test_confidence_capped_at_one(self):
        r = self.frs.analyze_failure("timeout connection timeout rate limit throttled", -1)
        # pattern matches should push confidence toward 1.0
        assert r["confidence"] <= 1.0

    def test_timeout_recovery_strategies(self):
        r = self.frs.analyze_failure("timed out", 0)
        assert len(r["recovery_strategies"]) > 0
        assert "Increase timeout values" in r["recovery_strategies"]

    def test_permission_recovery_strategies(self):
        r = self.frs.analyze_failure("permission denied", 0)
        assert "Run with elevated privileges" in r["recovery_strategies"]

    def test_rate_limited_recovery_strategies(self):
        r = self.frs.analyze_failure("rate limit", 0)
        assert "Implement delays between requests" in r["recovery_strategies"]

    def test_network_error_recovery_strategies(self):
        r = self.frs.analyze_failure("network unreachable", 0)
        assert "Check network connectivity" in r["recovery_strategies"]

    def test_alternative_tools_nmap(self):
        r = self.frs.analyze_failure("nmap: error", 0)
        assert "rustscan" in r["alternative_tools"]

    def test_alternative_tools_hydra(self):
        r = self.frs.analyze_failure("hydra failed", 0)
        assert "medusa" in r["alternative_tools"]

    def test_alternative_tools_unknown(self):
        r = self.frs.analyze_failure("unknown_tool_xyz failed", 0)
        assert r["alternative_tools"] == []

    def test_extract_tool_name(self):
        assert self.frs._extract_tool_name("nmap error") == "nmap"

    def test_extract_tool_name_unknown(self):
        assert self.frs._extract_tool_name("random stuff") == "unknown"
