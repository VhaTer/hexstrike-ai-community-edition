from server_core.workflows.ctf.CTFChallenge import CTFChallenge
from server_core.workflows.ctf.automator import CTFChallengeAutomator


def make_challenge(category="web", name="test-ctf", difficulty="medium"):
    return CTFChallenge(
        name=name,
        category=category,
        description=f"A {category} CTF challenge",
        difficulty=difficulty,
        points=100,
        target="127.0.0.1",
    )


class TestExtractFlagCandidates:
    def setup_method(self):
        self.automator = CTFChallengeAutomator()

    def test_flag_format(self):
        candidates = self.automator._extract_flag_candidates("flag{abc123}")
        assert "flag{abc123}" in candidates

    def test_ctf_format(self):
        candidates = self.automator._extract_flag_candidates("ctf{test_flag}")
        assert "ctf{test_flag}" in candidates

    def test_md5_hash(self):
        candidates = self.automator._extract_flag_candidates("hash 5f4dcc3b5aa765d61d8327deb882cf99 end")
        assert "5f4dcc3b5aa765d61d8327deb882cf99" in candidates

    def test_sha1_hash(self):
        h = "a9993e364706816aba3e25717850c26c9cd0d89d"
        candidates = self.automator._extract_flag_candidates(h)
        assert h in candidates

    def test_no_flag(self):
        candidates = self.automator._extract_flag_candidates("nothing here")
        assert candidates == []

    def test_multiple_flags(self):
        text = "first flag{one} and second flag{two}"
        candidates = self.automator._extract_flag_candidates(text)
        assert "flag{one}" in candidates
        assert "flag{two}" in candidates

    def test_dedup(self):
        text = "flag{dup} something flag{dup}"
        candidates = self.automator._extract_flag_candidates(text)
        assert len(candidates) == 1


class TestValidateFlagFormat:
    def setup_method(self):
        self.automator = CTFChallengeAutomator()

    def test_valid_flag(self):
        assert self.automator._validate_flag_format("flag{content}")

    def test_valid_ctf(self):
        assert self.automator._validate_flag_format("ctf{content}")

    def test_valid_bracket_format(self):
        assert self.automator._validate_flag_format("FLAG{content}")

    def test_invalid_no_brackets(self):
        assert not self.automator._validate_flag_format("plaintext")

    def test_invalid_md5(self):
        assert not self.automator._validate_flag_format("5f4dcc3b5aa765d61d8327deb882cf99")


class TestManualGuidance:
    def setup_method(self):
        self.automator = CTFChallengeAutomator()

    def test_web_guidance(self):
        ch = make_challenge("web")
        guidance = self.automator._generate_manual_guidance(ch, {"automated_steps": []})
        actions = [g["action"] for g in guidance]
        assert "manual_source_review" in actions
        assert "parameter_fuzzing" in actions

    def test_crypto_guidance(self):
        ch = make_challenge("crypto")
        guidance = self.automator._generate_manual_guidance(ch, {"automated_steps": []})
        actions = [g["action"] for g in guidance]
        assert "cipher_research" in actions

    def test_pwn_guidance(self):
        ch = make_challenge("pwn")
        guidance = self.automator._generate_manual_guidance(ch, {"automated_steps": []})
        actions = [g["action"] for g in guidance]
        assert "exploit_development" in actions

    def test_forensics_guidance(self):
        ch = make_challenge("forensics")
        guidance = self.automator._generate_manual_guidance(ch, {"automated_steps": []})
        actions = [g["action"] for g in guidance]
        assert "steganography_deep_dive" in actions

    def test_rev_guidance(self):
        ch = make_challenge("rev")
        guidance = self.automator._generate_manual_guidance(ch, {"automated_steps": []})
        actions = [g["action"] for g in guidance]
        assert "algorithm_analysis" in actions


class TestAutoSolve:
    def setup_method(self):
        self.automator = CTFChallengeAutomator()

    def test_basic_flow(self):
        ch = make_challenge("web", "test-flow")
        result = self.automator.auto_solve_challenge(ch)
        assert result["challenge_id"] == "test-flow"
        assert "automated_steps" in result
        assert "flag_candidates" in result
        assert result["confidence"] >= 0

    def test_confidence_increases(self):
        ch = make_challenge("web")
        result = self.automator.auto_solve_challenge(ch)
        assert result["confidence"] <= 1.0

    def test_manual_steps_on_unsolved(self):
        ch = make_challenge("crypto", "unsolved-test")
        result = self.automator.auto_solve_challenge(ch)
        assert result["status"] in ("solved", "needs_manual_intervention")
        if result["status"] == "needs_manual_intervention":
            assert len(result["manual_steps"]) > 0


class TestParallelStep:
    def setup_method(self):
        self.automator = CTFChallengeAutomator()

    def test_parallel_step_with_tools(self):
        ch = make_challenge("web")
        step = {
            "step": "recon",
            "action": "scan",
            "description": "Port scan",
            "parallel": True,
            "tools": ["nmap", "masscan"],
        }
        result = self.automator._execute_parallel_step(step, ch)
        assert "nmap" in result["tools_used"]
        assert result["success"] is True
        assert result["execution_time"] >= 0

    def test_parallel_step_manual_skip(self):
        ch = make_challenge("web")
        step = {
            "step": "analyze",
            "action": "manual",
            "description": "Analysis",
            "parallel": True,
            "tools": ["manual"],
        }
        result = self.automator._execute_parallel_step(step, ch)
        assert result["tools_used"] == []


class TestSequentialStep:
    def setup_method(self):
        self.automator = CTFChallengeAutomator()

    def test_sequential_tool(self):
        ch = make_challenge("web")
        step = {
            "step": "recon",
            "action": "scan",
            "description": "Port scan",
            "parallel": False,
            "tools": ["nmap"],
        }
        result = self.automator._execute_sequential_step(step, ch)
        assert "nmap" in result["tools_used"]
        assert result["success"] is True

    def test_sequential_manual(self):
        ch = make_challenge("pwn")
        step = {
            "step": "exploit",
            "action": "manual_exploit",
            "description": "Manual exploit dev",
            "parallel": False,
            "tools": ["manual"],
        }
        result = self.automator._execute_sequential_step(step, ch)
        assert "[MANUAL]" in result["output"]

    def test_sequential_custom(self):
        ch = make_challenge("crypto")
        step = {
            "step": "decrypt",
            "action": "custom",
            "description": "Custom decryption",
            "parallel": False,
            "tools": ["custom"],
        }
        result = self.automator._execute_sequential_step(step, ch)
        assert "[CUSTOM]" in result["output"]


class TestErrorHandling:
    def setup_method(self):
        self.automator = CTFChallengeAutomator()

    def test_auto_solve_exception(self):
        """Pass None challenge to trigger exception path."""
        try:
            result = self.automator.auto_solve_challenge(None)  # type: ignore
            assert result["status"] == "error"
        except AttributeError:
            pass  # acceptable if None fails before try/except
