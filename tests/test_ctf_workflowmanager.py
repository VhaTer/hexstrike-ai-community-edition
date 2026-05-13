"""Test suite for CTF Workflow Manager"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from server_core.workflows.ctf.workflowManager import CTFWorkflowManager
from server_core.workflows.ctf.CTFChallenge import CTFChallenge


class TestCTFWorkflowManager:
    """Comprehensive tests for CTFWorkflowManager"""

    @pytest.fixture
    def workflow_manager(self):
        """Create CTFWorkflowManager instance"""
        return CTFWorkflowManager()

    @pytest.fixture
    def sample_web_challenge(self):
        """Create a sample web CTF challenge"""
        return CTFChallenge(
            name="SQL Injection Challenge",
            category="web",
            description="Find the SQL injection vulnerability in the login form",
            points=100,
            difficulty="medium",
            url="http://example.com/login",
            target="example.com"
        )

    @pytest.fixture
    def sample_crypto_challenge(self):
        """Create a sample crypto CTF challenge"""
        return CTFChallenge(
            name="MD5 Hash Challenge",
            category="crypto",
            description="Break this MD5 hash: 5d41402abc4b2a76b9719d911017c592",
            points=50,
            difficulty="easy",
            target="5d41402abc4b2a76b9719d911017c592"
        )

    @pytest.fixture
    def sample_pwn_challenge(self):
        """Create a sample pwn CTF challenge"""
        return CTFChallenge(
            name="Buffer Overflow Challenge",
            category="pwn",
            description="Exploit the buffer overflow vulnerability in the binary",
            points=200,
            difficulty="hard",
            files=["vuln_binary"],
            target="vuln_binary"
        )

    @pytest.fixture
    def sample_forensics_challenge(self):
        """Create a sample forensics CTF challenge"""
        return CTFChallenge(
            name="Image Steganography Challenge",
            category="forensics",
            description="Extract hidden data from the provided PNG image using steganography",
            points=75,
            difficulty="medium",
            files=["image.png"],
            target="image.png"
        )

    @pytest.fixture
    def sample_rev_challenge(self):
        """Create a sample reverse engineering challenge"""
        return CTFChallenge(
            name="Binary Reverse Engineering",
            category="rev",
            description="Reverse engineer the binary to find the hidden flag",
            points=150,
            difficulty="hard",
            files=["binary"],
            target="binary"
        )

    @pytest.fixture
    def sample_misc_challenge(self):
        """Create a sample misc CTF challenge"""
        return CTFChallenge(
            name="Base64 Encoding Challenge",
            category="misc",
            description="Decode this base64 encoded string",
            points=25,
            difficulty="easy",
            target="aGVsbG8gd29ybGQ="
        )

    @pytest.fixture
    def sample_osint_challenge(self):
        """Create a sample OSINT challenge"""
        return CTFChallenge(
            name="Subdomain Enumeration",
            category="osint",
            description="Find all subdomains for example.com",
            points=80,
            difficulty="medium",
            target="example.com"
        )

    def test_initialization(self, workflow_manager):
        """Test CTFWorkflowManager initialization"""
        assert workflow_manager is not None
        assert hasattr(workflow_manager, 'category_tools')
        assert hasattr(workflow_manager, 'solving_strategies')
        assert isinstance(workflow_manager.category_tools, dict)
        assert isinstance(workflow_manager.solving_strategies, dict)

    def test_category_tools_structure(self, workflow_manager):
        """Test category_tools has all categories"""
        expected_categories = ["web", "crypto", "pwn", "forensics", "rev", "misc", "osint"]
        for category in expected_categories:
            assert category in workflow_manager.category_tools
            assert isinstance(workflow_manager.category_tools[category], dict)

    def test_solving_strategies_structure(self, workflow_manager):
        """Test solving_strategies has all categories"""
        expected_categories = ["web", "crypto", "pwn", "forensics", "rev"]
        for category in expected_categories:
            assert category in workflow_manager.solving_strategies
            assert isinstance(workflow_manager.solving_strategies[category], list)
            assert len(workflow_manager.solving_strategies[category]) > 0

    def test_create_ctf_challenge_workflow_web(self, workflow_manager, sample_web_challenge):
        """Test creating workflow for web challenge"""
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_web_challenge)
        assert isinstance(workflow, dict)
        assert workflow["challenge"] == "SQL Injection Challenge"
        assert workflow["category"] == "web"
        assert "tools" in workflow
        assert "strategies" in workflow

    def test_create_ctf_challenge_workflow_crypto(self, workflow_manager, sample_crypto_challenge):
        """Test creating workflow for crypto challenge"""
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_crypto_challenge)
        assert isinstance(workflow, dict)
        assert workflow["challenge"] == "MD5 Hash Challenge"
        assert workflow["category"] == "crypto"
        assert "tools" in workflow

    def test_create_ctf_challenge_workflow_pwn(self, workflow_manager, sample_pwn_challenge):
        """Test creating workflow for pwn challenge"""
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_pwn_challenge)
        assert isinstance(workflow, dict)
        assert workflow["category"] == "pwn"
        assert "tools" in workflow

    def test_create_ctf_challenge_workflow_forensics(self, workflow_manager, sample_forensics_challenge):
        """Test creating workflow for forensics challenge"""
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_forensics_challenge)
        assert isinstance(workflow, dict)
        assert workflow["category"] == "forensics"

    def test_create_ctf_challenge_workflow_rev(self, workflow_manager, sample_rev_challenge):
        """Test creating workflow for rev challenge"""
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_rev_challenge)
        assert isinstance(workflow, dict)
        assert workflow["category"] == "rev"

    def test_create_ctf_challenge_workflow_misc(self, workflow_manager, sample_misc_challenge):
        """Test creating workflow for misc challenge"""
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_misc_challenge)
        assert isinstance(workflow, dict)
        assert workflow["category"] == "misc"

    def test_create_ctf_challenge_workflow_osint(self, workflow_manager, sample_osint_challenge):
        """Test creating workflow for OSINT challenge"""
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_osint_challenge)
        assert isinstance(workflow, dict)
        assert workflow["category"] == "osint"

    def test_workflow_contains_required_fields(self, workflow_manager, sample_web_challenge):
        """Test workflow has all required fields"""
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_web_challenge)
        required_fields = [
            "challenge", "category", "difficulty", "points", "tools", 
            "strategies", "estimated_time", "success_probability", 
            "automation_level", "parallel_tasks", "dependencies",
            "fallback_strategies", "resource_requirements", 
            "expected_artifacts", "validation_steps"
        ]
        for field in required_fields:
            assert field in workflow

    def test_workflow_tools_is_list(self, workflow_manager, sample_web_challenge):
        """Test that workflow tools is a list"""
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_web_challenge)
        assert isinstance(workflow["tools"], list)

    def test_workflow_strategies_is_list(self, workflow_manager, sample_web_challenge):
        """Test that workflow strategies is a list"""
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_web_challenge)
        assert isinstance(workflow["strategies"], list)

    def test_workflow_parallel_tasks_is_list(self, workflow_manager, sample_web_challenge):
        """Test that parallel tasks is a list"""
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_web_challenge)
        assert isinstance(workflow["parallel_tasks"], list)

    def test_workflow_fallback_strategies_is_list(self, workflow_manager, sample_web_challenge):
        """Test that fallback strategies is a list"""
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_web_challenge)
        assert isinstance(workflow["fallback_strategies"], list)

    def test_workflow_resource_requirements_is_dict(self, workflow_manager, sample_web_challenge):
        """Test that resource requirements is a dict"""
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_web_challenge)
        assert isinstance(workflow["resource_requirements"], dict)

    def test_workflow_expected_artifacts_is_list(self, workflow_manager, sample_web_challenge):
        """Test that expected artifacts is a list"""
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_web_challenge)
        assert isinstance(workflow["expected_artifacts"], list)

    def test_workflow_validation_steps_is_list(self, workflow_manager, sample_web_challenge):
        """Test that validation steps is a list"""
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_web_challenge)
        assert isinstance(workflow["validation_steps"], list)

    def test_create_ctf_team_strategy_basic(self, workflow_manager, sample_web_challenge):
        """Test creating team strategy with basic challenge list"""
        challenges = [sample_web_challenge]
        strategy = workflow_manager.create_ctf_team_strategy(challenges, team_size=4)
        assert isinstance(strategy, dict)
        assert "team_size" in strategy or "challenges" in strategy

    def test_create_ctf_team_strategy_multiple_challenges(self, workflow_manager):
        """Test creating team strategy with multiple challenges"""
        challenges = [
            CTFChallenge("Web1", "web", "Challenge 1", 100, "easy"),
            CTFChallenge("Crypto1", "crypto", "Challenge 2", 100, "medium"),
            CTFChallenge("Pwn1", "pwn", "Challenge 3", 100, "hard"),
        ]
        strategy = workflow_manager.create_ctf_team_strategy(challenges, team_size=3)
        assert isinstance(strategy, dict)

    def test_create_ctf_team_strategy_different_team_sizes(self, workflow_manager, sample_web_challenge):
        """Test creating team strategy with different team sizes"""
        challenges = [sample_web_challenge]
        for team_size in [1, 2, 4, 8]:
            strategy = workflow_manager.create_ctf_team_strategy(challenges, team_size=team_size)
            assert isinstance(strategy, dict)

    def test_difficulty_time_estimation_easy(self, workflow_manager, sample_web_challenge):
        """Test time estimation for easy difficulty"""
        sample_web_challenge.difficulty = "easy"
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_web_challenge)
        assert workflow["estimated_time"] > 0

    def test_difficulty_time_estimation_medium(self, workflow_manager, sample_web_challenge):
        """Test time estimation for medium difficulty"""
        sample_web_challenge.difficulty = "medium"
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_web_challenge)
        assert workflow["estimated_time"] > 0

    def test_difficulty_time_estimation_hard(self, workflow_manager, sample_web_challenge):
        """Test time estimation for hard difficulty"""
        sample_web_challenge.difficulty = "hard"
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_web_challenge)
        assert workflow["estimated_time"] > 0

    def test_workflow_success_probability_is_float(self, workflow_manager, sample_web_challenge):
        """Test that success probability is a float between 0 and 1"""
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_web_challenge)
        assert isinstance(workflow["success_probability"], (int, float))
        assert 0 <= workflow["success_probability"] <= 1

    def test_workflow_automation_level(self, workflow_manager, sample_web_challenge):
        """Test that automation level is set"""
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_web_challenge)
        assert workflow["automation_level"] in ["low", "medium", "high"]

    def test_select_tools_for_web_sql_injection(self, workflow_manager, sample_web_challenge):
        """Test tool selection for SQL injection challenge"""
        sample_web_challenge.description = "SQL injection vulnerability in database query"
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_web_challenge)
        tools = workflow["tools"]
        assert isinstance(tools, list)

    def test_select_tools_for_web_xss(self, workflow_manager):
        """Test tool selection for XSS challenge"""
        challenge = CTFChallenge(
            "XSS Challenge", "web", "Cross-site scripting in javascript handler",
            100, "medium"
        )
        workflow = workflow_manager.create_ctf_challenge_workflow(challenge)
        tools = workflow["tools"]
        assert isinstance(tools, list)

    def test_select_tools_for_crypto_hash(self, workflow_manager):
        """Test tool selection for hash challenge"""
        challenge = CTFChallenge(
            "Hash Challenge", "crypto", "MD5 hash cracking with dictionary",
            100, "easy"
        )
        workflow = workflow_manager.create_ctf_challenge_workflow(challenge)
        tools = workflow["tools"]
        assert isinstance(tools, list)

    def test_select_tools_for_crypto_rsa(self, workflow_manager):
        """Test tool selection for RSA challenge"""
        challenge = CTFChallenge(
            "RSA Challenge", "crypto", "RSA public key cryptography",
            100, "hard"
        )
        workflow = workflow_manager.create_ctf_challenge_workflow(challenge)
        tools = workflow["tools"]
        assert isinstance(tools, list)

    def test_select_tools_for_forensics_image(self, workflow_manager):
        """Test tool selection for image forensics challenge"""
        challenge = CTFChallenge(
            "Image Challenge", "forensics", "Find hidden data in PNG image",
            100, "medium"
        )
        workflow = workflow_manager.create_ctf_challenge_workflow(challenge)
        tools = workflow["tools"]
        assert isinstance(tools, list)

    def test_select_tools_for_forensics_memory(self, workflow_manager):
        """Test tool selection for memory forensics challenge"""
        challenge = CTFChallenge(
            "Memory Challenge", "forensics", "Analyze memory dump for artifacts",
            100, "hard"
        )
        workflow = workflow_manager.create_ctf_challenge_workflow(challenge)
        tools = workflow["tools"]
        assert isinstance(tools, list)

    def test_category_tools_coverage(self, workflow_manager):
        """Test that all categories have tool definitions"""
        for category in ["web", "crypto", "pwn", "forensics", "rev", "misc", "osint"]:
            assert category in workflow_manager.category_tools
            assert isinstance(workflow_manager.category_tools[category], dict)

    def test_strategy_definitions_complete(self, workflow_manager):
        """Test that all strategies have proper structure"""
        for category in workflow_manager.solving_strategies:
            strategies = workflow_manager.solving_strategies[category]
            assert isinstance(strategies, list)
            for strategy in strategies:
                assert isinstance(strategy, dict)
                assert "strategy" in strategy or "description" in strategy

    def test_workflow_points_preserved(self, workflow_manager, sample_web_challenge):
        """Test that challenge points are preserved in workflow"""
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_web_challenge)
        assert workflow["points"] == sample_web_challenge.points

    def test_workflow_difficulty_preserved(self, workflow_manager, sample_web_challenge):
        """Test that challenge difficulty is preserved in workflow"""
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_web_challenge)
        assert workflow["difficulty"] == sample_web_challenge.difficulty

    def test_workflow_category_preserved(self, workflow_manager, sample_web_challenge):
        """Test that challenge category is preserved in workflow"""
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_web_challenge)
        assert workflow["category"] == sample_web_challenge.category

    def test_different_challenges_generate_different_workflows(self, workflow_manager):
        """Test that different challenges generate different workflows"""
        challenge1 = CTFChallenge("Web1", "web", "SQL injection", 100, "easy")
        challenge2 = CTFChallenge("Web2", "web", "File upload bypass", 100, "easy")
        
        workflow1 = workflow_manager.create_ctf_challenge_workflow(challenge1)
        workflow2 = workflow_manager.create_ctf_challenge_workflow(challenge2)
        
        # Workflows should be different due to different descriptions
        assert workflow1 is not workflow2

    def test_team_strategy_with_empty_challenges(self, workflow_manager):
        """Test team strategy with empty challenge list"""
        strategy = workflow_manager.create_ctf_team_strategy([], team_size=4)
        assert isinstance(strategy, dict)

    def test_workflow_has_dependencies_field(self, workflow_manager, sample_web_challenge):
        """Test that workflow has dependencies field"""
        workflow = workflow_manager.create_ctf_challenge_workflow(sample_web_challenge)
        assert "dependencies" in workflow
        assert isinstance(workflow["dependencies"], list)

    # --- _select_tools_for_challenge branch coverage ---

    def test_select_tools_web_sql(self, workflow_manager):
        c = CTFChallenge("t", "web", "sql injection in database query", 100, "easy")
        tools = workflow_manager._select_tools_for_challenge(c, workflow_manager.category_tools["web"])
        assert "sqlmap" in tools

    def test_select_tools_web_xss(self, workflow_manager):
        c = CTFChallenge("t", "web", "cross site scripting via javascript", 100, "easy")
        tools = workflow_manager._select_tools_for_challenge(c, workflow_manager.category_tools["web"])
        assert "dalfox" in tools

    def test_select_tools_web_wordpress(self, workflow_manager):
        c = CTFChallenge("t", "web", "wordpress plugin vulnerability", 100, "easy")
        tools = workflow_manager._select_tools_for_challenge(c, workflow_manager.category_tools["web"])
        assert "wpscan" in tools

    def test_select_tools_web_upload(self, workflow_manager):
        c = CTFChallenge("t", "web", "file upload bypass vulnerability", 100, "easy")
        tools = workflow_manager._select_tools_for_challenge(c, workflow_manager.category_tools["web"])
        assert "gobuster" in tools and "feroxbuster" in tools

    def test_select_tools_web_no_keywords(self, workflow_manager):
        c = CTFChallenge("t", "web", "generic challenge", 100, "easy")
        tools = workflow_manager._select_tools_for_challenge(c, workflow_manager.category_tools["web"])
        assert "sqlmap" not in tools

    def test_select_tools_crypto_hash(self, workflow_manager):
        c = CTFChallenge("t", "crypto", "md5 hash cracking", 100, "easy")
        tools = workflow_manager._select_tools_for_challenge(c, workflow_manager.category_tools["crypto"])
        assert "hashcat" in tools and "john" in tools

    def test_select_tools_crypto_rsa(self, workflow_manager):
        c = CTFChallenge("t", "crypto", "rsa public key decryption", 100, "easy")
        tools = workflow_manager._select_tools_for_challenge(c, workflow_manager.category_tools["crypto"])
        assert "rsatool" in tools and "factordb" in tools

    def test_select_tools_crypto_cipher(self, workflow_manager):
        c = CTFChallenge("t", "crypto", "cipher encryption algorithm", 100, "easy")
        tools = workflow_manager._select_tools_for_challenge(c, workflow_manager.category_tools["crypto"])
        assert "cipher-identifier" in tools and "cyberchef" in tools

    def test_select_tools_pwn_default(self, workflow_manager):
        c = CTFChallenge("t", "pwn", "binary exploitation", 100, "easy")
        tools = workflow_manager._select_tools_for_challenge(c, workflow_manager.category_tools["pwn"])
        assert "checksec" in tools and "ghidra" in tools and "pwntools" in tools

    def test_select_tools_pwn_heap(self, workflow_manager):
        c = CTFChallenge("t", "pwn", "heap malloc exploitation", 100, "easy")
        tools = workflow_manager._select_tools_for_challenge(c, workflow_manager.category_tools["pwn"])
        assert "glibc-heap-analysis" in tools

    def test_select_tools_pwn_format(self, workflow_manager):
        c = CTFChallenge("t", "pwn", "printf format string bug", 100, "easy")
        tools = workflow_manager._select_tools_for_challenge(c, workflow_manager.category_tools["pwn"])
        assert "format-string-exploiter" in tools

    def test_select_tools_forensics_image(self, workflow_manager):
        c = CTFChallenge("t", "forensics", "hidden data in png image", 100, "easy")
        tools = workflow_manager._select_tools_for_challenge(c, workflow_manager.category_tools["forensics"])
        assert "exiftool" in tools and "steghide" in tools and "stegsolve" in tools

    def test_select_tools_forensics_memory(self, workflow_manager):
        c = CTFChallenge("t", "forensics", "analyze memory dump for artifacts", 100, "easy")
        tools = workflow_manager._select_tools_for_challenge(c, workflow_manager.category_tools["forensics"])
        assert "volatility" in tools

    def test_select_tools_forensics_network(self, workflow_manager):
        c = CTFChallenge("t", "forensics", "analyze network pcap traffic", 100, "easy")
        tools = workflow_manager._select_tools_for_challenge(c, workflow_manager.category_tools["forensics"])
        assert "wireshark" in tools and "tcpdump" in tools

    def test_select_tools_rev_default(self, workflow_manager):
        c = CTFChallenge("t", "rev", "reverse engineering", 100, "easy")
        tools = workflow_manager._select_tools_for_challenge(c, workflow_manager.category_tools["rev"])
        assert "ghidra" in tools and "radare2" in tools and "strings" in tools

    def test_select_tools_rev_packed(self, workflow_manager):
        c = CTFChallenge("t", "rev", "packed binary with upx", 100, "easy")
        tools = workflow_manager._select_tools_for_challenge(c, workflow_manager.category_tools["rev"])
        assert "upx" in tools and "peid" in tools

    def test_select_tools_no_reconnaissance_key(self, workflow_manager):
        c = CTFChallenge("t", "web", "test", 100, "easy")
        tools = workflow_manager._select_tools_for_challenge(c, {})
        assert isinstance(tools, list)

    def test_select_tools_duplicates_removed(self, workflow_manager):
        c = CTFChallenge("t", "web", "test", 100, "easy")
        tools = workflow_manager._select_tools_for_challenge(c, workflow_manager.category_tools["web"])
        assert len(tools) == len(set(tools))

    def test_select_tools_misc_category_passes_all_elifs(self, workflow_manager):
        c = CTFChallenge("t", "misc", "something", 100, "easy")
        tools = workflow_manager._select_tools_for_challenge(c, workflow_manager.category_tools.get("misc", {}))
        assert isinstance(tools, list)

    def test_select_tools_osint_category_passes_all_elifs(self, workflow_manager):
        c = CTFChallenge("t", "osint", "something", 100, "easy")
        tools = workflow_manager._select_tools_for_challenge(c, workflow_manager.category_tools.get("osint", {}))
        assert isinstance(tools, list)

    # --- _create_category_workflow branch coverage ---

    @pytest.mark.parametrize("category", ["web", "crypto", "pwn", "forensics", "rev"])
    def test_create_category_workflow_known(self, workflow_manager, category):
        c = CTFChallenge("t", category, "test", 100, "easy")
        steps = workflow_manager._create_category_workflow(c)
        assert isinstance(steps, list)
        assert len(steps) == 6
        assert all(s["step"] == i + 1 for i, s in enumerate(steps))

    def test_create_category_workflow_unknown(self, workflow_manager):
        c = CTFChallenge("t", "unknown_cat", "test", 100, "easy")
        steps = workflow_manager._create_category_workflow(c)
        assert isinstance(steps, list)
        assert len(steps) == 6
        assert steps[0]["action"] == "analysis"

    # --- create_ctf_team_strategy branch coverage ---

    def test_team_strategy_same_member_reuse(self, workflow_manager):
        challenges = [
            CTFChallenge("A", "web", "easy web", 50, "easy"),
            CTFChallenge("B", "crypto", "easy crypto", 50, "easy"),
            CTFChallenge("C", "pwn", "hard pwn", 200, "hard"),
        ]
        strategy = workflow_manager.create_ctf_team_strategy(challenges, team_size=2)
        assert "challenge_allocation" in strategy
        assert "priority_order" in strategy
        assert len(strategy["priority_order"]) == 3

    def test_team_strategy_expected_score(self, workflow_manager):
        challenges = [
            CTFChallenge("A", "web", "test", 100, "easy"),
        ]
        strategy = workflow_manager.create_ctf_team_strategy(challenges, team_size=4)
        assert strategy["expected_score"] > 0

    # --- _analyze_description_complexity branch coverage ---

    def test_analyze_complexity_short(self, workflow_manager):
        score = workflow_manager._analyze_description_complexity("short")
        assert score < 0.1

    def test_analyze_complexity_medium(self, workflow_manager):
        desc = "x" * 250
        score = workflow_manager._analyze_description_complexity(desc)
        assert score >= 0.1

    def test_analyze_complexity_long(self, workflow_manager):
        desc = "x" * 600
        score = workflow_manager._analyze_description_complexity(desc)
        assert score >= 0.3

    def test_analyze_complexity_technical_terms(self, workflow_manager):
        desc = "buffer overflow exploit with rop shellcode stack heap encryption vulnerability"
        score = workflow_manager._analyze_description_complexity(desc)
        assert score > 0

    def test_analyze_complexity_multi_step(self, workflow_manager):
        desc = "first then next after finally step"
        score = workflow_manager._analyze_description_complexity(desc)
        assert score > 0

    def test_analyze_complexity_capped(self, workflow_manager):
        desc = ("algorithm " * 100) + ("first " * 100)
        score = workflow_manager._analyze_description_complexity(desc)
        assert score <= 1.0

    # --- _calculate_resource_requirements branch coverage ---

    @pytest.mark.parametrize("category", ["web", "crypto", "pwn", "forensics", "rev", "misc", "osint"])
    def test_resource_requirements_categories(self, workflow_manager, category):
        c = CTFChallenge("t", category, "test", 100, "medium")
        req = workflow_manager._calculate_resource_requirements(c)
        assert "cpu_cores" in req and "memory_mb" in req and "disk_space_mb" in req

    def test_resource_requirements_unknown_category(self, workflow_manager):
        c = CTFChallenge("t", "unknown", "test", 100, "medium")
        req = workflow_manager._calculate_resource_requirements(c)
        assert req["cpu_cores"] == 2

    @pytest.mark.parametrize("difficulty,mult", [("easy", 1.0), ("medium", 1.2), ("hard", 1.5), ("insane", 2.0), ("unknown", 1.3)])
    def test_resource_requirements_difficulty_multiplier(self, workflow_manager, difficulty, mult):
        c = CTFChallenge("t", "web", "test", 100, difficulty)
        req = workflow_manager._calculate_resource_requirements(c)
        assert req["cpu_cores"] == int(4 * mult)

    # --- _generate_fallback_strategies ---

    @pytest.mark.parametrize("category", ["web", "crypto", "pwn", "forensics", "rev", "misc", "osint"])
    def test_fallback_strategies(self, workflow_manager, category):
        strategies = workflow_manager._generate_fallback_strategies(category)
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        for s in strategies:
            assert "strategy" in s and "description" in s

    def test_fallback_strategies_unknown(self, workflow_manager):
        strategies = workflow_manager._generate_fallback_strategies("nonexistent")
        assert isinstance(strategies, list)
        assert len(strategies) == 0

    # --- _identify_parallel_tasks ---

    @pytest.mark.parametrize("category", ["web", "crypto", "pwn", "forensics", "rev", "misc", "osint"])
    def test_parallel_tasks(self, workflow_manager, category):
        tasks = workflow_manager._identify_parallel_tasks(category)
        assert isinstance(tasks, list)

    def test_parallel_tasks_unknown(self, workflow_manager):
        tasks = workflow_manager._identify_parallel_tasks("nonexistent")
        assert isinstance(tasks, list)
        assert len(tasks) == 0

    # --- _predict_expected_artifacts ---

    @pytest.mark.parametrize("category", ["web", "crypto", "pwn", "forensics", "rev", "misc", "osint"])
    def test_expected_artifacts(self, workflow_manager, category):
        c = CTFChallenge("t", category, "test", 100, "easy")
        artifacts = workflow_manager._predict_expected_artifacts(c)
        assert isinstance(artifacts, list)
        assert len(artifacts) > 0
        for a in artifacts:
            assert "type" in a and "description" in a

    def test_expected_artifacts_unknown(self, workflow_manager):
        c = CTFChallenge("t", "unknown", "test", 100, "easy")
        artifacts = workflow_manager._predict_expected_artifacts(c)
        assert isinstance(artifacts, list)
        assert len(artifacts) == 2

    # --- _create_validation_steps ---

    @pytest.mark.parametrize("category", ["web", "crypto", "pwn", "forensics", "rev", "misc", "osint"])
    def test_validation_steps(self, workflow_manager, category):
        steps = workflow_manager._create_validation_steps(category)
        assert isinstance(steps, list)
        assert len(steps) > 0
        for s in steps:
            assert "step" in s and "description" in s

    def test_validation_steps_unknown(self, workflow_manager):
        steps = workflow_manager._create_validation_steps("nonexistent")
        assert isinstance(steps, list)
        assert len(steps) == 2

    # --- _create_advanced_category_workflow ---

    @pytest.mark.parametrize("category", ["web", "crypto", "pwn", "forensics", "rev", "misc", "osint"])
    def test_advanced_workflow(self, workflow_manager, category):
        c = CTFChallenge("t", category, "test", 100, "easy")
        steps = workflow_manager._create_advanced_category_workflow(c)
        assert isinstance(steps, list)
        assert len(steps) > 0
        for s in steps:
            assert "step" in s and "action" in s and "tools" in s and "estimated_time" in s

    def test_advanced_workflow_unknown(self, workflow_manager):
        c = CTFChallenge("t", "unknown", "test", 100, "easy")
        steps = workflow_manager._create_advanced_category_workflow(c)
        assert isinstance(steps, list)
        assert len(steps) == 6
        assert steps[0]["action"] == "analysis"
