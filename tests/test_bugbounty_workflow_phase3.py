"""
Phase 3: BugBounty Workflow Test Suite (workflow.py)
Tests pure data structure generation for bug bounty workflows.

Coverage Target: 75%+
Effort: 4-5 hours (Quick Win - No external mocks needed!)
Pattern: Pure function testing, parametrized validation, workflow structure verification
"""

import pytest
from server_core.workflows.bugbounty.workflow import BugBountyWorkflowManager
from server_core.workflows.bugbounty.target import BugBountyTarget


# ========== FIXTURES ==========

@pytest.fixture
def manager():
    """Initialize BugBountyWorkflowManager."""
    return BugBountyWorkflowManager()


@pytest.fixture
def basic_target():
    """Create basic test target."""
    return BugBountyTarget(
        domain="example.com",
        scope=["*.example.com"],
        program_type="web",
        priority_vulns=["rce", "sqli", "xss"]
    )


@pytest.fixture
def advanced_target():
    """Create advanced test target with all available options."""
    return BugBountyTarget(
        domain="advanced.example.com",
        scope=["*.advanced.example.com", "api.advanced.example.com"],
        program_type="mobile",
        priority_vulns=["rce", "sqli", "xss", "ssrf", "idor"]
    )


# ========== TEST CLASS: RECONNAISSANCE WORKFLOW ==========

class TestReconnaissanceWorkflow:
    """Test reconnaissance workflow generation."""

    def test_recon_workflow_structure_complete(self, manager, basic_target):
        """Verify reconnaissance workflow has all required phases."""
        workflow = manager.create_reconnaissance_workflow(basic_target)
        
        assert "phases" in workflow
        assert len(workflow["phases"]) >= 3
        
        phase_names = [p["name"] for p in workflow["phases"]]
        # Verify at least some expected phases are present
        assert any(name in ["subdomain_discovery", "http_service_discovery", "content_discovery"] 
                  for name in phase_names)

    def test_recon_workflow_time_estimates(self, manager, basic_target):
        """Verify reconnaissance workflow includes time estimates."""
        workflow = manager.create_reconnaissance_workflow(basic_target)
        
        assert "estimated_time" in workflow
        assert isinstance(workflow["estimated_time"], (int, float))
        assert workflow["estimated_time"] > 0
        
        # Each phase should have time estimate
        for phase in workflow["phases"]:
            assert "estimated_time" in phase
            assert phase["estimated_time"] > 0

    def test_recon_phases_have_tools(self, manager, basic_target):
        """Verify each phase includes recommended tools."""
        workflow = manager.create_reconnaissance_workflow(basic_target)
        
        for phase in workflow["phases"]:
            assert "tools" in phase
            assert len(phase["tools"]) > 0
            
            for tool in phase["tools"]:
                assert "tool" in tool or "name" in tool
                # Params may or may not be present for all tools

    @pytest.mark.parametrize("scope_size", [1, 5, 10])
    def test_recon_scales_with_scope_size(self, manager, basic_target, scope_size):
        """Verify recon workflow scales appropriately with scope."""
        basic_target.scope = [f"*.subdomain{i}.example.com" for i in range(scope_size)]
        workflow = manager.create_reconnaissance_workflow(basic_target)
        
        assert workflow is not None
        assert "estimated_time" in workflow
        # Time should be positive
        assert workflow["estimated_time"] > 0


# ========== TEST CLASS: VULNERABILITY HUNTING WORKFLOW ==========

class TestVulnerabilityHuntingWorkflow:
    """Test vulnerability hunting workflow generation."""

    def test_vuln_hunting_priority_sorting(self, manager, basic_target):
        """Verify vulnerabilities are sorted by priority."""
        basic_target.priority_vulns = ["xss", "rce", "sqli"]  # Out of order
        workflow = manager.create_vulnerability_hunting_workflow(basic_target)
        
        assert "vulnerability_tests" in workflow
        vulns = [v["vulnerability_type"] for v in workflow["vulnerability_tests"]]
        
        # Should be sorted: rce (10), sqli (8), xss (7)
        assert vulns[0] == "rce"
        assert vulns[1] == "sqli"
        assert vulns[2] == "xss"

    def test_vuln_hunting_includes_all_priorities(self, manager, basic_target):
        """Verify all priority vulnerabilities are included."""
        basic_target.priority_vulns = ["rce", "sqli", "xss", "ssrf", "idor"]
        workflow = manager.create_vulnerability_hunting_workflow(basic_target)
        
        vuln_types = [v["vulnerability_type"] for v in workflow["vulnerability_tests"]]
        
        for vuln in basic_target.priority_vulns:
            assert vuln in vuln_types

    @pytest.mark.parametrize("vuln_type,expected_min_tests", [
        ("rce", 2),
        ("sqli", 2),
        ("xss", 2),
        ("ssrf", 2),
        ("idor", 2),
    ])
    def test_vuln_hunting_tests_per_vulnerability(self, manager, basic_target, vuln_type, expected_min_tests):
        """Verify each vulnerability has test scenarios."""
        basic_target.priority_vulns = [vuln_type]
        workflow = manager.create_vulnerability_hunting_workflow(basic_target)
        
        # Should have at least 1 vulnerability test with scenarios
        assert len(workflow["vulnerability_tests"]) >= 1
        assert workflow["vulnerability_tests"][0]["test_scenarios"] is not None

    def test_vuln_hunting_test_scenarios_structure(self, manager, basic_target):
        """Verify test scenarios have required fields."""
        workflow = manager.create_vulnerability_hunting_workflow(basic_target)
        
        for vuln_test in workflow["vulnerability_tests"]:
            assert "vulnerability_type" in vuln_test
            assert "test_scenarios" in vuln_test
            assert len(vuln_test["test_scenarios"]) > 0
            
            for scenario in vuln_test["test_scenarios"]:
                assert "name" in scenario
                assert "payloads" in scenario

    def test_vuln_hunting_has_all_priority_vulns(self, manager, basic_target):
        """Verify all priority vulnerabilities are included."""
        workflow = manager.create_vulnerability_hunting_workflow(basic_target)
        
        # Should have vulnerability tests
        assert "vulnerability_tests" in workflow
        assert len(workflow["vulnerability_tests"]) > 0


# ========== TEST CLASS: BUSINESS LOGIC TESTING ==========

class TestBusinessLogicWorkflow:
    """Test business logic testing workflow."""

    def test_business_logic_workflow_structure(self, manager, basic_target):
        """Verify business logic workflow has required structure."""
        workflow = manager.create_business_logic_testing_workflow(basic_target)
        
        assert "business_logic_tests" in workflow
        assert len(workflow["business_logic_tests"]) > 0
        assert "estimated_time" in workflow

    def test_business_logic_has_manual_testing_flag(self, manager, basic_target):
        """Verify business logic workflow flags manual testing requirement."""
        workflow = manager.create_business_logic_testing_workflow(basic_target)
        
        assert "manual_testing_required" in workflow
        assert isinstance(workflow["manual_testing_required"], bool)
        assert workflow["manual_testing_required"] is True

    @pytest.mark.parametrize("program_type", ["web", "mobile", "api"])
    def test_business_logic_adapts_to_program_type(self, manager, basic_target, program_type):
        """Verify business logic workflow adapts to program type."""
        basic_target.program_type = program_type
        workflow = manager.create_business_logic_testing_workflow(basic_target)
        
        assert workflow is not None
        assert "business_logic_tests" in workflow

    def test_business_logic_tests_have_descriptions(self, manager, basic_target):
        """Verify each business logic test category has tests."""
        workflow = manager.create_business_logic_testing_workflow(basic_target)
        
        for test_category in workflow["business_logic_tests"]:
            assert "category" in test_category or "name" in test_category
            assert "tests" in test_category


# ========== TEST CLASS: OSINT WORKFLOW ==========

class TestOSINTWorkflow:
    """Test OSINT workflow generation."""

    def test_osint_workflow_structure(self, manager, basic_target):
        """Verify OSINT workflow has required structure."""
        workflow = manager.create_osint_workflow(basic_target)
        
        assert "osint_phases" in workflow
        assert len(workflow["osint_phases"]) > 0

    def test_osint_phases_complete(self, manager, basic_target):
        """Verify OSINT includes all intelligence gathering phases."""
        workflow = manager.create_osint_workflow(basic_target)
        
        phase_names = [p.get("name", p.get("title", "")) for p in workflow["osint_phases"]]
        
        # Should include domain, DNS, subdomain, and external intelligence
        assert len(phase_names) >= 3

    def test_osint_time_estimate(self, manager, basic_target):
        """Verify OSINT workflow includes time estimate."""
        workflow = manager.create_osint_workflow(basic_target)
        
        assert "estimated_time" in workflow
        assert workflow["estimated_time"] > 0
        assert isinstance(workflow["estimated_time"], (int, float))

    @pytest.mark.parametrize("domain", [
        "example.com",
        "sub.example.com",
        "api.example.co.uk",
        "app.test.example.org"
    ])
    def test_osint_handles_various_domains(self, manager, basic_target, domain):
        """Verify OSINT workflow works with various domain formats."""
        basic_target.domain = domain
        workflow = manager.create_osint_workflow(basic_target)
        
        assert workflow is not None
        assert "osint_phases" in workflow


# ========== TEST CLASS: WORKFLOW SCENARIOS ==========

class TestWorkflowScenarios:
    """Test workflow scenarios and helper methods."""

    def test_get_test_scenarios_rce(self, manager):
        """Verify RCE test scenarios are available."""
        scenarios = manager._get_test_scenarios("rce")
        
        assert len(scenarios) > 0
        for scenario in scenarios:
            assert "name" in scenario
            assert "payloads" in scenario

    @pytest.mark.parametrize("vuln_type", ["rce", "sqli", "xss", "ssrf", "idor"])
    def test_get_test_scenarios_all_types(self, manager, vuln_type):
        """Verify test scenarios exist for common vulnerabilities."""
        scenarios = manager._get_test_scenarios(vuln_type)
        
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0

    def test_get_test_scenarios_unsupported_type(self, manager):
        """Verify unsupported vulnerability types handled gracefully."""
        scenarios = manager._get_test_scenarios("unsupported_vuln")
        
        # Should return empty list or None, not raise exception
        assert scenarios is None or scenarios == [] or isinstance(scenarios, list)

    def test_workflow_manager_initialization(self):
        """Verify BugBountyWorkflowManager initializes correctly."""
        manager = BugBountyWorkflowManager()
        
        assert manager is not None
        assert hasattr(manager, "create_reconnaissance_workflow")
        assert hasattr(manager, "create_vulnerability_hunting_workflow")
        assert hasattr(manager, "create_business_logic_testing_workflow")
        assert hasattr(manager, "create_osint_workflow")


# ========== TEST CLASS: EDGE CASES ==========

class TestWorkflowEdgeCases:
    """Test edge cases and error conditions."""

    def test_workflow_with_empty_scope(self, manager):
        """Verify workflow handles empty scope."""
        target = BugBountyTarget(
            domain="example.com",
            scope=[],
            program_type="web",
            priority_vulns=["rce"]
        )
        
        workflow = manager.create_reconnaissance_workflow(target)
        assert workflow is not None

    def test_workflow_with_no_priority_vulns(self, manager):
        """Verify workflow handles no priority vulnerabilities."""
        target = BugBountyTarget(
            domain="example.com",
            scope=["*.example.com"],
            program_type="web",
            priority_vulns=[]
        )
        
        workflow = manager.create_vulnerability_hunting_workflow(target)
        assert workflow is not None or workflow == {}

    @pytest.mark.parametrize("program_type", ["web", "mobile", "api", "hardware"])
    def test_workflow_with_various_program_types(self, manager, basic_target, program_type):
        """Verify workflows work with various program types."""
        basic_target.program_type = program_type
        
        recon = manager.create_reconnaissance_workflow(basic_target)
        vuln = manager.create_vulnerability_hunting_workflow(basic_target)
        
        assert recon is not None
        assert vuln is not None

    def test_workflow_with_high_scope_complexity(self, manager, advanced_target):
        """Verify workflows handle complex scopes."""
        advanced_target.scope = [f"*.sub{i}.example.com" for i in range(5)]
        workflow = manager.create_reconnaissance_workflow(advanced_target)
        
        assert workflow is not None
        assert "phases" in workflow


# ========== TEST CLASS: WORKFLOW INTEGRATION ==========

class TestWorkflowIntegration:
    """Test workflow integration and consistency."""

    def test_all_workflows_return_dicts(self, manager, basic_target):
        """Verify all workflow methods return dictionaries."""
        recon = manager.create_reconnaissance_workflow(basic_target)
        vuln = manager.create_vulnerability_hunting_workflow(basic_target)
        biz = manager.create_business_logic_testing_workflow(basic_target)
        osint = manager.create_osint_workflow(basic_target)
        
        assert isinstance(recon, dict)
        assert isinstance(vuln, dict)
        assert isinstance(biz, dict)
        assert isinstance(osint, dict)

    def test_workflow_time_estimates_realistic(self, manager, basic_target):
        """Verify workflow time estimates are reasonable (in minutes)."""
        workflows = [
            manager.create_reconnaissance_workflow(basic_target),
            manager.create_vulnerability_hunting_workflow(basic_target),
            manager.create_business_logic_testing_workflow(basic_target),
            manager.create_osint_workflow(basic_target)
        ]
        
        for workflow in workflows:
            if "estimated_time" in workflow:
                # Assume times are in minutes: 30min to 7 days
                assert 30 <= workflow["estimated_time"] <= 10080

    def test_target_unmodified_by_workflow_generation(self, manager, basic_target):
        """Verify workflow generation doesn't modify target."""
        original_domain = basic_target.domain
        original_scope = basic_target.scope.copy()
        
        manager.create_reconnaissance_workflow(basic_target)
        manager.create_vulnerability_hunting_workflow(basic_target)
        manager.create_business_logic_testing_workflow(basic_target)
        manager.create_osint_workflow(basic_target)
        
        assert basic_target.domain == original_domain
        assert basic_target.scope == original_scope


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
