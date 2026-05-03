from server_core.intelligence.intelligent_decision_engine import IntelligentDecisionEngine
from shared.target_profile import TargetProfile
from shared.target_types import TargetType


def test_analyze_target_basic_web():
    ide = IntelligentDecisionEngine()
    profile = ide.analyze_target("http://example.com")
    assert isinstance(profile, TargetProfile)
    assert profile.target_type == TargetType.WEB_APPLICATION


def test_create_attack_chain_contains_steps():
    ide = IntelligentDecisionEngine()
    profile = TargetProfile(target="http://example.com")
    profile.target_type = TargetType.WEB_APPLICATION
    profile.technologies = []
    profile.confidence_score = 0.8
    chain = ide.create_attack_chain(profile, objective="quick")
    assert chain.steps
    # success probability should be calculated
    assert hasattr(chain, "success_probability")
