import pytest
from server_core.workflows.ctf.CTFChallenge import CTFChallenge
from server_core.workflows.ctf.coordinator import CTFTeamCoordinator


@pytest.fixture
def challenges():
    return [
        CTFChallenge(description="test challenge", name="web-easy", category="web", points=100, difficulty="easy"),
        CTFChallenge(description="test challenge", name="web-hard", category="web", points=500, difficulty="hard"),
        CTFChallenge(description="test challenge", name="crypto-med", category="crypto", points=300, difficulty="medium"),
        CTFChallenge(description="test challenge", name="pwn-insane", category="pwn", points=1000, difficulty="insane"),
    ]


@pytest.fixture
def team_skills():
    return {
        "alice": ["web", "crypto"],
        "bob": ["pwn", "binary"],
        "charlie": ["forensics", "osint"],
    }


@pytest.fixture
def coordinator():
    return CTFTeamCoordinator()


class TestCTFTeamCoordinator:
    def test_optimize_strategy_returns_correct_structure(self, coordinator, challenges, team_skills):
        strategy = coordinator.optimize_team_strategy(challenges, team_skills)
        assert "assignments" in strategy
        assert "priority_queue" in strategy
        assert "collaboration_opportunities" in strategy
        assert strategy["estimated_total_score"] >= 0

    def test_assignments_by_skill_match(self, coordinator, challenges, team_skills):
        strategy = coordinator.optimize_team_strategy(challenges, team_skills)
        assignments = strategy["assignments"]
        # Alice has web skill, should get web challenges
        alice_challenges = [a["challenge"].name for a in assignments.get("alice", [])]
        # Bob has pwn, should get pwn-insane
        bob_challenges = [a["challenge"].name for a in assignments.get("bob", [])]

    def test_priority_queue_sorted_by_score(self, coordinator, challenges, team_skills):
        strategy = coordinator.optimize_team_strategy(challenges, team_skills)
        queue = strategy["priority_queue"]
        for i in range(len(queue) - 1):
            assert queue[i]["priority"] >= queue[i + 1]["priority"]

    def test_collaboration_on_hard_challenges(self, coordinator, challenges, team_skills):
        opportunities = coordinator._identify_collaboration_opportunities(challenges, team_skills)
        for opp in opportunities:
            assert opp["challenge"] in ["web-hard", "pwn-insane"]
            assert len(opp["recommended_team"]) >= 2

    def test_no_collaboration_on_easy(self, coordinator):
        easy_challenges = [CTFChallenge(description="test challenge", name="easy-web", category="web", points=50, difficulty="easy")]
        skills = {"alice": ["web"], "bob": ["crypto"]}
        opportunities = coordinator._identify_collaboration_opportunities(easy_challenges, skills)
        assert len(opportunities) == 0

    def test_estimate_solve_time_with_skills(self, coordinator):
        challenge = CTFChallenge(description="test challenge", name="test", category="web", points=100, difficulty="medium")
        with_skill = {"web": True}
        without_skill = {"web": False}
        time_with = coordinator._estimate_solve_time(challenge, with_skill)
        time_without = coordinator._estimate_solve_time(challenge, without_skill)
        assert time_with < time_without
        assert time_with == int(3600 * 0.7)

    def test_estimate_solve_time_all_difficulties(self, coordinator):
        skills = {"web": False}
        for diff, expected_base in [("easy", 1800), ("medium", 3600), ("hard", 7200),
                                     ("insane", 14400), ("unknown", 5400)]:
            c = CTFChallenge(description="test challenge", name="test", category="web", points=100, difficulty=diff)
            t = coordinator._estimate_solve_time(c, skills)
            assert t == expected_base

    def test_assign_challenges_optimally(self, coordinator, challenges, team_skills):
        member_challenge_scores = {}
        for member in team_skills.keys():
            member_challenge_scores[member] = []
            for challenge in challenges:
                member_challenge_scores[member].append({
                    "challenge": challenge,
                    "score": 100,
                    "estimated_time": 3600,
                })
        assignments = coordinator._assign_challenges_optimally(member_challenge_scores)
        assert len(assignments) == len(team_skills)
        # Greedy: one challenge per member per round
        all_assigned = sum(len(v) for v in assignments.values())
        assert all_assigned == len(team_skills)

    def test_empty_challenges(self, coordinator, team_skills):
        strategy = coordinator.optimize_team_strategy([], team_skills)
        assert strategy["assignments"] == {"alice": [], "bob": [], "charlie": []}
        assert strategy["priority_queue"] == []

    def test_empty_team(self, coordinator, challenges):
        strategy = coordinator.optimize_team_strategy(challenges, {})
        assert strategy["assignments"] == {}
        assert strategy["priority_queue"] == []

    def test_single_member_team(self, coordinator):
        challenge = CTFChallenge(description="test challenge", name="solo", category="web", points=100, difficulty="easy")
        strategy = coordinator.optimize_team_strategy([challenge], {"alice": ["web"]})
        assert len(strategy["assignments"]["alice"]) == 1

    def test_resource_sharing_in_strategy(self, coordinator, challenges, team_skills):
        strategy = coordinator.optimize_team_strategy(challenges, team_skills)
        assert "resource_sharing" in strategy
        assert "time_allocation" in strategy
