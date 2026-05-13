from server_core.workflows.ctf.coordinator import CTFTeamCoordinator
from server_core.workflows.ctf.CTFChallenge import CTFChallenge


class TestCTFTeamCoordinator:
    def test_optimize_team_strategy(self):
        coord = CTFTeamCoordinator()
        challenges = [
            CTFChallenge("Web1", "web", "SQL injection", 100, "medium"),
        ]
        team_skills = {"alice": ["web"], "bob": ["pwn"]}
        strategy = coord.optimize_team_strategy(challenges, team_skills)
        assert "assignments" in strategy
        assert "priority_queue" in strategy

    def test_collaboration_opportunities_hard_challenge(self):
        coord = CTFTeamCoordinator()
        challenges = [
            CTFChallenge("HardWeb", "web", "Complex exploit", 500, "hard"),
        ]
        team_skills = {"alice": ["web"], "bob": ["web", "crypto"]}
        strategy = coord.optimize_team_strategy(challenges, team_skills)
        assert len(strategy["collaboration_opportunities"]) >= 1
        opp = strategy["collaboration_opportunities"][0]
        assert opp["challenge"] == "HardWeb"
        assert len(opp["recommended_team"]) >= 2

    def test_estimate_solve_time_with_skill(self):
        coord = CTFTeamCoordinator()
        c = CTFChallenge("Test", "web", "desc", 100, "medium")
        time = coord._estimate_solve_time(c, {"web": True})
        assert time == 2520

    def test_estimate_solve_time_without_skill(self):
        coord = CTFTeamCoordinator()
        c = CTFChallenge("Test", "web", "desc", 100, "medium")
        time = coord._estimate_solve_time(c, {"web": False})
        assert time == 3600
