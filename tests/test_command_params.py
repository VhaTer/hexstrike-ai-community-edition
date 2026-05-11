from server_core.command_params import rebuild_command_with_params


class TestRebuildCommandWithParams:
    def test_no_matching_params(self):
        r = rebuild_command_with_params("nmap", "nmap -sV target", {})
        assert r == "nmap -sV target"

    def test_nmap_timeout(self):
        r = rebuild_command_with_params("nmap", "nmap target", {"timeout": 30})
        assert "--timeout 30" in r

    def test_gobuster_threads(self):
        r = rebuild_command_with_params("gobuster", "gobuster dir", {"threads": 50})
        assert "-t 50" in r

    def test_feroxbuster_delay(self):
        r = rebuild_command_with_params("feroxbuster", "feroxbuster", {"delay": 2})
        assert "--delay 2" in r

    def test_nmap_timing(self):
        r = rebuild_command_with_params("nmap", "nmap target", {"timing": "-T4"})
        assert "-T4" in r

    def test_nuclei_concurrency(self):
        r = rebuild_command_with_params("nuclei", "nuclei", {"concurrency": 10})
        assert "-c 10" in r

    def test_nuclei_rate_limit(self):
        r = rebuild_command_with_params("nuclei", "nuclei", {"rate-limit": 100})
        assert "-rl 100" in r

    def test_multiple_params(self):
        r = rebuild_command_with_params("nmap", "nmap target", {"timeout": 60, "timing": "-T4"})
        assert "--timeout 60" in r
        assert "-T4" in r

    def test_tool_mismatch(self):
        """Param for wrong tool should be ignored."""
        r = rebuild_command_with_params("nmap", "nmap target", {"threads": 50})
        assert r == "nmap target"

    def test_ffuf_threads(self):
        r = rebuild_command_with_params("ffuf", "ffuf target", {"threads": 10})
        assert "-t 10" in r
