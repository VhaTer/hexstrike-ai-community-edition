from server_core.operation_types import determine_operation_type


class TestDetermineOperationType:
    def test_known_tool(self):
        assert determine_operation_type("nmap") == "network_discovery"
        assert determine_operation_type("gobuster") == "web_discovery"
        assert determine_operation_type("nuclei") == "vulnerability_scanning"

    def test_unknown_tool(self):
        assert determine_operation_type("nonexistent_tool") == "unknown_operation"
