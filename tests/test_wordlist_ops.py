"""
Test suite for wordlist operations module.
Tests: mcp_tools/ops/wordlist.py
Coverage target: 50%+
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from mcp_tools.ops.wordlist import register_wordlist_tools


class TestWordlistOperations:
    """Test wordlist management and retrieval operations."""

    @pytest.fixture
    def mock_mcp(self):
        """Create mock MCP instance."""
        mcp = Mock()
        # Create a mapping to capture registered tools
        mcp._tools = {}
        
        def tool_decorator():
            def decorator(fn):
                mcp._tools[fn.__name__] = fn
                return fn
            return decorator
        
        mcp.tool = tool_decorator
        return mcp

    @pytest.fixture
    def mock_hexstrike_client(self):
        """Create mock hexstrike client."""
        return Mock()

    @pytest.fixture
    def setup_tools(self, mock_mcp, mock_hexstrike_client):
        """Register tools and return them for testing."""
        register_wordlist_tools(mock_mcp, mock_hexstrike_client)
        return mock_mcp._tools, mock_hexstrike_client

    @pytest.mark.asyncio
    async def test_wordlist_get_success(self, setup_tools):
        """Test retrieving a specific wordlist by ID."""
        tools, mock_client = setup_tools
        wordlist_get = tools['wordlist_get']

        mock_client.safe_get.return_value = {
            "success": True,
            "wordlist": {
                "id": "rockyou",
                "path": "/usr/share/wordlists/rockyou.txt",
                "type": "password",
                "size": 14344391,
                "speed": "slow",
                "language": "english",
                "coverage": "broad"
            }
        }

        result = await wordlist_get("rockyou")

        assert result["success"] is True
        assert result["wordlist"]["id"] == "rockyou"
        assert result["wordlist"]["path"] == "/usr/share/wordlists/rockyou.txt"
        mock_client.safe_get.assert_called_with("api/wordlists/rockyou")

    @pytest.mark.asyncio
    async def test_wordlist_get_not_found(self, setup_tools):
        """Test retrieving non-existent wordlist."""
        tools, mock_client = setup_tools
        wordlist_get = tools['wordlist_get']

        mock_client.safe_get.return_value = {
            "success": False,
            "error": "Wordlist not found",
            "status": 404
        }

        result = await wordlist_get("non_existent")

        assert result["success"] is False
        assert result["status"] == 404

    @pytest.mark.asyncio
    async def test_wordlist_get_all_success(self, setup_tools):
        """Test retrieving all wordlists."""
        tools, mock_client = setup_tools
        wordlist_get_all = tools['wordlist_get_all']

        mock_client.safe_get.return_value = {
            "success": True,
            "wordlists": [
                {
                    "id": "rockyou",
                    "path": "/usr/share/wordlists/rockyou.txt",
                    "type": "password"
                },
                {
                    "id": "dirb-common",
                    "path": "/usr/share/dirb/wordlists/common.txt",
                    "type": "directory"
                },
                {
                    "id": "sqlmap-payloads",
                    "path": "/usr/share/sqlmap/payloads.txt",
                    "type": "payload"
                }
            ]
        }

        result = await wordlist_get_all()

        assert result["success"] is True
        assert len(result["wordlists"]) == 3
        assert result["wordlists"][0]["id"] == "rockyou"
        assert result["wordlists"][1]["type"] == "directory"
        mock_client.safe_get.assert_called_with("api/wordlists")

    @pytest.mark.asyncio
    async def test_wordlist_get_all_empty(self, setup_tools):
        """Test retrieving all wordlists when none exist."""
        tools, mock_client = setup_tools
        wordlist_get_all = tools['wordlist_get_all']

        mock_client.safe_get.return_value = {
            "success": True,
            "wordlists": []
        }

        result = await wordlist_get_all()

        assert result["success"] is True
        assert len(result["wordlists"]) == 0

    @pytest.mark.asyncio
    async def test_wordlist_get_path_success(self, setup_tools):
        """Test retrieving wordlist file path."""
        tools, mock_client = setup_tools
        wordlist_get_path = tools['wordlist_get_path']

        mock_client.safe_get.return_value = {
            "success": True,
            "path": "/usr/share/wordlists/rockyou.txt"
        }

        result = await wordlist_get_path("rockyou")

        assert result["success"] is True
        assert result["path"] == "/usr/share/wordlists/rockyou.txt"
        mock_client.safe_get.assert_called_with("api/wordlists/rockyou/path")

    @pytest.mark.asyncio
    async def test_wordlist_get_path_not_found(self, setup_tools):
        """Test retrieving path for non-existent wordlist."""
        tools, mock_client = setup_tools
        wordlist_get_path = tools['wordlist_get_path']

        mock_client.safe_get.return_value = {
            "success": False,
            "error": "Wordlist not found",
            "status": 404
        }

        result = await wordlist_get_path("missing")

        assert result["success"] is False
        assert result["status"] == 404

    @pytest.mark.asyncio
    async def test_wordlist_find_best_by_type(self, setup_tools):
        """Test finding best wordlist by type."""
        tools, mock_client = setup_tools
        wordlist_find_best = tools['wordlist_find_best']

        mock_client.safe_post.return_value = {
            "success": True,
            "wordlist": {
                "id": "rockyou",
                "path": "/usr/share/wordlists/rockyou.txt",
                "type": "password",
                "speed": "slow",
                "coverage": "broad"
            }
        }

        criteria = {"type": "password"}
        result = await wordlist_find_best(criteria)

        assert result["success"] is True
        assert result["wordlist"]["type"] == "password"
        assert result["wordlist"]["id"] == "rockyou"
        mock_client.safe_post.assert_called_with("api/wordlists/bestmatch", criteria)

    @pytest.mark.asyncio
    async def test_wordlist_find_best_by_speed(self, setup_tools):
        """Test finding wordlist by speed criteria."""
        tools, mock_client = setup_tools
        wordlist_find_best = tools['wordlist_find_best']

        mock_client.safe_post.return_value = {
            "success": True,
            "wordlist": {
                "id": "small-password-list",
                "path": "/wordlists/small.txt",
                "type": "password",
                "speed": "fast"
            }
        }

        criteria = {"speed": "fast"}
        result = await wordlist_find_best(criteria)

        assert result["success"] is True
        assert result["wordlist"]["speed"] == "fast"

    @pytest.mark.asyncio
    async def test_wordlist_find_best_by_tool(self, setup_tools):
        """Test finding wordlist by tool requirement."""
        tools, mock_client = setup_tools
        wordlist_find_best = tools['wordlist_find_best']

        mock_client.safe_post.return_value = {
            "success": True,
            "wordlist": {
                "id": "dirb-common",
                "path": "/usr/share/dirb/wordlists/common.txt",
                "type": "directory",
                "tool": ["dirb", "wfuzz"]
            }
        }

        criteria = {"tool": "dirb"}
        result = await wordlist_find_best(criteria)

        assert result["success"] is True
        assert "dirb" in result["wordlist"]["tool"]

    @pytest.mark.asyncio
    async def test_wordlist_find_best_multiple_criteria(self, setup_tools):
        """Test finding wordlist with multiple criteria."""
        tools, mock_client = setup_tools
        wordlist_find_best = tools['wordlist_find_best']

        mock_client.safe_post.return_value = {
            "success": True,
            "wordlist": {
                "id": "rockyou-fast-subset",
                "path": "/wordlists/rockyou-fast.txt",
                "type": "password",
                "speed": "fast",
                "language": "english"
            }
        }

        criteria = {
            "type": "password",
            "speed": "fast",
            "language": "english"
        }
        result = await wordlist_find_best(criteria)

        assert result["success"] is True
        assert result["wordlist"]["type"] == "password"
        assert result["wordlist"]["speed"] == "fast"

    @pytest.mark.asyncio
    async def test_wordlist_find_best_not_found(self, setup_tools):
        """Test when no matching wordlist found."""
        tools, mock_client = setup_tools
        wordlist_find_best = tools['wordlist_find_best']

        mock_client.safe_post.return_value = {
            "success": False,
            "error": "No matching wordlist found"
        }

        criteria = {"type": "non_existent_type"}
        result = await wordlist_find_best(criteria)

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_wordlist_save_new(self, setup_tools):
        """Test saving a new wordlist."""
        tools, mock_client = setup_tools
        wordlist_save = tools['wordlist_save']

        mock_client.safe_post.return_value = {
            "success": True,
            "message": "Wordlist saved successfully"
        }

        wordlist_info = {
            "path": "/custom/wordlists/my_list.txt",
            "type": "password",
            "recommended_for": ["hashcat", "john"],
            "description": "Custom password list",
            "size": 1000000,
            "speed": "medium",
            "language": "english"
        }

        result = await wordlist_save("custom_list", wordlist_info)

        assert result["success"] is True
        mock_client.safe_post.assert_called_once()
        call_args = mock_client.safe_post.call_args
        assert call_args[0][0] == "api/wordlists/custom_list"

    @pytest.mark.asyncio
    async def test_wordlist_save_update_existing(self, setup_tools):
        """Test updating an existing wordlist."""
        tools, mock_client = setup_tools
        wordlist_save = tools['wordlist_save']

        mock_client.safe_post.return_value = {
            "success": True,
            "message": "Wordlist updated successfully"
        }

        updated_info = {
            "path": "/custom/wordlists/updated_list.txt",
            "type": "password",
            "recommended_for": ["hashcat"],
            "description": "Updated custom list"
        }

        result = await wordlist_save("custom_list", updated_info)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_wordlist_save_minimal_info(self, setup_tools):
        """Test saving wordlist with minimal required info."""
        tools, mock_client = setup_tools
        wordlist_save = tools['wordlist_save']

        mock_client.safe_post.return_value = {
            "success": True,
            "message": "Wordlist saved"
        }

        minimal_info = {
            "path": "/wordlists/minimal.txt",
            "type": "directory",
            "recommended_for": ["dirb"]
        }

        result = await wordlist_save("minimal_list", minimal_info)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_wordlist_save_failure_invalid_path(self, setup_tools):
        """Test saving wordlist with invalid path."""
        tools, mock_client = setup_tools
        wordlist_save = tools['wordlist_save']

        mock_client.safe_post.return_value = {
            "success": False,
            "error": "Path does not exist"
        }

        invalid_info = {
            "path": "/nonexistent/path/list.txt",
            "type": "password",
            "recommended_for": ["hashcat"]
        }

        result = await wordlist_save("invalid_list", invalid_info)

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_wordlist_operations_chain(self, setup_tools):
        """Test typical workflow: get all -> find best -> get path."""
        tools, mock_client = setup_tools

        # Mock responses
        mock_client.safe_get.side_effect = [
            # Response for wordlist_get_all
            {
                "success": True,
                "wordlists": [
                    {"id": "rockyou", "type": "password"},
                    {"id": "dirb-common", "type": "directory"}
                ]
            },
            # Response for wordlist_get_path
            {
                "success": True,
                "path": "/usr/share/wordlists/rockyou.txt"
            }
        ]

        mock_client.safe_post.return_value = {
            "success": True,
            "wordlist": {
                "id": "rockyou",
                "type": "password",
                "path": "/usr/share/wordlists/rockyou.txt"
            }
        }

        # Execute workflow
        result1 = await tools['wordlist_get_all']()
        assert result1["success"] is True
        assert len(result1["wordlists"]) == 2

        result2 = await tools['wordlist_find_best']({"type": "password"})
        assert result2["success"] is True
        assert result2["wordlist"]["id"] == "rockyou"

        result3 = await tools['wordlist_get_path']("rockyou")
        assert result3["success"] is True
        assert "rockyou.txt" in result3["path"]

    @pytest.mark.asyncio
    async def test_wordlist_find_best_common_criteria(self, setup_tools):
        """Test common real-world criteria patterns."""
        tools, mock_client = setup_tools
        wordlist_find_best = tools['wordlist_find_best']

        test_cases = [
            {"type": "password", "speed": "fast"},
            {"tool": "hashcat"},
            {"type": "directory", "tool": ["dirb", "wfuzz"]},
            {"language": "english", "coverage": "broad"},
            {"type": "payload"}
        ]

        mock_client.safe_post.return_value = {
            "success": True,
            "wordlist": {"id": "found"}
        }

        for criteria in test_cases:
            result = await wordlist_find_best(criteria)
            assert result["success"] is True
            mock_client.safe_post.assert_called_with("api/wordlists/bestmatch", criteria)

    @pytest.mark.asyncio
    async def test_wordlist_concurrent_operations(self, setup_tools):
        """Test concurrent wordlist operations."""
        tools, mock_client = setup_tools

        mock_client.safe_get.return_value = {
            "success": True,
            "wordlist": {"id": "test"}
        }

        # Run multiple operations concurrently
        results = await asyncio.gather(
            tools['wordlist_get']("list1"),
            tools['wordlist_get']("list2"),
            tools['wordlist_get']("list3")
        )

        assert len(results) == 3
        assert all(r["success"] is True for r in results)
        assert mock_client.safe_get.call_count == 3

    @pytest.mark.asyncio
    async def test_wordlist_save_with_all_optional_fields(self, setup_tools):
        """Test saving wordlist with all optional fields."""
        tools, mock_client = setup_tools
        wordlist_save = tools['wordlist_save']

        mock_client.safe_post.return_value = {
            "success": True,
            "message": "Complete wordlist saved"
        }

        complete_info = {
            "path": "/wordlists/complete.txt",
            "type": "password",
            "recommended_for": ["hashcat", "john", "aircrack"],
            "description": "Complete password dictionary with optimizations",
            "size": 14344391,
            "tool": ["hashcat", "john"],
            "speed": "slow",
            "language": "english",
            "coverage": "broad",
            "format": "txt"
        }

        result = await wordlist_save("complete_list", complete_info)

        assert result["success"] is True
        call_args = mock_client.safe_post.call_args
        assert call_args[0][1] == complete_info
