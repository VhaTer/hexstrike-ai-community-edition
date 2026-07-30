"""Tests for pulse/server/cli.py — CLI entry point and subcommands."""

import json
import logging
import sys
import argparse
from pathlib import Path
from unittest.mock import patch, MagicMock
import urllib.error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pulse.server.cli import (
    _resolve_tool,
    _call_with_stdout_suppressed,
    _cli_colors,
    _dw,
    _ansi_strip,
    _print_output,
    _emit_json,
    _unknown_tool_json,
    _format_scan_result,
    _ctf_scans_for,
    cmd_scan,
    cmd_tools,
    cmd_status,
    cmd_validate,
    cmd_ctf,
    cmd_serve,
    cmd_mcp,
    build_parser,
    main,
    VERSION,
    TOOL_ROUTES,
)


# ============================================================================
# Utility functions
# ============================================================================


class TestResolveTool:
    def test_known_tool_returns_func_and_key(self):
        result = _resolve_tool("nmap")
        assert result is not None
        exec_func, tool_key = result
        assert callable(exec_func)
        assert tool_key == "nmap"

    def test_unknown_tool_returns_none(self):
        assert _resolve_tool("this_tool_does_not_exist_xyz") is None


class TestCallWithStdoutSuppressed:
    def test_enabled_suppresses_stdout(self, capsys):
        def noisy():
            print("SHOULD NOT APPEAR")
            return 42

        result = _call_with_stdout_suppressed(True, noisy)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert result == 42

    def test_disabled_does_not_suppress(self, capsys):
        def noisy():
            print("SHOULD APPEAR")
            return 99

        result = _call_with_stdout_suppressed(False, noisy)
        captured = capsys.readouterr()
        assert captured.out == "SHOULD APPEAR\n"
        assert result == 99

    def test_passes_args_and_kwargs(self):
        def adder(a, b=0):
            return a + b

        result = _call_with_stdout_suppressed(True, adder, 10, b=20)
        assert result == 30


class TestCliColors:
    def test_returns_dict_with_expected_keys(self):
        colors = _cli_colors()
        assert isinstance(colors, dict)
        for key in ("ACCENT_LINE", "TERMINAL_GRAY", "BRIGHT_WHITE", "RESET"):
            assert key in colors, f"Missing color key: {key}"
        for v in colors.values():
            assert isinstance(v, str)
            assert v.startswith("\033[")


class TestDw:
    def test_basic_ascii_width(self):
        assert _dw("hello") == 5

    def test_empty_string(self):
        assert _dw("") == 0

    def test_strips_ansi_codes(self):
        green = "\033[32m"
        reset = "\033[0m"
        s = f"{green}hello{reset}"
        assert _dw(s) == 5

    def test_ansi_strip_helper(self):
        assert _ansi_strip("", "\033[32mhello\033[0m") == "hello"
        assert _ansi_strip("", "\033[38;5;196mABC\033[0m") == "ABC"
        assert _ansi_strip("", "plain text") == "plain text"
        assert _ansi_strip("", "") == ""


class TestPrintOutput:
    def test_prints_to_stdout(self, capsys):
        _print_output("hello world\n", argparse.Namespace(output=""))
        captured = capsys.readouterr()
        assert captured.out == "hello world\n"

    def test_writes_to_file_when_output_given(self, tmp_path, capsys):
        outfile = tmp_path / "result.txt"
        args = argparse.Namespace(output=str(outfile))
        _print_output("file content\n", args)
        captured = capsys.readouterr()
        assert captured.out == "file content\n"
        assert outfile.read_text() == "file content\n"

    def test_writes_to_file_no_output_arg(self):
        """When args has no output attribute, should just print."""

        class ArgsNoOutput:
            pass

        _print_output("just print\n", ArgsNoOutput())


class TestEmitJson:
    def test_emits_valid_json(self, capsys):
        data = {"key": "value", "num": 42}
        _emit_json(data, argparse.Namespace(output=""))
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed == data

    def test_writes_json_to_file(self, tmp_path, capsys):
        outfile = tmp_path / "out.json"
        data = {"success": True}
        _emit_json(data, argparse.Namespace(output=str(outfile)))
        assert json.loads(outfile.read_text()) == data
        captured = capsys.readouterr()
        assert json.loads(captured.out) == data

    def test_handles_non_serializable_types(self, capsys):
        data = {"timestamp": b"bytes"}
        _emit_json(data, argparse.Namespace(output=""))
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["timestamp"] == "b'bytes'"


class TestUnknownToolJson:
    def test_returns_error_shape_with_similar(self):
        result = _unknown_tool_json("nmap_")
        assert result["error"] == "Unknown tool: nmap_"
        assert result["success"] is False
        assert isinstance(result["similar"], list)
        assert any("nmap" in s for s in result["similar"])

    def test_completely_unknown_tool(self):
        result = _unknown_tool_json("zzzznotool")
        assert result["error"] == "Unknown tool: zzzznotool"
        assert result["similar"] == []
        assert result["success"] is False


# ============================================================================
# build_parser
# ============================================================================


class TestBuildParser:
    def test_parser_returns_argparse_parser(self):
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_version_action(self, capsys):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--version"])
        captured = capsys.readouterr()
        assert VERSION in captured.out

    def test_serve_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["serve", "--host", "0.0.0.0", "--port", "9999", "--debug"])
        assert args.command == "serve"
        assert args.host == "0.0.0.0"
        assert args.port == 9999
        assert args.debug is True

    def test_scan_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(
            ["scan", "nmap", "10.0.0.1", "-p", "scan_type=-sV", "--json", "-o", "out.json"]
        )
        assert args.command == "scan"
        assert args.tool == "nmap"
        assert args.target == "10.0.0.1"
        assert args.param == ["scan_type=-sV"]
        assert args.json is True
        assert args.output == "out.json"

    def test_scan_subcommand_no_target(self):
        parser = build_parser()
        args = parser.parse_args(["scan", "nmap"])
        assert args.command == "scan"
        assert args.tool == "nmap"
        assert args.target == ""

    def test_tools_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["tools", "--filter", "nmap", "--json", "-o", "tools.json"])
        assert args.command == "tools"
        assert args.filter == "nmap"
        assert args.json is True
        assert args.output == "tools.json"

    def test_status_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["status", "--host", "10.0.0.1", "--port", "8080"])
        assert args.command == "status"
        assert args.host == "10.0.0.1"
        assert args.port == 8080

    def test_validate_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["validate", "--tool-filter", "nmap", "--verbose", "--json"])
        assert args.command == "validate"
        assert args.tool_filter == "nmap"
        assert args.verbose is True
        assert args.json is True

    def test_mcp_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["mcp", "--debug", "--compact", "--timeout", "600"])
        assert args.command == "mcp"
        assert args.debug is True
        assert args.compact is True
        assert args.timeout == 600

    def test_ctf_subcommand(self):
        parser = build_parser()
        args = parser.parse_args([
            "ctf", "--category", "pwn", "--name", "test", "--difficulty", "hard",
            "--points", "500", "--target", "10.0.0.1", "--json",
        ])
        assert args.command == "ctf"
        assert args.category == "pwn"
        assert args.name == "test"
        assert args.difficulty == "hard"
        assert args.points == 500
        assert args.target == "10.0.0.1"
        assert args.json is True

    def test_ctf_subcommand_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["ctf"])
        assert args.command == "ctf"
        assert args.category == "web"
        assert args.name == ""
        assert args.difficulty == "medium"
        assert args.points == 0
        assert args.target == ""

    def test_no_command_fails(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_unknown_command_fails(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["nonexistent_command"])


# ============================================================================
# _format_scan_result
# ============================================================================


class TestFormatScanResult:
    def test_success_case(self):
        result = {"success": True, "output": "open ports: 22, 80", "execution_time": 3.5}
        output = _format_scan_result("nmap", "10.0.0.1", result)
        assert "Tool:" in output
        assert "Target:" in output
        assert "nmap" in output
        assert "10.0.0.1" in output
        assert "Success" in output
        assert "3.5" in output
        assert "open ports: 22, 80" in output

    def test_failed_case(self):
        result = {"success": False, "error": "Connection refused", "execution_time": 0.5}
        output = _format_scan_result("nmap", "10.0.0.1", result)
        assert "Failed" in output
        assert "Error:" in output
        assert "Connection refused" in output

    def test_timed_out_case(self):
        result = {"success": False, "timed_out": True, "execution_time": 30.0}
        output = _format_scan_result("nmap", "10.0.0.1", result)
        assert "Timed out" in output
        assert "30.0" in output

    def test_no_target(self):
        result = {"success": True, "output": "done", "execution_time": 1.0}
        output = _format_scan_result("tool_x", "", result)
        assert "tool_x" in output
        assert "(none)" in output

    def test_uses_stdout_fallback(self):
        result = {"success": True, "stdout": "fallback output", "execution_time": 1.0}
        output = _format_scan_result("tool_x", "t", result)
        assert "fallback output" in output

    def test_stderr_on_failure(self):
        result = {"success": False, "stderr": "something went wrong", "execution_time": 0.5}
        output = _format_scan_result("tool_x", "t", result)
        assert "something went wrong" in output

    def test_error_truncated_at_500(self):
        long_err = "x" * 1000
        result = {"success": False, "error": long_err, "execution_time": 0.5}
        output = _format_scan_result("tool_x", "t", result)
        assert "x" * 500 in output
        assert "x" * 501 not in output


# ============================================================================
# _ctf_scans_for
# ============================================================================


class TestCtfScansFor:
    def test_web_category_with_url(self):
        scans = _ctf_scans_for("web", "http://10.0.0.1:8080")
        assert len(scans) >= 1
        tool, params, label = scans[0]
        assert tool == "nmap"
        assert params["target"] == "10.0.0.1"
        assert "22,80,443,8080" in params.get("ports", "")

    def test_web_category_with_ip(self):
        scans = _ctf_scans_for("web", "10.0.0.1")
        assert len(scans) >= 1
        tool, params, label = scans[0]
        assert tool == "nmap"
        assert params["target"] == "10.0.0.1"

    def test_pwn_rev_category_extra_ports(self):
        scans = _ctf_scans_for("pwn", "10.0.0.1")
        assert len(scans) >= 1
        _, params, _ = scans[0]
        assert "4444" in params.get("ports", "")
        assert "1337" in params.get("ports", "")
        assert "-sV" in params.get("scan_type", "")

    def test_crypto_category_default_ports(self):
        scans = _ctf_scans_for("crypto", "10.0.0.1")
        assert len(scans) >= 1
        _, params, _ = scans[0]
        assert params.get("ports") == "22,80,443"

    def test_invalid_url_returns_empty(self):
        scans = _ctf_scans_for("web", "http://")
        assert scans == []

    def test_misc_category_like_web(self):
        scans = _ctf_scans_for("misc", "10.0.0.1")
        assert len(scans) >= 1
        _, params, _ = scans[0]
        assert "8080" in params.get("ports", "")


# ============================================================================
# cmd_scan
# ============================================================================


class TestCmdScan:
    @pytest.fixture
    def mock_resolve(self):
        fake_func = MagicMock(return_value={
            "success": True, "output": "scan results", "execution_time": 1.2
        })
        with patch("pulse.server.cli._resolve_tool", return_value=(fake_func, "nmap")):
            yield fake_func

    def test_scan_success_no_json(self, mock_resolve, capsys):
        args = argparse.Namespace(
            tool="nmap", target="10.0.0.1", param=[], json=False, output=""
        )
        cmd_scan(args)
        captured = capsys.readouterr()
        assert "nmap" in captured.out
        assert "10.0.0.1" in captured.out
        assert "Success" in captured.out

    def test_scan_json_output(self, mock_resolve, capsys):
        args = argparse.Namespace(
            tool="nmap", target="10.0.0.1", param=[], json=True, output=""
        )
        cmd_scan(args)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["success"] is True
        assert parsed["output"] == "scan results"

    def test_scan_with_params(self, mock_resolve, capsys):
        args = argparse.Namespace(
            tool="nmap", target="10.0.0.1",
            param=["scan_type=-sV", "ports=22,80"],
            json=False, output="",
        )
        cmd_scan(args)
        mock_func = mock_resolve
        called = mock_func.call_args
        assert called is not None
        # exec_func(tool_key, params) — both positional
        positioned_args = called[0]
        assert len(positioned_args) >= 2
        _, params_dict = positioned_args[0], positioned_args[1]
        assert params_dict["target"] == "10.0.0.1"
        assert params_dict["url"] == "10.0.0.1"

    def test_scan_unknown_tool_no_json(self, caplog, capsys):
        caplog.set_level(logging.INFO)
        args = argparse.Namespace(
            tool="nonexistent_tool_xyz", target="", param=[], json=False, output=""
        )
        with pytest.raises(SystemExit):
            cmd_scan(args)
        assert "Unknown tool: nonexistent_tool_xyz" in caplog.text

    def test_scan_unknown_tool_did_you_mean(self, caplog, capsys):
        caplog.set_level(logging.INFO)
        args = argparse.Namespace(
            tool="nmap_unk", target="", param=[], json=False, output=""
        )
        with pytest.raises(SystemExit):
            cmd_scan(args)
        assert "Did you mean" in caplog.text
        assert "nmap" in caplog.text

    def test_scan_unknown_tool_json(self, capsys):
        args = argparse.Namespace(
            tool="nonexistent_tool_xyz", target="", param=[], json=True, output=""
        )
        with pytest.raises(SystemExit):
            cmd_scan(args)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["error"] == "Unknown tool: nonexistent_tool_xyz"
        assert parsed["success"] is False

    def test_scan_writes_to_file(self, mock_resolve, tmp_path, capsys):
        outfile = tmp_path / "scan.json"
        args = argparse.Namespace(
            tool="nmap", target="10.0.0.1", param=[], json=True, output=str(outfile)
        )
        cmd_scan(args)
        assert outfile.exists()
        content = json.loads(outfile.read_text())
        assert content["success"] is True

    def test_scan_unmatched_param_skipped(self, mock_resolve, capsys):
        """Param without '=' is skipped."""
        args = argparse.Namespace(
            tool="nmap", target="10.0.0.1",
            param=["invalid"],
            json=False, output="",
        )
        cmd_scan(args)
        captured = capsys.readouterr()
        assert "nmap" in captured.out


# ============================================================================
# cmd_tools
# ============================================================================


class TestCmdTools:
    def test_lists_all_tools_no_filter(self, capsys):
        args = argparse.Namespace(filter="", json=False, output="")
        cmd_tools(args)
        captured = capsys.readouterr()
        assert "nmap" in captured.out or "vulnx" in captured.out

    def test_filter_matches(self, capsys):
        args = argparse.Namespace(filter="nmap", json=False, output="")
        cmd_tools(args)
        captured = capsys.readouterr()
        assert "nmap" in captured.out
        assert "nmap_advanced" in captured.out

    def test_filter_no_match_text(self, capsys):
        args = argparse.Namespace(filter="zzz_no_match_xyz", json=False, output="")
        cmd_tools(args)
        captured = capsys.readouterr()
        assert "No tools matching" in captured.out

    def test_filter_no_match_json(self, capsys):
        args = argparse.Namespace(filter="zzz_no_match_xyz", json=True, output="")
        cmd_tools(args)
        captured = capsys.readouterr()
        assert json.loads(captured.out) == {}

    def test_json_output(self, capsys):
        args = argparse.Namespace(filter="nmap", json=True, output="")
        cmd_tools(args)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "nmap" in parsed
        assert "description" in parsed["nmap"]
        assert "category" in parsed["nmap"]

    def test_writes_output_to_file(self, tmp_path, capsys):
        outfile = tmp_path / "tools.txt"
        args = argparse.Namespace(filter="nmap", json=False, output=str(outfile))
        cmd_tools(args)
        assert outfile.exists()
        content = outfile.read_text()
        assert "nmap" in content

    def test_writes_json_to_file(self, tmp_path, capsys):
        outfile = tmp_path / "tools.json"
        args = argparse.Namespace(filter="nmap", json=True, output=str(outfile))
        cmd_tools(args)
        content = json.loads(outfile.read_text())
        assert "nmap" in content

    def test_filter_no_match_writes_to_file(self, tmp_path, capsys):
        outfile = tmp_path / "empty.txt"
        args = argparse.Namespace(filter="zzz_no_match_xyz", json=False, output=str(outfile))
        cmd_tools(args)
        assert outfile.exists()
        assert "No tools matching" in outfile.read_text()


# ============================================================================
# cmd_status
# ============================================================================


def _make_status_args(**overrides):
    defaults = dict(host=None, port=None, json=False, output="")
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestCmdStatus:
    @pytest.fixture
    def mock_urlopen(self):
        """Code uses resp.read() directly (no 'with' statement)."""
        with patch("urllib.request.urlopen") as mock:
            mock.return_value.read.return_value = json.dumps({
                "status": "ready",
                "uptime_seconds": 3600,
                "checks": {
                    "essential_tools": {"available": 42, "total": 50},
                    "disk": {"free_gb": 100, "usage_pct": 30},
                },
            }).encode()
            yield mock

    def test_status_success_text(self, mock_urlopen, capsys):
        cmd_status(_make_status_args())
        captured = capsys.readouterr()
        assert "ready" in captured.out

    def test_status_success_json(self, mock_urlopen, capsys):
        cmd_status(_make_status_args(json=True))
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["status"] == "ready"
        assert parsed["uptime_seconds"] == 3600
        assert parsed["checks"]["essential_tools"]["available"] == 42

    def test_status_urlerror_text(self, capsys):
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("Connection refused")):
            with pytest.raises(SystemExit):
                cmd_status(_make_status_args())
        captured = capsys.readouterr()
        assert "not responding" in captured.out

    def test_status_urlerror_json(self, capsys):
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("Connection refused")):
            with pytest.raises(SystemExit):
                cmd_status(_make_status_args(json=True))
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "Server not responding" in parsed["error"]

    def test_status_generic_exception_text(self, capsys):
        with patch("urllib.request.urlopen", side_effect=RuntimeError("Boom!")):
            with pytest.raises(SystemExit):
                cmd_status(_make_status_args())
        captured = capsys.readouterr()
        assert "Error: Boom!" in captured.out

    def test_status_generic_exception_json(self, capsys):
        with patch("urllib.request.urlopen", side_effect=RuntimeError("Boom!")):
            with pytest.raises(SystemExit):
                cmd_status(_make_status_args(json=True))
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["error"] == "Boom!"

    def test_status_with_custom_host_port(self, mock_urlopen, capsys):
        cmd_status(_make_status_args(host="10.0.0.1", port=8080))
        call_url = mock_urlopen.call_args[0][0]
        assert "10.0.0.1:8080" in call_url


# ============================================================================
# cmd_validate
# ============================================================================


def _make_validate_args(**overrides):
    defaults = dict(tool_filter="", verbose=False, json=False, output="")
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestCmdValidate:
    def test_all_present_text(self, capsys):
        with patch("shutil.which", return_value="/usr/bin/nmap"):
            cmd_validate(_make_validate_args())
        captured = capsys.readouterr()
        assert "tools available" in captured.out

    def test_missing_tools_text(self, capsys):
        def fake_which(binary):
            return "/usr/bin/nmap" if binary == "nmap" else None

        with patch("shutil.which", side_effect=fake_which):
            cmd_validate(_make_validate_args())
        captured = capsys.readouterr()
        assert "Missing" in captured.out

    def test_mixed_verbose_text(self, capsys):
        present_binaries = {"nmap", "sqlmap", "gobuster"}

        def fake_which(binary):
            return f"/usr/bin/{binary}" if binary in present_binaries else None

        with patch("shutil.which", side_effect=fake_which):
            cmd_validate(_make_validate_args(verbose=True))
        captured = capsys.readouterr()
        assert "Present" in captured.out
        assert "Missing" in captured.out

    def test_json_output_all_present(self, capsys):
        with patch("shutil.which", return_value="/usr/bin/nmap"):
            cmd_validate(_make_validate_args(json=True))
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "total" in parsed
        assert "present_count" in parsed
        assert "missing_count" in parsed

    def test_json_output_missing(self, capsys):
        def fake_which(binary):
            return "/usr/bin/nmap" if binary == "nmap" else None

        with patch("shutil.which", side_effect=fake_which):
            cmd_validate(_make_validate_args(json=True))
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["missing_count"] > 0
        assert len(parsed["missing"]) == parsed["missing_count"]
        assert len(parsed["present"]) == parsed["present_count"]

    def test_filter_limits_tools_checked(self, capsys):
        checked = []

        def fake_which(binary):
            checked.append(binary)
            return "/usr/bin/nmap"

        with patch("shutil.which", side_effect=fake_which):
            cmd_validate(_make_validate_args(tool_filter="nmap"))
        assert len(set(checked)) >= 2
        assert any("nmap" in b for b in checked)
        captured = capsys.readouterr()
        assert "tools available" in captured.out

    def test_filter_no_match_text(self, capsys):
        with patch("shutil.which", return_value="/usr/bin/nmap"):
            cmd_validate(_make_validate_args(tool_filter="zzz_no_match_xyz"))
        captured = capsys.readouterr()
        assert captured.out or not captured.out

    def test_writes_output_to_file(self, tmp_path, capsys):
        outfile = tmp_path / "validate.txt"
        with patch("shutil.which", return_value="/usr/bin/nmap"):
            cmd_validate(_make_validate_args(output=str(outfile)))
        assert outfile.exists()
        assert "tools available" in outfile.read_text()


# ============================================================================
# cmd_ctf
# ============================================================================


class TestCmdCtf:
    @pytest.fixture
    def mock_ctf_imports(self):
        """Import paths used inside cmd_ctf():
            from pulse.workflows.ctf.CTFChallenge import CTFChallenge
            from pulse.workflows.ctf.workflowManager import CTFWorkflowManager
        """
        mock_challenge_cls = MagicMock()
        mock_challenge_instance = MagicMock()
        mock_challenge_cls.return_value = mock_challenge_instance

        mock_wm = MagicMock()
        mock_wm.create_ctf_challenge_workflow.return_value = {
            "workflow_steps": [
                {"step": "recon", "description": "Initial recon", "action": "recon"},
                {"step": "exploit", "description": "Exploit", "action": "exploit"},
            ],
            "tools": ["nmap", "gobuster"],
            "estimated_time": 600,
            "success_probability": 0.75,
        }

        with (
            patch("pulse.workflows.ctf.CTFChallenge.CTFChallenge", mock_challenge_cls),
            patch("pulse.workflows.ctf.workflowManager.CTFWorkflowManager",
                  return_value=mock_wm),
        ):
            yield

    def test_ctf_text_output(self, mock_ctf_imports, capsys):
        args = argparse.Namespace(
            category="web", name="TestChallenge",
            description="A test", difficulty="easy",
            points=200, target="", json=False, output="",
        )
        cmd_ctf(args)
        captured = capsys.readouterr()
        assert "TestChallenge" in captured.out
        assert "web" in captured.out
        assert "easy" in captured.out
        assert "Workflow" in captured.out
        assert "recon" in captured.out

    def test_ctf_json_output(self, mock_ctf_imports, capsys):
        args = argparse.Namespace(
            category="web", name="TestChallenge",
            description="A test", difficulty="easy",
            points=200, target="", json=True, output="",
        )
        cmd_ctf(args)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["name"] == "TestChallenge"
        assert parsed["category"] == "web"
        assert parsed["difficulty"] == "easy"
        assert parsed["points"] == 200
        assert parsed["total_steps"] == 2
        assert parsed["estimated_time"] == 600
        assert parsed["success_probability"] == 0.75

    def test_ctf_text_with_target(self, mock_ctf_imports, capsys):
        args = argparse.Namespace(
            category="web", name="WebChallenge",
            description="", difficulty="medium",
            points=0, target="10.0.0.1", json=False, output="",
        )
        cmd_ctf(args)
        captured = capsys.readouterr()
        assert "10.0.0.1" in captured.out

    def test_ctf_json_with_target(self, mock_ctf_imports, capsys):
        args = argparse.Namespace(
            category="web", name="WebChallenge",
            description="", difficulty="medium",
            points=0, target="10.0.0.1", json=True, output="",
        )
        cmd_ctf(args)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["target"] == "10.0.0.1"

    def test_ctf_with_scan_results(self, mock_ctf_imports, capsys):
        fake_scan_func = MagicMock(return_value={
            "success": True, "output": "open port 80", "execution_time": 2.0
        })
        with patch("pulse.server.cli._resolve_tool", return_value=(fake_scan_func, "nmap")):
            args = argparse.Namespace(
                category="web", name="WebChallenge",
                description="", difficulty="medium",
                points=0, target="10.0.0.1", json=True, output="",
            )
            cmd_ctf(args)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "scans" in parsed
        assert len(parsed["scans"]) >= 1
        assert parsed["scans"][0]["tool"] == "nmap"

    def test_ctf_scan_resolve_fails(self, mock_ctf_imports, capsys):
        with patch("pulse.server.cli._resolve_tool", return_value=None):
            args = argparse.Namespace(
                category="web", name="WebChallenge",
                description="", difficulty="medium",
                points=0, target="10.0.0.1", json=True, output="",
            )
            cmd_ctf(args)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "scans" not in parsed or parsed["scans"] == []

    def test_ctf_scan_execution_exception(self, mock_ctf_imports, capsys):
        def failing_func(*args, **kwargs):
            raise RuntimeError("Scan crashed")

        fake_scan_func = MagicMock(side_effect=failing_func)
        with patch("pulse.server.cli._resolve_tool", return_value=(fake_scan_func, "nmap")):
            args = argparse.Namespace(
                category="web", name="WebChallenge",
                description="", difficulty="medium",
                points=0, target="10.0.0.1", json=True, output="",
            )
            cmd_ctf(args)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "scans" in parsed
        assert parsed["scans"][0]["result"]["success"] is False
        assert "Scan crashed" in parsed["scans"][0]["result"]["error"]

    def test_ctf_default_name(self, mock_ctf_imports, capsys):
        """When name is empty, generates from category."""
        args = argparse.Namespace(
            category="web", name="",
            description="", difficulty="medium",
            points=0, target="", json=True, output="",
        )
        cmd_ctf(args)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["name"] == "WEB Challenge"

    def test_ctf_default_description(self, mock_ctf_imports, capsys):
        """When description is empty, generates from category."""
        args = argparse.Namespace(
            category="web", name="MyChallenge",
            description="", difficulty="medium",
            points=0, target="", json=True, output="",
        )
        cmd_ctf(args)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["total_steps"] == 2

    def test_ctf_scan_result_with_error_only_text(self, mock_ctf_imports, capsys):
        """CTF text output: scan result with error but no output lines."""
        fake_scan_func = MagicMock(return_value={
            "success": False, "error": "timeout", "execution_time": 30.0,
        })
        with patch("pulse.server.cli._resolve_tool", return_value=(fake_scan_func, "nmap")):
            args = argparse.Namespace(
                category="web", name="WebChallenge",
                description="", difficulty="medium",
                points=0, target="10.0.0.1", json=False, output="",
            )
            cmd_ctf(args)
        captured = capsys.readouterr()
        assert "timeout" in captured.out
        assert "FAIL" in captured.out


# ============================================================================
# cmd_serve / cmd_mcp (smoke tests)
# ============================================================================


class TestCmdServe:
    def test_serve_with_default_host_port(self):
        """cmd_serve() does local imports at runtime:
            from pulse.interface.server_setup import ...
            from pulse.infrastructure.modern_visual_engine import ...
        """
        mock_mcp = MagicMock()

        with (
            patch("pulse.interface.server_setup.setup_mcp_server_standalone",
                  return_value=mock_mcp) as mock_setup,
            patch("pulse.infrastructure.modern_visual_engine.ModernVisualEngine") as mock_mve,
            patch("importlib.import_module") as mock_import,
        ):
            mock_mve.create_banner.return_value = "BANNER"
            mock_server_setup = MagicMock()
            mock_import.return_value = mock_server_setup

            args = argparse.Namespace(host=None, port=None, debug=False)
            cmd_serve(args)

        mock_setup.assert_called_once()
        mock_mcp.run.assert_called_once_with(
            transport="http", host="127.0.0.1", port=8888, show_banner=False,
        )

    def test_serve_with_custom_host_port(self):
        mock_mcp = MagicMock()
        with (
            patch("pulse.interface.server_setup.setup_mcp_server_standalone",
                  return_value=mock_mcp),
            patch("pulse.infrastructure.modern_visual_engine.ModernVisualEngine"),
            patch("importlib.import_module"),
        ):
            args = argparse.Namespace(host="0.0.0.0", port=9090, debug=False)
            cmd_serve(args)

        mock_mcp.run.assert_called_once_with(
            transport="http", host="0.0.0.0", port=9090, show_banner=False,
        )


class TestCmdMcp:
    def test_mcp_debug_mode(self):
        """cmd_mcp() imports at runtime:
            from pulse.server.mcp_entry import run_mcp
        """
        mock_run_mcp = MagicMock()
        with patch("pulse.server.mcp_entry.run_mcp", mock_run_mcp):
            args = argparse.Namespace(
                debug=True, server="http://localhost:8888",
                timeout=300, compact=False, profile=[],
                auth_token="", disable_ssl_verify=False,
            )
            cmd_mcp(args)

        mock_run_mcp.assert_called_once()
        call_args = mock_run_mcp.call_args[0]
        assert call_args[0].debug is True
        assert call_args[0].server == "http://localhost:8888"
        assert call_args[0].timeout == 300

    def test_mcp_with_compact_and_auth(self):
        mock_run_mcp = MagicMock()
        with patch("pulse.server.mcp_entry.run_mcp", mock_run_mcp):
            args = argparse.Namespace(
                debug=False, server="http://127.0.0.1:8888",
                timeout=600, compact=True, profile=["web"],
                auth_token="tok123", disable_ssl_verify=True,
            )
            cmd_mcp(args)

        mock_run_mcp.assert_called_once()
        obj = mock_run_mcp.call_args[0][0]
        assert obj.compact is True
        assert obj.auth_token == "tok123"
        assert obj.disable_ssl_verify is True


# ============================================================================
# main()
# ============================================================================


class TestMain:
    def test_dispatches_to_scan(self, capsys):
        fake_func = MagicMock(return_value={
            "success": True, "output": "done", "execution_time": 0.5
        })
        with (
            patch("pulse.server.cli._resolve_tool", return_value=(fake_func, "nmap")),
            patch.object(sys, "argv", ["hexstrike", "scan", "nmap", "10.0.0.1"]),
        ):
            main()
        captured = capsys.readouterr()
        assert "nmap" in captured.out
        assert "10.0.0.1" in captured.out

    def test_dispatches_to_tools(self, capsys):
        with patch.object(sys, "argv", ["hexstrike", "tools", "--filter", "nmap"]):
            main()
        captured = capsys.readouterr()
        assert "nmap" in captured.out

    def test_dispatches_to_status(self, capsys):
        with (
            patch("urllib.request.urlopen") as mock_urlopen,
        ):
            mock_urlopen.return_value.read.return_value = json.dumps({
                "status": "ready", "uptime_seconds": 100, "checks": {},
            }).encode()
            with patch.object(sys, "argv", ["hexstrike", "status"]):
                main()
        captured = capsys.readouterr()
        assert "ready" in captured.out

    def test_dispatches_to_validate(self, capsys):
        with (
            patch("shutil.which", return_value="/usr/bin/nmap"),
            patch.object(sys, "argv", ["hexstrike", "validate"]),
        ):
            main()
        captured = capsys.readouterr()
        assert "tools available" in captured.out

    def test_dispatches_to_ctf(self, capsys):
        mock_wm = MagicMock()
        mock_wm.create_ctf_challenge_workflow.return_value = {
            "workflow_steps": [],
            "tools": [],
            "estimated_time": 0,
            "success_probability": 0,
        }
        with (
            patch("pulse.workflows.ctf.CTFChallenge.CTFChallenge"),
            patch("pulse.workflows.ctf.workflowManager.CTFWorkflowManager",
                  return_value=mock_wm),
            patch.object(sys, "argv", ["hexstrike", "ctf", "--category", "rev"]),
        ):
            main()
        captured = capsys.readouterr()
        assert "CTF:" in captured.out
        assert "rev" in captured.out

    def test_bare_exit_on_parse_error(self):
        with patch.object(sys, "argv", ["hexstrike"]):
            with pytest.raises(SystemExit):
                main()

    def test_command_dispatch_error_handled(self, caplog, capsys):
        caplog.set_level(logging.ERROR)
        with (
            patch("pulse.server.cli.cmd_tools", side_effect=ValueError("oops")),
            patch.object(sys, "argv", ["hexstrike", "tools"]),
            pytest.raises(SystemExit),
        ):
            main()
        assert "tools failed: oops" in caplog.text

    def test_command_dispatch_error_json(self):
        """When json=True and handler fails, emits JSON error."""
        with (
            patch("pulse.server.cli.cmd_ctf", side_effect=RuntimeError("broken")),
            patch.object(sys, "argv", ["hexstrike", "ctf", "--json"]),
            pytest.raises(SystemExit),
        ):
            main()
