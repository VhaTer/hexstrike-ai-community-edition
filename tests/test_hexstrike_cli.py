import json
import subprocess
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
HEXSTRIKE = ROOT / "pulse/server/cli.py"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HEXSTRIKE), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_ctf_json_is_valid_and_uses_args():
    result = run_cli(
        "ctf",
        "--category", "web",
        "--difficulty", "hard",
        "--name", "Admin Portal",
        "--points", "500",
        "--target", "10.10.10.10",
        "--json",
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["name"] == "Admin Portal"
    assert data["category"] == "web"
    assert data["difficulty"] == "hard"
    assert data["points"] == 500
    assert data["target"] == "10.10.10.10"
    assert data["total_steps"] == len(data["steps"])


def test_tools_json_is_valid():
    result = run_cli("tools", "--filter", "nmap", "--json")

    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert "nmap" in data


def test_scan_json_suppresses_runner_stdout(monkeypatch):
    import pulse.server.cli as hexstrike

    def noisy_resolve(tool_name):
        print("resolver noise")

        def noisy_exec(tool_key, params):
            print("runner noise")
            return {"success": True, "output": "clean", "params": params}

        return noisy_exec, tool_name

    monkeypatch.setattr(hexstrike, "_resolve_tool", noisy_resolve)
    args = SimpleNamespace(
        tool="fake_tool",
        target="example.com",
        param=[],
        json=True,
        output="",
    )

    stdout = StringIO()
    with redirect_stdout(stdout):
        hexstrike.cmd_scan(args)

    data = json.loads(stdout.getvalue())
    assert data["success"] is True
    assert data["output"] == "clean"
    assert "resolver noise" not in stdout.getvalue()
    assert "runner noise" not in stdout.getvalue()
