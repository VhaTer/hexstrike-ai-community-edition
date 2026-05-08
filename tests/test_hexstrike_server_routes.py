import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch


class FakeMCP:
    def __init__(self):
        self._additional_http_routes = []
        self.routes = {}
        self.route_order = []

    def custom_route(self, path, methods, name=None, include_in_schema=True):
        def decorator(fn):
            self.routes[(path, tuple(methods))] = fn
            self.route_order.append((path, tuple(methods)))
            return fn
        return decorator


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_register_http_routes_adds_health_and_ping(tmp_path):
    from hexstrike_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")
    assets_dir = static_dir / "assets"
    assets_dir.mkdir()

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)

    assert ("/health", ("GET",)) in mcp.routes
    assert ("/ping", ("GET",)) in mcp.routes
    assert ("/web-dashboard", ("GET",)) in mcp.routes
    assert ("/dashboard", ("GET",)) in mcp.routes
    assert mcp.route_order.index(("/health", ("GET",))) < mcp.route_order.index(("/{filename:str}", ("GET",)))
    assert mcp.route_order.index(("/ping", ("GET",))) < mcp.route_order.index(("/{filename:str}", ("GET",)))


def test_health_route_returns_healthy_json(tmp_path):
    from hexstrike_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    health_route = mcp.routes[("/health", ("GET",))]

    with patch("hexstrike_server._build_dashboard_response", return_value={"status": "healthy", "version": "test"}):
        response = run(health_route(MagicMock()))

    assert response.status_code == 200
    assert json.loads(response.body) == {"status": "healthy", "version": "test"}


def test_health_route_returns_500_when_dashboard_state_is_error(tmp_path):
    from hexstrike_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    health_route = mcp.routes[("/health", ("GET",))]

    with patch("hexstrike_server._build_dashboard_response", return_value={"status": "error", "error": "boom"}):
        response = run(health_route(MagicMock()))

    assert response.status_code == 500
    assert json.loads(response.body) == {"status": "error", "error": "boom"}


def test_ping_route_returns_ok(tmp_path):
    from hexstrike_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    ping_route = mcp.routes[("/ping", ("GET",))]
    response = run(ping_route(MagicMock()))

    assert response.status_code == 200
    assert json.loads(response.body) == {"status": "ok", "server": "hexstrike-ai-pulse"}


def test_build_dashboard_status_degraded_when_tools_missing():
    from hexstrike_server import _build_dashboard_response

    no_tools = {tool: False for tools in (
        ["nmap", "curl", "python3", "subfinder", "amass", "httpx", "katana",
         "nikto", "sqlmap", "gobuster", "ffuf", "nuclei",
         "airmon-ng", "airodump-ng", "aircrack-ng",
         "msfconsole", "searchsploit"]
    ) for tool in [tools] if isinstance(tools, list) for t in tools}
    # Simpler: build the full set of all known tools
    all_tools = {
        "nmap": False, "curl": False, "python3": False,
        "subfinder": False, "amass": False, "httpx": False, "katana": False,
        "nikto": False, "sqlmap": False, "gobuster": False, "ffuf": False, "nuclei": False,
        "airmon-ng": False, "airodump-ng": False, "aircrack-ng": False,
        "msfconsole": False, "searchsploit": False,
    }

    with patch("hexstrike_server._get_tool_availability", return_value=all_tools):
        result = _build_dashboard_response()

    assert result["status"] == "degraded"
    assert result["all_essential_tools_available"] is False


def test_build_dashboard_status_healthy_when_all_tools_present():
    from hexstrike_server import _build_dashboard_response

    all_tools_present = {
        "nmap": True, "curl": True, "python3": True,
        "subfinder": True, "amass": True, "httpx": True, "katana": True,
        "nikto": True, "sqlmap": True, "gobuster": True, "ffuf": True, "nuclei": True,
        "airmon-ng": True, "airodump-ng": True, "aircrack-ng": True,
        "msfconsole": True, "searchsploit": True,
    }

    with patch("hexstrike_server._get_tool_availability", return_value=all_tools_present):
        result = _build_dashboard_response()

    assert result["status"] == "healthy"
    assert result["all_essential_tools_available"] is True


def test_health_route_returns_500_when_degraded(tmp_path):
    from hexstrike_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    health_route = mcp.routes[("/health", ("GET",))]

    with patch("hexstrike_server._build_dashboard_response", return_value={"status": "degraded", "all_essential_tools_available": False}):
        response = run(health_route(MagicMock()))

    assert response.status_code == 500
    assert json.loads(response.body)["status"] == "degraded"
