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


def _mock_dashboard_state(server_status="healthy", version="0.11.0"):
    """Return a mock _collect_dashboard_state() result matching the Prefab pipeline."""
    return {
        "overview": {
            "server_status": server_status,
            "version": version,
            "uptime_seconds": 3600,
            "tools_count": 130,
            "memory": "3.2/7.8 GB",
            "uptime": "1h 0m",
        },
        "scope": {"active_target": None, "target_type": None},
        "surface": {},
        "findings": [],
        "plan": {},
        "active": {},
        "history": [],
        "rl": {},
        "err": {},
        "perf": {},
        "cache_status": {},
        "trends": {},
        "sessions": {},
        "confirmations": {},
        "netio": {},
        "async_scans_summary": "",
        "next_suggested_tool": {},
    }


def test_resolve_static_dir_defaults_to_repo_frontend():
    from pulse.server.web_server import _resolve_static_dir

    resolved = _resolve_static_dir()
    assert resolved.name == "server_static"
    assert (resolved / "index.html").exists()


def test_register_http_routes_adds_health_and_ping(tmp_path):
    from pulse.server.web_server import register_http_routes

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


def test_health_route_returns_ready_when_all_ok(tmp_path):
    from pulse.server.web_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    health_route = mcp.routes[("/health", ("GET",))]

    all_tools_ok = {
        "nmap": True, "curl": True, "python3": True,
        "subfinder": True, "amass": True, "httpx": True, "katana": True,
        "nikto": True, "sqlmap": True, "gobuster": True, "ffuf": True, "nuclei": True,
        "airmon-ng": True, "airodump-ng": True, "aircrack-ng": True,
        "msfconsole": True, "searchsploit": True,
    }

    with (
        patch("pulse.server.web_server._get_tool_availability", return_value=all_tools_ok),
        patch("pulse.server.web_server.shutil.disk_usage", return_value=MagicMock(free=50 * 1024**3, total=100 * 1024**3)),
    ):
        response = run(health_route(MagicMock()))

    assert response.status_code == 200
    data = json.loads(response.body)
    assert data["status"] == "ready"
    assert data["checks"]["essential_tools"]["status"] == "ok"
    assert data["checks"]["disk"]["status"] == "ok"


def test_health_route_returns_503_when_degraded(tmp_path):
    from pulse.server.web_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    health_route = mcp.routes[("/health", ("GET",))]

    no_tools = {
        "nmap": False, "curl": False, "python3": False,
        "subfinder": False, "amass": False, "httpx": False, "katana": False,
        "nikto": False, "sqlmap": False, "gobuster": False, "ffuf": False, "nuclei": False,
        "airmon-ng": False, "airodump-ng": False, "aircrack-ng": False,
        "msfconsole": False, "searchsploit": False,
    }

    with patch("pulse.server.web_server._get_tool_availability", return_value=no_tools):
        response = run(health_route(MagicMock()))

    assert response.status_code == 503
    data = json.loads(response.body)
    assert data["status"] == "degraded"
    assert data["checks"]["essential_tools"]["status"] == "degraded"


def test_health_route_returns_500_on_exception(tmp_path):
    from pulse.server.web_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    health_route = mcp.routes[("/health", ("GET",))]

    with patch("pulse.server.web_server._get_tool_availability", side_effect=RuntimeError("boom")):
        response = run(health_route(MagicMock()))

    assert response.status_code == 500
    data = json.loads(response.body)
    assert data["status"] == "error"


def test_ping_route_returns_ok(tmp_path):
    from pulse.server.web_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    ping_route = mcp.routes[("/ping", ("GET",))]
    response = run(ping_route(MagicMock()))

    assert response.status_code == 200
    assert json.loads(response.body) == {"status": "ok", "server": "hexstrike-ai-pulse"}


def test_root_static_rejects_path_traversal(tmp_path):
    from pulse.server.web_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("top-secret", encoding="utf-8")

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    static_route = mcp.routes[("/{filename:str}", ("GET",))]

    request = MagicMock(path_params={"filename": "../secret.txt"})
    response = run(static_route(request))
    assert response.status_code == 404

    (static_dir / "ok.txt").write_text("fine", encoding="utf-8")
    request_ok = MagicMock(path_params={"filename": "ok.txt"})
    response_ok = run(static_route(request_ok))
    assert response_ok.status_code == 200


def test_build_dashboard_status_degraded_when_tools_missing():
    from pulse.server.web_server import _build_dashboard_response

    with patch("pulse.interface.pulse_app._collect_dashboard_state",
               return_value=_mock_dashboard_state("degraded")):
        with patch("pulse.interface.pulse_app.get_tool_intelligence", return_value=[]):
            result = _build_dashboard_response()

    assert result["status"] == "degraded"


def test_build_dashboard_status_healthy_when_all_tools_present():
    from pulse.server.web_server import _build_dashboard_response

    with patch("pulse.interface.pulse_app._collect_dashboard_state",
               return_value=_mock_dashboard_state("healthy")):
        with patch("pulse.interface.pulse_app.get_tool_intelligence", return_value=[]):
            result = _build_dashboard_response()

    assert result["status"] == "healthy"



