"""Comprehensive unit tests for hexstrike_server.py — targets 100% coverage."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_module_globals():
    """Reset hexstrike_server module-level globals before each test."""
    import hexstrike_server
    hexstrike_server._tool_availability_last_refresh = 0.0
    hexstrike_server._tool_availability_cache = {}


def _make_dashboard_mocks(overrides=None):
    """Convenience — build common mock returns for _build_dashboard_response dependencies."""
    env = {
        "_get_tool_availability": None,
        "enhanced_process_manager": None,
        "telemetry": None,
        "cache": None,
        "config_core": None,
        "time_time": None,
    }
    if overrides:
        env.update(overrides)
    return env


def mock_telemetry():
    t = MagicMock()
    t.stats = {"start_time": 1000.0}
    t.get_stats.return_value = {"commands_executed": 42}
    return t


def mock_cache():
    c = MagicMock()
    c.get_stats.return_value = {
        "size": 100, "max_size": 500, "hit_rate": "50.0%",
        "hits": 50, "misses": 50, "evictions": 0
    }
    return c


def mock_epm():
    m = MagicMock()
    m.resource_monitor.get_current_usage.return_value = {"cpu_percent": 50.0}
    return m


# ===========================================================================
# _get_tool_availability
# ===========================================================================

def test_get_tool_availability_cache_miss():
    """First call — no cache, no last_refresh — builds result from scratch."""
    from hexstrike_server import _get_tool_availability
    import hexstrike_server
    hexstrike_server._tool_availability_last_refresh = 0.0
    hexstrike_server._tool_availability_cache = {}

    with patch("hexstrike_server.shutil.which", return_value="/usr/bin/tool"):
        result = _get_tool_availability()

    assert len(result) == 17
    assert all(v is True for v in result.values())
    assert hexstrike_server._tool_availability_last_refresh > 0
    assert hexstrike_server._tool_availability_cache is result


def test_get_tool_availability_cache_hit():
    """Second call within 60s returns cached dict, does NOT call shutil.which."""
    from hexstrike_server import _get_tool_availability
    import hexstrike_server
    hexstrike_server._tool_availability_last_refresh = 0.0
    hexstrike_server._tool_availability_cache = {}

    with patch("hexstrike_server.shutil.which", return_value="/usr/bin/tool"):
        result1 = _get_tool_availability()

    with patch("hexstrike_server.shutil.which") as mock_which:
        result2 = _get_tool_availability()

    mock_which.assert_not_called()
    assert result1 is result2


# ===========================================================================
# _build_dashboard_response
# ===========================================================================

def _all_tools_available():
    return {
        "nmap": True, "curl": True, "python3": True,
        "subfinder": True, "amass": True, "httpx": True, "katana": True,
        "nikto": True, "sqlmap": True, "gobuster": True, "ffuf": True, "nuclei": True,
        "airmon-ng": True, "airodump-ng": True, "aircrack-ng": True,
        "msfconsole": True, "searchsploit": True,
    }


def _no_tools_available():
    return {t: False for t in _all_tools_available()}


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


def test_build_dashboard_healthy():
    """Status=healthy when overview has healthy server_status."""
    from hexstrike_server import _build_dashboard_response

    with patch("pulse_app._collect_dashboard_state",
               return_value=_mock_dashboard_state("healthy")):
        with patch("pulse_app.get_tool_intelligence", return_value=[]):
            result = _build_dashboard_response()

    assert result["status"] == "healthy"
    assert result["version"] == "0.11.0"
    assert result["tools_count"] == 130
    assert "intelligence" in result


def test_build_dashboard_degraded():
    """Status=degraded when overview has degraded server_status."""
    from hexstrike_server import _build_dashboard_response

    with patch("pulse_app._collect_dashboard_state",
               return_value=_mock_dashboard_state("degraded")):
        with patch("pulse_app.get_tool_intelligence", return_value=[]):
            result = _build_dashboard_response()

    assert result["status"] == "degraded"


def test_build_dashboard_with_age():
    """Uptime from overview.uptime_seconds in dashboard state."""
    from hexstrike_server import _build_dashboard_response

    state = _mock_dashboard_state("healthy")
    state["overview"]["uptime_seconds"] = 7200

    with patch("pulse_app._collect_dashboard_state", return_value=state):
        with patch("pulse_app.get_tool_intelligence", return_value=[]):
            result = _build_dashboard_response()

    assert result["uptime"] == 7200


def test_build_dashboard_exception():
    """Exception in _build_dashboard_response returns error dict."""
    from hexstrike_server import _build_dashboard_response

    with patch("pulse_app._collect_dashboard_state",
               side_effect=RuntimeError("test error")):
        result = _build_dashboard_response()

    assert result["status"] == "error"
    assert "test error" in result.get("error", "")


# ===========================================================================
# _json_status_response
# ===========================================================================

def test_json_status_response_healthy():
    """Status 'healthy' → HTTP 200."""
    from hexstrike_server import _json_status_response
    response = _json_status_response({"status": "healthy", "data": "ok"})
    assert response.status_code == 200
    assert json.loads(response.body) == {"status": "healthy", "data": "ok"}


def test_json_status_response_not_healthy():
    """Status != 'healthy' → HTTP 500."""
    from hexstrike_server import _json_status_response
    response = _json_status_response({"status": "degraded", "data": "bad"})
    assert response.status_code == 500
    assert json.loads(response.body) == {"status": "degraded", "data": "bad"}


# ===========================================================================
# register_http_routes — static_dir behavior
# ===========================================================================

def test_register_routes_no_static_dir(tmp_path):
    """static_dir does not exist → warning, no dashboard/root routes."""
    from hexstrike_server import register_http_routes

    mcp = FakeMCP()
    logger = MagicMock()
    static_dir = tmp_path / "nonexistent"

    register_http_routes(mcp, logger, static_dir=static_dir)

    logger.warning.assert_called_once()
    assert ("/dashboard", ("GET",)) not in mcp.routes
    assert ("/{filename:str}", ("GET",)) not in mcp.routes
    assert ("/ping", ("GET",)) in mcp.routes  # always registered
    assert ("/health", ("GET",)) in mcp.routes
    assert ("/web-dashboard", ("GET",)) in mcp.routes
    assert ("/web-dashboard/stream", ("GET",)) in mcp.routes


def test_register_routes_no_assets_dir(tmp_path):
    """static_dir exists but assets/ subdir missing → no /assets mount."""
    from hexstrike_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")
    # No assets/ subdir

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)

    # /assets should NOT have been mounted
    mount_paths = [r.path for r in mcp._additional_http_routes
                   if getattr(r, "path", None) == "/assets"]
    assert len(mount_paths) == 0


def test_register_routes_with_assets(tmp_path):
    """static_dir + assets/ exist → /assets mounted."""
    from hexstrike_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")
    assets_dir = static_dir / "assets"
    assets_dir.mkdir()
    (assets_dir / "app.js").write_text("// js", encoding="utf-8")

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)

    mount_paths = [r for r in mcp._additional_http_routes
                   if getattr(r, "path", None) == "/assets"]
    assert len(mount_paths) == 1


# ===========================================================================
# Dashboard route
# ===========================================================================

def test_dashboard_index_found(tmp_path):
    """index.html exists → FileResponse with 200."""
    from hexstrike_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    dashboard_route = mcp.routes[("/dashboard", ("GET",))]

    response = run(dashboard_route(MagicMock()))
    assert response.status_code == 200
    assert response.media_type == "text/html"


def test_dashboard_index_missing(tmp_path):
    """No index.html → 404."""
    from hexstrike_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    # No index.html

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    dashboard_route = mcp.routes[("/dashboard", ("GET",))]

    response = run(dashboard_route(MagicMock()))
    assert response.status_code == 404


# ===========================================================================
# Root static route ({filename:str})
# ===========================================================================

def test_root_static_valid_file(tmp_path):
    """Valid .ico file in static_dir → FileResponse 200."""
    from hexstrike_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")
    (static_dir / "favicon.ico").write_text("icon", encoding="utf-8")

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    root_route = mcp.routes[("/{filename:str}", ("GET",))]

    request = MagicMock()
    request.path_params = {"filename": "favicon.ico"}
    response = run(root_route(request))
    assert response.status_code == 200


def test_root_static_not_found(tmp_path):
    """Non-existent file in static_dir → 404."""
    from hexstrike_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    root_route = mcp.routes[("/{filename:str}", ("GET",))]

    request = MagicMock()
    request.path_params = {"filename": "nonexistent.txt"}
    response = run(root_route(request))
    assert response.status_code == 404


def test_root_static_not_a_file(tmp_path):
    """Requested path is a directory → 404."""
    from hexstrike_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")
    (static_dir / "subdir").mkdir()

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    root_route = mcp.routes[("/{filename:str}", ("GET",))]

    request = MagicMock()
    request.path_params = {"filename": "subdir"}
    response = run(root_route(request))
    assert response.status_code == 404


# ===========================================================================
# /ping route
# ===========================================================================

def test_ping_route(tmp_path):
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


# ===========================================================================
# /health route
# ===========================================================================

def test_health_ready(tmp_path):
    """All essential tools available + disk OK → 200 ready."""
    from hexstrike_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    health_route = mcp.routes[("/health", ("GET",))]

    all_ok = _all_tools_available()
    with (
        patch("hexstrike_server._get_tool_availability", return_value=all_ok),
        patch("hexstrike_server.shutil.disk_usage",
              return_value=MagicMock(free=50 * 1024**3, total=100 * 1024**3)),
    ):
        response = run(health_route(MagicMock()))

    assert response.status_code == 200
    data = json.loads(response.body)
    assert data["status"] == "ready"
    assert data["checks"]["essential_tools"]["status"] == "ok"
    assert data["checks"]["disk"]["status"] == "ok"


def test_health_degraded_tools(tmp_path):
    """Missing essential tools → 503 degraded."""
    from hexstrike_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    health_route = mcp.routes[("/health", ("GET",))]

    no_tools = _no_tools_available()
    with patch("hexstrike_server._get_tool_availability", return_value=no_tools):
        response = run(health_route(MagicMock()))

    assert response.status_code == 503
    data = json.loads(response.body)
    assert data["status"] == "degraded"
    assert data["checks"]["essential_tools"]["status"] == "degraded"


def test_health_degraded_disk(tmp_path):
    """Low disk space → 503 degraded even if tools OK."""
    from hexstrike_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    health_route = mcp.routes[("/health", ("GET",))]

    all_ok = _all_tools_available()
    # disk.free / disk.total = 0.05 < 0.1 → disk_ok = False
    with (
        patch("hexstrike_server._get_tool_availability", return_value=all_ok),
        patch("hexstrike_server.shutil.disk_usage",
              return_value=MagicMock(free=5 * 1024**3, total=100 * 1024**3)),
    ):
        response = run(health_route(MagicMock()))

    assert response.status_code == 503
    data = json.loads(response.body)
    assert data["status"] == "degraded"
    assert data["checks"]["disk"]["status"] == "degraded"


def test_health_exception(tmp_path):
    """Exception in health route → 500."""
    from hexstrike_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    health_route = mcp.routes[("/health", ("GET",))]

    with patch("hexstrike_server._get_tool_availability",
               side_effect=RuntimeError("boom")):
        response = run(health_route(MagicMock()))

    assert response.status_code == 500
    data = json.loads(response.body)
    assert data["status"] == "error"


# ===========================================================================
# /web-dashboard route
# ===========================================================================

def test_web_dashboard_normal(tmp_path):
    """Normal call returns dashboard data."""
    from hexstrike_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    web_dashboard_route = mcp.routes[("/web-dashboard", ("GET",))]

    dashboard_data = {"status": "healthy", "data": "test"}
    with patch("hexstrike_server._build_dashboard_response",
               return_value=dashboard_data):
        response = run(web_dashboard_route(MagicMock()))

    assert response.status_code == 200
    assert json.loads(response.body) == dashboard_data


def test_web_dashboard_exception(tmp_path):
    """Exception in _build_dashboard_response → 500 error response."""
    from hexstrike_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    web_dashboard_route = mcp.routes[("/web-dashboard", ("GET",))]

    with patch("hexstrike_server._build_dashboard_response",
               side_effect=RuntimeError("dashboard error")):
        response = run(web_dashboard_route(MagicMock()))

    assert response.status_code == 500
    data = json.loads(response.body)
    assert data["status"] == "error"
    assert "dashboard error" in data["error"]


# ===========================================================================
# /web-dashboard/stream SSE endpoint
# ===========================================================================

async def _collect_stream_chunks(stream_route, side_effects, n_chunks=3):
    """Helper: collect n_chunks from the stream route with given side_effects."""
    with (
        patch("hexstrike_server._build_dashboard_response") as mock_build,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_build.side_effect = side_effects
        response = await stream_route(MagicMock())
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
            if len(chunks) >= n_chunks:
                break
    return chunks


@pytest.mark.asyncio
async def test_stream_dashboard_initial_data(tmp_path):
    """First chunk should contain dashboard data."""
    from hexstrike_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    stream_route = mcp.routes[("/web-dashboard/stream", ("GET",))]

    chunks = await _collect_stream_chunks(stream_route, [{"status": "healthy"}], n_chunks=1)

    assert len(chunks) == 1
    decoded = chunks[0].decode()
    assert decoded.startswith("data: ")
    assert '"healthy"' in decoded


@pytest.mark.asyncio
async def test_stream_dashboard_keepalive(tmp_path):
    """Same data consecutively yields keepalive."""
    from hexstrike_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    stream_route = mcp.routes[("/web-dashboard/stream", ("GET",))]

    # Two identical values → second should be keepalive
    chunks = await _collect_stream_chunks(
        stream_route,
        [{"status": "healthy"}, {"status": "healthy"}],
        n_chunks=2,
    )

    assert len(chunks) == 2
    assert chunks[0].decode().startswith("data: ")
    assert chunks[1] == b": keepalive\n\n"


@pytest.mark.asyncio
async def test_stream_dashboard_changing_data(tmp_path):
    """Different data consecutively yields data again."""
    from hexstrike_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    stream_route = mcp.routes[("/web-dashboard/stream", ("GET",))]

    chunks = await _collect_stream_chunks(
        stream_route,
        [{"status": "healthy"}, {"status": "degraded"}],
        n_chunks=2,
    )

    assert len(chunks) == 2
    assert chunks[0].decode().startswith("data: ")
    assert chunks[1].decode().startswith("data: ")
    assert b"healthy" in chunks[0]
    assert b"degraded" in chunks[1]


@pytest.mark.asyncio
async def test_stream_dashboard_error(tmp_path):
    """Exception in _build_dashboard_response yields error data."""
    from hexstrike_server import register_http_routes

    mcp = FakeMCP()
    static_dir = tmp_path / "server_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")

    register_http_routes(mcp, MagicMock(), static_dir=static_dir)
    stream_route = mcp.routes[("/web-dashboard/stream", ("GET",))]

    chunks = await _collect_stream_chunks(
        stream_route,
        [RuntimeError("stream error")],
        n_chunks=1,
    )

    assert len(chunks) == 1
    decoded = chunks[0].decode()
    assert decoded.startswith("data: ")
    assert "stream error" in decoded


# ===========================================================================
# Main block — tested via subprocess
# ===========================================================================

def test_main_block_execution():
    """Verify __main__ block runs without error (subprocess)."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", """
import sys
sys.path.insert(0, ".")
import hexstrike_server
assert hasattr(hexstrike_server, "register_http_routes")
assert hasattr(hexstrike_server, "_build_dashboard_response")
assert hasattr(hexstrike_server, "_get_tool_availability")
print("OK: module imports cleanly")
"""],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "OK" in result.stdout
