import logging

from server_core.setup_logging import _SuppressMCPProbeAccess, _SuppressStartupNoise


def make_record(message: str, logger_name: str = "test") -> logging.LogRecord:
    return logging.LogRecord(
        name=logger_name,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_startup_noise_filter_drops_worker_task_lines():
    noise_filter = _SuppressStartupNoise()

    assert noise_filter.filter(make_record("Starting worker 'D3V#123' with the following tasks:")) is False
    assert noise_filter.filter(make_record("* trace(message: str, ...)")) is False
    assert noise_filter.filter(make_record("* fail(message: str, ...)")) is False
    assert noise_filter.filter(make_record("* sleep(seconds: float, ...)")) is False


def test_startup_noise_filter_keeps_useful_startup_lines():
    noise_filter = _SuppressStartupNoise()

    assert noise_filter.filter(make_record("🚀 Starting HexStrike Pulse Standalone Server")) is True
    assert noise_filter.filter(make_record("📦 Resources MCP registered: health://server, scan://{target}/{tool}")) is True


def test_mcp_probe_access_filter_drops_browser_get_mcp_lines():
    access_filter = _SuppressMCPProbeAccess()

    assert access_filter.filter(make_record('127.0.0.1:56920 - "GET /mcp HTTP/1.1" 404 Not Found', "uvicorn.access")) is False
    assert access_filter.filter(make_record('127.0.0.1:56920 - "GET /health HTTP/1.1" 200 OK', "uvicorn.access")) is True
