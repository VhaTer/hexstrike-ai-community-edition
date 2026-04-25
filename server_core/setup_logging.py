import logging
import re
import sys
from shared.colored_formatter import ColoredFormatter

_ANSI_ESCAPE = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
_CLF_ACCESS = re.compile(r' - \[\d{2}/\w+/\d{4} \d{2}:\d{2}:\d{2}\]| -\s*$')

class _StripCLFNoise(logging.Filter):
    """Remove redundant CLF timestamp and trailing dash from Werkzeug access log lines."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        clean = _CLF_ACCESS.sub('', msg)
        record.msg = clean
        record.args = None
        return True

class _PlainFormatter(logging.Formatter):
    """Formatter that strips ANSI escape codes — safe for log files and grep."""

    def format(self, record):
        formatted = super().format(record)
        return _ANSI_ESCAPE.sub('', formatted)


class _SuppressStartupNoise(logging.Filter):
    """Drop low-value framework startup noise from the normal operator console."""

    _NOISE_MARKERS = (
        "Starting worker 'D3V#",
        "* trace(message: str, ...)",
        "* fail(message: str, ...)",
        "* sleep(seconds: float, ...)",
        "StreamableHTTP session manager started",
        "StreamableHTTP session manager shutting down",
        "Starting MCP server 'hexstrike-ai pulse' with transport 'http' on",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(marker in msg for marker in self._NOISE_MARKERS)


class _SuppressMCPProbeAccess(logging.Filter):
    """Hide browser-style GET probes to /mcp; they are noisy and operationally useless."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return '"GET /mcp HTTP/' not in msg

def setup_logging(log_file: str = 'hexstrike.log') -> logging.Logger:
    """Setup enhanced logging: colored console output + ANSI-stripped file output."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicate entries on re-call
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    fmt = ColoredFormatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fmt._stream = console_handler.stream
    console_handler.setFormatter(fmt)
    console_handler.addFilter(_SuppressStartupNoise())
    root.addHandler(console_handler)

    # File handler — plain text, no ANSI codes
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(_PlainFormatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        file_handler.addFilter(_SuppressStartupNoise())
        root.addHandler(file_handler)
    except PermissionError:
        root.warning("Could not open log file %s — logging to console only.", log_file)

    # Strip redundant CLF timestamp and trailing dash from Werkzeug access log lines
    logging.getLogger('werkzeug').addFilter(_StripCLFNoise())
    logging.getLogger('uvicorn.access').addFilter(_SuppressMCPProbeAccess())

    # FastMCP / MCP worker startup chatter uses its own loggers and handlers.
    # Add the same filter at logger-level so the noise is dropped before any
    # formatter/handler emits it.
    for logger_name in (
        'fastmcp.server.mixins.transport',
        'mcp.server.streamable_http_manager',
        'docket.worker',
    ):
        logging.getLogger(logger_name).addFilter(_SuppressStartupNoise())

    # Suppress Werkzeug's startup banner lines (e.g. "Serving Flask app",
    # "Development server" warning, "Running on ...").
    class _SuppressWerkzeugBanner(logging.Filter):
        _BANNER_PREFIXES = ('WARNING: This is a development server',)

        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return not any(msg.__contains__(p) for p in self._BANNER_PREFIXES)

    logging.getLogger('werkzeug').addFilter(_SuppressWerkzeugBanner())

    return root
