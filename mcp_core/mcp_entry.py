"""
mcp_core/mcp_entry.py

Entry point for stdio transport used by Claude Desktop
and other MCP clients that speak stdio.

hexstrike_mcp.py calls run_mcp().
"""

import os
import sys
import time
import threading
import logging
import fcntl
import atexit
from mcp_core.server_setup import setup_mcp_server_standalone, _scan_cache

logger = logging.getLogger(__name__)

_LOCK_PATH = "/tmp/hexstrike_mcp.lock"
_LOCK_TTL = 5
_lock_fh = None


def _seed_scan_cache(logger):
    """Populate _scan_cache with demo scan data when HEXSTRIKE_SEED_SCANS is set.

    The target is read from the env var (e.g. "scanme.nmap.org").
    Uses real CLI scan results collected during Session 30 testing.
    """
    seed_target = os.environ.get("HEXSTRIKE_SEED_SCANS", "").strip()
    if not seed_target:
        return
    logger.debug("🌱 Seeding _scan_cache for %s ...", seed_target)

    def _add(tool, stdout, exec_time, success=True, stderr=""):
        now = time.time()
        entry = {
            "tool": tool, "target": seed_target, "timestamp": now,
            "result": {
                "success": success, "stdout": stdout, "stderr": stderr,
                "output": stdout, "error": "", "returncode": 0,
                "execution_time": exec_time, "timed_out": False,
                "partial_results": False,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now)),
            },
        }
        _scan_cache.set(f"seed:{tool}:{seed_target}", entry, execution_time=exec_time)
        from server_core.operational_metrics import _op_metrics
        _op_metrics.record({"tool": tool, "success": success,
                            "duration": exec_time, "timed_out": False, "cache_hit": False})

    _add("nmap", """Starting Nmap 7.98 ( https://nmap.org ) at 2026-05-16 11:22 +0200
Nmap scan report for scanme.nmap.org (45.33.32.156)
Host is up.

PORT    STATE    SERVICE VERSION
22/tcp  filtered ssh
80/tcp  filtered http
443/tcp filtered https

Nmap done: 1 IP address (1 host up) scanned in 3.48 seconds
""", exec_time=3.63)

    _add("whatweb", """http://scanme.nmap.org [200 OK] Apache[2.4.7], Country[RESERVED][ZZ],
  Google-Analytics[Universal] [UA-11009417-1], HTML5,
  HTTPServer[Ubuntu Linux][Apache/2.4.7 (Ubuntu)], IP[45.33.32.156],
  Script, Title[Go ahead and ScanMe!]
""", exec_time=11.98, stderr="ERROR Opening: https://scanme.nmap.org - Connection refused")

    _add("nuclei", """[INF] Scan completed in 45.70s. 0 matches found.
""", exec_time=45.7)

    _add("nikto", """Nikto scan - target not reachable, request timed out after 180s.
""", exec_time=180.0, success=False, stderr="Connection timeout")

    logger.debug("✅ Seeded _scan_cache with 4 entries for %s", seed_target)


def _prewarm_singletons(logger):
    """Pre-initialize heavy singletons in background thread.

    Avoids first-access latency (~50-200ms for decision engine,
    ~5-20ms for tool stats store) when the user issues their first
    dashboard or planning request.
    """
    def _warm():
        logger.debug("🔍 Pre-warming lazy singletons...")
        try:
            from server_core.singletons import get_decision_engine, get_tool_stats_store
            eng = get_decision_engine()
            eng._get_parameter_optimizer()
            get_tool_stats_store()
            logger.debug("✅ Pre-warming complete")
        except Exception as exc:
            logger.debug("⚠️ Pre-warming skipped: %s", exc)

    threading.Thread(target=_warm, daemon=True, name="prewarm").start()


def _read_pid_from_lock() -> int | None:
    """Read PID from existing lock file. Returns None if missing/corrupt/empty."""
    try:
        with open(_LOCK_PATH, "r") as f:
            content = f.read().strip()
            if content and content.isdigit():
                return int(content)
    except (FileNotFoundError, ValueError, OSError):
        pass
    return None


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running (Linux /proc)."""
    try:
        return os.path.isdir(f"/proc/{pid}")
    except Exception:
        return False


def _acquire_lock(logger):
    """Prevent duplicate MCP instances (fcntl + PID combined — like nginx/postgres).

    Strategy:
    1. Read existing lock file, check /proc/<PID> for alive instance.
    2. If alive → exit 1 (duplicate). If dead → kernel already released fcntl lock.
    3. TTL safety net: remove stale file >5s.
    4. Open file, acquire fcntl.flock() (auto-released by kernel on crash/SIGKILL).
    5. Write own PID to file (truncated first = clean shutdown signal).
    """
    global _lock_fh

    # 1. PID health check — detect alive instances without relying on fcntl
    existing_pid = _read_pid_from_lock()
    if existing_pid is not None and _is_pid_alive(existing_pid):
        logger.error(f"💥 Another HexStrike MCP instance is running (PID {existing_pid}). Exiting.")
        sys.exit(1)

    # 2. TTL safety net for edge cases (PID reuse, /proc unavailable, etc.)
    try:
        age = time.time() - os.path.getmtime(_LOCK_PATH)
        if age > _LOCK_TTL:
            os.unlink(_LOCK_PATH)
    except (FileNotFoundError, OSError):
        pass

    # 3. Open file and acquire fcntl lock (auto-released by kernel on crash)
    try:
        _lock_fh = open(_LOCK_PATH, "w")
        fcntl.flock(_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        logger.error("💥 Another HexStrike MCP instance is already running (fcntl conflict). Exiting.")
        sys.exit(1)

    # 4. Write PID — truncate first (= clean shutdown marker: empty file)
    _lock_fh.truncate(0)
    _lock_fh.write(str(os.getpid()))
    _lock_fh.flush()

    atexit.register(_release_lock)


def _release_lock():
    """Release MCP lock. Registered as atexit handler.

    Truncates file (signals clean shutdown → empty file = intentional exit).
    Releases fcntl lock and removes lock file.
    """
    global _lock_fh
    if _lock_fh is not None and not _lock_fh.closed:
        try:
            _lock_fh.truncate(0)
            _lock_fh.flush()
            fcntl.flock(_lock_fh, fcntl.LOCK_UN)
            _lock_fh.close()
        except Exception:
            logger.warning("mcp_entry: failed to release lock file", exc_info=True)
    try:
        os.unlink(_LOCK_PATH)
    except FileNotFoundError:
        pass


def run_mcp(args, logger):
    """Run the HexStrike MCP server with configurable transport."""
    transport = getattr(args, 'transport', 'stdio')

    # Acquire lock only for stdio (prevents duplicate Claude Desktop instances)
    if transport == 'stdio':
        _acquire_lock(logger)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("🔍 Debug logging enabled")

    logger.info("🚀 Starting HexStrike AI-PULSE MCP server (standalone)")

    try:
        mcp = setup_mcp_server_standalone(logger)
        logger.info("✅ HexStrike AI-PULSE MCP server ready")

        # Seed scan cache with demo data if HEXSTRIKE_SEED_SCANS is set
        _seed_scan_cache(logger)

        # Pre-warm heavy singletons in background so the first user
        # request doesn't block on lazy initialization.
        _prewarm_singletons(logger)

        if transport == 'http':
            host = getattr(args, 'host', '127.0.0.1')
            port = getattr(args, 'port', 8888)
            from hexstrike_server import register_http_routes
            register_http_routes(mcp, logger)
            logger.info(f"🌐 HTTP transport on {host}:{port} — MCP SSE at /mcp")
            mcp.run(transport="http", host=host, port=port, show_banner=False, log_level="WARNING", stateless=True)
        else:
            # stdio transport for Claude Desktop / MCP clients
            mcp.run(show_banner=False, log_level="WARNING")

    except Exception as e:
        logger.error(f"💥 Error starting MCP server: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
