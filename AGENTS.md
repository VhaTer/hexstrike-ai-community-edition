# AGENTS.md

## Guidance for agentic coding assistants working in this repository

## Quick orientation

- **Project:** HexStrike AI Community Edition — FastMCP 3.x (Python 3.8+)
- **Primary entrypoints:** `hexstrike_mcp.py` (MCP server), `hexstrike_server.py` (HTTP gateway)
- **Phase:** Phase 2 complete (115 tools, 12 `*_direct.py` modules)
- **Config:** `config.py`, `core/config_core.py`, use `config_core.get(key, default)`
- **Tools registry:** `tool_registry.py` + gateway routing in `mcp_tools/gateway.py`
- **Tests:** 200+ test functions, 48 test classes — comprehensive coverage

## Build, lint, test commands

### Environment setup

```bash
python3 -m venv hexstrike-env
source hexstrike-env/bin/activate
python3 -m pip install -r requirements.txt
```

### Run the server

```bash
python3 hexstrike_server.py
```

### Run the MCP client

```bash
hexstrike-env/bin/python3 hexstrike_mcp.py --server http://localhost:8888
```

### Tests

Comprehensive test coverage (200+ test functions, 48 test classes):

```bash
# Run all tests
hexstrike-env/bin/pytest tests/ -v

# WiFi penetration testing (37 functions, 12 classes)
hexstrike-env/bin/pytest tests/test_wifi_mcp_tools.py -v

# Endpoint/tool coverage (36+ classes: system, network, web, cloud, crypto, etc.)
hexstrike-env/bin/pytest tests/test_endpoints_exist.py -v

# Single test file
hexstrike-env/bin/pytest tests/path/to/test_file.py

# Single test function
hexstrike-env/bin/pytest tests/test_file.py::TestClass::test_function_name
```

**Key test patterns:**
- Use venv pytest: `hexstrike-env/bin/pytest` (not system pytest)
- Patch module-level imports for mocking: `patch("mcp_core.wifi_direct.wifi_exec")`
- Mock context: `ctx = MagicMock()`, `ctx.info = AsyncMock()`, etc.
- All 37/37 WiFi tests passing ✅

### Lint/format

No explicit lint/format configuration was found (no `pyproject.toml`, `setup.cfg`,
or `tox.ini`). Do not introduce auto-format-only changes unless requested.

If you add a linter, follow existing conventions and keep diffs scoped.

## Code style guidelines

### General conventions

- Follow existing patterns and keep changes minimal.
- Avoid large formatting-only diffs.
- Keep PRs small and focused (see `.github/CONTRIBUTING.md`).

### Indentation and whitespace

- `.editorconfig` specifies: spaces, 2-space indent, LF, final newline.
- Respect existing files even if they deviate (avoid reformatting).

### Imports

- Prefer standard ordering: stdlib, third-party, local modules.
- Use explicit imports for project modules (e.g., `from core import ...`).
- Keep import lists stable; do not reorder for style only.

### Typing

- Use type hints on public functions and significant internal APIs.
- Prefer `Dict[str, Any]`, `Optional[T]`, and dataclasses where already used.
- Keep typing consistent with existing style in the file you edit.

### Naming

- Modules/files: `snake_case.py`.
- Classes: `PascalCase`.
- Functions/variables: `snake_case`.
- Constants: `UPPER_SNAKE_CASE`.
- Use descriptive names for tools and endpoints (see `mcp_tools/*`).

### Logging and errors

- Prefer the `logging` module; reuse existing loggers in each module.
- Follow existing patterns for structured error returns, e.g.
  `{"error": "message", "success": False}`.
- When integrating tool execution, surface recovery or escalation info if present.

### API and tool patterns

- Gateway tools are registered via `mcp.tool()` decorators (see `mcp_tools/gateway.py`).
- When calling tools, validate required params and fill defaults like existing logic.
- When adding tools, keep signature docstrings short and consistent.

### FastMCP 3.x Context Integration (Phase 2 Pattern)

All tools use `ctx: Context` for LLM-aware logging and progress reporting:

```python
from fastmcp import FastMCP, Context
from fastmcp.dependencies import CurrentContext
import mcp_core.wifi_direct as _wifi_direct  # module-level import

@mcp.tool(description="Start WiFi monitor mode")
async def airmon_ng(
    ctx: Context = CurrentContext(),
    interface: str = "wlan0",
    action: str = "start"
) -> Dict[str, Any]:
    await ctx.info(f"Starting {action} on {interface}")
    await ctx.report_progress(0, 100)
    
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, lambda: _wifi_direct.wifi_exec("airmon_ng", {...})
    )
    return result
```

**Key principles:**
- Module-level imports enable test mocking: `patch("mcp_core.wifi_direct.wifi_exec")`
- Use `run_in_executor` for blocking tool calls
- Call `ctx.info()` / `ctx.error()` to stream output to LLM
- Call `ctx.report_progress()` for long-running operations
- Always await context methods (they're async)

### Direct Execution Pattern (12 modules, 115 tools)

Backend executors live in `mcp_core/{module}_direct.py`:

```python
def _require(data: dict, *keys) -> dict:
    """Validate required params."""
    for key in keys:
        if not data.get(key, ""):
            return {"success": False, "error": f"Missing: {key}"}
    return {}

def _my_tool(data: dict) -> dict:
    err = _require(data, "target")
    if err: return err
    cmd = f"tool {data['target']}"
    return execute_command(cmd)

_HANDLERS = {"my_tool": _my_tool}

def example_exec(tool: str, data: dict) -> dict:
    handler = _HANDLERS.get(tool)
    if not handler:
        return {"success": False, "error": f"Unknown: {tool}"}
    return handler(data)
```

Gateway routes via `DIRECT_ROUTES` in `mcp_tools/gateway.py`:
```python
DIRECT_ROUTES = {
    "airmon_ng": (wifi_exec, "airmon_ng"),
    "nmap": (net_scan_exec, "nmap"),
    # 110+ tools routed directly (no Flask)
}
```

### Configuration

- Read config values via `core.config_core.get(key, default)`.
- Avoid hard-coding wordlist paths; use `config_core.get_word_list_path(...)`.

### External tools

- Many security tools are external and not installed via pip.
- If you add new tool integrations, document required binaries and parameters.

## Contribution guidelines (from repo docs)

- Keep PRs focused; avoid unrelated formatting changes.
- Prefer PRs under 300 changed lines; avoid exceeding 500 lines.
- Use clear commit messages: `type: short description`.
- Branch naming: `feature/...`, `fix/...`, `docs/...`, `refactor/...`.
- AI assistance is allowed but must be reviewed and tested.

## Repo-specific notes

- **Phase 2 Status:** 115 tools migrated to `*_direct.py` modules, 70+ using `ctx: Context`, 15-20 with progress reporting
- **Direct execution:** Gateway bypasses Flask for 110+ routed tools → massive performance gain
- **Test coverage:** 200+ test functions, 48 test classes (wifi_pentest, endpoints_exist)
- **Server binds to** `127.0.0.1` by default; override via `HEXSTRIKE_HOST`.
- **API token:** set `HEXSTRIKE_API_TOKEN` to enable bearer auth.
- **Default API port:** `HEXSTRIKE_PORT` (default 8888).
- **Command timeouts:** `COMMAND_TIMEOUT` in config.
- **Known:** ANSI colors in output can break JSON in IDE — remove in Phase 3

## When in doubt

- Mirror the style of the file you are editing.
- Keep diffs minimal and avoid global refactors.
- Prefer adding tests when adding new logic, even if the suite is small.
