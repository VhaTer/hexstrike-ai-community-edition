# Known Issues

## FastMCP 3.2.4 — `get_prompt()` kwargs passed as `version` parameter

**Bug**: `mcp.get_prompt(name, arguments)` and `local_provider.get_prompt(name, arguments)` pass the arguments dict as the `version` parameter of `_get_prompt()`, which expects `VersionSpec | None`. The internal call at `local_provider.py:440` fails with:

```
'dict' object has no attribute 'matches'
```

**Impact**: Can't test prompts via FastMCP server API. Prompts appear in `list_prompts()` but `get_prompt()` returns `None` for every prompt.

**Workaround** (applied S59): Define prompt functions at module level (not as closures inside `register_prompts()`), test them directly by importing and calling with `await fn(**kwargs)`. `register_prompts()` wraps them with `mcp.prompt(name=...)(fn)` for server registration.

**Detection date**: 2026-05-28 (Session 59)
**Status**: Upstream bug, no fix available yet.
