"""
Pulse Dashboard — Section registry and orchestration (data layer).

Each section = one panel with its own data source (get_* function),
estimated token cost, and display rules.

DESIGN: Pure data layer — NO imports from pulse_app.py.
Function references are injected by pulse_app.py at init time
to avoid circular imports.

Usage (in pulse_app.py):
    from server_core.dashboard_sections import SectionConfig, build_registry, auto_detect_sections
    
    _SECTION_REGISTRY = build_registry({
        "header": SectionConfig("header", "HEADER", get_overview, cost_est=500, always=True),
        ...
    })
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Section configuration ────────────────────────────────────────────────────


@dataclass
class SectionConfig:
    """Configuration for a single dashboard section.

    Attributes:
        name: Short identifier (e.g. "header", "surface")
        label: Human-readable label (e.g. "HEADER", "SURFACE")
        data_func: Function to call to load this section's data.
            Signature: (target: str | None = None) -> dict
        cost_est: Estimated token cost when loading this section.
        always: If True, always include in auto-detect.
        depends: Section name that must be relevant for this one to be auto-detected.
        condition: Optional function () -> bool to check if this section has data.
        requires_target: If True, skip when no active target.
    """

    name: str
    label: str
    data_func: Callable[[str | None], dict]
    cost_est: int = 500
    always: bool = False
    depends: str | None = None
    condition: Callable[[], bool] | None = None
    requires_target: bool = False


def build_registry(configs: list[SectionConfig]) -> dict[str, SectionConfig]:
    """Build a section registry dict from a list of SectionConfigs.

    Called by pulse_app.py at module init to inject function references.
    """
    return {c.name: c for c in configs}


# ── Workflow detection helpers (pure logic on _scan_cache) ────────────────────


def cache_for_target(scan_cache: dict, target: str) -> list[dict]:
    """Return all scan cache entries for a given target."""
    result = []
    for v in scan_cache.values():
        if isinstance(v, dict) and v.get("target") == target:
            result.append(v)
    return result


def detect_workflow_state(
    scan_cache: dict, scope_func: Callable[[str | None], dict]
) -> tuple[str | None, str, dict]:
    """Detect current workflow state from scan cache and session data.

    Returns (current_step, next_step, context_dict).
    """
    scope = scope_func(None)
    active = scope.get("active_target") if isinstance(scope, dict) else None
    if not active:
        if len(scan_cache) == 0:
            return None, "overview", {"reason": "No scans have been run yet"}
        return "overview", "scope", {
            "reason": "No active target detected — scans exist but no target selected"
        }

    entries = cache_for_target(scan_cache, active)
    tools_run = {e.get("tool") for e in entries}

    has_nmap = bool(tools_run & {"nmap", "nmap_advanced"})
    has_ports = False
    if has_nmap:
        for e in entries:
            if e.get("tool") in ("nmap", "nmap_advanced"):
                output = str(e.get("result", {}).get("output", "") or "")
                if any("/open" in line for line in output.splitlines()):
                    has_ports = True
                    break

    has_findings = bool(tools_run & {"nuclei", "nikto"})
    has_plan = "plan" in str({e.get("tool") for e in entries}).lower()

    context: dict[str, Any] = {
        "active_target": active,
        "tools_run": sorted(tools_run),
        "tools_count": len(tools_run),
    }

    if not has_nmap:
        return "scope", "surface", {
            **context,
            "reason": f"Target {active} detected but not yet scanned",
        }
    if not has_findings:
        return "surface", "findings", {
            **context,
            "reason": f"Ports detected on {active} — need vulnerability scan",
        }
    if not has_plan:
        return "findings", "plan", {
            **context,
            "reason": f"Vulnerabilities found on {active} — plan exploitation",
        }
    return "plan", "exploit", {
        **context,
        "reason": f"Full recon complete on {active} — ready to exploit",
    }


# ── Section auto-detection ───────────────────────────────────────────────────


def auto_detect_sections(
    registry: dict[str, SectionConfig],
    scan_cache: dict,
    scope_func: Callable[[str | None], dict],
    has_async_scans_fn: Callable[[], bool] = lambda: False,
    has_failures_fn: Callable[[], bool] = lambda: False,
    missing_tools_getter: Callable[[], list] = lambda: [],
) -> list[str]:
    """Return the list of sections relevant to the current session state.

    Does NOT load any data — only inspects the scan cache and session state.
    Uses injected callables for all external state checks.
    """
    state, next_step, ctx = detect_workflow_state(scan_cache, scope_func)

    always = {name for name, c in registry.items() if c.always}
    relevant = set(always)

    target = ctx.get("active_target")

    if state is None and next_step == "overview":
        return [s for s in registry if s in always]

    # Workflow-driven additions
    if target:
        if next_step in ("surface", "findings", "plan", "exploit"):
            relevant.add("surface")
        if next_step in ("findings", "plan", "exploit"):
            relevant.add("findings")
        if next_step in ("plan", "exploit"):
            relevant.add("plan")

    # Condition-driven additions
    conditional_checks = {
        "history": lambda: len(scan_cache) > 0,
        "errors": has_failures_fn,
        "cache": lambda: len(scan_cache) > 0,
        "active": has_async_scans_fn,
        "async": has_async_scans_fn,
        "missing": lambda: len(missing_tools_getter()) > 0,
        "performance": lambda: len(scan_cache) > 5,
    }

    for name, config in registry.items():
        if name in relevant:
            continue
        if config.condition is not None:
            if config.condition():
                relevant.add(name)
        elif name in conditional_checks:
            if conditional_checks[name]():
                relevant.add(name)

    # Stable ordering
    display_order = [
        "header", "scope", "surface", "findings", "plan",
        "history", "active", "async", "errors", "performance",
        "cache", "missing", "intel", "trends", "sessions",
        "confirmations", "netio",
    ]
    return [s for s in display_order if s in relevant]


def load_section(
    name: str, registry: dict[str, SectionConfig], target: str | None = None
) -> dict:
    """Load data for a single section by calling its data function.

    Args:
        name: Section identifier.
        registry: SectionConfig registry (injected).
        target: Optional target to pass to the data function.

    Returns:
        dict with section, label, data, cost_est, load_time_ms.
    """
    config = registry.get(name)
    if config is None:
        return {
            "section": name,
            "error": f"Unknown section: {name}",
            "cost_est": 0,
            "load_time_ms": 0,
        }

    start = time.monotonic()
    try:
        data = config.data_func(target)
        elapsed = time.monotonic() - start
        return {
            "section": name,
            "label": config.label,
            "data": data,
            "cost_est": config.cost_est,
            "load_time_ms": round(elapsed * 1000),
        }
    except Exception as e:
        logger.debug("Failed to load section '%s': %s", name, e)
        return {
            "section": name,
            "label": config.label,
            "error": str(e),
            "cost_est": config.cost_est,
            "load_time_ms": 0,
        }


def cost_for_sections(
    sections: list[str], registry: dict[str, SectionConfig]
) -> dict:
    """Estimate token cost for a list of sections without loading them."""
    per_section = {}
    total = 0
    for name in sections:
        config = registry.get(name)
        est = config.cost_est if config else 500
        per_section[name] = est
        total += est
    return {"total_cost": total, "per_section": per_section}
