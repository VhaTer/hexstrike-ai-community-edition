"""
mcp_core/ctf_engine.py

HexStrike AI-PULSE — CTF Intelligence Engine

Bridges the V6 CTF intelligence layer (CTFWorkflowManager, CTFToolManager,
CTFChallengeAutomator, CTFTeamCoordinator) with the FastMCP 3.x native
execution stack (run_security_tool via *_direct.py modules).

This is the missing link in the V6 → Pulse migration:
  V6:    CTFChallengeAutomator → simulated execute_command (Flask stubs)
  Pulse: CTFChallengeAutomator → run_security_tool_direct() → *_direct.py → real subprocess

Exposed as MCP tools via register_ctf_tools(mcp):
  - ctf_solve    : Auto-solve a CTF challenge (all categories)
  - ctf_team     : Team strategy optimizer for CTF competitions
  - ctf_tools    : Get tool arsenal and optimized commands for a category
  - ctf_analyze  : Analyze a challenge and return workflow + strategies

The key design principle: V6 classes are kept intact.
This module adds only the execution bridge and MCP registration.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import json
import os
import re

from fastmcp import FastMCP, Context
from fastmcp.server.dependencies import get_context

from server_core.workflows.ctf.CTFChallenge import CTFChallenge
from server_core.workflows.ctf.toolManager import CTFToolManager
from server_core.singletons import get_ctf_manager, get_ctf_tools, get_ctf_automator, get_ctf_coordinator
from mcp_core.server_setup import _scan_cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Execution bridge — connects CTFChallengeAutomator to run_security_tool()
# ---------------------------------------------------------------------------

async def _execute_ctf_step_real(
    step: Dict[str, Any],
    challenge: CTFChallenge,
    ctx: Context,
) -> Dict[str, Any]:
    """
    Execute a single CTF workflow step using the real HexStrike execution stack.

    Replaces CTFChallengeAutomator._execute_parallel_step() and
    _execute_sequential_step() which only simulated execution.

    For tools available in DIRECT_TOOLS: calls run_security_tool() via
    asyncio executor → execute_command() → real subprocess.

    For manual/custom steps: returns structured guidance without execution.
    """
    step_result = {
        "step":           step.get("step", 0),
        "action":         step.get("action", "unknown"),
        "description":    step.get("description", ""),
        "success":        False,
        "output":         "",
        "tools_executed": [],
        "tools_skipped":  [],
        "execution_time": 0.0,
        "flag_candidates": [],
        "artifacts":      [],
    }

    import time
    t0 = time.perf_counter()

    tools = step.get("tools", [])
    is_parallel = step.get("parallel", False)
    target = challenge.target or challenge.url or challenge.name

    # Filter executable tools vs manual/custom guidance steps
    executable = [t for t in tools if t not in ("manual", "custom", "python", "sage",
                                                  "ida", "x64dbg", "ollydbg", "burpsuite",
                                                  "wireshark", "audacity", "maltego")]
    manual_tools = [t for t in tools if t not in executable]

    step_result["tools_skipped"] = manual_tools

    if not executable:
        # Pure manual step — return structured guidance
        step_result["success"] = True
        step_result["output"] = f"[MANUAL] {step['description']}"
        step_result["execution_time"] = round(time.perf_counter() - t0, 3)
        return step_result

    # Coroutine factory for a single tool call
    async def _run_tool(tool_name: str) -> Dict[str, Any]:
        try:
            from mcp_core.server_setup import get_direct_tools, _normalize_tool_result
            import asyncio as _asyncio
            direct_tools = get_direct_tools()
            route = direct_tools.get(tool_name.lower())
            if not route:
                return {"tool": tool_name, "result": {}, "success": False,
                        "error": f"Tool not in DIRECT_TOOLS: {tool_name}"}
            exec_func, tool_key = route
            params = {"target": target}
            loop = _asyncio.get_running_loop()
            raw = await loop.run_in_executor(None, lambda: exec_func(tool_key, params))
            result = _normalize_tool_result(raw)
            return {"tool": tool_name, "result": result, "success": result.get("success", False)}
        except Exception as exc:
            logger.warning("CTF engine: %s failed: %s", tool_name, exc)
            return {"tool": tool_name, "result": {}, "success": False, "error": str(exc)}

    # Parallel vs sequential execution
    if is_parallel and len(executable) > 1:
        await ctx.info(f"⚡ [{step['action']}] running {len(executable)} tools in parallel")
        tasks = [_run_tool(t) for t in executable]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    else:
        results = []
        for t in executable:
            res = await _run_tool(t)
            results.append(res)

    # Aggregate results
    combined_output = []
    any_success = False

    for r in results:
        if isinstance(r, Exception):
            combined_output.append(f"[ERROR] {r}")
            continue
        tool = r.get("tool", "?")
        result = r.get("result", {})
        success = r.get("success", False)

        if success:
            any_success = True
            step_result["tools_executed"].append(tool)
            output = result.get("output", "")
            if output:
                combined_output.append(f"[{tool}]\n{output[:2000]}")
        else:
            err = r.get("error", result.get("error", "failed"))
            combined_output.append(f"[{tool}] ✗ {err[:200]}")

    step_result["success"] = any_success
    step_result["output"] = "\n\n".join(combined_output)
    step_result["execution_time"] = round(time.perf_counter() - t0, 3)

    # Extract flag candidates from aggregated output
    import re
    flag_patterns = [
        r'flag\{[^}]+\}',
        r'FLAG\{[^}]+\}',
        r'ctf\{[^}]+\}',
        r'CTF\{[^}]+\}',
        r'[a-zA-Z0-9_]+\{[^}]{3,50}\}',
        r'[0-9a-f]{32}',
        r'[0-9a-f]{40}',
        r'[0-9a-f]{64}',
    ]
    candidates = set()
    for pattern in flag_patterns:
        candidates.update(re.findall(pattern, step_result["output"], re.IGNORECASE))
    step_result["flag_candidates"] = list(candidates)

    return step_result


# ---------------------------------------------------------------------------
# Binary triage — file + checksec + strings in parallel
# ---------------------------------------------------------------------------

async def _triage_binary(file_path: str) -> Dict[str, Any]:
    """
    Run binary triage: file type, checksec protections, strings analysis.

    Returns a dict with binary characteristics:
      binary_type, architecture, stripped, not_stripped, debug_info,
      protections{canary,nx,pie,relro}, flags_in_strings,
      interesting_keywords[], raw{file, checksec, strings}.
    """
    result = {
        "binary_type": "",
        "architecture": "",
        "stripped": None,
        "not_stripped": None,
        "debug_info": None,
        "protections": {},
        "flags_in_strings": False,
        "flag_value": None,
        "interesting_keywords": [],
        "raw": {},
    }

    if not file_path or not os.path.isfile(file_path):
        result["error"] = f"File not found: {file_path}"
        return result

    async def _run(cmd: str) -> str:
        loop = asyncio.get_running_loop()
        try:
            proc = await loop.run_in_executor(
                None, lambda: __import__("subprocess").run(
                    cmd, shell=True, capture_output=True, text=True, timeout=15
                )
            )
            return proc.stdout.strip()
        except Exception as exc:
            logger.warning("Triage subprocess failed: %s", exc)
            return ""

    file_out, strings_out = await asyncio.gather(
        _run(f"file {file_path}"),
        _run(f"strings -n 4 {file_path}"),
    )

    result["raw"]["file"] = file_out
    result["raw"]["strings"] = strings_out[:2000] if strings_out else ""

    # Parse file output
    if file_out:
        result["binary_type"] = file_out
        arch_match = re.search(r"(x86-?64|i386|ARM\b|AArch64|MIPS)", file_out)
        if arch_match:
            result["architecture"] = arch_match.group(1)
        result["stripped"] = "not stripped" not in file_out
        result["not_stripped"] = "not stripped" in file_out

    # Parse checksec output — try JSON first, fallback to text
    loop = asyncio.get_running_loop()
    prots = {}
    checksec_raw = ""
    try:
        proc = await loop.run_in_executor(
            None, lambda: __import__("subprocess").run(
                f"checksec --file={file_path} --format=json",
                shell=True, capture_output=True, text=True, timeout=15,
            )
        )
        checksec_raw = proc.stdout.strip()
        if checksec_raw:
            data = json.loads(checksec_raw)
            if isinstance(data, dict):
                prots = {k.lower(): v for k, v in data.items()
                         if k.lower() in ("canary", "nx", "pie", "relro", "rpath", "fortified")}
            elif isinstance(data, list) and data:
                prots = {k.lower(): v for k, v in data[0].items()
                         if k.lower() in ("canary", "nx", "pie", "relro", "rpath", "fortified")}
    except Exception:
        prots = {}

    if not prots:
        try:
            proc = await loop.run_in_executor(
                None, lambda: __import__("subprocess").run(
                    f"checksec --file={file_path}",
                    shell=True, capture_output=True, text=True, timeout=15,
                )
            )
            checksec_raw = proc.stdout.strip()
            if checksec_raw:
                for line in checksec_raw.split("\n"):
                    line = line.strip().lower()
                    if "canary" in line:
                        prots["canary"] = "yes" in line or "true" in line
                    elif "nx" in line:
                        prots["nx"] = "yes" in line or "true" in line
                    elif "pie" in line:
                        prots["pie"] = "yes" in line or "true" in line
                    elif "relro" in line:
                        if "full" in line:
                            prots["relro"] = "full"
                        elif "partial" in line:
                            prots["relro"] = "partial"
                        else:
                            prots["relro"] = "no"
                    elif "rpath" in line:
                        prots["rpath"] = "yes" in line
                    elif "fortified" in line:
                        prots["fortified"] = "yes" in line or "true" in line
        except Exception:
            pass

    result["raw"]["checksec"] = checksec_raw
    result["protections"] = prots

    # Parse strings output for flags and keywords
    if strings_out:
        flag_match = re.search(r"HTB\{[^}]+\}|flag\{[^}]+\}|FLAG\{[^}]+\}|CTF\{[^}]+\}", strings_out)
        if flag_match:
            result["flags_in_strings"] = True
            result["flag_value"] = flag_match.group(0)

        keywords = [w for w in ("HTB", "flag", "win", "shell", "admin",
                                 "password", "secret", "key", "root")
                    if w.lower() in strings_out.lower()]
        result["interesting_keywords"] = keywords

    # Seed _scan_cache with triage results for future ctf_analyze/get_plan calls
    try:
        import time
        import uuid
        from mcp_core.server_setup import _scan_cache, _cache_key_for
        session_id = f"ctf_triage_{uuid.uuid4().hex[:8]}"
        for tool_name, raw_output in (("file", result["raw"].get("file", "")),
                                       ("checksec", result["raw"].get("checksec", "")),
                                       ("strings", result["raw"].get("strings", ""))):
            if raw_output:
                cache_key = _cache_key_for(session_id, tool_name, file_path, {})
                _scan_cache.set(cache_key, {
                    "tool": tool_name,
                    "target": file_path,
                    "result": {"success": True, "output": raw_output, "stdout": raw_output, "stderr": "", "returncode": 0},
                    "timestamp": time.time(),
                    "execution_time": 0.1
                }, ttl=7200)
    except Exception:
        logger.debug("Failed to seed _scan_cache with triage results", exc_info=True)

    return result


# ---------------------------------------------------------------------------
# Workflow refinement — adjust plan based on triage data
# ---------------------------------------------------------------------------

def _refine_workflow_with_triage(
    workflow: Dict[str, Any],
    triage: Dict[str, Any],
) -> Dict[str, Any]:
    """Adjust the workflow plan based on binary triage results."""

    refined = dict(workflow)
    refined["triage_refined"] = True
    refined["triage_notes"] = []

    # Rule 1: flag already in strings → immediate return
    if triage.get("flags_in_strings") and triage.get("flag_value"):
        flag = triage["flag_value"]
        refined["estimated_time"] = 60
        refined["success_probability"] = 1.0
        refined["triage_notes"].append(f"Flag found in strings: {flag}")
        refined["workflow_steps"] = [{
            "step": 1, "action": "flag_immediate",
            "description": f"Flag already visible in binary strings: {flag}",
            "parallel": False, "tools": ["strings"],
            "estimated_time": 10,
        }]
        return refined

    stripped = triage.get("stripped")
    not_stripped = triage.get("not_stripped")
    prots = triage.get("protections", {})

    # Rule 2: not stripped + has debug info → remove ghidra/ida, reduce time
    heavy_re = re.compile(r"\b(ghidra|ida\b|radare2)\b", re.IGNORECASE)
    heavy_tools = {"ghidra", "ida", "radare2"}

    if not_stripped:
        refined["triage_notes"].append(
            "Binary not stripped — ghidra/ida removed, estimated time reduced"
        )
        # Remove heavy RE tools from workflow_steps
        new_steps = []
        for step in workflow.get("workflow_steps", []):
            step_tools = step.get("tools", [])
            filtered = [t for t in step_tools if t.lower() not in heavy_tools]
            if not filtered:
                continue
            step["tools"] = filtered
            step["estimated_time"] = max(60, int(step.get("estimated_time", 300) * 0.3))
            new_steps.append(step)
        refined["workflow_steps"] = new_steps or [{
            "step": 1, "action": "binary_analysis",
            "description": "Binary analysis (not stripped, ghidra skipped)",
            "parallel": True,
            "tools": ["strings", "objdump", "checksec"],
            "estimated_time": 180,
        }]
        # Reduce overall estimate
        base_time = workflow.get("estimated_time", 3600)
        refined["estimated_time"] = max(120, int(base_time * 0.3))
        refined["success_probability"] = min(
            0.95, workflow.get("success_probability", 0.5) + 0.25
        )

    # Rule 3: stripped + PIE + canary → suggest gdb/ltrace before ghidra
    elif stripped and prots.get("pie") and prots.get("canary"):
        refined["triage_notes"].append(
            "Stripped + PIE + canary — suggesting gdb/ltrace before heavy RE"
        )
        # Insert a lightweight step before heavy RE
        existing = workflow.get("workflow_steps", [])
        lightweight = {
            "step": 1, "action": "lightweight_binary_recon",
            "description": "Lightweight recon before heavy RE: gdb, ltrace, execute_code extraction",
            "parallel": True,
            "tools": ["gdb", "ltrace", "strings", "objdump"],
            "estimated_time": 300,
        }
        refined["workflow_steps"] = [lightweight] + [
            {**s, "step": s["step"] + 1} for s in existing
        ]
        refined["success_probability"] = min(
            0.95, workflow.get("success_probability", 0.5) + 0.1
        )

    # Rule 4: fewer protections → higher success proba
    weak_count = sum(
        1 for k in ("canary", "nx", "pie") if prots.get(k) is False
    )
    if weak_count >= 2:
        refined["success_probability"] = min(
            0.95, refined.get("success_probability", 0.5) + 0.2
        )
        refined["triage_notes"].append(
            f"{weak_count} protections absent — success probability boosted"
        )

    # Rule 5: ELF binary → remove PE-specific tools
    if "ELF" in triage.get("binary_type", ""):
        pe_tools = {"peid", "upx", "pe-sieve", "pedumper"}
        new_steps = []
        for step in refined.get("workflow_steps", []):
            step["tools"] = [t for t in step.get("tools", [])
                             if t.lower() not in pe_tools]
            new_steps.append(step)
        refined["workflow_steps"] = new_steps

    return refined


# ---------------------------------------------------------------------------
# MCP tool registration
# ---------------------------------------------------------------------------

def register_ctf_tools(mcp: FastMCP) -> None:
    """Register CTF intelligence tools on the FastMCP server."""

    @mcp.tool(
        name="ctf_analyze",
        description=(
            "Analyze a CTF challenge and return a complete workflow plan: "
            "tool selection, strategies, time estimate, success probability, "
            "resource requirements, parallel task groups, and validation steps. "
            "Uses the V6 CTFWorkflowManager intelligence layer. "
            "Does NOT execute tools — use ctf_solve for real execution. "
            "Provide file_path for binary challenges (rev/pwn) to get triage-based refinement."
        ),
        timeout=30.0,
    )
    async def ctf_analyze(
        name: str,
        category: str,
        description: str,
        difficulty: str = "unknown",
        target: str = "",
        points: int = 0,
        file_path: str = "",
    ) -> Dict[str, Any]:
        """
        Analyze a CTF challenge and generate a full workflow plan with tool selection.

        Call FIRST when starting a CTF challenge — before scanning or solving.
        Uses CTFWorkflowManager to select tools and estimate time/success probability
        based on challenge category, description, and difficulty.
        When file_path is provided (rev/pwn categories), runs binary triage
        (file, checksec, strings) and refines the workflow accordingly.

        Returns: success, challenge name, category, difficulty, triage (if file_path),
        triage_refined (bool), workflow { tools[], workflow_steps[][],
        estimated_time, success_probability, parallel_groups, fallback_strategies }

        Args:
            name:        Challenge name (e.g. 'Don't Panic')
            category:    web | crypto | pwn | forensics | rev | misc | osint
            description: Challenge description text — drives keyword-based tool selection
            difficulty:  easy | medium | hard | insane | unknown
            target:      URL, IP, or binary path (optional, can be set later via scan)
            points:      Point value (optional, for scoring)
            file_path:   Path to the challenge binary (rev/pwn only). Triggers triage + refinement.

        Do NOT use for non-CTF targets — use scan() + get_plan() instead.
        Do NOT skip this step — ctf_analyze() is required before ctf_solve() for optimal results.
        For rev/pwn challenges, always provide file_path for triage-based accuracy.

        Example: ctf_analyze('Don\\'t Panic', 'pwn', 'Reverse a Rust binary...', 'medium', file_path='./dontpanic')
        Example: ctf_analyze('Web100', 'web', 'SQL injection...', 'easy', 'http://target:8080')
        Next: scan(target) for recon, then ctf_solve(name, ...) for exploitation
        """
        ctx = get_context()
        await ctx.info(f"🧠 CTFWorkflowManager analyzing [{category.upper()}] {name}")

        challenge = CTFChallenge(
            name=name,
            category=category,
            description=description,
            difficulty=difficulty,
            target=target,
            points=points,
        )

        ctf = get_ctf_manager()
        workflow = ctf.create_ctf_challenge_workflow(challenge)

        result = {
            "success":     True,
            "challenge":   name,
            "category":    category,
            "difficulty":  difficulty,
            "workflow":    workflow,
        }

        # Binary triage if file_path provided
        triage_result = {}
        triage_refined = False

        # If no explicit file_path, check for cached binary triage in _scan_cache
        if not file_path:
            target = challenge.target or challenge.url or challenge.name
            if target and category in ("rev", "pwn"):
                cached_triage = None
                for k, v in _scan_cache.items():
                    if v.get("target") == target and v.get("tool") in ("file", "checksec", "strings", "binary_triage"):
                        cached_triage = v.get("result", {})
                        break
                if cached_triage:
                    await ctx.info(f"🔍 Using cached binary triage for {target}")
                    workflow = _refine_workflow_with_triage(workflow, cached_triage)
                    triage_refined = True
                    result["triage"] = cached_triage
                    result["triage_refined"] = True
                    result["workflow"] = workflow

        if file_path:
            await ctx.info(f"🔍 Running binary triage on: {file_path}")
            triage_result = await _triage_binary(file_path)

            if triage_result.get("flags_in_strings") and triage_result.get("flag_value"):
                workflow = _refine_workflow_with_triage(workflow, triage_result)
                triage_refined = True
                await ctx.info(
                    f"🚩 Flag found in strings: {triage_result['flag_value']} — "
                    f"returning immediate solve"
                )
            elif triage_result.get("not_stripped"):
                workflow = _refine_workflow_with_triage(workflow, triage_result)
                triage_refined = True
                await ctx.info(
                    f"✅ Binary not stripped — workflow refined, "
                    f"estimated time reduced to ~{workflow.get('estimated_time', 0)//60} min"
                )
            elif triage_result.get("stripped") and \
                 triage_result.get("protections", {}).get("pie") and \
                 triage_result.get("protections", {}).get("canary"):
                workflow = _refine_workflow_with_triage(workflow, triage_result)
                triage_refined = True
                await ctx.info(
                    f"🔧 Binary stripped + PIE + canary — lightweight recon step added"
                )

            result["triage"] = triage_result
            result["triage_refined"] = triage_refined
            if triage_refined:
                result["workflow"] = workflow

        await ctx.info(
            f"✅ Analysis complete — {len(workflow.get('tools', []))} tools selected | "
            f"~{workflow.get('estimated_time', 0)//60} min | "
            f"success: {workflow.get('success_probability', 0):.0%}"
            + (" | triage: refined" if triage_refined else "")
        )

        return result

    @mcp.tool(
        name="ctf_tools",
        description=(
            "Get the tool arsenal and executable commands for a CTF challenge category. "
            "Returns suggested tools with their CLI commands from CTFToolManager, "
            "plus the full category arsenal. Useful for understanding what tools are "
            "available before running ctf_solve()."
        ),
        timeout=10.0,
    )
    async def ctf_tools_list(
        category: str,
        description: str = "",
        target: str = "",
    ) -> Dict[str, Any]:
        """
        Get the complete CTF tool arsenal and optimized commands for a category.

        Returns suggested tools (keyword-matched to description), their executable commands,
        and the full category arsenal listing. Use to preview tools before running ctf_solve().

        Args:
            category:    web | crypto | pwn | forensics | rev | misc | osint
            description: Challenge description for context-aware tool selection (keyword-driven)
            target:      Target IP/URL for command generation (optional, appended to tool commands)

        Call AFTER ctf_analyze() to review available tools for the challenge.
        Do NOT use for non-CTF recon — use scan() or search_tools() instead.

        Example: ctf_tools('web', 'Bypass path restrictions and access admin panel', 'http://target:8080')
        Next: ctf_solve() to execute the workflow with these tools
        """
        ctx = get_context()
        tool_mgr = get_ctf_tools()

        suggested = tool_mgr.suggest_tools_for_challenge(description, category) if description else []
        category_tools = {}
        for cat_key in tool_mgr.tool_categories:
            if cat_key.startswith(category):
                category_tools[cat_key] = tool_mgr.tool_categories[cat_key]

        # Build executable commands for suggested tools
        commands = {}
        for tool in suggested[:10]:
            try:
                commands[tool] = tool_mgr.get_tool_command(tool, target or "TARGET")
            except Exception:
                commands[tool] = f"{tool} {target or 'TARGET'}"

        await ctx.info(f"🛠️ CTFToolManager: {len(suggested)} tools for [{category.upper()}]")

        return {
            "success":          True,
            "category":         category,
            "suggested_tools":  suggested,
            "commands":         commands,
            "category_arsenal": category_tools,
        }

    @mcp.tool(
        name="ctf_solve",
        description=(
            "Execute the CTF workflow and attempt to solve a challenge using real tools. "
            "Runs the workflow steps from CTFWorkflowManager through Pulse's execution "
            "bridge — each step runs an actual security tool, flag is extracted from real "
            "output. Set dry_run=True to preview without executing. "
            "Call AFTER ctf_analyze() for best results — standalone also works with description."
        ),
        timeout=600.0,  # CTF solving can take up to 10 minutes
    )
    async def ctf_solve(
        name: str,
        category: str,
        description: str,
        difficulty: str = "unknown",
        target: str = "",
        points: int = 0,
        dry_run: bool = False,
        max_steps: int = 8,
    ) -> Dict[str, Any]:
        """
        Execute the CTF workflow and attempt to solve the challenge using real security tools.

        Each workflow step runs an actual tool via Pulse's execution bridge — not simulated.
        Flags are extracted from real tool output using regex patterns. Returns solved status,
        flag candidates, step-by-step results, and confidence score.

        Returns: success, challenge, category, status (solved/in_progress/manual),
        steps_executed[], steps_planned[], flag_candidates[], flag (if found),
        confidence (0-1), manual_guidance[], summary.

        Args:
            name:        Challenge name (any string, used for logging)
            category:    web | crypto | pwn | forensics | rev | misc | osint
            description: Challenge description text — drives tool selection
            difficulty:  easy | medium | hard | insane | unknown
            target:      URL, IP, or binary path for tool execution
            points:      Point value (optional)
            dry_run:     If True, plan only — no tool execution (default: False)
            max_steps:   Maximum workflow steps to execute (default: 8)

        Call AFTER ctf_analyze() for optimal tool selection.
        Do NOT use for non-CTF targets — use scan() + get_findings() + get_plan() instead.
        dry_run=True is recommended first to preview before executing potentially destructive steps.

        Example: ctf_solve('Dynamic Paths', 'web', 'Bypass path restrictions...', 'easy', 'http://target:8080')
        Example: ctf_solve('crypto100', 'crypto', 'RSA with small e...', 'medium', dry_run=True)
        Next: check flag in response. If not found, review manual_guidance[] for next steps.
        """
        ctx = get_context()

        challenge = CTFChallenge(
            name=name,
            category=category,
            description=description,
            difficulty=difficulty,
            target=target or name,
            points=points,
        )

        ctf = get_ctf_manager()
        automator = get_ctf_automator()
        workflow = ctf.create_ctf_challenge_workflow(challenge)
        steps = workflow.get("workflow_steps", [])[:max_steps]

        est_min = workflow.get("estimated_time", 3600) // 60
        success_prob = workflow.get("success_probability", 0.55)

        await ctx.report_progress(0, len(steps))
        await ctx.info(
            f"🏴 CTF Solve — [{category.upper()}] {name} | {difficulty} | {points}pts\n"
            f"Tools: {', '.join(workflow.get('tools', [])[:6])}\n"
            f"Estimated: ~{est_min} min | Success: {success_prob:.0%}\n"
            f"Steps: {len(steps)} | Mode: {'DRY RUN' if dry_run else 'LIVE'}"
        )

        result = {
            "challenge":       name,
            "category":        category,
            "status":          "in_progress",
            "steps_executed":  [],
            "steps_planned":   [],
            "flag_candidates": [],
            "flag":            None,
            "confidence":      0.0,
            "manual_guidance": [],
            "workflow_plan":   workflow,
        }

        if dry_run:
            # Plan mode — return workflow without executing
            for step in steps:
                result["steps_planned"].append({
                    "step":        step["step"],
                    "action":      step["action"],
                    "description": step["description"],
                    "tools":       step.get("tools", []),
                    "parallel":    step.get("parallel", False),
                    "est_min":     step.get("estimated_time", 600) // 60,
                })
            result["status"] = "planned"
            await ctx.info(f"📋 Dry run complete — {len(steps)} steps planned")
            return result

        # Live execution
        flag_found = False
        for i, step in enumerate(steps):
            await ctx.report_progress(i + 1, len(steps))
            await ctx.info(
                f"⚙️ Step {step['step']}/{len(steps)} — {step['action']}: {step['description']}"
            )

            step_result = await _execute_ctf_step_real(step, challenge, ctx)
            result["steps_executed"].append(step_result)

            # Collect flag candidates
            candidates = step_result.get("flag_candidates", [])
            result["flag_candidates"].extend(candidates)

            # Update confidence
            if step_result.get("success"):
                result["confidence"] = min(1.0, result["confidence"] + 0.1)

            # Check if flag found
            for candidate in candidates:
                if automator._validate_flag_format(candidate):
                    result["flag"] = candidate
                    result["status"] = "solved"
                    flag_found = True
                    await ctx.info(f"🚩 FLAG FOUND: {candidate}")
                    break

            if flag_found:
                break

        # If not solved, generate manual guidance
        if not flag_found:
            result["status"] = "needs_manual_intervention"
            result["manual_guidance"] = automator._generate_manual_guidance(challenge, {
                "automated_steps": [
                    {"tools_used": s.get("tools_executed", [])}
                    for s in result["steps_executed"]
                ]
            })
            await ctx.info(
                f"⚠️ Auto-solve incomplete — {len(result['flag_candidates'])} flag candidates found\n"
                f"Manual guidance: {len(result['manual_guidance'])} suggestions"
            )

        result["confidence"] = round(result["confidence"], 3)
        await ctx.report_progress(len(steps), len(steps))

        return result

    @mcp.tool(
        name="ctf_team",
        description=(
            "Optimize CTF team strategy: assign challenges to team members based on "
            "skills, estimate solve times, identify collaboration opportunities. "
            "Uses CTFTeamCoordinator from the V6 intelligence layer."
        ),
        timeout=30.0,
    )
    async def ctf_team(
        team_skills: Dict[str, List[str]],
        challenges: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Optimize CTF team strategy — assign members to challenges for max score.

        Uses CTFTeamCoordinator with Hungarian algorithm approximation to assign
        team members to challenges based on their skills. Returns assignments,
        coverage gaps, collaboration opportunities, and estimated score.

        Returns: success, strategy{assignments{}, coverage_gaps[], expected_score,
        collaboration_opportunities[]}.

        Args:
            team_skills: {"alice": ["web", "crypto"], "bob": ["pwn", "rev"], ...}
            challenges:  [{"name": "...", "category": "...", "description": "...",
                           "difficulty": "...", "points": 100}, ...]

        Call after ctf_analyze() on all challenges to plan team workflow.
        Do NOT use for solo CTF — just use ctf_analyze() + ctf_solve() directly.

        Example: ctf_team({"alice": ["web", "crypto"], "bob": ["pwn", "rev"]},
                          [{"name": "Web100", "category": "web", "points": 100}])
        Next: team members run ctf_solve() on their assigned challenges
        """
        ctx = get_context()
        coordinator = get_ctf_coordinator()

        challenge_objects = [
            CTFChallenge(
                name=c.get("name", "unknown"),
                category=c.get("category", "misc"),
                description=c.get("description", ""),
                difficulty=c.get("difficulty", "unknown"),
                points=c.get("points", 0),
                target=c.get("target", ""),
            )
            for c in challenges
        ]

        await ctx.info(
            f"👥 CTFTeamCoordinator — {len(team_skills)} members, "
            f"{len(challenge_objects)} challenges"
        )

        strategy = coordinator.optimize_team_strategy(challenge_objects, team_skills)

        # Summary for ctx.info
        total_expected = sum(
            sum(ch.get("points", 0) * ch.get("success_probability", 0.5)
                for ch in assignments)
            for assignments in strategy.get("assignments", {}).values()
        )

        await ctx.info(
            f"✅ Strategy optimized — "
            f"expected score: {total_expected:.0f} pts | "
            f"{len(strategy.get('collaboration_opportunities', []))} collaboration opportunities"
        )

        return {
            "success":  True,
            "strategy": strategy,
        }

    logger.info(
        "🏴 CTF Engine registered: ctf_analyze, ctf_tools, ctf_solve, ctf_team"
    )
