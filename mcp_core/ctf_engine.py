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

from fastmcp import FastMCP, Context
from fastmcp.server.dependencies import get_context

from server_core.workflows.ctf.CTFChallenge import CTFChallenge
from server_core.workflows.ctf.toolManager import CTFToolManager
from server_core.singletons import get_ctf_manager, get_ctf_tools, get_ctf_automator, get_ctf_coordinator

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
            "Does NOT execute tools — use ctf_solve for real execution."
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
    ) -> Dict[str, Any]:
        """
        Analyze a CTF challenge and return a full workflow plan.

        Args:
            name:        Challenge name
            category:    web | crypto | pwn | forensics | rev | misc | osint
            description: Challenge description (drives tool selection)
            difficulty:  easy | medium | hard | insane | unknown
            target:      URL, IP, or binary path (optional)
            points:      Point value (optional)
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

        await ctx.info(
            f"✅ Analysis complete — {len(workflow.get('tools', []))} tools selected | "
            f"~{workflow.get('estimated_time', 0)//60} min | "
            f"success: {workflow.get('success_probability', 0):.0%}"
        )

        return {
            "success":     True,
            "challenge":   name,
            "category":    category,
            "difficulty":  difficulty,
            "workflow":    workflow,
        }

    @mcp.tool(
        name="ctf_tools",
        description=(
            "Get the complete CTF tool arsenal for a category with optimized commands. "
            "Returns executable commands from CTFToolManager for a given challenge type."
        ),
        timeout=10.0,
    )
    async def ctf_tools_list(
        category: str,
        description: str = "",
        target: str = "",
    ) -> Dict[str, Any]:
        """
        Get CTF tool arsenal and optimized commands for a category.

        Args:
            category:    web | crypto | pwn | forensics | rev | misc | osint
            description: Challenge description for context-aware tool selection
            target:      Target for command generation (optional)
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
            "Attempt to auto-solve a CTF challenge by executing the V6 workflow steps "
            "with real HexStrike tools. Each step runs actual pentesting tools via "
            "run_security_tool() and flags are extracted from real output. "
            "Set dry_run=true to plan without executing."
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
        Execute the CTF workflow and attempt to solve the challenge.

        Args:
            name:        Challenge name
            category:    web | crypto | pwn | forensics | rev | misc | osint
            description: Challenge description
            difficulty:  easy | medium | hard | insane | unknown
            target:      URL, IP, or binary path
            points:      Point value
            dry_run:     If True, plan only — no tool execution
            max_steps:   Maximum workflow steps to execute (default 8)
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
        Optimize CTF team strategy.

        Args:
            team_skills: {"alice": ["web", "crypto"], "bob": ["pwn", "rev"], ...}
            challenges:  [{"name": "...", "category": "...", "description": "...",
                           "difficulty": "...", "points": 100}, ...]
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
