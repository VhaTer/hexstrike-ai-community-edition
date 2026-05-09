"""
mcp_core/bugbounty_engine.py

HexStrike AI-PULSE — Bug Bounty Intelligence Engine

Bridges BugBountyWorkflowManager (V6 intelligence layer) with FastMCP 3.x.

BugBountyWorkflowManager implements 4 complete workflow generators:
  - create_reconnaissance_workflow()      : subdomain + HTTP + content + param discovery
  - create_vulnerability_hunting_workflow(): priority-ranked vuln testing per target profile
  - create_business_logic_testing_workflow(): auth bypass, authz flaws, race conditions
  - create_osint_workflow()               : domain + social + email + tech intelligence

This module adds the MCP registration layer + real execution bridge.
All V6 classes kept intact — zero modification.

Exposed via register_bugbounty_tools(mcp):
  - bb_recon      : Full reconnaissance workflow plan + real execution
  - bb_hunt       : Vulnerability hunting workflow (priority-ranked)
  - bb_business   : Business logic test plan
  - bb_osint      : OSINT gathering workflow
  - bb_full       : Complete bug bounty engagement (all 4 phases)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_context

from server_core.workflows.bugbounty.target import BugBountyTarget
from server_core.singletons import get_bugbounty_manager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Execution bridge — executes recon/vuln tools via run_security_tool()
# ---------------------------------------------------------------------------

async def _execute_bb_phase(
    phase: Dict[str, Any],
    target_domain: str,
    ctx,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Execute a single bug bounty workflow phase using real HexStrike tools.

    For each tool in the phase: calls run_security_tool() if available in
    DIRECT_TOOLS, otherwise returns structured guidance.
    """
    phase_result = {
        "name":           phase.get("name", "unknown"),
        "description":    phase.get("description", ""),
        "tools_executed": [],
        "tools_skipped":  [],
        "outputs":        [],
        "success":        False,
        "execution_time": 0.0,
    }

    if dry_run:
        phase_result["success"] = True
        phase_result["tools_skipped"] = [
            t.get("tool", "?") for t in phase.get("tools", [])
        ]
        return phase_result

    import time
    t0 = time.perf_counter()
    tools = phase.get("tools", [])

    # Tools not in DIRECT_TOOLS — need manual execution
    MANUAL_TOOLS = {
        "assetfinder", "dnsrecon", "certificate_transparency",
        "social_mapper", "linkedin_scraper", "hunter_io",
        "haveibeenpwned", "email_validator", "builtwith",
        "wappalyzer", "shodan", "jwt_tool", "race_the_web",
        "upload_scanner", "dirsearch",
    }

    async def _run_tool(tool_name: str, params: Dict) -> Dict:
        if tool_name in MANUAL_TOOLS:
            return {"tool": tool_name, "success": False, "manual": True,
                    "message": f"[MANUAL] {tool_name} — requires manual execution"}
        try:
            from mcp_core.server_setup import run_security_tool
            import json
            # Merge target into params
            exec_params = {"target": target_domain, **params}
            result = await run_security_tool(
                ctx=ctx,
                tool_name=tool_name,
                parameters=json.dumps(exec_params),
            )
            return {"tool": tool_name, "success": result.get("success", False),
                    "output": result.get("output", "")[:2000], "manual": False}
        except Exception as exc:
            return {"tool": tool_name, "success": False, "manual": False,
                    "error": str(exc)[:200]}

    # Execute all tools in the phase concurrently
    tool_coroutines = [
        _run_tool(t.get("tool", ""), t.get("params", {}))
        for t in tools
    ]
    results = await asyncio.gather(*tool_coroutines, return_exceptions=True)

    for r in results:
        if isinstance(r, Exception):
            phase_result["tools_skipped"].append(str(r)[:100])
            continue
        tool = r.get("tool", "?")
        if r.get("manual"):
            phase_result["tools_skipped"].append(tool)
        elif r.get("success"):
            phase_result["tools_executed"].append(tool)
            if r.get("output"):
                phase_result["outputs"].append(
                    {"tool": tool, "output": r["output"]}
                )
        else:
            phase_result["tools_skipped"].append(tool)

    phase_result["success"] = len(phase_result["tools_executed"]) > 0
    phase_result["execution_time"] = round(time.perf_counter() - t0, 3)
    return phase_result


# ---------------------------------------------------------------------------
# MCP tool registration
# ---------------------------------------------------------------------------

def register_bugbounty_tools(mcp: FastMCP) -> None:
    """Register Bug Bounty intelligence tools on the FastMCP server."""

    @mcp.tool(
        name="bb_recon",
        description=(
            "Bug bounty reconnaissance workflow: subdomain enumeration, HTTP service "
            "discovery, content discovery, parameter discovery. "
            "Uses BugBountyWorkflowManager (V6 intelligence layer). "
            "Set dry_run=true to get the workflow plan without executing tools."
        ),
        timeout=600.0,
    )
    async def bb_recon(
        domain: str,
        scope: Optional[List[str]] = None,
        out_of_scope: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Full bug bounty reconnaissance workflow.

        Args:
            domain:       Target root domain (e.g. 'example.com')
            scope:        In-scope domains/IPs (default: [domain])
            out_of_scope: Out-of-scope items
            dry_run:      Plan only — no tool execution
        """
        ctx = get_context()
        target = BugBountyTarget(
            domain=domain,
            scope=scope or [domain],
            out_of_scope=out_of_scope or [],
        )

        bb = get_bugbounty_manager()
        workflow = bb.create_reconnaissance_workflow(target)

        phases = workflow.get("phases", [])
        est_min = workflow.get("estimated_time", 0) // 60
        tools_count = workflow.get("tools_count", 0)

        mode = "DRY RUN" if dry_run else "LIVE"
        await ctx.info(
            f"🎯 BB Recon — {domain} | {len(phases)} phases | "
            f"~{est_min} min | {tools_count} tools | {mode}"
        )
        await ctx.report_progress(0, len(phases))

        phase_results = []
        for i, phase in enumerate(phases):
            await ctx.info(f"⚙️ Phase {i+1}/{len(phases)}: {phase['name']}")
            result = await _execute_bb_phase(phase, domain, ctx, dry_run=dry_run)
            phase_results.append(result)
            await ctx.report_progress(i + 1, len(phases))

        executed = sum(1 for p in phase_results if p["success"])
        await ctx.info(f"✅ Recon complete — {executed}/{len(phases)} phases succeeded")

        return {
            "success":       executed > 0 or dry_run,
            "domain":        domain,
            "workflow_plan": workflow,
            "phase_results": phase_results,
            "mode":          mode,
            "summary": {
                "phases_total":    len(phases),
                "phases_executed": executed,
                "estimated_min":   est_min,
            },
        }

    @mcp.tool(
        name="bb_hunt",
        description=(
            "Bug bounty vulnerability hunting workflow. "
            "Prioritizes vulnerability types by impact (RCE > SQLi > SSRF > IDOR > XSS ...) "
            "and generates targeted test plans with tools and payloads. "
            "Customize priority_vulns to focus on specific vulnerability classes."
        ),
        timeout=60.0,
    )
    async def bb_hunt(
        domain: str,
        priority_vulns: Optional[List[str]] = None,
        program_type: str = "web",
    ) -> Dict[str, Any]:
        """
        Vulnerability hunting workflow (priority-ranked by impact).

        Args:
            domain:         Target domain
            priority_vulns: Vuln types to test: rce, sqli, ssrf, idor, xss, lfi, xxe, csrf
            program_type:   web | api | mobile | iot
        """
        ctx = get_context()
        target = BugBountyTarget(
            domain=domain,
            program_type=program_type,
            priority_vulns=priority_vulns or ["rce", "sqli", "ssrf", "idor", "xss", "lfi"],
        )

        bb = get_bugbounty_manager()
        workflow = bb.create_vulnerability_hunting_workflow(target)

        tests = workflow.get("vulnerability_tests", [])
        est_min = workflow.get("estimated_time", 0) // 60

        await ctx.info(
            f"🔍 BB Hunt — {domain} | {len(tests)} vuln types | "
            f"~{est_min} min | priority score: {workflow.get('priority_score', 0)}"
        )

        # Format test scenarios for easy reading
        for test in tests[:8]:
            scenarios = test.get("test_scenarios", [])
            payload_preview = ""
            if scenarios:
                first = scenarios[0]
                payloads = first.get("payloads", [])[:2]
                payload_preview = f" | payloads: {', '.join(str(p)[:30] for p in payloads)}"
            await ctx.info(
                f"  [{test['priority']:2d}] {test['vulnerability_type'].upper()} — "
                f"tools: {', '.join(test['tools'])}{payload_preview}"
            )

        return {
            "success":  True,
            "domain":   domain,
            "workflow": workflow,
            "summary": {
                "vulns_prioritized": len(tests),
                "estimated_min":     est_min,
                "priority_score":    workflow.get("priority_score", 0),
                "top_priority":      tests[0]["vulnerability_type"] if tests else "none",
            },
        }

    @mcp.tool(
        name="bb_business",
        description=(
            "Bug bounty business logic testing plan. "
            "Covers: authentication bypass (JWT, OAuth, password reset), "
            "authorization flaws (IDOR, privilege escalation, RBAC bypass), "
            "business process manipulation (race conditions, price manipulation, "
            "workflow state), input validation bypass (file upload, content-type). "
            "Returns structured test plan — manual + automated steps."
        ),
        timeout=15.0,
    )
    async def bb_business(
        domain: str,
    ) -> Dict[str, Any]:
        """
        Business logic vulnerability testing plan.

        Args:
            domain: Target domain
        """
        ctx = get_context()
        target = BugBountyTarget(domain=domain)

        bb = get_bugbounty_manager()
        workflow = bb.create_business_logic_testing_workflow(target)

        categories = workflow.get("business_logic_tests", [])
        est_h = workflow.get("estimated_time", 480) // 3600

        await ctx.info(
            f"🧩 BB Business Logic — {domain} | "
            f"{len(categories)} categories | ~{est_h}h | manual testing required"
        )

        for cat in categories:
            tests = cat.get("tests", [])
            automated = [t["name"] for t in tests if t.get("method") == "automated"]
            manual = [t["name"] for t in tests if t.get("method") == "manual"]
            await ctx.info(
                f"  {cat['category']}: "
                f"{len(automated)} automated, {len(manual)} manual"
            )

        return {
            "success":  True,
            "domain":   domain,
            "workflow": workflow,
            "summary": {
                "categories":  len(categories),
                "estimated_h": est_h,
                "manual_required": workflow.get("manual_testing_required", True),
                "total_tests": sum(
                    len(c.get("tests", [])) for c in categories
                ),
            },
        }

    @mcp.tool(
        name="bb_osint",
        description=(
            "Bug bounty OSINT gathering workflow: domain intelligence (WHOIS, DNS, "
            "cert transparency), social media intel (Sherlock), email intelligence, "
            "technology intelligence (Shodan, Wappalyzer, BuiltWith). "
            "Returns structured OSINT plan — most tools require manual execution."
        ),
        timeout=15.0,
    )
    async def bb_osint(
        domain: str,
    ) -> Dict[str, Any]:
        """
        OSINT gathering workflow for bug bounty.

        Args:
            domain: Target domain
        """
        ctx = get_context()
        target = BugBountyTarget(domain=domain)

        bb = get_bugbounty_manager()
        workflow = bb.create_osint_workflow(target)

        phases = workflow.get("osint_phases", [])
        est_min = workflow.get("estimated_time", 240) // 60

        await ctx.info(
            f"🕵️ BB OSINT — {domain} | {len(phases)} phases | ~{est_min} min"
        )
        for phase in phases:
            tools = [t.get("tool", "?") for t in phase.get("tools", [])]
            await ctx.info(f"  {phase['name']}: {', '.join(tools)}")

        return {
            "success":  True,
            "domain":   domain,
            "workflow": workflow,
            "intel_types": workflow.get("intelligence_types", []),
            "summary": {
                "phases":        len(phases),
                "estimated_min": est_min,
            },
        }

    @mcp.tool(
        name="bb_full",
        description=(
            "Full bug bounty engagement: runs all 4 phases in sequence — "
            "OSINT → Reconnaissance → Vulnerability Hunting → Business Logic. "
            "Use dry_run=true for a complete engagement plan without execution. "
            "Live mode executes recon tools in real-time and returns combined results."
        ),
        timeout=900.0,  # Full engagement can take 15 minutes
    )
    async def bb_full(
        domain: str,
        scope: Optional[List[str]] = None,
        out_of_scope: Optional[List[str]] = None,
        priority_vulns: Optional[List[str]] = None,
        program_type: str = "web",
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Complete bug bounty engagement — all 4 phases.

        Args:
            domain:         Target root domain
            scope:          In-scope items
            out_of_scope:   Out-of-scope items
            priority_vulns: Vuln types: rce, sqli, ssrf, idor, xss, lfi, xxe, csrf
            program_type:   web | api | mobile | iot
            dry_run:        Plan only — no execution
        """
        ctx = get_context()
        mode = "DRY RUN" if dry_run else "LIVE"

        target = BugBountyTarget(
            domain=domain,
            scope=scope or [domain],
            out_of_scope=out_of_scope or [],
            program_type=program_type,
            priority_vulns=priority_vulns or ["rce", "sqli", "ssrf", "idor", "xss", "lfi"],
        )

        bb = get_bugbounty_manager()

        # Generate all 4 workflows
        osint_wf    = bb.create_osint_workflow(target)
        recon_wf    = bb.create_reconnaissance_workflow(target)
        hunt_wf     = bb.create_vulnerability_hunting_workflow(target)
        business_wf = bb.create_business_logic_testing_workflow(target)

        total_min = (
            osint_wf.get("estimated_time", 0) +
            recon_wf.get("estimated_time", 0) +
            hunt_wf.get("estimated_time", 0) +
            business_wf.get("estimated_time", 0)
        ) // 60

        await ctx.info(
            f"🚀 Full BB Engagement — {domain} | {mode} | ~{total_min} min total"
        )
        await ctx.report_progress(0, 4)

        # Phase 1 — OSINT (plan only, mostly manual tools)
        await ctx.info("Phase 1/4 — OSINT")
        await ctx.report_progress(1, 4)

        # Phase 2 — Recon (executable)
        await ctx.info("Phase 2/4 — Reconnaissance")
        recon_results = []
        for phase in recon_wf.get("phases", []):
            r = await _execute_bb_phase(phase, domain, ctx, dry_run=dry_run)
            recon_results.append(r)
        await ctx.report_progress(2, 4)

        # Phase 3 — Vuln hunt (plan)
        await ctx.info("Phase 3/4 — Vulnerability Hunting Plan")
        await ctx.report_progress(3, 4)

        # Phase 4 — Business logic (plan)
        await ctx.info("Phase 4/4 — Business Logic Plan")
        await ctx.report_progress(4, 4)

        await ctx.info(f"✅ Full BB Engagement complete — {domain}")

        return {
            "success":      True,
            "domain":       domain,
            "mode":         mode,
            "engagement": {
                "osint":          osint_wf,
                "reconnaissance": recon_wf,
                "vuln_hunting":   hunt_wf,
                "business_logic": business_wf,
                "recon_results":  recon_results,
            },
            "summary": {
                "total_estimated_min": total_min,
                "recon_phases":        len(recon_wf.get("phases", [])),
                "vuln_types":          len(hunt_wf.get("vulnerability_tests", [])),
                "business_categories": len(business_wf.get("business_logic_tests", [])),
                "osint_phases":        len(osint_wf.get("osint_phases", [])),
            },
        }

    logger.info(
        "🎯 Bug Bounty Engine registered: bb_recon, bb_hunt, bb_business, bb_osint, bb_full"
    )
