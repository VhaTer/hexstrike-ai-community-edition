"""
mcp_core/cve_engine.py

HexStrike AI-PULSE — CVE Intelligence Engine

Bridges CVEIntelligenceManager (V6 intelligence layer) with FastMCP 3.x.

CVEIntelligenceManager already implements:
  - fetch_latest_cves()        : NVD API v2.0 real-time queries
  - analyze_cve_exploitability(): CVSS v3.1 deep analysis + exploit indicators
  - search_existing_exploits() : GitHub + Metasploit + Exploit-DB + PacketStorm

This module adds only the MCP registration layer — no business logic duplication.

Exposed as MCP tools via register_cve_tools(mcp):
  - cve_fetch    : Latest CVEs from NVD (real-time, filtered by severity/time)
  - cve_analyze  : Deep exploitability analysis for a specific CVE ID
  - cve_exploits : Search for existing exploits (GitHub/Metasploit/Exploit-DB)
  - cve_intel    : Full intelligence report (fetch + analyze + exploits in one call)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_context

from server_core.singletons import get_cve_intelligence

logger = logging.getLogger(__name__)


def register_cve_tools(mcp: FastMCP) -> None:
    """Register CVE intelligence tools on the FastMCP server."""

    @mcp.tool(
        name="cve_fetch",
        description=(
            "Fetch latest CVEs from NVD (National Vulnerability Database) in real-time. "
            "Returns CVEs published or modified within the specified timeframe, "
            "filtered by severity. Uses NVD API v2.0. "
            "Severity options: HIGH,CRITICAL | CRITICAL | HIGH | MEDIUM | LOW | ALL"
        ),
        timeout=60.0,  # NVD API can be slow — 6s rate limit between requests
    )
    async def cve_fetch(
        hours: int = 24,
        severity: str = "HIGH,CRITICAL",
    ) -> Dict[str, Any]:
        """
        Fetch latest CVEs from NVD.

        Args:
            hours:    Lookback window in hours (default 24)
            severity: Comma-separated severity levels: HIGH,CRITICAL | MEDIUM | LOW | ALL
        """
        ctx = get_context()
        await ctx.info(f"🔍 Fetching CVEs — last {hours}h | severity: {severity}")
        await ctx.report_progress(0, 100)

        import asyncio
        loop = asyncio.get_event_loop()
        cve_manager = get_cve_intelligence()

        result = await loop.run_in_executor(
            None,
            lambda: cve_manager.fetch_latest_cves(hours=hours, severity_filter=severity)
        )

        await ctx.report_progress(100, 100)

        if result.get("success"):
            cves = result.get("cves", [])
            await ctx.info(
                f"✅ {len(cves)} CVEs found | "
                f"Sources: {', '.join(result.get('data_sources', []))}"
            )
        else:
            await ctx.error(f"❌ CVE fetch failed: {result.get('error', 'unknown')}")

        return result

    @mcp.tool(
        name="cve_analyze",
        description=(
            "Deep exploitability analysis for a specific CVE. "
            "Queries NVD API for CVSS v3.1 metrics, calculates exploitability score, "
            "identifies exploit type (RCE/SQLi/XSS/etc), checks for public exploit references. "
            "Returns: exploitability_score, priority (IMMEDIATE/HIGH/MEDIUM/LOW), "
            "attack_vector, weaponization_level, active_exploitation assessment."
        ),
        timeout=30.0,
    )
    async def cve_analyze(
        cve_id: str,
    ) -> Dict[str, Any]:
        """
        Analyze exploitability for a specific CVE ID.

        Args:
            cve_id: CVE identifier (e.g. 'CVE-2024-1234')
        """
        ctx = get_context()
        cve_id = cve_id.strip().upper()

        if not cve_id.startswith("CVE-"):
            return {
                "success": False,
                "error": f"Invalid CVE ID format: '{cve_id}' — expected 'CVE-YYYY-NNNNN'",
                "cve_id": cve_id,
            }

        await ctx.info(f"🔬 Analyzing exploitability: {cve_id}")
        await ctx.report_progress(0, 100)

        import asyncio
        loop = asyncio.get_event_loop()
        cve_manager = get_cve_intelligence()

        result = await loop.run_in_executor(
            None,
            lambda: cve_manager.analyze_cve_exploitability(cve_id)
        )

        await ctx.report_progress(100, 100)

        if result.get("success"):
            await ctx.info(
                f"✅ {cve_id} — "
                f"CVSS: {result.get('cvss_score', 'N/A')} | "
                f"Severity: {result.get('severity', 'N/A')} | "
                f"Exploitability: {result.get('exploitability_level', 'N/A')} | "
                f"Priority: {result.get('threat_intelligence', {}).get('recommended_priority', 'N/A')}"
            )
        else:
            await ctx.error(f"❌ Analysis failed for {cve_id}: {result.get('error', 'unknown')}")

        return result

    @mcp.tool(
        name="cve_exploits",
        description=(
            "Search for existing exploits and PoCs for a CVE across multiple sources: "
            "GitHub repositories, Metasploit framework modules, Exploit-DB references, "
            "PacketStorm Security. "
            "Returns exploit list with reliability ratings (EXCELLENT/GOOD/FAIR/UNVERIFIED), "
            "direct URLs, and author information."
        ),
        timeout=45.0,
    )
    async def cve_exploits(
        cve_id: str,
    ) -> Dict[str, Any]:
        """
        Search for existing exploits for a specific CVE.

        Args:
            cve_id: CVE identifier (e.g. 'CVE-2024-1234')
        """
        ctx = get_context()
        cve_id = cve_id.strip().upper()

        if not cve_id.startswith("CVE-"):
            return {
                "success": False,
                "error": f"Invalid CVE ID format: '{cve_id}' — expected 'CVE-YYYY-NNNNN'",
                "cve_id": cve_id,
            }

        await ctx.info(
            f"🔎 Searching exploits for {cve_id} — "
            f"GitHub + Metasploit + Exploit-DB + PacketStorm"
        )
        await ctx.report_progress(0, 100)

        import asyncio
        loop = asyncio.get_event_loop()
        cve_manager = get_cve_intelligence()

        result = await loop.run_in_executor(
            None,
            lambda: cve_manager.search_existing_exploits(cve_id)
        )

        await ctx.report_progress(100, 100)

        if result.get("success"):
            exploits = result.get("exploits", [])
            summary = result.get("search_summary", {})
            await ctx.info(
                f"✅ {len(exploits)} exploits found for {cve_id} | "
                f"GitHub: {summary.get('github_repos', 0)} | "
                f"Metasploit: {summary.get('metasploit_modules', 0)} | "
                f"Exploit-DB: {summary.get('exploit_db_refs', 0)}"
            )
        else:
            await ctx.error(
                f"❌ Exploit search failed for {cve_id}: {result.get('error', 'unknown')}"
            )

        return result

    @mcp.tool(
        name="cve_intel",
        description=(
            "Full CVE intelligence report: fetch + analyze + exploit search in one call. "
            "For a given CVE ID, returns: CVSS analysis, exploitability score, priority, "
            "threat intelligence, AND all known exploits from GitHub/Metasploit/Exploit-DB. "
            "Use this for comprehensive CVE assessment before exploitation decisions."
        ),
        timeout=90.0,  # fetch + analyze + search = up to 90s
    )
    async def cve_intel(
        cve_id: str,
    ) -> Dict[str, Any]:
        """
        Full intelligence report for a CVE: analysis + exploit search combined.

        Args:
            cve_id: CVE identifier (e.g. 'CVE-2024-1234')
        """
        ctx = get_context()
        cve_id = cve_id.strip().upper()

        if not cve_id.startswith("CVE-"):
            return {
                "success": False,
                "error": f"Invalid CVE ID format: '{cve_id}'",
                "cve_id": cve_id,
            }

        await ctx.info(f"🧠 Full CVE Intel: {cve_id}")
        await ctx.report_progress(0, 100)

        import asyncio
        loop = asyncio.get_event_loop()
        cve_manager = get_cve_intelligence()

        # Run analysis and exploit search concurrently
        await ctx.info("Step 1/2 — Exploitability analysis + exploit search (parallel)")
        await ctx.report_progress(10, 100)

        analysis_task = loop.run_in_executor(
            None, lambda: cve_manager.analyze_cve_exploitability(cve_id)
        )
        exploits_task = loop.run_in_executor(
            None, lambda: cve_manager.search_existing_exploits(cve_id)
        )

        analysis, exploits = await asyncio.gather(analysis_task, exploits_task)

        await ctx.report_progress(90, 100)

        # Compute combined risk score
        exploitability_score = analysis.get("exploitability_score", 0.0)
        exploit_count = len(exploits.get("exploits", []))
        has_metasploit = exploits.get("search_summary", {}).get("metasploit_modules", 0) > 0
        cvss_score = analysis.get("cvss_score", 0.0)

        # Risk multiplier: having Metasploit module = significantly higher risk
        risk_score = exploitability_score * 0.5 + (cvss_score / 10.0) * 0.3
        if has_metasploit:
            risk_score = min(1.0, risk_score + 0.3)
        if exploit_count > 3:
            risk_score = min(1.0, risk_score + 0.1)

        if risk_score >= 0.85:
            risk_level = "CRITICAL"
        elif risk_score >= 0.65:
            risk_level = "HIGH"
        elif risk_score >= 0.40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        await ctx.report_progress(100, 100)
        await ctx.info(
            f"✅ {cve_id} Intel Complete — "
            f"Risk: {risk_level} ({risk_score:.2f}) | "
            f"CVSS: {cvss_score} | "
            f"Exploits: {exploit_count} | "
            f"Metasploit: {'YES 🚨' if has_metasploit else 'no'}"
        )

        return {
            "success":           analysis.get("success", False),
            "cve_id":            cve_id,
            "risk_score":        round(risk_score, 3),
            "risk_level":        risk_level,
            "analysis":          analysis,
            "exploits":          exploits,
            "intel_summary": {
                "cvss_score":           cvss_score,
                "severity":             analysis.get("severity", "UNKNOWN"),
                "exploitability_level": analysis.get("exploitability_level", "UNKNOWN"),
                "priority":             analysis.get("threat_intelligence", {})
                                                .get("recommended_priority", "UNKNOWN"),
                "exploit_count":        exploit_count,
                "has_metasploit":       has_metasploit,
                "active_exploitation":  analysis.get("threat_intelligence", {})
                                                .get("active_exploitation", False),
                "attack_vector":        analysis.get("attack_vector", "UNKNOWN"),
                "exploit_indicators":   analysis.get("threat_intelligence", {})
                                                .get("exploit_indicators", []),
            },
        }

    logger.info(
        "🔍 CVE Engine registered: cve_fetch, cve_analyze, cve_exploits, cve_intel"
    )
