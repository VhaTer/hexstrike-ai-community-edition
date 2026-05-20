"""
mcp_core/parameter_optimizer.py

Port of V6 ParameterOptimizer class to FastMCP 3.x Phase 3.

Original V6 class: hexstrike_server.py#L4635-L4876
Part of the IntelligentDecisionEngine subsystem.

Optimizes security tool parameters based on:
- Detected technology stack (TechProfile from TechnologyDetector)
- Execution profile: stealth / normal / aggressive
- System resource state (CPU/memory)

Usage:
    from mcp_core.technology_detector import TechProfile
    from mcp_core.parameter_optimizer import ParameterOptimizer

    optimizer = ParameterOptimizer()
    params = optimizer.optimize("gobuster", {"url": "https://example.com"}, tech_profile)
    # params["threads"] reduced, params["delay"] added if WAF detected
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Literal

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PSUTIL_AVAILABLE = False


def _check_psutil():
    try:
        import psutil
        return True
    except ImportError:
        return False

from mcp_core.technology_detector import TechProfile

logger = logging.getLogger(__name__)

Profile = Literal["stealth", "normal", "aggressive"]


# ---------------------------------------------------------------------------
# ParameterOptimizer
# ---------------------------------------------------------------------------

class ParameterOptimizer:
    """
    Intelligent parameter optimizer for HexStrike security tools.

    Ported from V6 hexstrike_server.py#L4635-L4876.
    Implements three optimization profiles (stealth/normal/aggressive)
    with automatic WAF detection response and resource-aware tuning.
    """

    # Base parameters per tool — from V6 _get_base_parameters()
    _BASE_PARAMS: Dict[str, Dict[str, Any]] = {
        "nmap": {
            "scan_type": "-sS",
            "ports": "1-1000",
            "additional_args": "-T4",
        },
        "nmap_advanced": {
            "scan_type": "-sS",
            "ports": "1-65535",
            "additional_args": "-sV -sC -T4",
        },
        "gobuster": {
            "mode": "dir",
            "wordlist": "/usr/share/wordlists/dirb/common.txt",
            "threads": 20,
            "additional_args": "",
        },
        "ffuf": {
            "wordlist": "/usr/share/wordlists/dirb/common.txt",
            "match_codes": "200,204,301,302,307,401,403",
            "threads": 40,
        },
        "feroxbuster": {
            "wordlist": "/usr/share/wordlists/dirb/common.txt",
            "threads": 10,
        },
        "sqlmap": {
            "additional_args": "--batch --level=1 --risk=1",
        },
        "nuclei": {
            "severity": "critical,high,medium",
            "additional_args": "-c 25",
        },
        "nikto": {
            "additional_args": "",
        },
        "hydra": {
            "additional_args": "-t 4",
        },
        "wpscan": {
            "additional_args": "--enumerate ap,at",
        },
    }

    # Optimization profiles — from V6 optimization_profiles dict
    _PROFILES: Dict[str, Dict[str, Dict[str, Any]]] = {
        "nmap": {
            "stealth":    {"additional_args": "-sS -T2 --max-retries 1 --host-timeout 300s"},
            "normal":     {"additional_args": "-sS -sV -T4 --max-retries 2"},
            "aggressive": {"additional_args": "-sS -sV -sC -O -T5 --max-retries 3 --min-rate 1000"},
        },
        "gobuster": {
            "stealth":    {"threads": 5,  "additional_args": "-t 5 -to 30s"},
            "normal":     {"threads": 20, "additional_args": ""},
            "aggressive": {"threads": 50, "additional_args": "-t 50 -to 5s"},
        },
        "ffuf": {
            "stealth":    {"threads": 5,  "additional_args": "-p 1"},
            "normal":     {"threads": 40, "additional_args": ""},
            "aggressive": {"threads": 100,"additional_args": ""},
        },
        "sqlmap": {
            "stealth":    {"additional_args": "--batch --level=1 --risk=1 --delay=2 --randomize-agent"},
            "normal":     {"additional_args": "--batch --level=2 --risk=2"},
            "aggressive": {"additional_args": "--batch --level=3 --risk=3 --threads=10"},
        },
        "nuclei": {
            "stealth":    {"severity": "critical,high", "additional_args": "-c 5 -rate-limit 10"},
            "normal":     {"severity": "critical,high,medium", "additional_args": "-c 25"},
            "aggressive": {"severity": "critical,high,medium,low", "additional_args": "-c 50"},
        },
        "hydra": {
            "stealth":    {"additional_args": "-t 1 -W 3"},
            "normal":     {"additional_args": "-t 4"},
            "aggressive": {"additional_args": "-t 16 -f"},
        },
    }

    def optimize(
        self,
        tool_name: str,
        params: Dict[str, Any],
        tech_profile: TechProfile | None = None,
        profile: Profile = "normal",
    ) -> Dict[str, Any]:
        """
        Optimize tool parameters based on tech profile and execution profile.

        V6 flow:
          1. Get base parameters
          2. Apply technology stack optimizations (WAF → stealth)
          3. Apply resource-aware tuning (high CPU → reduce threads)
          4. Apply profile template
          5. Return enriched params dict

        Args:
            tool_name:    Tool key matching DIRECT_TOOLS (e.g. 'gobuster')
            params:       Original parameters dict from the MCP call
            tech_profile: Detected technology profile (optional)
            profile:      Execution profile: stealth / normal / aggressive

        Returns:
            Optimized parameters dict (copy of params with adjustments applied)
        """
        result = dict(params)
        tool   = tool_name.lower()
        # Track keys explicitly set by the caller — resource tuning must not touch these
        caller_keys: set = set(params.keys())

        # 1. Merge base params (only for keys not already set by caller)
        base = self._BASE_PARAMS.get(tool, {})
        for key, value in base.items():
            if key not in result:
                result[key] = value

        # 2. Technology-aware optimizations
        forced_stealth = False
        if tech_profile:
            result, forced_stealth = self._apply_tech_optimizations(
                tool, result, tech_profile
            )

        # 3. Apply profile template (stealth override if WAF detected)
        #    Must run before resource tuning so base_val comparison is clean
        effective_profile: Profile = "stealth" if forced_stealth else profile
        result = self._apply_profile(tool, result, effective_profile, caller_keys)

        # 4. Resource-aware tuning — after profile, and never touches caller-set keys
        result = self._apply_resource_tuning(tool, result, caller_keys)

        # 5. Metadata for transparency
        result["_optimizer"] = {
            "profile":       effective_profile,
            "forced_stealth": forced_stealth,
            "tech_summary":  tech_profile.summary() if tech_profile else "not detected",
        }

        logger.debug(
            f"ParameterOptimizer: {tool} profile={effective_profile} "
            f"forced_stealth={forced_stealth}"
        )

        return result

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _apply_tech_optimizations(
        self,
        tool: str,
        params: Dict[str, Any],
        tech: TechProfile,
    ) -> tuple[Dict[str, Any], bool]:
        """
        Apply technology stack-aware optimizations.
        Returns (params, forced_stealth).
        Mirrors V6 _apply_technology_optimizations().
        """
        forced_stealth = False

        # --- WAF/CDN detected → force stealth ---
        if tech.is_waf:
            forced_stealth = True
            logger.info(
                f"ParameterOptimizer: WAF detected ({', '.join(tech.security)}) "
                f"→ forcing stealth mode for {tool}"
            )

        # --- WordPress-specific optimizations ---
        if tech.is_wordpress:
            if tool == "gobuster":
                existing = params.get("additional_args", "")
                wp_extensions = "php,html,txt,xml,json,bak"
                if "-x" not in existing:
                    params["additional_args"] = f"{existing} -x {wp_extensions}".strip()
                # Inject known WP paths
                params["_wp_paths"] = [
                    "/wp-content/", "/wp-admin/", "/wp-includes/",
                    "/wp-login.php", "/xmlrpc.php",
                ]
            elif tool == "nuclei":
                tags = params.get("additional_args", "")
                if "wordpress" not in tags:
                    params["additional_args"] = f"{tags} -tags wordpress".strip()
            elif tool == "wpscan":
                params["additional_args"] = "--enumerate ap,at,cb,dbe"

        # --- PHP-specific ---
        if tech.is_php:
            if tool in ("gobuster", "ffuf", "feroxbuster"):
                existing = params.get("additional_args", "")
                if "-x" not in existing and ".php" not in existing:
                    params["additional_args"] = f"{existing} -x php,php3,php4,php5,phtml".strip()

        # --- Web server-specific ---
        if "apache" in tech.web_servers:
            if tool == "gobuster":
                existing = params.get("additional_args", "")
                if "-x" not in existing:
                    params["additional_args"] = f"{existing} -x php,html,conf,htaccess".strip()
        elif "nginx" in tech.web_servers:
            if tool == "gobuster":
                existing = params.get("additional_args", "")
                if "-x" not in existing:
                    params["additional_args"] = f"{existing} -x json,html,conf".strip()

        # --- Framework-specific ---
        if "django" in tech.frameworks and tool == "gobuster":
            params["_django_paths"] = ["/admin/", "/api/", "/static/", "/media/"]
        if "rails" in tech.frameworks and tool == "gobuster":
            params["_rails_paths"] = ["/admin", "/api/v1", "/assets/", "/rails/info"]

        # --- .NET specific ---
        if "dotnet" in tech.languages:
            if tool in ("gobuster", "ffuf"):
                existing = params.get("additional_args", "")
                if "-x" not in existing:
                    params["additional_args"] = f"{existing} -x aspx,asp,html,config".strip()

        return params, forced_stealth

    def _apply_resource_tuning(
        self,
        tool: str,
        params: Dict[str, Any],
        caller_keys: set | None = None,
    ) -> Dict[str, Any]:
        """
        Reduce concurrency under high CPU/memory load.
        Mirrors V6 PerformanceMonitor-based resource tuning.
        Never overrides keys explicitly set by the caller.
        """
        if not _PSUTIL_AVAILABLE:
            return params

        protected = caller_keys or set()

        try:
            cpu    = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory().percent

            # High load → reduce threads
            if cpu > 85 or memory > 90:
                for thread_key in ("threads", "concurrency", "workers"):
                    if thread_key in params and thread_key not in protected:
                        original = params[thread_key]
                        params[thread_key] = max(1, int(original // 2))
                        logger.debug(
                            f"ParameterOptimizer: high load (CPU={cpu}% MEM={memory}%) "
                            f"→ {thread_key} {original}→{params[thread_key]}"
                        )
        except Exception:
            pass  # psutil errors are non-fatal

        return params

    def _apply_profile(
        self,
        tool: str,
        params: Dict[str, Any],
        profile: Profile,
        caller_keys: set | None = None,
    ) -> Dict[str, Any]:
        """
        Apply profile template overrides.
        Mirrors V6 _apply_profile_optimizations().

        Rules:
        - Numeric params (threads, concurrency): profile wins ONLY if caller
          did not explicitly set them (i.e. value came from base params).
        - String params (additional_args): APPEND profile flags to existing
          value rather than overwriting — preserves tech optimizations.
        - Empty string profile values are ignored.
        - Scan-type flags (-sS, -sV, -sn, etc.) stripped from additional_args
          when caller explicitly set scan_type to avoid conflict.
        """
        tool_profiles = self._PROFILES.get(tool, {})
        profile_params = tool_profiles.get(profile, {})
        base_params    = self._BASE_PARAMS.get(tool, {})
        caller_keys    = caller_keys or set()

        for key, value in profile_params.items():
            current = params.get(key)

            # additional_args — append rather than overwrite
            if key == "additional_args":
                if not value:  # profile says empty — keep existing
                    continue
                profile_val = value.strip()
                # Strip scan-type flags (-sX) if caller explicitly set scan_type
                if "scan_type" in caller_keys:
                    import re
                    profile_val = re.sub(r'-s[A-Za-z]+\s*', '', profile_val).strip()
                if not profile_val:
                    continue
                existing = (current or "").strip()
                if profile_val not in existing:
                    params[key] = f"{existing} {profile_val}".strip()
                continue

            # Numeric params — only override if value is still the base default
            # (meaning the caller did not explicitly set it)
            if isinstance(value, (int, float)):
                base_val = base_params.get(key)
                if current == base_val:  # still at default → profile can override
                    params[key] = value
                # else: caller set it explicitly → keep caller value
                continue

            # All other params (severity, etc.) — profile wins
            params[key] = value

        return params

    def handle_failure(
        self,
        tool: str,
        params: Dict[str, Any],
        failure_type: str,
    ) -> Dict[str, Any]:
        """
        Adaptive recovery after tool failure.
        Mirrors V6 handle_tool_failure() + FailureRecoverySystem.

        failure_type: "timeout" | "rate_limited" | "connection_refused"
        """
        recovery = dict(params)

        if failure_type == "timeout":
            # Double timeout, halve threads
            if "timeout" in recovery:
                recovery["timeout"] = recovery["timeout"] * 2
            for thread_key in ("threads", "concurrency"):
                if thread_key in recovery:
                    recovery[thread_key] = max(1, recovery[thread_key] // 2)
            logger.info(f"ParameterOptimizer: timeout recovery for {tool} — reduced concurrency")

        elif failure_type == "rate_limited":
            # Force stealth profile
            recovery = self._apply_profile(tool, recovery, "stealth")
            logger.info(f"ParameterOptimizer: rate_limited recovery for {tool} → stealth profile")

        elif failure_type == "connection_refused":
            # Try alternative ports or reduce aggressiveness
            logger.info(f"ParameterOptimizer: connection_refused for {tool} — no automatic recovery")

        return recovery
