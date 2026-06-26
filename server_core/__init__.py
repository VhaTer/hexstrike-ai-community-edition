"""
HexStrike Core Module
Shared infrastructure components for the HexStrike framework.

Track 2 / Chemin C refactor:
- Only active-path components are imported eagerly here
- Intelligence/workflow classes are available but NOT instantiated at import time
- Use server_core.singletons.get_*() accessors for lazy singleton access
"""

import importlib
import sys

# ── Re-exported names — imported lazily on first access ──────────────────────
# Format: name → module_path (same name in target)
#         name → (module_path, original_name) for aliased re-exports

_LAZY: dict = {
    # Active path
    "HexStrikeCache":                  "server_core.cache",
    "ModernVisualEngine":              "server_core.modern_visual_engine",
    "_execute_command":                ("server_core.command_executor", "execute_command"),
    "_execute_command_with_recovery":  ("server_core.recovery_executor", "execute_command_with_recovery"),
    "EnhancedProcessManager":          "server_core.enhanced_process_manager",
    "TechnologyDetector":              "server_core.technology_detector",
    "ParameterOptimizer":              "server_core.parameter_optimizer",
    "RateLimitDetector":               "server_core.rate_limit_detector",
    "_op_metrics":                     "server_core.operational_metrics",
    "HexStrikeLoggingMiddleware":      "server_core.hexstrike_middleware",
    "HexStrikeSessionMiddleware":      "server_core.hexstrike_middleware",
    "_rebuild_command_with_params":    ("server_core.command_params", "rebuild_command_with_params"),
    "_determine_operation_type":       ("server_core.operation_types", "determine_operation_type"),
    "file_manager":                    "server_core.file_ops",
    "TelemetryCollector":              "server_core.telemetry_collector",
    # Error handling
    "ErrorType":                       "server_core.error_handling",
    "RecoveryAction":                  "server_core.error_handling",
    "ErrorContext":                    "server_core.error_handling",
    "IntelligentErrorHandler":         "server_core.error_handling",
    "GracefulDegradation":             "server_core.error_handling",
    # Active path singletons
    "cache":                           "server_core.singletons",
    "telemetry":                       "server_core.singletons",
    "enhanced_process_manager":        "server_core.singletons",
    "error_handler":                   "server_core.singletons",
    "degradation_manager":             "server_core.singletons",
    "ROCKYOU_PATH":                    "server_core.singletons",
    "COMMON_DIRB_PATH":                "server_core.singletons",
    "COMMON_DIRSEARCH_PATH":           "server_core.singletons",
    # Intelligence classes (not instantiated — use get_*() from singletons)
    "CVEIntelligenceManager":          "server_core.intelligence.cve_intelligence_manager",
    "IntelligentDecisionEngine":       "server_core.intelligence.intelligent_decision_engine",
    "AIExploitGenerator":              "server_core.ai_exploit_generator",
    "VulnerabilityCorrelator":         "server_core.vulnerability_correlator",
    "BugBountyWorkflowManager":        "server_core.workflows.bugbounty.workflow",
    "FileUploadTestingFramework":      "server_core.workflows.bugbounty.testing",
    "BugBountyTarget":                 "server_core.workflows.bugbounty.target",
    "CTFChallenge":                    "server_core.workflows.ctf.CTFChallenge",
    "CTFWorkflowManager":              "server_core.workflows.ctf.workflowManager",
    "CTFToolManager":                  "server_core.workflows.ctf.toolManager",
    "CTFChallengeAutomator":           "server_core.workflows.ctf.automator",
    "CTFTeamCoordinator":              "server_core.workflows.ctf.coordinator",
    # Misc
    "SessionStore":                    "server_core.session_store",
    "WordlistStore":                   "server_core.wordlist_store",
    "env_manager":                     "server_core.python_env_manager",
    "FailureRecoverySystem":           "server_core.failure_recovery_system",
    "PerformanceMonitor":              "server_core.performance_monitor",
}


def __getattr__(name: str):
    """Lazy import on first attribute access."""
    if name in _LAZY:
        spec = _LAZY[name]
        if isinstance(spec, str):
            mod = importlib.import_module(spec)
            val = getattr(mod, name, mod)
        else:
            mod = importlib.import_module(spec[0])
            val = getattr(mod, spec[1])
        setattr(sys.modules[__name__], name, val)
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Active path
    "HexStrikeCache", "ModernVisualEngine", "TelemetryCollector",
    "EnhancedProcessManager", "TechnologyDetector", "ParameterOptimizer",
    "RateLimitDetector", "HexStrikeLoggingMiddleware", "HexStrikeSessionMiddleware",
    "ErrorType", "RecoveryAction", "ErrorContext",
    "IntelligentErrorHandler", "GracefulDegradation",
    "_execute_command", "_execute_command_with_recovery",
    "_rebuild_command_with_params", "_determine_operation_type", "file_manager",
    # Active singletons
    "cache", "telemetry", "enhanced_process_manager",
    "error_handler", "degradation_manager",
    "ROCKYOU_PATH", "COMMON_DIRB_PATH", "COMMON_DIRSEARCH_PATH",
    # Intelligence classes (not instantiated)
    "CVEIntelligenceManager", "IntelligentDecisionEngine",
    "AIExploitGenerator", "VulnerabilityCorrelator",
    "BugBountyWorkflowManager", "FileUploadTestingFramework", "BugBountyTarget",
    "CTFChallenge", "CTFWorkflowManager", "CTFToolManager",
    "CTFChallengeAutomator", "CTFTeamCoordinator",
    # Misc
    "SessionStore", "WordlistStore", "env_manager",
    "FailureRecoverySystem", "PerformanceMonitor", "_op_metrics",
]
