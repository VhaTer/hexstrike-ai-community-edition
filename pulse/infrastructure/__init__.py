"""
HexStrike Core Module
Shared infrastructure components for the HexStrike framework.

Track 2 / Chemin C refactor:
- Only active-path components are imported eagerly here
- Intelligence/workflow classes are available but NOT instantiated at import time
- Use pulse.infrastructure.singletons.get_*() accessors for lazy singleton access
"""

import importlib
import sys

# ── Re-exported names — imported lazily on first access ──────────────────────
# Format: name → module_path (same name in target)
#         name → (module_path, original_name) for aliased re-exports

_LAZY: dict = {
    # Active path
    "HexStrikeCache":                  "pulse.infrastructure.cache",
    "ModernVisualEngine":              "pulse.infrastructure.modern_visual_engine",
    "_execute_command":                ("pulse.infrastructure.command_executor", "execute_command"),
    "_execute_command_with_recovery":  ("pulse.infrastructure.recovery_executor", "execute_command_with_recovery"),
    "EnhancedProcessManager":          "pulse.infrastructure.enhanced_process_manager",
    "TechnologyDetector":              "pulse.infrastructure.technology_detector",
    "ParameterOptimizer":              "pulse.infrastructure.parameter_optimizer",
    "RateLimitDetector":               "pulse.infrastructure.rate_limit_detector",
    "_op_metrics":                     "pulse.infrastructure.operational_metrics",
    "HexStrikeLoggingMiddleware":      "pulse.infrastructure.hexstrike_middleware",
    "HexStrikeSessionMiddleware":      "pulse.infrastructure.hexstrike_middleware",
    "_rebuild_command_with_params":    ("pulse.infrastructure.command_params", "rebuild_command_with_params"),
    "_determine_operation_type":       ("pulse.infrastructure.operation_types", "determine_operation_type"),
    "file_manager":                    "pulse.infrastructure.file_ops",
    "TelemetryCollector":              "pulse.infrastructure.telemetry_collector",
    # Error handling
    "ErrorType":                       "pulse.infrastructure.error_handling",
    "RecoveryAction":                  "pulse.infrastructure.error_handling",
    "ErrorContext":                    "pulse.infrastructure.error_handling",
    "IntelligentErrorHandler":         "pulse.infrastructure.error_handling",
    "GracefulDegradation":             "pulse.infrastructure.error_handling",
    # Active path singletons
    "cache":                           "pulse.infrastructure.singletons",
    "telemetry":                       "pulse.infrastructure.singletons",
    "enhanced_process_manager":        "pulse.infrastructure.singletons",
    "error_handler":                   "pulse.infrastructure.singletons",
    "degradation_manager":             "pulse.infrastructure.singletons",
    "ROCKYOU_PATH":                    "pulse.infrastructure.singletons",
    "COMMON_DIRB_PATH":                "pulse.infrastructure.singletons",
    "COMMON_DIRSEARCH_PATH":           "pulse.infrastructure.singletons",
    # Intelligence classes (not instantiated — use get_*() from singletons)
    "CVEIntelligenceManager":          "pulse.intelligence.cve_intelligence_manager",
    "IntelligentDecisionEngine":       "pulse.intelligence.intelligent_decision_engine",
    "AIExploitGenerator":              "pulse.intelligence.ai_exploit_generator",
    "VulnerabilityCorrelator":         "pulse.intelligence.vulnerability_correlator",
    "BugBountyWorkflowManager":        "pulse.workflows.bugbounty.workflow",
    "FileUploadTestingFramework":      "pulse.workflows.bugbounty.testing",
    "BugBountyTarget":                 "pulse.workflows.bugbounty.target",
    "CTFChallenge":                    "pulse.workflows.ctf.CTFChallenge",
    "CTFWorkflowManager":              "pulse.workflows.ctf.workflowManager",
    "CTFToolManager":                  "pulse.workflows.ctf.toolManager",
    "CTFChallengeAutomator":           "pulse.workflows.ctf.automator",
    "CTFTeamCoordinator":              "pulse.workflows.ctf.coordinator",
    # Misc
    "SessionStore":                    "pulse.infrastructure.session_store",
    "WordlistStore":                   "pulse.infrastructure.wordlist_store",
    "env_manager":                     "pulse.infrastructure.python_env_manager",
    "FailureRecoverySystem":           "pulse.infrastructure.failure_recovery_system",
    "PerformanceMonitor":              "pulse.infrastructure.performance_monitor",
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
