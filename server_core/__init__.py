"""
HexStrike Core Module
Shared infrastructure components for the HexStrike framework.

Track 2 / Chemin C refactor:
- Only active-path components are imported eagerly here
- Intelligence/workflow classes are available but NOT instantiated at import time
- Use server_core.singletons.get_*() accessors for lazy singleton access
"""

# ── Active path — always needed ───────────────────────────────────────────────
from .cache import HexStrikeCache
from .modern_visual_engine import ModernVisualEngine
from .command_executor import execute_command as _execute_command
from .recovery_executor import execute_command_with_recovery as _execute_command_with_recovery
from .enhanced_process_manager import EnhancedProcessManager
from .technology_detector import TechnologyDetector
from .parameter_optimizer import ParameterOptimizer
from .rate_limit_detector import RateLimitDetector
from .operational_metrics import _op_metrics
from .hexstrike_middleware import HexStrikeLoggingMiddleware, HexStrikeSessionMiddleware
from .command_params import rebuild_command_with_params as _rebuild_command_with_params
from .operation_types import determine_operation_type as _determine_operation_type
from .file_ops import file_manager
from .telemetry_collector import TelemetryCollector
from .error_handling import (
    ErrorType,
    RecoveryAction,
    ErrorContext,
    IntelligentErrorHandler,
    GracefulDegradation,
)

# ── Active path singletons ────────────────────────────────────────────────────
from .singletons import (
    cache,
    telemetry,
    enhanced_process_manager,
    error_handler,
    degradation_manager,
    ROCKYOU_PATH,
    COMMON_DIRB_PATH,
    COMMON_DIRSEARCH_PATH,
)

# ── Intelligence / workflow classes — importable but NOT instantiated ─────────
# Use get_*() from singletons for lazy singleton access:
#   from server_core.singletons import get_cve_intelligence
from .intelligence.cve_intelligence_manager import CVEIntelligenceManager
from .intelligence.intelligent_decision_engine import IntelligentDecisionEngine
from .ai_exploit_generator import AIExploitGenerator
from .vulnerability_correlator import VulnerabilityCorrelator
from .workflows.bugbounty.workflow import BugBountyWorkflowManager
from .workflows.bugbounty.testing import FileUploadTestingFramework
from .workflows.bugbounty.target import BugBountyTarget
from .workflows.ctf.CTFChallenge import CTFChallenge
from .workflows.ctf.workflowManager import CTFWorkflowManager
from .workflows.ctf.toolManager import CTFToolManager
from .workflows.ctf.automator import CTFChallengeAutomator
from .workflows.ctf.coordinator import CTFTeamCoordinator

# ── Misc ──────────────────────────────────────────────────────────────────────
from .session_store import SessionStore
from .wordlist_store import WordlistStore
from .python_env_manager import env_manager
from .failure_recovery_system import FailureRecoverySystem
from .performance_monitor import PerformanceMonitor

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
