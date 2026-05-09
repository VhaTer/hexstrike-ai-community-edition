"""
HexStrike shared singletons — lazy loading.

Objects in Group A (active path) are instantiated eagerly at import time
because they are always needed by the active runtime.

Objects in Group B (intelligence / workflows) use lazy accessors: they are
instantiated only on first access, so server startup does not pay the memory
or init cost for components that may never be used in a given session.

Usage — active path (unchanged):
    from server_core.singletons import cache, telemetry, enhanced_process_manager
    from server_core.singletons import ROCKYOU_PATH, COMMON_DIRB_PATH

Usage — intelligence layer (lazy):
    from server_core.singletons import get_cve_intelligence
    cve_intel = get_cve_intelligence()   # instantiated on first call only
"""

import server_core.config_core as config_core

# ── GROUP A: Active path — always instantiated ────────────────────────────────

from .cache import HexStrikeCache
from .enhanced_process_manager import EnhancedProcessManager
from .error_handling import IntelligentErrorHandler, GracefulDegradation
from .enhanced_command_executor import telemetry  # noqa: F401 — re-export

cache                = HexStrikeCache()
enhanced_process_manager = EnhancedProcessManager()
error_handler        = IntelligentErrorHandler()
degradation_manager  = GracefulDegradation()

# Wordlist paths — resolved once, used by *_direct.py
ROCKYOU_PATH         = config_core.get_word_list_path("rockyou")
COMMON_DIRB_PATH     = config_core.get_word_list_path("common_dirb")
COMMON_DIRSEARCH_PATH = config_core.get_word_list_path("common_dirsearch")


# ── GROUP B: Lazy — instantiated on first access ──────────────────────────────
# These were always constructed at startup in the original singletons.py but
# are never imported by the active MCP path (server_setup.py, *_direct.py).
# They now live behind accessor functions so startup stays lightweight.

_session_store        = None
_wordlist_store       = None
_cve_intelligence     = None
_exploit_generator    = None
_vulnerability_correlator = None
_decision_engine      = None
_bugbounty_manager    = None
_fileupload_framework = None
_ctf_manager          = None
_ctf_tools            = None
_ctf_automator        = None
_ctf_coordinator      = None


def get_session_store():
    global _session_store
    if _session_store is None:
        from .session_store import SessionStore
        _session_store = SessionStore()
    return _session_store


def get_wordlist_store():
    global _wordlist_store
    if _wordlist_store is None:
        from .wordlist_store import WordlistStore
        _wordlist_store = WordlistStore()
    return _wordlist_store


def get_cve_intelligence():
    """CVEIntelligenceManager — NVD API + GitHub exploit search."""
    global _cve_intelligence
    if _cve_intelligence is None:
        from .intelligence.cve_intelligence_manager import CVEIntelligenceManager
        _cve_intelligence = CVEIntelligenceManager()
    return _cve_intelligence


def get_exploit_generator():
    """AIExploitGenerator — Python exploit templates from CVE data."""
    global _exploit_generator
    if _exploit_generator is None:
        from .ai_exploit_generator import AIExploitGenerator
        _exploit_generator = AIExploitGenerator()
    return _exploit_generator


def get_vulnerability_correlator():
    """VulnerabilityCorrelator — multi-stage attack chain discovery."""
    global _vulnerability_correlator
    if _vulnerability_correlator is None:
        from .vulnerability_correlator import VulnerabilityCorrelator
        _vulnerability_correlator = VulnerabilityCorrelator()
    return _vulnerability_correlator


def get_decision_engine():
    """IntelligentDecisionEngine — tool selection + attack chain planning."""
    global _decision_engine
    if _decision_engine is None:
        from .intelligence.intelligent_decision_engine import IntelligentDecisionEngine
        _decision_engine = IntelligentDecisionEngine()
    return _decision_engine


def get_bugbounty_manager():
    """BugBountyWorkflowManager — recon → hunting → business logic phases."""
    global _bugbounty_manager
    if _bugbounty_manager is None:
        from .workflows.bugbounty.workflow import BugBountyWorkflowManager
        _bugbounty_manager = BugBountyWorkflowManager()
    return _bugbounty_manager


def get_fileupload_framework():
    global _fileupload_framework
    if _fileupload_framework is None:
        from .workflows.bugbounty.testing import FileUploadTestingFramework
        _fileupload_framework = FileUploadTestingFramework()
    return _fileupload_framework


def get_ctf_manager():
    """CTFWorkflowManager — per-category CTF workflows with parallel execution."""
    global _ctf_manager
    if _ctf_manager is None:
        from .workflows.ctf.workflowManager import CTFWorkflowManager
        _ctf_manager = CTFWorkflowManager()
    return _ctf_manager


def get_ctf_tools():
    global _ctf_tools
    if _ctf_tools is None:
        from .workflows.ctf.toolManager import CTFToolManager
        _ctf_tools = CTFToolManager()
    return _ctf_tools


def get_ctf_automator():
    global _ctf_automator
    if _ctf_automator is None:
        from .workflows.ctf.automator import CTFChallengeAutomator
        _ctf_automator = CTFChallengeAutomator()
    return _ctf_automator


def get_ctf_coordinator():
    global _ctf_coordinator
    if _ctf_coordinator is None:
        from .workflows.ctf.coordinator import CTFTeamCoordinator
        _ctf_coordinator = CTFTeamCoordinator()
    return _ctf_coordinator


# ── Backward-compat aliases (deprecated — use get_*() accessors) ─────────────
# These allow existing code that does `from server_core.singletons import X`
# to keep working. They resolve lazily on first attribute access via a
# module-level __getattr__.

_LAZY_ALIASES = {
    "session_store":          get_session_store,
    "wordlist_store":         get_wordlist_store,
    "cve_intelligence":       get_cve_intelligence,
    "exploit_generator":      get_exploit_generator,
    "vulnerability_correlator": get_vulnerability_correlator,
    "decision_engine":        get_decision_engine,
    "bugbounty_manager":      get_bugbounty_manager,
    "fileupload_framework":   get_fileupload_framework,
    "ctf_manager":            get_ctf_manager,
    "ctf_tools":              get_ctf_tools,
    "ctf_automator":          get_ctf_automator,
    "ctf_coordinator":        get_ctf_coordinator,
}


def __getattr__(name: str):
    """Module-level lazy attribute resolution for backward-compat aliases."""
    if name in _LAZY_ALIASES:
        return _LAZY_ALIASES[name]()
    raise AttributeError(f"module 'server_core.singletons' has no attribute {name!r}")
