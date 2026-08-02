"""Dashboard UIs — pure prefab_ui rendering, wired by pulse_app.py (R5).

Each module builds a PrefabApp from data; pulse_app.py registers them on
the FastMCPApp via @app.ui() pass-through wrappers (the "passerelle").
"""

from pulse.interface.dashboards._helpers import (
    _fmt_duration, _fmt_rl_summary, _rl_variant,
    _NST_VARIANTS, _RL_VARIANTS,
)
from pulse.interface.dashboards.ctf_dashboard import ctf_dashboard
from pulse.interface.dashboards.pentest_report import pentest_report
from pulse.interface.dashboards.pulse_dashboard import (
    _STATE_KEY_SOURCES, _build_ui_state, pulse_dashboard,
)
from pulse.interface.dashboards.recon_summary import recon_summary

__all__ = [
    "_build_ui_state", "_STATE_KEY_SOURCES",
    "_fmt_duration", "_fmt_rl_summary", "_rl_variant",
    "_NST_VARIANTS", "_RL_VARIANTS",
    "ctf_dashboard", "pentest_report", "pulse_dashboard", "recon_summary",
]
