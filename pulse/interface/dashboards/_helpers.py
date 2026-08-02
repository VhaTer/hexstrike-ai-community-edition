"""Shared display helpers for the dashboard UIs (extracted from pulse_app, R5)."""

import time


# Badge variant for RL profile
_RL_VARIANTS = {"stealth": "destructive", "conservative": "warning", "normal": "default", "aggressive": "secondary"}
def _rl_variant(profile: str) -> str:
    return _RL_VARIANTS.get(profile, "default")


# Badge variant for next suggested tool priority
_NST_VARIANTS = {"critical": "destructive", "high": "warning", "medium": "default", "low": "secondary"}


def _fmt_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "\u2014"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h {m}m"
    elif m:
        return f"{m}m {s}s"
    return f"{s}s"


def _fmt_rl_summary(rl: dict) -> str:
    """Format rate limit summary line."""
    n = rl.get("event_count", 0)
    if n == 0:
        return "No rate limit events detected"
    last = rl.get("last_detected")
    if last:
        ago = _fmt_duration(time.time() - last)
        return f"{n} event(s) \u00b7 Last {ago} ago"
    return f"{n} event(s)"
