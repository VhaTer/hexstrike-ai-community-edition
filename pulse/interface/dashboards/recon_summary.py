"""Recon summary dashboard — extracted from pulse_app.py (R5)."""

import time

from prefab_ui.app import PrefabApp
from prefab_ui.components import (
    Badge, Card, CardContent, Column,
    DataTable, DataTableColumn, Heading, Icon, Metric,
    Muted, Row, Separator, Table, TableBody,
    TableCaption, TableCell, TableHead, TableHeader,
    TableRow, Text,
)

from pulse.interface.dashboards._helpers import _fmt_duration

_risk_color = {"high": "destructive", "medium": "warning", "low": "success", "unknown": "secondary"}


def recon_summary(target: str) -> PrefabApp:
    """Reconnaissance summary — ports, tech, cache, history."""
    from pulse.interface.pulse_app import _cache_for_target, get_history, get_surface  # lazy (avoid import cycle)

    surface = get_surface(target)
    history = get_history(target, limit=20)
    cache_entries = _cache_for_target(target) if target else []

    ports = surface.get("ports", [])
    techs = surface.get("technologies", [])
    risk = surface.get("risk_level", "unknown")

    now = time.time()
    cache_info = [
        {"tool": e.get("tool", "?"), "age": _fmt_duration(now - e.get("timestamp", now)) if e.get("timestamp") else "?"}
        for e in cache_entries[-10:]
    ]

    with PrefabApp() as app:
        with Column(gap=4, css_class="p-4"):
            with Row(gap=3, align="center"):
                Icon(name="compass", size="default")
                Heading(f"Recon — {target}", css_class="text-lg font-bold")
                Badge(risk.upper(), variant=_risk_color.get(risk, "outline"))
            with Row(gap=4, css_class="flex-wrap"):
                Metric(label="Open Ports", value=str(len(ports)))
                Metric(label="Technologies", value=str(len(techs)))
                Metric(label="Cache entries", value=str(len(cache_entries)))
                Metric(label="History entries", value=str(len(history)))
            Separator()
            if ports:
                Muted("Ports & Services", css_class="text-xs uppercase tracking-wider")
                with Table():
                    TableCaption(content="Port scan results")
                    with TableHeader():
                        with TableRow():
                            with TableHead(): Text("Port")
                            with TableHead(): Text("Service")
                            with TableHead(): Text("State")
                    with TableBody():
                        for p in ports:
                            with TableRow():
                                with TableCell(): Text(str(p.get("port", "")))
                                with TableCell(): Text(p.get("service", ""))
                                with TableCell(): Text(p.get("state", ""))
            if techs:
                Separator()
                Muted("Technology Stack", css_class="text-xs uppercase tracking-wider")
                with Card():
                    with CardContent(css_class="p-3"):
                        with Row(gap=2, css_class="flex-wrap"):
                            for t in techs:
                                Badge(t, variant="outline")
            if cache_info:
                Separator()
                Muted("Cache (recent)", css_class="text-xs uppercase tracking-wider")
                with DataTable(
                    columns=[
                        DataTableColumn(key="tool", header="Tool"),
                        DataTableColumn(key="age", header="Age"),
                    ],
                    rows=cache_info,
                ):
                    pass
            if history:
                Separator()
                Muted("Recent Tools", css_class="text-xs uppercase tracking-wider")
                with DataTable(
                    columns=[
                        DataTableColumn(key="tool", header="Tool"),
                        DataTableColumn(key="execution_display", header="Time"),
                        DataTableColumn(key="status", header="Status"),
                    ],
                    rows=history[-8:],
                ):
                    pass
    return app
