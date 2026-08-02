"""CTF challenge tracker dashboard — extracted from pulse_app.py (R5)."""

from prefab_ui.app import PrefabApp
from prefab_ui.components import (
    Alert, AlertDescription, AlertTitle,
    Badge, Card, CardContent,
    Column, Heading, Icon, Muted, Row, Separator, Text,
)
from prefab_ui.components.charts import BarChart, ChartSeries

from pulse.infrastructure.singletons import get_ctf_manager
from pulse.tools.tool_registry_v2 import _registry


def ctf_dashboard() -> PrefabApp:
    """CTF challenge tracker — categories, tools, progress."""
    cm = get_ctf_manager()
    cats = cm.category_tools if hasattr(cm, "category_tools") else {}
    cats = cats or {}
    categories = sorted(cats.keys())
    _cat_tool_list = {
        c: [t for subs in (cats.get(c, {}) or {}).values() for t in subs]
        for c in categories
    }

    cat_chart = [
        {"category": c, "tools": len(_cat_tool_list.get(c, []))}
        for c in categories
    ]

    with PrefabApp() as app:
        with Column(gap=4, css_class="p-4"):
            with Row(gap=3, align="center"):
                Icon(name="swords", size="default")
                Heading("CTF Challenge Dashboard", css_class="text-lg font-bold")
                Badge(f"{len(categories)} categories", variant="secondary")
            Separator()
            if categories:
                with Row(gap=4, css_class="items-start"):
                    with Column(gap=2, css_class="flex-1"):
                        Muted("Categories & Tools", css_class="text-xs uppercase tracking-wider")
                        for cat in categories:
                            tools = _cat_tool_list.get(cat, [])
                            with Card():
                                with CardContent(css_class="p-3"):
                                    with Row(gap=2, align="center"):
                                        Badge(cat, variant="default")
                                        Text(f"{len(tools)} tools")
                                    Text(", ".join(tools[:8]), css_class="text-xs text-muted mt-1")
                    with Column(gap=2, css_class="w-64"):
                        Muted("Tool Coverage", css_class="text-xs uppercase tracking-wider")
                        if cat_chart:
                            with Card():
                                with CardContent(css_class="p-3"):
                                    BarChart(
                                        data=cat_chart,
                                        series=[ChartSeries(data_key="tools", label="Tools")],
                                        x_axis="category",
                                        show_legend=False,
                                    )
            else:
                with Alert(variant="info"):
                    AlertTitle("No CTF data")
                    AlertDescription("No CTF categories loaded. Run a CTF challenge first.")
    return app
