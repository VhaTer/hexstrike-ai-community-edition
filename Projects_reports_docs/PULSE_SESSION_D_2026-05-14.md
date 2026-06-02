# Pulse Session D — 2026-05-14

De : Claude
Pour : HexDevMaster
Validé par : Nexus

---

## Diagnostic — 5 tests failed

### Les 2 fails identifiés (test_hexstrike_server_routes.py)

```
FAILED test_build_dashboard_status_degraded_when_tools_missing
FAILED test_build_dashboard_status_healthy_when_all_tools_present
```

**Root cause** : `_build_dashboard_response()` appelle
`enhanced_process_manager.resource_monitor.get_current_usage()` sans mock.
En environnement test, `enhanced_process_manager` est le vrai singleton —
`resource_monitor.get_current_usage()` peut lever une exception ou retourner
des données inattendues selon l'état du système.

**Fix** :

```python
def test_build_dashboard_status_healthy_when_all_tools_present():
    from hexstrike_server import _build_dashboard_response

    all_tools_present = {
        "nmap": True, "curl": True, "python3": True,
        "subfinder": True, "amass": True, "httpx": True, "katana": True,
        "nikto": True, "sqlmap": True, "gobuster": True, "ffuf": True,
        "nuclei": True, "airmon-ng": True, "airodump-ng": True,
        "aircrack-ng": True, "msfconsole": True, "searchsploit": True,
    }
    mock_usage = {"cpu_percent": 10, "memory_percent": 20, "disk_percent": 30}

    with (
        patch("hexstrike_server._get_tool_availability", return_value=all_tools_present),
        patch("hexstrike_server.enhanced_process_manager") as mock_pm,
    ):
        mock_pm.resource_monitor.get_current_usage.return_value = mock_usage
        result = _build_dashboard_response()

    assert result["status"] == "healthy"
    assert result["all_essential_tools_available"] is True


def test_build_dashboard_status_degraded_when_tools_missing():
    from hexstrike_server import _build_dashboard_response

    no_tools = {tool: False for tool in [
        "nmap", "curl", "python3", "subfinder", "amass", "httpx", "katana",
        "nikto", "sqlmap", "gobuster", "ffuf", "nuclei",
        "airmon-ng", "airodump-ng", "aircrack-ng", "msfconsole", "searchsploit",
    ]}
    mock_usage = {"cpu_percent": 10, "memory_percent": 20, "disk_percent": 30}

    with (
        patch("hexstrike_server._get_tool_availability", return_value=no_tools),
        patch("hexstrike_server.enhanced_process_manager") as mock_pm,
    ):
        mock_pm.resource_monitor.get_current_usage.return_value = mock_usage
        result = _build_dashboard_response()

    assert result["status"] == "degraded"
    assert result["all_essential_tools_available"] is False
```

**Vérifie les 3 autres fails** — non identifiés dans les logs fournis :

```bash
pytest tests/test_hexstrike_server_routes.py -v --tb=short 2>&1 | head -60
```

Probablement le même pattern — mock `enhanced_process_manager` manquant.

---

## Renderer Prefab UI — Solution Session D

### Root cause confirmée (analysée dans le handoff Session C)

Deux renderers incompatibles :
- **CDN** : `window.__FASTMCP_PREVIEW__` → `<sW>` → Kitchen Sink (ignore les données)
- **Bundled** : `window.__FASTMCP_PREVIEW__` → `<n6n>` → Kitchen Sink

**Seul chemin valide** :
Bundled sans `__FASTMCP_PREVIEW__` + data dans `<script id="prefab:initial-data">`
dans `<head>` avant le JS bundlé → `Jor()` trouve les données au module init →
`lze=false` → `<Qor>` rend le vrai view tree.

### Strategy Session D — Custom HTML wrapper

`pulse_working.html` est la bonne direction. Script à affiner :

```python
# fix_export.py — version propre Session D
import json
import subprocess
import re
from pathlib import Path

def generate_pulse_html():
    # 1. Générer l'export bundlé standard
    result = subprocess.run(
        ["python3", "export_dashboard.py", "--bundled"],
        capture_output=True, text=True
    )
    bundled_path = Path("/tmp/pulse_bundled.html")
    html = bundled_path.read_text(encoding="utf-8")

    # 2. Extraire la data state du PrefabApp
    from pulse_app import pulse_dashboard
    app_instance = pulse_dashboard()
    state_json = json.dumps(app_instance.state, ensure_ascii=False, indent=None)

    # 3. Injecter dans <head> AVANT le script bundlé
    # La clé : id="prefab:initial-data" — Jor() cherche exactement ce id
    data_tag = f'<script id="prefab:initial-data" type="application/json">{state_json}</script>'

    # 4. Retirer __FASTMCP_PREVIEW__ s'il existe
    html = re.sub(r'window\.__FASTMCP_PREVIEW__\s*=\s*[^;]+;', '', html)

    # 5. Placer data_tag dans <head> avant le premier <script src=...>
    html = re.sub(
        r'(<script[^>]+src[^>]+>)',
        f'{data_tag}\n\\1',
        html,
        count=1
    )

    output = Path("/tmp/pulse_working.html")
    output.write_text(html, encoding="utf-8")
    print(f"Generated: {output} ({output.stat().st_size // 1024} KB)")
    return output

if __name__ == "__main__":
    generate_pulse_html()
```

### Vérification browser

```bash
# Générer
python3 fix_export.py

# Ouvrir dans Chromium
chromium /tmp/pulse_working.html &

# Si blank — inspecter la console pour :
# 1. "Jor() returns null" → data pas trouvée, vérifier id="prefab:initial-data"
# 2. "t.some is not a function" → ForEach key mismatch (voir ci-dessous)
# 3. "Connection error" → lze=true encore actif, __FASTMCP_PREVIEW__ pas retiré
```

### Fix possible — ForEach "t.some is not a function"

Si l'erreur persiste avec `ForEach`, c'est un mismatch entre le key attendu
par le renderer et la structure des données.

**Vérification** :

```python
# Dans pulse_app.py, get_scope() retourne :
"tools_used": [{"name": t} for t in all_tools]

# Le ForEach dans la vue accède :
with ForEach("scope_tools") as tool:
    Badge(tool.name, variant="outline")
```

Si le renderer attend une liste de strings et non de dicts → changer :

```python
# Option A — liste de strings
"scope_tools": sorted(active["tools"])  # ["nmap", "gobuster", ...]

# Et dans la vue :
with ForEach("scope_tools") as tool:
    Badge(tool, variant="outline")  # tool est directement la string
```

Si `surface_techs` a le même problème :
```python
# surface_techs est déjà une liste de strings → OK
"surface_techs": sorted(techs)  # ["Apache", "PHP", ...]
```

---

## Session D — Layout redesign (3+2+1 grid)

### Objectif

Passer du layout linéaire actuel (8 sections empilées) au layout C&C
validé par Nexus : données terrain organisées en grille, header compact,
footer stats.

### Structure cible

```
┌─────────────────────────────────────────────────────────┐
│ HEADER — logo · version · uptime · RAM · tools · status │
├─────────────────────────────────────────────────────────┤
│ SCOPE BAR — target · type · objectif · phase            │
├─────────────────┬──────────────────┬────────────────────┤
│ SURFACE         │ FINDINGS         │ PLAN IDE           │
│ ports, techs,   │ critical/high/   │ steps numérotés    │
│ risk, WAF       │ medium/info      │ score live visible │
├─────────────────┴──────────────────┴────────────────────┤
│ HISTORY (scope-filtered)  │ TOOLS recommandés           │
│ tool/target/age/status    │ grid 3x2, score live        │
├───────────────────────────┴─────────────────────────────┤
│ INTELLIGENCE — baseline vs live vs blended              │
├─────────────────────────────────────────────────────────┤
│ FOOTER — exéc · cache hit · timeouts · IDE adaptatif    │
└─────────────────────────────────────────────────────────┘
```

### Implémentation Prefab

```python
@app.ui()
def pulse_dashboard() -> PrefabApp:
    """Open the Pulse dashboard."""
    # ... data loading (inchangé) ...

    with Column(gap=0) as view:

        # ── Header ─────────────────────────────────────────────────────
        with Row(gap=2, align="center", css_class="p-3 border-b flex-wrap"):
            Badge(f"{rx_version}", variant="destructive")
            Text(" · ")
            Text(f"{rx_uptime}")
            Text(" · ")
            Text(f"{rx_ram}")
            Text(" · ")
            Text(f"{rx_tools}")
            Text(" · ")
            Badge(f"{rx_status}", variant=rx_status_var)

        # ── Scope bar ──────────────────────────────────────────────────
        with Row(gap=3, align="center",
                 css_class="p-2 px-4 bg-muted/30 border-b flex-wrap"):
            Muted("SCOPE")
            Text(f"{rx_scope_target}", css_class="font-bold")
            Badge(f"{rx_scope_type}", variant="outline")
            Muted(f"{rx_scope_summary}")

        # ── Grid 3 colonnes ────────────────────────────────────────────
        with Row(gap=4, css_class="p-4 items-start"):

            # Col 1 — Surface
            with Column(gap=2, css_class="flex-1"):
                Muted("SURFACE", css_class="text-xs uppercase tracking-wider")
                with Card():
                    with CardContent(css_class="p-3"):
                        with Column(gap=2):
                            with Row(gap=2, align="center"):
                                Badge(f"{rx_risk}", variant=rx_risk_var)
                                Muted(f"{rx_ports_display}")
                            with Row(gap=1, css_class="flex-wrap"):
                                with ForEach("surface_ports") as p:
                                    Badge(p.service, variant="outline")
                            with Row(gap=1, css_class="flex-wrap"):
                                with ForEach("surface_techs") as t:
                                    Badge(t, variant="secondary")

            # Col 2 — Findings
            with Column(gap=2, css_class="flex-1"):
                Muted("FINDINGS", css_class="text-xs uppercase tracking-wider")
                DataTable(
                    columns=[
                        DataTableColumn(key="severity", header="Sev"),
                        DataTableColumn(key="finding",  header="Finding"),
                        DataTableColumn(key="details",  header="Details"),
                    ],
                    rows=Rx("findings"),
                )

            # Col 3 — Plan IDE
            with Column(gap=2, css_class="flex-1"):
                Muted("PLAN IDE", css_class="text-xs uppercase tracking-wider")
                Muted(f"{rx_plan_summary}")
                DataTable(
                    columns=[
                        DataTableColumn(key="num",          header="#"),
                        DataTableColumn(key="tool",         header="Tool"),
                        DataTableColumn(key="prob_display", header="Prob"),
                        DataTableColumn(key="eta_display",  header="ETA"),
                    ],
                    rows=Rx("plan_steps"),
                )

        Separator()

        # ── Grid 2 colonnes ────────────────────────────────────────────
        with Row(gap=4, css_class="p-4 items-start"):

            # Col 1 — History
            with Column(gap=2, css_class="flex-1"):
                Muted("HISTORY", css_class="text-xs uppercase tracking-wider")
                DataTable(
                    columns=[
                        DataTableColumn(key="tool",              header="Tool"),
                        DataTableColumn(key="age",               header="When"),
                        DataTableColumn(key="status",            header="✓"),
                        DataTableColumn(key="execution_display", header="Time"),
                    ],
                    rows=Rx("history"),
                )

            # Col 2 — Active + Tools recommandés
            with Column(gap=2, css_class="flex-1"):
                Muted("ACTIVE", css_class="text-xs uppercase tracking-wider")
                with Card():
                    with CardContent(css_class="p-3"):
                        with Row(gap=4):
                            Metric(label="Processes", value=Rx("active_processes"))
                            Metric(label="Workers",   value=Rx("active_workers"))
                            Metric(label="Queued",    value=Rx("active_queue"))

        Separator()

        # ── Intelligence ───────────────────────────────────────────────
        with Column(gap=2, css_class="p-4"):
            Muted("INTELLIGENCE", css_class="text-xs uppercase tracking-wider")
            DataTable(
                columns=[
                    DataTableColumn(key="tool",     header="Tool"),
                    DataTableColumn(key="baseline", header="Baseline"),
                    DataTableColumn(key="live",     header="Live"),
                    DataTableColumn(key="blended",  header="Blended"),
                    DataTableColumn(key="runs",     header="Runs"),
                ],
                rows=Rx("intelligence"),
            )

        Separator()

        # ── Footer ─────────────────────────────────────────────────────
        with Row(gap=4, align="center",
                 css_class="p-2 px-4 bg-muted/20 border-t flex-wrap"):
            Muted(f"{rx_version}")
            # Stats ops depuis _op_metrics — à ajouter dans state
            # Muted(f"108 exéc · 94% succès · cache 67% · 3 timeouts")
```

**Note** : le footer stats nécessite 4 nouvelles clés dans le state :
`total_runs_display`, `success_rate_display`, `cache_hit_display`,
`timeout_count_display`. À ajouter dans `get_overview()` ou directement
dans `pulse_dashboard()` depuis `_op_metrics.summary()`.

---

## Ordre d'exécution Session D

```
1. Fix 2 tests server_routes (mock enhanced_process_manager) — 15 min
2. Vérifier les 3 autres fails
3. pytest --tb=short → vérifier ≥ 2577 passed
4. python3 fix_export.py → tester pulse_working.html dans browser
5. Si blank → inspecter console (Jor null / t.some / lze)
6. Si rendu OK → implémenter layout 3+2+1 dans pulse_app.py
7. Régénérer pulse_working.html → valider rendu final
8. prefab serve fix — optionnel si le workflow export fonctionne
9. Claude Desktop config fix — tester pulse_dashboard() en MCP réel
```

---

## État Git avant commit

```
M  .gitignore
M  hexstrike.py
M  pulse_app.py
M  server_core/operational_metrics.py
?? export_dashboard.py
?? serve_pulse.py
?? tests/test_hexstrike_cli.py
?? tests/test_pulse_app.py
```

Décision avant commit — `export_dashboard.py` et `serve_pulse.py` :
- Si utilisés en workflow → committer avec doc dans README
- Si debug temporaire → ajouter à `.gitignore`

---

## Handoff attendu

`Projects_reports_docs/AGENT_HANDOFF_2026-05-14_SD.md` avec :
- fix tests : avant/après pytest
- renderer : pulse_working.html fonctionne ou bloqué sur quoi exactement
- layout redesign : implémenté ou pas
- décision export_dashboard.py / serve_pulse.py
- commit hash
