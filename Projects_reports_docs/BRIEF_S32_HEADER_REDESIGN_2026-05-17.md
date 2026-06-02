# S32 — Header Redesign

**Objectif :** Remplacer le header texte actuel par des icônes Lucide + sparklines CPU/RAM.

**Header actuel (pulse_app.py ~ligne 555-565) :**
```
PULSE v0.8.0 • up Xh Ym • RAM X/Y GB • N tools • ✓ healthy
```

**Nouveau header :**
```
🖥️ 45%  💾 6.2/16G  💿 72%  ═══ HexStrike PULSE v0.8.0 ═══ N tools
▁▃▅▇▃▅▂ sparkline CPU 30s
```

**Composants Prefab à utiliser :**
- `Icon(name="cpu", size="sm")` → CPU
- `Icon(name="hard-drive", size="sm")` → RAM/Disk
- `Sparkline(data=[...], variant="success", fill=True)` → mini graphiques
- `Tooltip(content="...", side="top")` → infos au survol
- `Progress(value=45)` → barres CPU/Disk

**Fichiers à modifier :**
- `pulse_app.py` : `pulse_dashboard()` (layout header) + `get_overview()` (backend data)
- `mcp_core/server_setup.py` : si besoin de data sparkline depuis ResourceMonitor

**Scope :** Header UNIQUEMENT. Pas de nouveaux panels.

**Tests :** `python -m pytest tests/test_pulse_app.py -v -q --tb=short`
