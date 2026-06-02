# Brief nouvelle session — 2026-05-16

## Contexte en 10 lignes

HexStrike Pulse v0.8.0 est un serveur MCP FastMCP 3.2.4 connecté à Claude Desktop via stdio.
130+ tools de pentest exposés comme DIRECT_TOOLS MCP.
Dashboard Pulse : 8 tools individuels légers dans pulse_app.py.
IntelligentDecisionEngine orchestre les plans d'attaque avec scores terrain adaptatifs.
ToolStatsStore feed fermé — blended_effectiveness() retourne des scores réels.
2577 tests, 0 regressions. Branch : feature/prefab-dashboard.

## Lancer le serveur

```bash
cd ~/hexstrike-ai-community-edition
source hexstrike-env/bin/activate
HEXSTRIKE_SEED_SCANS=scanme.nmap.org \
HEXSTRIKE_LOG_LEVEL=debug \
python3 hexstrike_mcp.py 2>pulse_debug.log &
```

## Les 8 tools disponibles

| Tool | Params | Taille |
|------|--------|--------|
| `get_overview` | — | ~105 tokens |
| `get_scope` | target | ~50 tokens |
| `get_surface` | target | ~42 tokens |
| `get_findings` | target | ~0 tokens |
| `get_plan` | target, objective | ~2855 tokens |
| `get_active_tools` | — | ~158 tokens |
| `get_history` | target, limit | ~0 tokens |
| `get_tool_intelligence` | — | ~441 tokens |

## Premier appel dans la nouvelle session

```
"Call get_overview then get_surface for scanme.nmap.org and show me the Pulse dashboard"
```

Claude reçoit les données, rend le dashboard HTML inline.

## Règle non-négociable

Dashboard : observe, plan, suggest. Pas d'exécution automatique.
