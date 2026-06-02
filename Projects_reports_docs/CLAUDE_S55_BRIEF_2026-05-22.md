# Session 55 — Search Transform Refactor (RegexSearch + Lean always_visible)

**Date** : 2026-05-22

## Résumé

Optimisation du token budget à l'initialisation du serveur MCP. `list_tools()` passe de 12 à 6 outils fixes. Tous les pentest tools et get_* individuels ne sont plus pré-chargés — découverts via `search_tools()` à la demande.

## Changements

### `mcp_core/server_setup.py` (3 edits)

1. **Import** : `BM25SearchTransform` → `RegexSearchTransform` (ligne 40-42)
2. **Config** (ligne 978-984) :
   - `always_visible` = `["scan", "get_live_dashboard", "scan_background", "pulse_dashboard"]`
   - `max_results` = 10 (était 15)
   - Commentaire mis à jour
3. **Commentaire ligne 2170** : `BM25SearchTransform` → `RegexSearchTransform`

### `AGENTS.md` (2 edits)

1. Key conventions : nouvelle section Search transform + Token budget
2. Nouvelle entrée Session 55

## Token budget

| Élément | Avant | Après |
|---------|-------|-------|
| always_visible | 10 tools (dont 3 pentest + 6 get_*) | 4 tools (scan, get_live_dashboard, scan_background, pulse_dashboard) |
| Meta-tools | search_tools + call_tool | search_tools + call_tool |
| **Total list_tools()** | **12** | **6** |
| Découverte pentest | pré-chargée inutilement | via search_tools() purpose-driven |
| max_results/search | 15 | 10 |

## Justification

- `get_live_dashboard` subsumme tous les get_* individuels (overview, surface, findings, plan, scope) → les avoir en always_visible est redondant
- Les pentest tools (nmap, whatweb, sqlmap) ne sont utiles qu'après un `scan()` → pas besoin de les pré-charger
- `RegexSearchTransform` = même économie de tokens que BM25, mais zéro overhead d'indexation, matching prévisible par pattern
- Le LLM voit 6 outils fixes, découvre le reste via `search_tools("recon")` ou `search_tools("nmap")` — consumption au moment du besoin

## Fichiers modifiés

```
mcp_core/server_setup.py      | 6 changes (3 edits)
AGENTS.md                      | 2 changes (new section + updated conventions)
```

## Tests

- 10 scan-related tests pass (test_server_setup_unit.py -k TestScanBackground + TestScan)
- Import runtime : RegexSearchTransform + serialize_tools_for_output_markdown OK
- Instantiation : max_results=10, always_visible OK

## RPI Status

- **IP** : `10.253.249.151` (réseau hotspot "retahv", phone tethering)
- **Host** : RPi, OpenSSH 9.2p1 Debian 12
- **User** : `HexStrikePI` (pas `pi`)
- **Auth** : `publickey` only (password désactivé)
- **Problème** : aucune clé de cette machine match l'`authorized_keys` du RPI. L'utilisateur avait copié une clé depuis `/etc/ssh/` mais aucune des 3 host keys (rsa/ecdsa/ed25519) ne fonctionne.
- **Next** : carte SD à re-flasher avec clé SSH pré-configurée, dispo dimanche au plus tard.
- **Validation** : `Projects_reports_docs/CHECKLIST_RPI_VALIDATION_2026-05-21.md` prête mais bloquée.

## Discussion technique (token budget)

- **CodeMode** (`fastmcp.experimental.transforms.code_mode`) : permet des niveaux de détail brief/detailed/full sur les search results. Identifié comme piste future mais trop experimental pour maintenant (sandbox + extra `fastmcp[code-mode]` requis).
- **Detail levels** non disponibles dans `RegexSearchTransform` standard — faudrait un custom transform.
- **Tags** (`tool.meta._fastmcp.tags`) : potentiel pour filtrage par catégorie, nécessite étiquetage de tous les outils → pas prioritaire.
- **Conclusion** : `RegexSearchTransform` avec `always_visible=4` + `max_results=10` est le bon équilibre tokens/accessibilité pour maintenant.

## Test count

- **2700+ passed, 1 skipped, 2 pre-existing flaky — 0 regressions** (stable depuis Session 54)
