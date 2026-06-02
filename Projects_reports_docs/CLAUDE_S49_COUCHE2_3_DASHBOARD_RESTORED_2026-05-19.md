# Session 49 — Couche 2 + 3 + Dashboard restoration
**Date :** 2026-05-19
**Auteur :** Claude
**Pour :** HexDevMaster

---

## Résumé

3 deliverables en 1 session :
1. **Couche 2** — System prompt `instructions=` sur FastMCP
2. **Couche 3** — `next_suggested_tool` dans toutes les réponses (exec tools + app tools)
3. **Dashboard Prefab restauré** — annulation de l'erreur S48

2639 tests, 0 régressions.

---

## 1. Dashboard Prefab restauré

**Problème :** Session 48 j'avais supprimé `pulse_dashboard()` (UI entry point) + tout le template Prefab sans que l'utilisateur le demande — malentendu sur "on peut enlever".

**Correctif :**
- `git checkout HEAD -- pulse_app.py` restauré à l'état S39
- Correctifs S45 ré-appliqués : URL routing (`_TOOLS_NEED_URL`, `_TOOLS_NEED_URL_AS_TARGET`), cache write-back dans `scan()`, TargetStore record_scan
- `prefab-ui>=0.14.0,<0.15.0` restauré dans `requirements.txt`
- Tests xfail retirés

**Fichiers touchés :** `pulse_app.py`, `requirements.txt`, `tests/test_pulse_app.py`

---

## 2. Couche 2 — System prompt `instructions=`

### Fichier créé : `mcp_core/instructions.py`

80 lignes, constant `INSTRUCTIONS` qui couvre :
- Entry points : `scan()`, `get_live_dashboard()`, `run_security_tool()`, `run_async_tool()`
- Workflow order : overview → scope → surface → findings → plan
- Workflow prompts MCP : bug_bounty_recon, wifi_attack_chain, ctf_web_challenge, ctf_challenge, smb_lateral_movement, cloud_security_audit
- MCP Resources : `targets://`, `target://{target}`, `health://server`, `skill://{category}/SKILL.md`
- Tool naming : underscores, typed wrappers, `_profile` param (stealth/normal/aggressive)
- Caching : auto-populated, repeated calls return instantly
- Destructive tools : Metasploit, aireplay-ng, responder, mitm6, mdk4 require confirmation
- Skills : `get_tool_skill(tool_name)` per-category guidance

### Injection : `mcp_core/server_setup.py:738`

```python
FastMCP(
    "hexstrike-ai pulse",
    instructions=os.environ.get("HEXSTRIKE_INSTRUCTIONS", INSTRUCTIONS),
    transforms=transforms,
)
```

Override via `HEXSTRIKE_INSTRUCTIONS` env var.

### Caveat connu

Claude Desktop ignore `instructions` (bug Anthropic, toujours ouvert mai 2026). Claude Code (terminal) l'utilise correctement. Injecté pour future-proof + Claude Code.

### Fichiers touchés
- `mcp_core/instructions.py` — **NOUVEAU**
- `mcp_core/server_setup.py` — `import os` + `instructions=` kwarg

---

## 3. Couche 3 — `next_suggested_tool`

### Architecture deux niveaux

**Niveau 1 — Exec tools (130+) :** `_suggest_next_tool(tool_name, output, target)` dans `server_setup.py`

Injectée dans `finalize()` après `_normalize_tool_result()`. Chaque appel `run_security_tool()` retourne automatiquement `next_suggested_tool: {tool, reason}`.

Règles basées sur parsing de stdout :
| Tool output | Suggestion |
|---|---|
| nmap → port 80/443 | whatweb (fingerprinting) |
| nmap → port 445 | smbmap (shares) |
| nmap → port 22 | hydra (creds) |
| nmap → DB ports | sqlmap (weak auth) |
| whatweb → WordPress | wpscan (plugins/users) |
| whatweb → Joomla | joomscan (extensions) |
| whatweb → Drupal | nuclei (CVEs) |
| whatweb → générique | gobuster (dirs) |
| nuclei/nikto → SQLi | sqlmap (confirm/exploit) |
| nuclei/nikto → XSS | dalfox (validate) |
| nuclei/nikto → SSL | testssl (inspect) |
| nuclei/nikto → SMB/EternalBlue | metasploit (exploit) |
| gobuster/ffuf → dirs trouvés | nuclei (vuln scan) |
| hydra → creds trouvés | metasploit (exploit) |
| smbmap → shares accessibles | metasploit (exploit) |

**Niveau 2 — App tools :** `_suggest_next_from_context(surface, findings)` dans `pulse_app.py`

Basée sur données structurées (pas de parsing stdout) :
- Surface ports+services → détermine next tool
- Findings → détermine next tool (SQLi→sqlmap, XSS→dalfox, etc.)
- Ajoutée dans : `scan()`, `get_surface()`, `_collect_dashboard_state()`, `get_live_dashboard()`

### Fichiers touchés
- `mcp_core/server_setup.py` — `_suggest_next_tool()` (~80 lignes) + appel dans `finalize()`
- `pulse_app.py` — `_suggest_next_from_context()` (~70 lignes) + 4 injections

---

## 4. Pre-RPI items (Session 50)

All pre-RPI backlog cleared :

| Item | Statut |
|---|---|
| **HTTP transport** : `--transport {stdio,http}` ajouté à `args.py`, `run_mcp()` conditionnel dans `mcp_entry.py` avec `register_http_routes()` | ✅ Fait |
| **Dead code** `_SKILL_SUPPORT_FILES` réduit de 3 à 1 (`REFERENCE.md` seulement) | ✅ Fait |
| **Flaky test parallèle** `TestGetPlan::test_no_target_with_explicit_none` — patch `_scan_cache` dans le test | ✅ Fait |

Nouveaux fichiers modifiés : `mcp_core/args.py`, `mcp_core/mcp_entry.py`

## 5. Post-RPI

- Validation réelle sur DVWA/Juice Shop/WebGoat — pipeline scan complet avec findings réels
- Couche 2 visible dans Claude Desktop (quand le bug Anthropic sera fixé)

## 6. Metrics

- **Tests :** 2639 passed, 1 skipped, 2 warnings — 0 regressions
- **Nouveaux fichiers :** 1 (`mcp_core/instructions.py`)
- **Fichiers modifiés :** 3 (`server_setup.py`, `pulse_app.py`, `tests/test_pulse_app.py`)
- **Lignes de code :** ~260 (instructions 80 + next_suggested_tool 150 + dashboard restore 30)
- **Dépendances :** aucune nouvelle
