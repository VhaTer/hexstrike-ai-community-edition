# Claude → OpenCode — Feedback & Roadmap Sync
# 2026-05-09

---

## Feedback on Night Session Work

OpenCode, ton travail de nuit est excellent. Voici un retour honnête et détaillé.

---

### What you got exactly right

**B1 — Cache key ordering fix**
C'est le bug le plus subtil et le plus impactant de la session. Calculer la clé
de cache APRÈS l'optimizer fait que les clés incluent les paramètres par défaut
injectés (scan_type, ports, additional_args). Deux appels identiques du point de
vue utilisateur donnaient des clés différentes. Ton fix est architecturalement
correct — la clé doit refléter l'intention utilisateur, pas l'output de
l'optimizer.

**B2 — wafw00f url→target fallback**
Bug silencieux critique. Le registry déclare `url`, le handler attendait `target`.
100% des appels wafw00f via schema LLM échouaient sans message d'erreur clair.
Ton pattern de normalisation (if `url` in data and `target` not in data) est
exactement le bon fix — minimal et non-destructif.

**B5 — ffuf URL doubling**
Simple, précis, correct. Le check `if "/FUZZ" in url` est la bonne approche.

**4 orphaned tools wired**
Correct. hcxpcapngtool, eaphammer, bettercap_wifi, mdk4 avaient des handlers
complets dans wifi_direct.py mais n'étaient pas dans DIRECT_TOOLS. Idem pour
enum4linux-ng, pwntools, jwt_analyzer, autopsy.

**autopsy handler rewrite**
La version précédente ignorait tous les paramètres et lançait une commande
hardcodée. Ta version avec image_path + case_name est correcte.

**Documentation quality**
AGENT_HANDFOFF_2026-05-09.md est exemplaire — bugs numérotés, fichiers précis,
lignes exactes, test results, remaining bugs clairement listés. Continue comme ça.

---

### One thing to verify before next session

**DIRECT_TOOLS count: 112**
Ton rapport dit 112 entrées. Avant ton travail nous étions à 106 + 6 ajoutés = ~112.
Mais vérifie qu'il n'y a pas de doublons — lors d'un edit intermédiaire, `wifite`
et `wifite2` ont été ajoutés deux fois dans le diff. Un `grep -c "wifi_exec"
mcp_core/server_setup.py` rapide confirmera.

---

### What we were doing before your session (context for sync)

Voici l'état exact du projet quand tu as pris le relai :

**Architecture actuelle — FastMCP 3.2.4, Python 3.13**

```
Entrypoint:   hexstrike_server.py (Starlette + FastMCP HTTP, port 8888)
MCP setup:    mcp_core/server_setup.py::setup_mcp_server_standalone()
Execution:    run_security_tool() → DIRECT_TOOLS → *_direct.py → execute_command()
Intelligence: server_core/intelligence/ (lazy singletons, 13.7% RAM startup)
```

**Versions et métriques actuelles**
- FastMCP: 3.2.4
- Tests: 1360 passed, 6 xfailed (avant ton commit — à revalider après merge)
- RAM startup: 13.7% (lazy singletons depuis v0.6.0)
- DIRECT_TOOLS: 112 entries (après tes ajouts)
- MCP tools exposés: ~130 (DIRECT_TOOLS + CTF/CVE/BugBounty engines + helpers)

**Ce qui a été accompli avant toi (par ordre chronologique)**

Phase 1-5 Stabilisation (v0.3.0):
  - Normalization pipeline: _normalize_result() + _normalize_tool_result()
  - finalize() closure: telemetry + metrics sur TOUS les exit paths
  - Session-scoped cache keys: {session_id}:{tool}:{target}
  - OperationalMetricsStore: metrics://tools resource
  - 1334 → 1360 tests

Track 1 FastMCP native (v0.4.0):
  - ctx.session_id pour isolation cache multi-client
  - mcp.tool(timeout=) per-category (300-1800s)
  - HexStrikeLoggingMiddleware + HexStrikeSessionMiddleware
  - ctx.sample() opt-in AI suggestions

Track 2 Flask removal (v0.5.0):
  - setup_mcp_server() supprimé
  - server_api/ 233 fichiers Flask supprimés
  - server_core/decorators.py supprimé
  - requirements.txt: flask retiré

Chemin C — V6 Intelligence bridging (v0.6.0):
  - Lazy singletons: 83.2% → 13.7% RAM
  - CTF Engine: ctf_analyze, ctf_tools, ctf_solve, ctf_team
  - CVE Engine: cve_fetch, cve_analyze, cve_exploits, cve_intel
  - Bug Bounty Engine: bb_recon, bb_hunt, bb_business, bb_osint, bb_full

---

## Roadmap — Next priorities (ordered)

### Priority 1 — B4: plan_attack/ctf_solve scan cache injection (MEDIUM)

**What:** plan_attack et ctf_solve construisent un plan from scratch sans
consulter les résultats de scan déjà en cache. Un nmap sur la cible a peut-être
déjà tourné — ces infos devraient enrichir le TargetProfile avant que l'IDE
génère la chain.

**Where:** mcp_core/server_setup.py::plan_attack tool (ligne ~960)
**How:**
```python
# Avant d'appeler _ide.analyze_target(target):
# 1. Récupérer tous les scans cachés pour ce target dans la session
cached_scans = {
    k.split(":")[1]: v["result"]
    for k, v in _scan_cache.items()
    if k.startswith(_session_id) and v.get("target") == target
}
# 2. Les injecter dans le TargetProfile
profile = await loop.run_in_executor(None, lambda: _ide.analyze_target(target))
if "nmap" in cached_scans:
    nmap_output = cached_scans["nmap"].get("output", "")
    # Parser l'output nmap pour enrichir profile.open_ports, profile.services
    profile = _enrich_profile_from_cache(profile, cached_scans)
```

### Priority 2 — IntelligentErrorHandler in run_security_tool() (MEDIUM)

**What:** Quand un tool échoue, classifier l'erreur et suggérer un outil
alternatif au lieu de retourner `error: unknown`. La classe existe déjà dans
server_core/error_handling.py — elle n'est pas branchée au flow actif.

**Where:** mcp_core/server_setup.py, dans finalize() quand result["success"] is False
**How:**
```python
# Dans finalize() sur le path failure:
if not result.get("success"):
    from server_core.singletons import error_handler
    error_ctx = error_handler.classify_error(
        result.get("error", ""), tool_name, target
    )
    alternative = error_handler.get_alternative_tool(tool_name, error_ctx)
    if alternative:
        result["suggested_alternative"] = alternative
        await ctx.info(f"💡 Try instead: {alternative}")
```

### Priority 3 — Coverage improvements on active modules (MEDIUM)

**Current coverage on critical active modules:**
- server_core/rate_limit_detector.py: 36% → target 70%
- mcp_core/technology_detector.py: 22% → target 60%
- server_core/parameter_optimizer.py: 44% → target 65%
- shared/target_profile.py: 44% → target 70%
- mcp_core/ctf_engine.py: 0% (new file) → target 60%
- mcp_core/cve_engine.py: 0% (new file) → target 60%
- mcp_core/bugbounty_engine.py: 0% (new file) → target 60%

**Run coverage:** `pytest tests/ --cov=mcp_core --cov=server_core --cov=shared
--cov-report=term-missing -q`

### Priority 4 — SECURITY.md (LOW, quick win)

**What:** Aucun fichier SECURITY.md. Requis pour les projets GitHub publics.
**Content needed:** responsible disclosure policy, supported versions, contact.
**Time:** 15 minutes.

### Priority 5 — CI/CD GitHub Actions (LOW)

**What:** Pas de pipeline CI. Chaque régression est détectée manuellement.
**Suggested workflow:** .github/workflows/ci.yml
```yaml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.13'}
      - run: pip install -r requirements.txt
      - run: pytest tests/ -q --tb=short
```
Note: Les tests qui appellent setup_mcp_server_standalone() sont lents (~5 min).
Envisager de les séparer en fast/slow avec markers pytest.

### Priority 6 — HTB live validation (WHEN VPN ACTIVE)

**What:** Valider B2 (param normalization) et B5 (ffuf) contre une vraie cible.
**Tools to test in order:**
1. wafw00f (url param) → `run_security_tool("wafw00f", {"url": "http://target"})`
2. ffuf (no URL doubling) → `run_security_tool("ffuf", {"url": "http://target/FUZZ", ...})`
3. nuclei with ports → `run_security_tool("nuclei", {"target": "...", "ports": "80,443"})`
4. hcxpcapngtool (newly wired)

---

## Architecture principles to respect

Ces principes ont été établis pendant la stabilisation et doivent être respectés :

1. **finalize() owns all exit paths** — telemetry + _op_metrics.record() sont
   appelés via finalize(). Ne jamais bypass finalize() pour un early return sans
   l'appeler.

2. **Lazy singletons** — ne jamais instancier CVEIntelligenceManager,
   CTFWorkflowManager, BugBountyWorkflowManager au niveau module. Utiliser
   get_cve_intelligence(), get_ctf_manager(), get_bugbounty_manager() de
   singletons.py. Le démarrage à 13.7% RAM dépend de ça.

3. **_normalize_tool_result() at MCP boundary** — tout résultat de *_direct.py
   passe par _normalize_tool_result() dans run_security_tool(). Ne pas bypasser
   cette normalisation même pour des tools custom.

4. **session-scoped cache keys** — format: {session_id}:{tool}:{target}[:{hash}].
   Ne jamais utiliser une clé non-sessionée pour _scan_cache.

5. **V6 classes intact** — CTFWorkflowManager, BugBountyWorkflowManager,
   CVEIntelligenceManager, IntelligentDecisionEngine ne sont PAS modifiés.
   On ajoute seulement des bridges MCP dans mcp_core/. Toute logique métier
   reste dans server_core/.

6. **Tests: 1360 minimum** — chaque session doit finir avec ≥ 1360 passed,
   6 xfailed. Toujours lancer `pytest tests/ -q` avant de commiter.
   Toujours avoir le venv actif: `source hexstrike-env/bin/activate`

---

## Branch state

- Active branch: feature/attack-intelligence
- master: v0.6.0 (ee7f201)
- Latest commit on feature: ton travail non encore commité (pending commit)
- Tag history: v0.3.0-stable-candidate, v0.4.0, v0.5.0, v0.6.0
- Next tag: v0.6.1 (bugfix release) after B4 + coverage improvements

---

## Quick commands for next session

```bash
# Toujours commencer par:
cd ~/hexstrike-ai-community-edition
source hexstrike-env/bin/activate
git status --short
pytest tests/ -q 2>&1 | tail -5

# Coverage:
pytest tests/ --cov=mcp_core --cov=server_core --cov=shared \
  --cov-report=term-missing -q 2>&1 | grep -E "TOTAL|rate_limit|optimizer|ctf_engine|cve_engine"

# Server live test:
python hexstrike_server.py &
sleep 4 && curl -s http://localhost:8888/health | python -m json.tool | grep memory
kill %1
```
