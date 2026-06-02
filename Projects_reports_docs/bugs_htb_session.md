# HexStrike Pulse — Bug Report (Live HTB Test Session)

Date: 2026-05-09
Target: 10.129.121.47 (HTB Starting Point)

## 🔴 Critiques

### B1 — Cache keyé par IP uniquement
- **Fichier:** `server_core/advanced_cache.py` / `mcp_core/server_setup.py:40-58`
- **Problème:** Le scan cache utilise seulement l'IP target comme clé. Ignore `ports`, `scan_type`, `additional_args`.
- **Impact:** `nmap -p 22,80` puis `nmap -p 1-10000` retournent le même résultat en cache.
- **Repro:** Lancer nmap avec des paramètres différents sur la même target → résultat identique.

### B2 — Paramètres inconsistants entre handlers
- **Fichier:** `mcp_core/web_scan_direct.py:49` (nikto), `mcp_core/web_recon_direct.py:144` (wafw00f), `mcp_core/web_probe_direct.py` (whatweb), `mcp_core/exploit_framework_direct.py:140` (searchsploit)
- **Problème:** `nikto`/`wafw00f` attendent `target`, `whatweb` attend `url`, `searchsploit` attend `query`. Pas de convention de nommage unifiée.
- **Impact:** L'utilisateur doit connaître le nom exact du paramètre pour chaque outil.

### B3 — Nuclei ignore les ports découverts
- **Fichier:** `mcp_core/misc_direct.py:_nuclei()`
- **Problème:** Nuclei reçoit une IP sans informations de ports → scanne seulement 80/443.
- **Impact:** Sur une machine avec Tomcat sur 8080/8443, nuclei ne trouve rien.

### B4 — plan_attack / ctf_solve ignore les résultats de scan
- **Fichier:** `server_core/intelligence/intelligent_decision_engine.py`
- **Problème:** L'intelligence engine génère des steps génériques sans intégrer les résultats de scan (ports découverts, services).
- **Impact:** `plan_attack` propose rustscan même si on a déjà scanné tous les ports.

## 🟡 Moyennes

### B5 — ffuf URL doublon (/FUZZ/FUZZ)
- **Fichier:** `mcp_core/web_fuzz_direct.py:68`
- **Problème:** Le handler ajoute `/FUZZ` automatiquement. Si l'utilisateur inclut déjà `/FUZZ` dans l'URL, ça devient `http://x/FUZZ/FUZZ`.
- **Fix:** Vérifier si FUZZ est déjà dans l'URL avant d'ajouter, ou documenter que l'URL ne doit pas contenir FUZZ.

### B6 — Paramètres de scan mélangés via cache
- **Fichier:** `mcp_core/server_setup.py` (scan cache)
- **Problème:** Les paramètres de calls successifs semblent se mélanger (warning "Duplicate port number" dans stderr).
- **Impact:** Confusion sur quel résultat correspond à quels paramètres.

### B7 — MCP transport timeout vs outil long
- **Fichiers:** `hexstrike-ai-mcp.json:13` (timeout: 300), `mcp_core/server_setup.py:208-226` (per-category timeouts)
- **Problème:** Le timeout MCP du client (OpenCode, ~30s par défaut) est plus court que le temps d'exécution de certains outils (rustscan full range, nikto, nuclei).
- **Ce qui existe déjà:** HexStrike a une stack de timeout complète et robuste :
  - `EnhancedCommandExecutor` (`server_core/enhanced_command_executor.py:191`) — `process.wait(timeout=300)` avec terminate() → kill() en cascade
  - `FastMCP mcp.tool(timeout=...)` — per-category timeout envelope via `anyio.fail_after()` (300-1800s selon catégorie)
  - `execute_command(timeout=300)` — timeout par défaut configurable via `COMMAND_TIMEOUT` dans `config.py`
  - Scan cache adaptive TTL — 30/60/90 min selon temps d'exécution
- **VRAI problème:** Ce n'est PAS un bug dans HexStrike. C'est un **décalage entre le timeout MCP transport** (côté client comme OpenCode) et le timeout interne des outils. Quand le MCP timeout, l'outil continue de tourner sur le serveur mais le résultat est perdu.
- **Note:** tools typés ont des timeouts par catégorie (300s nmap-recon, 600s web-vuln, 1800s password-cracking). Legacy tools (Phase 1) n'ont PAS de timeout FastMCP.
- **Fix possible:** Permettre de récupérer les résultats d'un tool qui a timeouté via le cache de scan (le résultat est stocké même si le transport MCP a abandonné).

### B8 — Gobuster ignore additional_args
- **Fichier:** `mcp_core/web_fuzz_direct.py` (handler gobuster)
- **Problème:** Le handler a `wordlist`, `mode` en paramètres dédiés mais `additional_args` n'est pas passé à la commande.

## 🔧 Fix List (Priorisée)

| Priorité | Bug | Action |
|---|---|---|
| **P0** | B1 — Cache IP only | Inclure les paramètres dans la clé de cache (concat target + ports + scan_type + additional_args) |
| **P0** | B2 — Paramètres inconsistants | Unifier les noms de paramètres dans tous les `*_direct.py` handlers : `target` pour les IPs, `url` pour les URLs, `query` pour les recherches |
| **P1** | B5 — ffuf URL doublon | Vérifier si FUZZ déjà dans l'URL avant d'ajouter |
| **P1** | B4 — plan_attack ignore scan results | Feed les résultats de scan dans l'intelligence engine avant de générer les steps |
| **P2** | B3 — nuclei ignore ports | Permettre de passer `ports` dans les paramètres nuclei |
| **P2** | B7 — MCP transport timeout | Le timeout interne est OK. Problème = récupérer les résultats après MCP timeout via le cache de scan. Ajouter un endpoint ou flow pour ça. |
| **P3** | B8 — Gobuster additional_args | Passer additional_args dans la commande gobuster |
| **P3** | B6 — Cache param mixing | Relié à B1 — sera fixé quand la clé de cache inclura les paramètres |
