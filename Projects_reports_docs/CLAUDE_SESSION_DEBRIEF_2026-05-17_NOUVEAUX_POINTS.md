# Session Debrief — Nouveaux points & réflexions
**Date :** 2026-05-17  
**Auteur :** Claude (Sonnet 4.6)  
**Pour :** HexDevMaster  
**Source :** Échanges directs avec HexReaper — session complète 2026-05-17

---

## 1. Validation live S30 — ce qui s'est passé vraiment

La session du 2026-05-16 était le "Test Claude Desktop" attendu depuis S30.
Chronologie honnête :

- **Problème initial :** deux instances `hexstrike_mcp.py` en conflit (PID 40018 + 40333) — Claude Desktop respawnait le process automatiquement en plus de l'instance manuelle. Fix : `pkill -f hexstrike_mcp.py` + relance unique en foreground.
- **Discovery des tools :** `get_overview` et `get_surface` n'existent pas comme tools MCP nommés dans cette session fraîche — Claude a utilisé `server_health` + `nmap` + `whatweb` à la place. La discovery via `search_tools` sur 130 tools a fonctionné.
- **Données live obtenues :**
  - `server_health` : ~1s, JSON complet
  - `nmap scanme.nmap.org` : 10.74s, 6 ports filtered, IP 45.33.32.156
  - `whatweb` : 13.2s, Apache 2.4.7 EOL, Ubuntu, HTML5, 200 OK
- **Dashboard :** rendu HTML inline dans Claude Desktop depuis données réelles — pas de prefab, pas de mock.

**Conclusion :** stack validée. Le "Test Claude" est passé 🟢.

---

## 2. Le dashboard HTML inline — shift de paradigme

Avant : `pulse_dashboard_sketch.html` — prefab statique, données mockées.  
Maintenant : Claude génère le dashboard HTML à la demande depuis les données MCP réelles.

**Comment ça marche exactement :**
```
MCP tools retournent JSON
    → Claude reçoit et parse
    → Claude appelle show_widget(html)
    → dashboard render inline dans la conversation
    → boutons sendPrompt() pour lancer des scans suivants
```

**Avantages :**
- Zéro iframe, zéro handshake, zéro CSP
- Données toujours fraîches depuis l'état réel du serveur
- Interactif (boutons dans le dashboard déclenchent des actions)
- Fonctionne sur stdio sans modification

**Limites honnêtes :**
- Claude contrôle le render — pas pixel-perfect, pas déterministe à 100%
- HTML régénéré à chaque session — pas de persistance
- Pour UI brandée stricte : PrefabApp sur HTTP reste la bonne réponse

Le POC complet est documenté dans `DISCORD_FASTMCP_PROOF_CLAUDE_RENDERING.md`.

---

## 3. Le cold start problem — LLM + MCP tools

Point soulevé par HexReaper sur l'expérience avec d'autres assistants (Cline, etc.) :

**Le problème :** un LLM fresh connecté à 130 tools MCP va souvent sur internet, cherche le README, le wiki — au lieu d'utiliser les tools directement. Comportement frustrant pour un utilisateur qui sait que ça marche.

**Pourquoi ça arrive :**
- Le LLM reçoit 130 tools → overwhelmed → fallback sur ce qu'il connaît
- Si les tool descriptions sont trop courtes, le LLM manque de confiance pour les utiliser
- Sans contexte injecté (userMemories, system prompt), chaque session repart de zéro

**Pourquoi ça marchait avec Claude ce soir :**
- userMemories avec contexte HexStrike injecté au démarrage
- Réflexe `search_tools` avant d'aller sur internet
- Pas de garantie que ça marche aussi bien avec n'importe quel LLM fresh

**La solution — 3 couches pour le plug-and-play effect :**

### Couche 1 — Tool descriptions riches (dans le code)
Chaque tool doit se "vendre" au LLM en 3 lignes :

```python
@mcp.tool()
def nmap(target: str, scan_type: str = "-sCV",
         ports: str = "", additional_args: str = "-T4 -Pn") -> dict:
    """
    Run nmap port scan against target. Use for initial recon on any host.
    Returns: dict with 'output' (raw nmap stdout), 'success' (bool), 
    'execution_time' (float). Always use BEFORE whatweb on new targets.
    Example: nmap('scanme.nmap.org') or nmap('192.168.1.1', ports='80,443')
    """
```

Actuellement les descriptions sont générées automatiquement via `_build_typed_tool_doc` — fonctionnel mais minimal. Le LLM sait que `target` est requis mais pas quand utiliser nmap vs rustscan vs masscan.

### Couche 2 — System prompt agent (à injecter)
Un court texte injecté au démarrage de chaque session :
```
Tu as accès à HexStrike AI-PULSE — 130 tools de sécurité offensive.
Commence TOUJOURS par search_tools() pour découvrir les tools disponibles.
N'utilise jamais internet pour des tâches recon — utilise les tools MCP directement.
Pour un nouveau target : nmap() → whatweb() → findings → dashboard.
```

### Couche 3 — Workflow hints dans les tool responses
Chaque tool retourne un champ `next_suggested_tool` :
```json
{
  "success": true,
  "output": "...",
  "execution_time": 10.74,
  "next_suggested_tool": "whatweb",
  "next_reason": "web ports 80/443 detected"
}
```

Avec ces 3 couches, n'importe quel LLM arrive sur HexStrike et sait quoi faire sans README, sans wiki, sans guidance manuelle. C'est le vrai plug-and-play.

---

## 4. État actuel des annotations nmap

Lu directement dans `mcp_core/server_setup.py` — les annotations sont générées via `_build_typed_tool_doc`. Ce que le LLM voit réellement :

```
Port scan and service detection

Args:
    target: Required parameter
    scan_type: Optional parameter. Default: '-sCV'
    ports: Optional parameter. Default: ''
    additional_args: Optional parameter. Default: '-T4 -Pn'

Returns:
    Tool execution results

Notes:
    This is a typed wrapper over run_security_tool(tool_name='nmap', ...)
```

Fonctionnel mais sans contexte d'usage, sans exemple, sans workflow hint.
`_build_typed_tool_doc` est le seul endroit à modifier — impact sur tous les tools en un seul changement. Ticket clair, pas de refacto.

---

## 5. PrefabApp vs MCP Apps — clarification écosystème

Point important soulevé en session — deux entités distinctes souvent confondues :

```
FastMCP (jlowin / communauté)
    └── PrefabApp / PrefabUI
            └── Composants UI : DataTable, Card, Badge, 
                Sparkline, Tooltip, Icon Lucide, Carousel...
            └── Suit le protocole MCP mais implémentation indépendante
            └── Problème : stdio + Claude Desktop = instable (iframe handshake)
            └── HTTP transport = stable

MCP Apps (Anthropic, lancé 26 janvier 2026)
    └── Extension officielle du protocole MCP
    └── @modelcontextprotocol/ext-apps
    └── Partenaires : Amplitude, Asana, Box, Canva, Figma, Slack...
    └── Iframes sandboxées — spec ouverte
    └── Supporté Claude.ai web + desktop, VS Code Insiders, Goose, ChatGPT
```

**Ce que ça veut dire pour HexStrike :**
- Notre Claude-side HTML rendering est valide et stable maintenant
- PrefabApp redevient option sérieuse si on passe sur HTTP transport
- MCP Apps officiel = cible long terme si on veut distribuer HexStrike à la communauté
- Pas urgent — à mettre en roadmap post-RPI vuln lab

---

## 6. Context managers PrefabUI — explication

Les composants PrefabUI utilisent le pattern Python `with ... :` pour gérer l'imbrication :

```python
# Correct — with obligatoire
with Card() as card:
    Title("Mon titre")
    with Row():
        Badge("OK")
        Sparkline(data)

# Bugué — manque le with (pas d'erreur levée, render silencieux cassé)
Card() as card:        # ← KeyError ou composant non attaché
    Tooltip("CPU"):    # ← S32 bug exact — fixé en S33
        Sparkline(data)
```

Le `with` fait :
- **Enter** : enregistre le composant comme parent actif dans le contexte courant
- **Exit** : ferme le composant et l'attache au parent au-dessus

C'est le bug S32 que HexDevMaster a fixé en S33 — `with` manquant devant `Tooltip` et `Row`, `Span` utilisé sans `content`. Piégeux car aucune exception levée — le render casse silencieusement.

---

## 7. RPI vuln lab — architecture cible

Clarification importante : le RPI est une **cible**, pas un serveur MCP.

```
Claude Desktop (Windows/Kali)
    → HexStrike MCP server (Kali, stdio)
        → 130 tools : nmap, whatweb, sqlmap, nikto...
            → RPI Ubuntu (réseau local isolé)
                → DVWA        (SQLi, XSS, LFI, CSRF)
                → Juice Shop  (OWASP Top 10, API)
                → WebGoat     (Java, pédagogique)
                → vAPI / crAPI (API vulnérables)
                → Metasploitable (network services)
                → VulnHub VMs  (CTF style)
```

Installation recommandée : **Docker sur Ubuntu** — propre, isolé, resetable entre tests.

**Valeur pour HexStrike :**
- Tester les 130 tools sur des cibles réelles contrôlées
- Valider le CTF engine et Bug Bounty engine dans des conditions réelles
- System Trends panel (S34) très utile — RPI a ~4GB RAM, voir les spikes sous scan
- Premier vrai test end-to-end de toute la stack

---

## 8. Transport layer stdio vs HTTP — point à creuser

Question ouverte pour HexDevMaster :

- `hexstrike_mcp.py` supporte-t-il déjà `--transport http` ou stdio uniquement ?
- FastMCP 3 supporte les deux transports nativement
- HTTP transport = PrefabApp stable + MCP Apps Anthropic possible
- stdio transport = actuel, validé, Claude Desktop natif

**Recommandation :** audit du transport layer en S35 — pas urgent mais important pour la roadmap distribution/communauté.

---

## 9. Roadmap consolidée post-session

```
S34  → Cache seed fix + Panels Batch 1 (Tool Perf, Cache, System Trends)
S35  → Panels Batch 2 (Sessions, Confirmations, Network I/O)
S35  → Wrappers get_overview / get_surface comme vrais tools MCP nommés
S35  → Audit transport layer stdio vs HTTP
S36  → Tag v0.10.0 + merge master
S36  → Tool descriptions enrichies (Couche 1 plug-and-play)
RPI  → Vuln lab Ubuntu (Docker : DVWA, Juice Shop, Metasploitable)
RPI  → Test end-to-end sur cibles réelles
?    → System prompt agent HexStrike (Couche 2 plug-and-play)
?    → Workflow hints dans tool responses (Couche 3 plug-and-play)
?    → HTTP transport + PrefabApp stable
?    → MCP Apps @ext-apps si distribution communauté
```

---

## 10. Note équipe

Ce projet a une discipline rare — briefs structurés, docs après chaque session, 0 regressions maintenu sur 2576 tests, changelogs propres. Le "docs writer" manque toujours mais les deux membres de l'équipe compensent bien.

Le vuln lab RPI va changer la nature des tests — passer de `scanme.nmap.org` filtré à des cibles réelles contrôlées. C'est le vrai milestone.

---

*Claude Sonnet 4.6 — HexStrike team — 2026-05-17*
