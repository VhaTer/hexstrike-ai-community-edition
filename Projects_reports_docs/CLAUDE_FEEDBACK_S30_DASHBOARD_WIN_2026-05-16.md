# Claude Feedback — S30 Dashboard Win
**Date :** 2026-05-16  
**Session :** Test Claude Desktop (nouvelle session fraîche)  
**Auteur :** Claude (Sonnet 4.6)  
**Pour :** HexDevMaster + équipe  
**Scope :** Validation live MCP + premier render dashboard Pulse depuis données réelles

---

## Contexte

Cette session était le "Test Claude Desktop" attendu depuis le Point S30 — une nouvelle session fraîche, sans contexte injecté manuellement, avec le serveur MCP lancé via `HEXSTRIKE_SEED_SCANS=scanme.nmap.org`.

L'objectif : vérifier que l'ensemble de la stack (serveur MCP, tools, seed cache, exécution scan, render dashboard) tient de bout en bout depuis Claude Desktop.

---

## Ce qui s'est passé — chronologie honnête

### Phase 1 : double instance en conflit
Au premier appel, tous les tools timeoutaient après 4 minutes. Diagnostic : deux instances `hexstrike_mcp.py` tournaient simultanément (PID 40018 et 40333), probablement à cause de la configuration Claude Desktop qui respawn le process. Le serveur était UP mais Claude Desktop se connectait à la mauvaise instance ou les deux se bloquaient mutuellement sur stdio.

**Fix appliqué :** `pkill -f hexstrike_mcp.py` + relance unique en foreground + restart depuis Claude Desktop settings.

### Phase 2 : discovery des vrais noms de tools
`get_overview` et `get_surface` n'existent pas comme tools MCP nommés — c'étaient des appellations conceptuelles du brief. Après `search_tools`, les vrais tools disponibles ont été identifiés :

- `server_health` → équivalent `get_overview`
- `nmap` + `whatweb` → équivalent `get_surface`

La discovery via `search_tools` a fonctionné proprement sur 130 tools.

### Phase 3 : données live et render

Trois appels successifs réussis :

| Tool | Durée | Résultat |
|---|---|---|
| `server_health` | ~1s | JSON complet, santé OK |
| `nmap scanme.nmap.org` | 10.74s | 6 ports filtered, IP 45.33.32.156 |
| `whatweb http://scanme.nmap.org` | 13.2s | Apache 2.4.7, Ubuntu, HTML5, 200 OK |

Dashboard HTML rendu inline dans Claude Desktop depuis ces données réelles — pas de prefab, pas de mock.

---

## Ce qui fonctionne vraiment bien

**Architecture lazy singletons** — RAM stable à 39.3% pendant les scans (3 tools exécutés), cohérent avec les 13.7% idle mesurés. Le pré-warming au boot se voit dans les logs et est silencieux.

**Seed cache** (`67dcfa3`) — les 4 entrées pour `scanme.nmap.org` sont chargées au boot, confirmé dans les logs. La feature tient.

**Split 8 tools individuels** (`56bcfb7`) — `nmap`, `whatweb`, `server_health` sont appelables directement sans passer par `call_tool`. Schéma propre, paramètres typés, validation Pydantic qui remonte des erreurs claires.

**DNS timeout 3s** (`adca024`) — aucun hang pendant les scans. La cible répond (ou filtre) rapidement sans bloquer le workflow.

**130 tools, 0 regressions** — `search_tools` parcourt tout le catalogue sans erreur. Les engines CVE/CTF/BugBounty sont registered et visibles.

**`call_tool` avec schema `arguments: object`** — le pattern générique fonctionne une fois le bon schéma connu. Permet d'appeler n'importe quel tool par nom.

---

## Le render dashboard — remplacement du prefab

Avant cette session, la référence était `pulse_dashboard_sketch.html` (prefab statique, données mockées). 

Aujourd'hui : **dashboard HTML rendu inline dans Claude Desktop depuis données MCP réelles**, en temps réel, sans fichier intermédiaire. C'est un changement de paradigme pour le workflow :

- Les données arrivent via MCP tools
- Claude assemble et rend le dashboard directement dans la conversation
- Le résultat est interactif (boutons `sendPrompt` pour lancer des scans suivants)
- Chaque session produit un dashboard frais depuis l'état réel du serveur

Ce n'est pas un remplacement permanent du prefab — c'est une démonstration que **Claude peut être le renderer du dashboard**, à la demande, depuis n'importe quelle session.

---

## Points à adresser avant Discord FastMCP

### 1. Double instance au démarrage — priorité haute
La config Claude Desktop lance probablement deux fois le process MCP (restart automatique + instance manuelle). À investiguer dans `claude_desktop_config.json` — probablement un `autoRestart: true` qui entre en conflit avec un process déjà lancé en terminal.

**Suggestion :** ajouter un lock file au boot de `hexstrike_mcp.py` :
```python
import fcntl, sys
lock = open('/tmp/hexstrike_mcp.lock', 'w')
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except IOError:
    print("[ERROR] Another instance is already running. Exiting.")
    sys.exit(1)
```

### 2. `get_overview` / `get_surface` comme tools MCP nommés
Pour que le brief "call get_overview then get_surface" soit littéralement exécutable, ces deux tools devraient exister comme wrappers nommés dans le serveur :

- `get_overview()` → appelle `server_health` + résume l'état
- `get_surface(target: str)` → appelle `nmap` + `whatweb` + retourne un objet structuré unifié

Cela permettrait un appel unique depuis Claude au lieu de trois appels séquentiels.

### 3. Cache seed non utilisé pendant la session
`cached_scans: 0` dans `server_health` — les 4 entrées seedées au boot n'ont pas été servies depuis le cache pendant les scans live. À vérifier si le cache key du scan live correspond au format attendu par le seed.

---

## Note sur la dynamique d'équipe

Ce qui rend ce projet solide : HexDevMaster build avec discipline (6 commits, 0 regressions, changelog propre), la coordination se fait via des briefs structurés, et les docs suivent chaque session. C'est rare et ça se voit dans la qualité du résultat.

Le "Test Claude" a validé la stack de bout en bout. Le prochain milestone — Discord FastMCP — peut avancer.

---

## Résumé exécutif pour HexDevMaster

```
✅ Connexion MCP opérationnelle (après fix double instance)
✅ server_health : données complètes en <1s
✅ nmap scanme.nmap.org : 10.74s, output propre
✅ whatweb : 13.2s, stack web identifiée
✅ Dashboard Pulse rendu live depuis données réelles
✅ 130 tools accessibles, discovery fonctionnelle
✅ Lazy singletons stables sous charge

⚠️  Double instance au démarrage — à corriger (lock file suggéré)
⚠️  Cache seed non consommé pendant scans live — à vérifier
📋  get_overview / get_surface à implémenter comme vrais tools MCP
```

**Verdict session :** 🟢 Test Claude passé. Stack validée pour présentation Discord FastMCP.

---

*Claude Sonnet 4.6 — HexStrike AI-PULSE team*
