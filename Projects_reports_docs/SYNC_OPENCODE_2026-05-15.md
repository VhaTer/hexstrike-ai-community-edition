# Sync OpenCode — 2026-05-15

De : Claude
Pour : OpenCode (HexDevMaster)
Validé par : Nexus

---

## Ce que Claude a fait pendant tes sessions

### Fix 1 — KeyError 'timestamp' (hexstrike_server.py)
`_build_dashboard_response()` accédait à `v["timestamp"]` avec `[]`.
State leak — 5 tests tombaient en suite complète, passaient en isolation.
Corrigé en `.get()` défensif sur toutes les clés.
**2580 passed, 0 failed ✅**

### Fix 2 — KeyError '$4' (workflowManager.py ligne 585)
Prompt `ctf_challenge` crashait sur difficulty inconnue comme `$4`.
```python
# Avant
multiplier = difficulty_multipliers[challenge.difficulty]
# Après
multiplier = difficulty_multipliers.get(
    challenge.difficulty,
    difficulty_multipliers["unknown"]
)
```
**Prompt ctf_challenge opérationnel ✅**

### Fix 3 — claude_desktop_config.json
Deux objets JSON au niveau racine = JSON invalide.
`wsl.exe` sans `-e` = args mal passés.
Corrigé — config valide avec `-e` flag.

---

## Ce que DeepWiki a audité (snapshot 03/05)

Audit sur ancienne version — certains points déjà corrigés.
Vérification faite sur code actuel. Voici ce qui reste vrai :

### Confirmés vrais aujourd'hui

**4 duplicate keys dans tool_registry.py** (silent data loss) :
- `nbtscan` — deux définitions, dernière gagne silencieusement
- `aircrack-ng` — deux endpoints différents ! (`wifi_pentest` vs `password_cracking`)
- `checkov` — deux effectiveness différents
- `terrascan` — deux descriptions différentes

**4 impacket tools avec typo `/api/tool/` (singulier)** :
```
line 575 : impacket-scripts  → /api/tool/active_directory/impacket
line 589 : impacket-spec     → /api/tool/active_directory/impacket/spec
line 601 : impacket-ad-enum  → /api/tool/active_directory/impacket
line 625 : impacket-remote-exec → /api/tool/active_directory/impacket
```
Tous les autres tools utilisent `/api/tools/` (pluriel).

**sublist3r endpoint inversé** (line 690) :
```
/api/osint/tools/sublist3r  ← tous les autres : /api/tools/osint/<name>
```

**auto_install_missing_apt_tools** (line ~1012) :
Feature CE — **décision Nexus : à supprimer**.
Jamais voulu dans Pulse. Nécessite root. Contredit la règle no-auto-execution.

**zap/zaproxy** (lines 355, 1406) :
Même endpoint `/api/tools/zap` mais :
- `zap` : target **optionnel**
- `zaproxy` : target **requis**

### Non critiques / déjà gérés

`aireplay_ng` vs `aireplay-ng` → table de mapping déjà en place dans
`server_setup.py` lignes 371 et 1298. Pas un bug actif.

`_CLASSIFY_PROMPT` catégories incomplètes → feature CE, probablement
obsolète maintenant que Claude Desktop orchestre. À évaluer séparément.

---

## Tâches tool_registry.py — à faire après dashboard

**Priorité 1 — duplicate keys**
Garder la meilleure définition :
- `nbtscan` → garder seconde (verbose + timeout en optional)
- `aircrack-ng` → garder première (endpoint wifi_pentest, type:list sur capture_files)
- `checkov` → garder effectiveness 0.87
- `terrascan` → garder effectiveness 0.85

**Priorité 2 — supprimer auto_install_missing_apt_tools**
Décision Nexus. Feature CE non voulue.

**Priorité 3 — corriger typos**
- 4 impacket : `/api/tool/` → `/api/tools/`
- sublist3r : `/api/osint/tools/` → `/api/tools/osint/`
- auto_install : `api/tools/` → `/api/tools/` (leading slash manquant)

**Priorité 4 — zap/zaproxy**
Décider lequel est canonique et aligner le comportement target.

---

## Objectif immédiat — dashboard dans Claude Desktop

Root cause trouvée par toi : `BM25SearchTransform` avec `always_visible`
hardcodé cachait `pulse_dashboard`.
Fix appliqué : `pulse_dashboard` ajouté à `always_visible`.
**2580 passed, 0 regressions ✅**

### Étape suivante — test réel

```bash
# Terminal WSL
cd ~/hexstrike-ai-community-edition
source hexstrike-env/bin/activate
python3 hexstrike_mcp.py
```

Puis dans Claude Desktop :
```
"Open the Pulse dashboard"
```

Si ça rend → Session D terminée, on commit.
Si blank ou erreur → colle les logs ici.

---

## Règle rappel — non négociable

```
Dashboard : observe, plan, suggest.
Pas d'exécution automatique.
auto_install_missing_apt_tools → à virer.
```
