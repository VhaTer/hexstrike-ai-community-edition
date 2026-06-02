# GPT Audit S31 — `hexstrike.py` CLI Review

**Date :** 2026-05-14  
**Auteur :** GPT / Codex audit  
**Pour :** HexDevMaster + Claude  
**Scope :** `hexstrike.py`, README CLI, tests CLI existants  
**Mode :** audit only — aucun fix appliqué dans cette session

---

## Résumé

Le CLI `hexstrike.py` n'est pas un simple stub : il existe déjà une vraie base avec 7 subcommands, routage direct vers les modules `*_direct.py`, sortie colorée partielle, `--json`, `-o`, validation outils, status serveur, MCP bridge et workflow CTF.

Mais il n'est pas encore au niveau "operator-grade" attendu pour HexStrike Pulse.

Deux problèmes bloquants ont été confirmés :

1. `ctf` crash systématiquement à cause de variables non définies.
2. `scan --json` peut produire une sortie non JSON parce que les logs/visuals du runner polluent `stdout`.

Ces deux points doivent être corrigés avant de considérer le CLI stable.

---

## Commandes vérifiées

```bash
hexstrike-env/bin/python3 hexstrike.py --version
# OK → hexstrike 0.8.0

hexstrike-env/bin/python3 hexstrike.py --help
# OK → liste serve/scan/tools/status/validate/mcp/ctf

hexstrike-env/bin/python3 hexstrike.py tools --filter nmap --json
# OK → JSON propre

hexstrike-env/bin/python3 hexstrike.py ctf --category web --json
# FAIL → NameError: name 'name' is not defined

hexstrike-env/bin/python3 -m pytest tests/test_real_integration.py::TestCLIScan::test_cli_json_flag -q --tb=short
# FAIL → JSONDecodeError sur r.stdout
```

---

## Faits établis

### 1. Surface CLI existante

`hexstrike.py` expose actuellement :

- `serve` — démarre le serveur HTTP/SSE ;
- `scan` — exécute un outil directement sans serveur ;
- `tools` — liste les outils disponibles ;
- `status` — vérifie `/health` ;
- `validate` — vérifie les binaires dans `PATH` ;
- `mcp` — lance le bridge stdio ;
- `ctf` — génère un workflow CTF.

Le README documente bien ces 7 commandes.

### 2. UX déjà partiellement travaillée

Points positifs :

- `argparse` structuré ;
- `--json` sur plusieurs commandes ;
- `-o/--output` sur les commandes principales ;
- suggestions sur outil inconnu ;
- couleurs et box drawing via `ModernVisualEngine`;
- format humain pour `scan`, `status`, `validate`;
- `tools --filter` utile.

Le CLI n'est donc pas "à zéro". Il faut le durcir.

---

## Findings bloquants

### Finding 1 — `ctf` est cassé

Dans `hexstrike.py`, `cmd_ctf(args)` construit `CTFChallenge` avec :

```python
name=name,
category=category,
description=description,
difficulty=difficulty,
points=points,
target=target,
```

Mais ces variables ne sont jamais assignées depuis `args`.

Résultat observé :

```text
NameError: name 'name' is not defined
```

Impact :

- commande documentée mais inutilisable ;
- README claim faux pour `ctf`;
- risque de casser une démo CLI ;
- test manquant.

Fix attendu :

- assigner `name = args.name or ...`;
- `category = args.category`;
- `description = args.description`;
- `difficulty = args.difficulty`;
- `points = args.points`;
- `target = args.target`;
- ajouter test `ctf --json`.

### Finding 2 — `scan --json` n'est pas fiable

Le test ciblé échoue :

```text
FAILED tests/test_real_integration.py::TestCLIScan::test_cli_json_flag
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

Cause probable :

- les fonctions directes ou `execute_command` impriment des banners/logs sur `stdout`;
- `cmd_scan()` sérialise bien `result` en JSON ensuite ;
- mais le `stdout` final contient déjà du texte avant le JSON.

Impact :

- `--json` non scriptable ;
- pipes cassés ;
- automation impossible ;
- contradiction avec README et docs API.

Règle CLI à imposer :

```text
If --json is set, stdout must contain JSON only.
All logs/progress/banners must go to stderr or be suppressed.
```

Fix attendu :

- dans `cmd_scan`, capturer stdout des `exec_func(...)` via `redirect_stdout`;
- si `args.json`, ne pas rejouer ces logs sur stdout ;
- éventuellement les envoyer vers stderr en mode `--verbose`;
- préserver JSON propre sur stdout ;
- ajouter test qui fait `json.loads(stdout)`.

---

## Findings qualité

### Finding 3 — `DIRECT_ROUTES` duplique une vérité serveur

`hexstrike.py` maintient son propre `DIRECT_ROUTES`, très long.

Risque :

- divergence avec `mcp_core/server_setup.py`;
- outil ajouté côté MCP mais absent CLI ;
- binaire renommé d'un côté seulement ;
- maintenance lourde.

Recommandation :

- court terme : garder tel quel mais ajouter test de cohérence ;
- moyen terme : extraire une registry commune CLI/MCP.

### Finding 4 — `scan` n'a pas de confirmation destructive

Le CLI peut appeler directement des outils sensibles selon les routes disponibles.

Exemples à traiter avec prudence :

- `metasploit`;
- `sqlmap`;
- `hydra`;
- `hashcat`;
- `aireplay_ng`;
- `responder`;
- `mitm6`.

Recommandation :

- ajouter un marquage `destructive/sensitive` ;
- demander confirmation interactive par défaut ;
- permettre bypass explicite avec `--yes` pour automation ;
- ne jamais bloquer les commandes passives.

### Finding 5 — UX humaine bonne mais incomplète

Le CLI a déjà des couleurs, mais il manque encore des conventions "operator-grade" :

- `--no-color`;
- `--quiet`;
- `--verbose` global ;
- `--debug` global ou par commande ;
- exit codes documentés ;
- erreurs homogènes ;
- aide plus lisible par groupes ;
- `examples` par tool courant.

### Finding 6 — `scan` syntax docs divergentes

Le parser actuel attend :

```bash
python3 hexstrike.py scan <tool> <target> -p key=val
```

Mais l'epilog montre :

```bash
python3 hexstrike.py scan nmap target=scanme.nmap.org scan_type=-sV
```

Cette forme n'est pas alignée avec le parser actuel, sauf si elle est passée comme target et arguments invalides.

Recommandation :

- corriger les exemples dans l'epilog ;
- garder un seul style officiel.

---

## Priorité de correction proposée

### P0 — Stabiliser promesses existantes

1. Fix `ctf` variables non définies.
2. Fix `scan --json` pour stdout JSON strict.
3. Ajouter tests ciblés :
   - `ctf --json`;
   - `scan --json` parseable ;
   - `tools --json` parseable ;
   - unknown tool exit code.

### P1 — Rendre le CLI fiable pour opérateur

1. Ajouter `--no-color`.
2. Ajouter `--quiet` / `--verbose`.
3. Standardiser exit codes.
4. Nettoyer les exemples help.
5. Ajouter confirmation pour outils sensibles.

### P2 — Monter le niveau UX

1. `hexstrike plan <target>` pour afficher IDE plan sans exécuter.
2. `hexstrike scope` pour afficher la cible active/cache.
3. `hexstrike history` pour scans récents.
4. `hexstrike intel` pour ToolStatsStore.
5. Tables plus lisibles pour `tools`, `validate`, `status`.

---

## Verdict

La base CLI est meilleure que prévu : les fondations sont là.

Mais deux bugs prouvent que l'agent précédent a probablement survolé cette facette :

- une commande documentée (`ctf`) crash ;
- une promesse centrale (`--json`) n'est pas fiable sur `scan`.

Avant de viser une UX "premium terminal", il faut d'abord garantir les contrats fondamentaux :

- commandes documentées fonctionnelles ;
- JSON strict quand demandé ;
- erreurs propres ;
- confirmation pour actions sensibles.

Ensuite seulement, on peut améliorer le style terminal avec couleurs, badges, tables, italic/dim text et une vraie ergonomie operator-grade.

---
echange GPT additional

"""
Je partirais du hexstrike.py existant, mais avec une refonte progressive et structurée.

Pourquoi :

Le CLI actuel a déjà une bonne surface : serve, scan, tools, status, validate, mcp, ctf.
Le README est déjà aligné sur cette idée de “unified CLI gateway”.
Les routes directes existent.
Il y a déjà des tests CLI, même s’ils révèlent des bugs.
Une refonte totale risquerait de casser le dashboard/serveur/MCP alors que la branche dashboard est déjà chargée.
Ma stratégie serait donc :

Phase 0 — Corriger les contrats cassés

Fix ctf.
Garantir --json = JSON strict stdout only.
Nettoyer les logs vers stderr.
Ajouter tests ciblés.
Phase 1 — Stabiliser l’ergonomie

--no-color.
--quiet.
--verbose.
exit codes propres.
erreurs homogènes.
aide CLI corrigée.
Phase 2 — Monter le niveau opérateur

tables propres ;
badges colorés ;
résumé scan plus lisible ;
suggestions tool/params ;
confirmation pour tools sensibles ;
meilleur validate.
Phase 3 — Pulse-native CLI

hexstrike plan <target>;
hexstrike scope;
hexstrike history;
hexstrike intel;
hexstrike dashboard ou hexstrike pulse.
Donc ma réponse nette : évolution forte sur base existante, pas rewrite.
On garde la compatibilité et on transforme progressivement le CLI en vraie interface opérateur.
"""
Nettoyage fait.

**Nettoyé**
- Restauré `scripts/patch_beartype_for_coverage.sh` et `scripts/restore_beartype.sh`.
- Supprimé le doublon sûr `reports/scripts/` après vérification blob-identique avec `HEAD`.
- Corrigé le bruit de mode Git sur WSL/Windows avec `core.filemode=false` localement.

**Préservé**
- Les anciens rapports `reports/http_*` sont conservés.
- Le travail dashboard reste intact : `pulse_app.py`, `server_core/operational_metrics.py`, `tests/test_pulse_app.py`, `export_dashboard.py`, `serve_pulse.py`.
- Le travail CLI P0 reste intact : `hexstrike.py`, `tests/test_hexstrike_cli.py`.

**État restant**
- `M .gitignore`
- `M hexstrike.py`
- `M pulse_app.py`
- `M server_core/operational_metrics.py`
- `?? export_dashboard.py`
- `?? serve_pulse.py`
- `?? tests/test_hexstrike_cli.py`
- `?? tests/test_pulse_app.py`
"""

"""
