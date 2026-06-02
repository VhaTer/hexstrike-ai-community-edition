# GPT Audit S30 — Prefab Dashboard State Review

**Date :** 2026-05-14  
**Auteur :** GPT / Codex audit  
**Pour :** HexDevMaster + Claude  
**Scope :** état réel du repo local vs handoffs récents S26/S27/S28/S29  
**Mode :** audit documentaire uniquement — aucun push, aucun commit, aucune modification code

---

## Préambule

Je suis intervenu comme troisième regard technique sur HexStrike AI-PULSE : audit des claims docs vs code, stabilisation Phase 1/2, détection de flows fragiles, correction de bugs runtime ciblés, structuration des handoffs, et clarification des frontières entre serveur MCP, moteur d'exécution, intelligence, cache, métriques et dashboard.

Mon rôle ici n'est pas de remplacer le duo existant, mais de vérifier froidement la cohérence entre ce qui est écrit, ce qui est réellement dans le code, et ce qui doit être considéré comme prêt avant push/PR.

### Rôles de l'équipe

- **HexDevMaster / opencode** : code guru méthodologique, décision finale, arbitrage qualité, validation avant push.
- **Claude** : cerveau orchestrator historique du projet, porteur de la vision initiale, planification des sessions et handoffs.
- **GPT / Codex audit** : reviewer pragmatique, vérifie les faits dans le repo, repère les incohérences, formalise les risques et prochaines vérifications.

---

## Contexte actuel

Le fait que GitHub ne reflète pas encore tout le travail est normal à ce stade.

L'agent travaille sur la branche dashboard avec instruction claire : **ne pas push tant que les tests ne sont pas terminés et que l'état n'est pas stable**.

Donc l'écart entre local et remote n'est pas une alerte en soi. C'est un état de travail contrôlé. Le point important est de bien distinguer :

- ce qui est déjà mergé/poussé ;
- ce qui est local et en cours ;
- ce qui est documenté dans les handoffs ;
- ce qui est réellement implémenté dans le code actuel.

---

## Sources vérifiées

Documents récents lus par date :

- `Projects_reports_docs/PULSE_S27_2026-05-14.md`
- `Projects_reports_docs/AGENT_HANDOFF_2026-05-14_S26.md`
- `Projects_reports_docs/PULSE_S23_2026-05-13.md`
- `Projects_reports_docs/PULSE_S23_VISION_2026-05-13.md`
- `Projects_reports_docs/PULSE_S22_BRIEF_2026-05-13.md`
- `Projects_reports_docs/REPORT_2026-05-13_v0.8.0.md`
- `AGENTS.md`

Code/local state observé :

- `pulse_app.py`
- `server_core/operational_metrics.py`
- `tests/test_pulse_app.py`
- `export_dashboard.py`
- `serve_pulse.py`
- état Git local sur `feature/prefab-dashboard`

---

## Faits établis

### 1. Le dossier legacy n'est plus la question principale

Les anciens axes ambigus `mcp_tools/` et `server_api/` ne sont plus présents comme dossiers actifs.

La dette Phase 1 restante n'est donc plus : "qui possède `server_api` ou `mcp_tools` ?"

La dette restante est plutôt :

- références mortes dans docs anciennes ;
- commentaires historiques qui mentionnent encore Flask/server_api ;
- wording à nettoyer dans certains rapports ;
- clarification officielle que la surface active est FastMCP direct + modules `mcp_core/`, `server_core/`, `shared/`, `pulse_app.py`.

### 2. S26/S27 sont historiques, pas source de vérité complète

`AGENT_HANDOFF_2026-05-14_S26.md` décrit un premier `pulse_app.py` très limité :

- un backend `get_pulse_metrics`;
- une UI `pulse_dashboard`;
- état standalone, non intégré serveur.

`PULSE_S27_2026-05-14.md` demande ensuite des panels performance/cache/security/rate-limit/intelligence.

Mais le code local actuel a déjà dépassé cette étape.

### 3. Le dashboard actuel suit plutôt S28/S29 + début Session C

`pulse_app.py` actuel contient déjà :

- `get_overview()`;
- `get_scope()`;
- `get_surface()`;
- `get_findings()`;
- `get_tool_intelligence()`;
- `get_recent_scans()`;
- `get_plan()`;
- `get_active_tools()`;
- `get_history()`;
- `pulse_dashboard()`.

Cela correspond davantage à la direction décrite dans `AGENTS.md` :

- Header + Scope ;
- Surface + Findings ;
- Plan IDE ;
- Tools actifs ;
- Historique ;
- Intelligence.

### 4. Les tests dashboard existent localement

`tests/test_pulse_app.py` existe en local non tracké selon l'état Git observé.

Une exécution ciblée précédente a donné :

```text
53 passed
```

Ce résultat est bon, mais il ne suffit pas encore pour push final. Il valide surtout le dashboard isolé.

### 5. L'état Git est cohérent avec une branche en chantier

Travail local observé :

- modifications dans `pulse_app.py`;
- ajout ou modification autour de `server_core/operational_metrics.py`;
- nouveaux fichiers utilitaires `export_dashboard.py`, `serve_pulse.py`;
- tests dashboard locaux ;
- quelques changements de mode sur scripts shell.

Ce n'est pas un problème tant que la règle reste : pas de push avant validation complète.

---

## Risques à surveiller

### Risque 1 — Docs S26/S27 dépassées

Les prochains agents peuvent lire S26/S27 et croire que le dashboard est encore au stade "premier panel".

Mitigation :

- garder S26/S27 comme historique ;
- créer un handoff S30/S31 qui devient la source de vérité récente ;
- marquer explicitement S26/S27 comme "historical, superseded by current dashboard branch state".

### Risque 2 — Dashboard qui glisse vers l'exécution

Le dashboard doit rester une interface d'observation, de planning et de suggestion.

Règle forte :

```text
Dashboard may observe, summarize, plan, and suggest.
Dashboard must not execute scans automatically without explicit user confirmation.
```

`get_plan()` peut produire une chaîne d'attaque. C'est acceptable si c'est un plan lisible, pas une exécution.

Toute fonction future du type `run_plan`, `start_scan`, `execute_tool`, `launch_attack_chain` doit être considérée sensible et nécessiter confirmation explicite.

### Risque 3 — Scope auto-détecté trop implicite

Le scope actif est déduit du cache de scans.

C'est pratique, mais il faut éviter que l'utilisateur pense agir sur une cible alors que le dashboard a auto-sélectionné une autre cible récente.

Mitigation :

- afficher la cible active clairement ;
- afficher l'âge du dernier scan ;
- afficher les outils ayant produit ce scope ;
- demander confirmation si une action exécutable est ajoutée plus tard.

### Risque 4 — Preview browser vs rendu MCP réel

`fastmcp dev` valide les tools/UI, mais n'est pas forcément un rendu identique au client MCP final.

Mitigation :

- garder `fastmcp dev` pour validation d'enregistrement ;
- garder `serve_pulse.py` comme preview locale si utile ;
- valider le rendu réel dans Claude Desktop ou client MCP compatible avant annoncer "UI done".

### Risque 5 — Mode bits parasites

Les scripts `scripts/patch_beartype_for_coverage.sh` et `scripts/restore_beartype.sh` ont montré des changements de mode.

Mitigation :

- vérifier si ces changements sont intentionnels ;
- sinon les restaurer avant commit pour éviter du bruit dans la PR.

---

## Suite d'audit proposée

### Étape A — Établir la vérité dashboard actuelle

Objectif : produire une table simple :

| Élément | Doc S26/S27 | Code actuel | Statut |
|---|---|---|---|
| Header | partiel | présent | OK |
| Scope | absent S26 | présent | OK |
| Surface | absent S26 | présent | OK |
| Findings | absent S26 | présent | OK |
| Plan IDE | future Session C | présent | à vérifier |
| Active tools | future Session C | présent | à vérifier |
| History | future Session C | présent | à vérifier |
| Execution auto | interdit | non observé dans dashboard | OK sous réserve |

### Étape B — Vérifier les tests avant push

Ordre recommandé :

```bash
python -m pytest tests/test_pulse_app.py -v -q --tb=short
python -m pytest tests/ --ignore=tests/test_real_integration.py --ignore=tests/test_server_setup_standalone.py -n auto -q
```

Si les tests passent :

- documenter le résultat ;
- vérifier le diff ;
- décider du commit ;
- push uniquement après arbitrage HexDevMaster.

### Étape C — Nettoyer le diff avant PR

À vérifier avant commit :

- `.gitignore` : ajout `debug_app.py` intentionnel ?
- scripts shell : mode `100755 -> 100644` intentionnel ?
- `export_dashboard.py` et `serve_pulse.py` : debug tools à garder ou non ?
- `tests/test_pulse_app.py` : doit être tracké si dashboard livré.

### Étape D — Écrire le handoff final

Créer ensuite :

```text
Projects_reports_docs/AGENT_HANDOFF_2026-05-14_S30.md
```

Contenu minimal :

- résumé réel du dashboard actuel ;
- liste des fichiers modifiés ;
- tests exécutés ;
- décision sur preview/export files ;
- risques restants ;
- statut push/PR.

---

## Verdict provisoire

Le travail local est cohérent avec une branche dashboard en chantier avancé.

Le point fort : la direction est maintenant claire — dashboard Prefab centré terrain, branché sur métriques, cache, scope, findings, IDE et process manager.

Le point faible : les docs récentes ne sont plus parfaitement alignées avec le code actuel, parce que le travail a avancé plus vite que les handoffs.

Décision recommandée :

1. ne pas push immédiatement ;
2. finir la validation tests ;
3. nettoyer le diff ;
4. écrire un handoff S30 propre ;
5. préparer PR seulement après validation HexDevMaster.

