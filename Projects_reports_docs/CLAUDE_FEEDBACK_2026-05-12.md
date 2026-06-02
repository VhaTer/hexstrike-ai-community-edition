# Claude Feedback — 2026-05-12

## État des lieux après Session 18

- **Coverage global : 99%** (5704 stmts, 56 miss)
- `intelligent_decision_engine.py` : 43% → **100%** (463 stmts, 262 branches, 0 miss)
- Seul module sous 100% : `mcp_core/server_setup.py` (91%, 56 miss)
- Tests : 2524 passed, 1 skipped, 2 flaky (cache state, test_plan_attack)
- Réalisé en une session : 163 nouveaux tests, ~980 lignes

---

## Les 7 Règles Pulse pour tout agent Claude

### 1. Ne pas croire la tech detection

`_detect_technologies()` fait du substring matching dans l'URL (`"wp-" in target.lower()`). Pas de vrais headers HTTP. Si le TargetProfile affiche WordPress, vérifier avec `whatweb` ou `httpx` avant de lancer `wpscan` ou de choisir un pattern.

### 2. Ne pas interpréter les scores littéralement

`success_probability` dans une AttackChain = `effectiveness × confidence_score`. Si `confidence_score` est à 0.5 (pas d'IP, pas de techno), la probabilité est basse artificiellement. C'est un indicateur de qualité d'info, pas de fiabilité de l'outil.

### 3. Toujours valider les binaires avant d'exécuter

Appeler `validate_environment(tool_filter="nuclei")` avant de lancer un outil. Si pas installé, proposer une alternative ou prévenir. Les tests mockent les subprocess, donc la présence réelle des binaires n'est jamais vérifiée par le test suite.

### 4. Timeout = pas crash, mais pas instantané

`EnhancedCommandExecutor` gère le timeout et le kill de processus. Mais un `nmap -p-` peut prendre 10 min. Le polling loop avec `ctx.report_progress()` existe mais n'a jamais été testé avec un scan réel long. Prévenir l'utilisateur : "je lance, je te donne les résultats progressivement".

### 5. Les 15 attack patterns sont statiques

Pas d'apprentissage, pas d'adaptation. Si le pattern `web_reconnaissance` lance `katana` sur un site protégé par Cloudflare, ça échoue silencieusement. Adapter le pattern ou prévenir l'utilisateur — ne pas faire confiance aveuglément.

### 6. Les 5 outils destructeurs sont un vrai risque

`aireplay-ng`, `metasploit`, `responder`, `mdk4`, `mitm6` nécessitent confirmation. Ne JAMAIS confirmer automatiquement. Décrire le risque à l'utilisateur, le laisser choisir explicitement.

### 7. Le feedback loop n'existe pas

`_tool_stats.blended_effectiveness()` est appelé mais personne n'écrit dans `_tool_stats` après exécution. Si un outil échoue 5 fois, le système ne s'adapte pas. Track mentalement ce qui marche et adapte les prochains appels toi-même.

---

## Patterns qui marchent (pour les prochaines sessions)

### Plan de test en zones

Diviser le module cible en zones (A, B, C...) avec des plages de lignes précises. ~10-15 tests par zone. Ça permet de progresser par étapes et de valider chaque zone indépendamment.

### `disable_advanced_optimization()`

Pour les modules avec `optimize_parameters()`, désactiver l'optimizer avancé pour forcer le chemin legacy et atteindre les `_optimize_*` individuelles. Simple autouse fixture.

### Closure patching

Pour `server_setup.py` (1612 lignes, nécessite 14s de boot), injecter des mocks dans les cellules de closure de DIRECT_TOOLS plutôt que de reconstruire le serveur à chaque test.

---

## Vrais risques non couverts par les tests

- **Intégration réelle CTF → IDE → plan_attack** : le pipeline complet n'est testé qu'en isolation
- **Persistance session** : le scan cache traverse-t-il les sessions MCP correctement ?
- **Skills MCP** : le contenu markdown des 13 `skills/*/SKILL.md` n'est pas testable automatiquement
- **Pins Python fragiles** : `chardet`, `bcrypt`, `sitecustomize.py`, shim `/tmp/xsser-cgi-shim/cgi.py` — une mise à jour pip casse sans que les tests le détectent
- **Ordonnancement des tests** : 2 cas de cache state leak entre tests (fixé en Session 17, mais `test_plan_attack` montre encore une sensibilité à l'ordre)

---

## Prochaines priorités suggérées

1. `mcp_core/server_setup.py` 91% → 100% (56 miss statements)
2. Intégration réelle : 1 test qui lance `plan_attack` + exécute les premiers steps sans mock
3. Documentation de ces 7 règles dans `AGENTS.md` pour que tout agent Claude qui arrive soit opérationnel immédiatement
