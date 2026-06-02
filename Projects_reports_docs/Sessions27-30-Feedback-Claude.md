# Feedback Claude — Sessions 27→30
# Date : 2026-05-15
# De : Claude
# Pour : HexDevMaster
# Validé par : Nexus

---

## Vue d'ensemble

4 sessions denses et bien ordonnées. La progression est logique :
cleanup → dashboard complet → performance. Zéro régression sur 2580 tests.
Le projet est dans le meilleur état technique depuis le début.

---

## Session 27 — Cleanup + Prefab 6 panels

### Ce qui est bien
5 deps orphelines retirées proprement (selenium, aiohttp, mitmproxy,
beautifulsoup4, webdriver-manager). C'était attendu depuis la décision
de Nexus de virer Flask/React — le requirements.txt reflète enfin
les vraies dépendances du projet.

6 panels Prefab livrés en une session — rythme soutenu.

### Point à noter
Branch feature/prefab-dashboard mergée et taguée v0.8.0 dans AGENTS.md
mais le tag était déjà v0.8.0 depuis S21. À clarifier — est-ce v0.9.0
ou un re-tag du même commit ?

---

## Session 28 — Header + Scope

### Ce qui est bien
get_scope() avec auto-détection de la cible depuis _scan_cache —
c'est exactement la bonne approche. Le dashboard s'adapte au contexte
actif sans que l'utilisateur ait à saisir quoi que ce soit.

_op_metrics.system_metrics() qui retourne memory_total_gb —
correction propre, utilisée dans le header RAM display.

5 dev panels retirés — le dashboard se recentre sur les données terrain.
Décision conforme à la vision de Nexus : "observe, plan, suggest".

### Point à noter
Legacy panels conservés dans cette session (System Resources,
Recent Activity, Intelligence DataTable) — normalement remplacés
en S29. Vérifier qu'ils ont bien été retirés ou remplacés.

---

## Session 29 — Dashboard complet

### Ce qui est excellent
Parsing terrain réel : get_surface() parse stdout nmap/whatweb,
get_findings() parse nuclei/nikto. C'est du vrai travail terrain —
pas des données mockées. Le dashboard consomme des sorties d'outils réels.

_cache_for_target(target) : helper qui filtre _scan_cache par
cible à travers toutes les sessions. Exactement ce que le senior pentest
attend — historique filtré par scope, pas global.

get_plan() branché sur IntelligentDecisionEngine — la boucle
IDE → dashboard est fermée. Les recommandations sont celles que l'IDE
calcule avec les vrais scores terrain (post ToolStatsStore feed S26).

Binary names underscore→hyphen unifiés — cohérence avec tool_registry.py.

### Points à surveiller

Parser fragile sur get_findings()
Regex sur stdout terminal nuclei/nikto — fonctionnel mais cassant.
Si nuclei change son format de sortie entre versions, les findings
deviennent silencieusement vides sans erreur visible.
Recommandation : ajouter un fallback qui logue un warning si 0 findings
parsés sur un scan nuclei connu.

Risk level heuristique trop simple dans get_surface()
Le count de ports brut ne reflète pas le vrai risque.
WAF détecté + WordPress + port 3306 exposé = high même avec 3 ports.
L'IDE (_determine_risk_level()) a une logique plus sophistiquée.
A aligner progressivement — pas urgent pour la session actuelle.

get_pulse_data() timeout via stdio
10 sous-tools agrégés en un seul appel. Question clé :
sont-ils appelés en asyncio.gather() ou séquentiellement ?
Si séquentiel → 10 × latence = timeout garanti sur stdio.
Fix recommandé :

    async def get_pulse_data(target=None):
        results = await asyncio.gather(
            get_overview(),
            get_scope(),
            get_surface(target),
            get_findings(target),
            get_plan(target),
            get_active_tools(),
            get_history(target),
            get_intelligence_stats(),
            return_exceptions=True
        )

---

## Session 30 — Lazy init fixes (meilleur travail technique des 4 sessions)

### Ce qui est excellent

_prewarm_singletons() background thread
C'est la bonne solution au timeout stdio. En pré-initialisant
get_decision_engine() + get_tool_stats_store() après le start
serveur, le premier appel get_pulse_data() ne bloque plus 100-350ms.
Pattern propre, non-bloquant, cohérent avec l'architecture lazy singleton.

ParameterOptimizer() lazy sur classe
_get_parameter_optimizer() comme méthode lazy au lieu d'instance
module-level — exactement le même pattern que les autres singletons.
Cohérence architecturale maintenue.

CTFAutomator — suppression des singletons dupliqués
CTFWorkflowManager() et CTFToolManager() en module-level dans
automator.py étaient des doublons des singletons centraux.
self._manager / self._tools lazy via get_ctf_manager() /
get_ctf_tools() — correction propre.

resolve_data_dir() + ensure_data_dir() dans config_core.py
Thread-safe, idempotent. Centralise la gestion des data dirs.
Élimine les os.makedirs() redondants.

### Point à vérifier
_prewarm_singletons() est lancé en thread background après server start.
Si le thread plante, le serveur continue mais le premier appel sera lent.
Vérifier que le thread a un try/except global avec logging :

    def _prewarm_singletons():
        try:
            get_decision_engine()
            get_tool_stats_store()
            logger.info("Singletons pre-warmed")
        except Exception as e:
            logger.warning(f"Pre-warm failed: {e}")

---

## Ce qui reste a faire — priorites

P0 — Tester get_pulse_data() via MCP maintenant
Le _prewarm_singletons() de S30 devrait resoudre le timeout.
Relancer le serveur et appeler get_pulse_data() depuis Claude Desktop.

P1 — asyncio.gather() dans get_pulse_data()
Si timeout persiste → implémenter gather() sur les 10 sous-tools.
Gain estimé : 60-80% de réduction de latence.

P2 — tool_registry.py restant
Points DeepWiki non encore traités :
- sublist3r endpoint inversé
- zap/zaproxy conflit target optionnel vs requis
- auto_install_missing_apt_tools → décision Nexus : à virer
- _CLASSIFY_PROMPT catégories manquantes (feature CE — évaluer)

P3 — Post RPI terrain
- get_findings() parser : warning si 0 findings sur scan connu
- get_surface() risk level : aligner avec logique IDE
- Tests intégration réels (Phase 5 plan stabilisation)

---

## Métriques globales sessions 27-30

Tests S27 début : 2524 — S30 fin : 2580
Regressions : 0
Deps orphelines retirées : 5
Dashboard panels : 8
Lazy init issues résolus : 3
get_pulse_data timeout : en cours de résolution

---

## Mot de Claude

Sessions propres, bien documentées, zéro régression.
Le dashboard existe et répond. Le prewarm est en place.
La prochaine étape c'est de voir les données s'afficher
dans Claude Desktop — ce moment-là sera le vrai v0.9.0.
