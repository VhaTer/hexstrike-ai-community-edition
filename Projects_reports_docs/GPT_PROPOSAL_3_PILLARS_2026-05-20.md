# Proposition GPT — 3 piliers architecturaux

## Contexte
GPT a analysé le repo HexStrike et proposé une vision "operational brain" avec 3 piliers. Voici ce qu'il propose vs où nous en sommes.

---

## Pilier 1 — Event Bus + State Engine

**GPT propose :**
- Centralized event bus avec events typés (tool.completed, scan.started, target.found, error.raised)
- State engine avec reducers (pattern Redux) : chaque event → nouveau state
- Dashboard = projection pure du state. Plus de 20 appels parallèles indépendants.
- Le state engine est la source de vérité unique.

**Notre codebase :** Rien. Le seul "pub/sub" est `LogSubscriber` dans `setup_logging.py` qui stream les logs. Le dashboard lit 20+ data sources synchrones via `_collect_dashboard_state()` / `asyncio.gather()`. Zéro state machine, zéro reducer, zéro event bus.

---

## Pilier 2 — Dashboard multi-vues (workspaces/tabs)

**GPT propose :**
- Au lieu d'un plateau vertical de 10 sections : tabs séparés (Scan / Monitor / Intelligence / History)
- Chaque tab = un workspace avec ses propres panels
- L'agent peut passer d'une vue à l'autre, le state suit
- Architecture Prefab `@app.ui()` multi-entry ou Carousel/routing

**Notre codebase :** `pulse_app.py` ~2030 lignes, un seul template Prefab monolithique avec 10 sections empilées verticalement. Pas de système de tabs, pas de workspace switching, pas de routing entre vues.

---

## Pilier 3 — Orchestration adaptative + confidence loop

**GPT propose :**
- `scan()` ne lance plus N tools séquentiellement
- Boucle : exécute outil → parse résultat → ajuste `confidence_level` dynamique → choisit prochain outil
- Phases (Recon → Vuln Scan → Exploit → Report) avec état partagé
- `IntelligentDecisionEngine` utilisé en temps réel, pas en pré-plan statique

**Notre codebase :**
- Partiel. `_execute_bb_phase()` dans `bugbounty_engine.py` gère des phases séquentielles avec outils concurrents dans chaque phase
- `CTFWorkflowManager` existe
- `IntelligentDecisionEngine` donne un plan statique au début — pas de boucle d'adaptation runtime
- `next_suggested_tool` (Couche 3) est un conseil post-hoc basé sur l'output, pas une décision en temps réel
- Aucun état partagé entre les appels outil

---

## Où nous en sommes aujourd'hui

| Pilier | Status |
|--------|--------|
| Event bus | Rien. Seul pub/sub = LogSubscriber pour les logs |
| State engine | Rien. Dashboard = asyncio.gather() sur 20 sources indépendantes |
| Dashboard multi-vues | Un seul template Prefab monolithique (~2030 lignes). Pas de tabs/routing |
| Orchestration adaptative | Partiel : `_execute_bb_phase()` existe, `IntelligentDecisionEngine` donne un plan statique, `next_suggested_tool` est un conseil post-hoc |
| Boucle confidence | Pas implémenté. Aucun état partagé entre les appels outil |

---

## Discussion ouverte

GPT a identifié des patterns valides (event-driven, state projection, orchestration adaptative) qui sont absents de notre codebase. Les fondations existent pour certains aspects (Couche 3, bug bounty phases, decision engine) mais ne sont pas reliées entre elles dans une architecture cohérente.

**Question :** Est-ce qu'on veut aller vers cette direction (refactor architectural majeur — semaines, pas heures) ou garder l'approche actuelle (monolithique, synchrone, plate) et l'améliorer incrémentalement ?
