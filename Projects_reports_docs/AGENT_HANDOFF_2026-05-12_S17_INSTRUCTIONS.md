# Instructions pour OpenCode — Session 17 (2026-05-12)

De : Claude (supervisor)
Pour : OpenCode (sub-agent)

---

## Contexte rapide

Le projet HexStrike AI-PULSE est sur `feature/attack-intelligence`.
État actuel après Session 16 :
- 2328 tests passed, 1 skipped
- Coverage globale : **92%** (5555 stmts, 363 miss)
- 55/63 modules à 100%

---

## Objectif Session 17

**`mcp_core/server_setup.py` : 83% → 90%+**

C'est la priorité absolue selon le plan de stabilisation (`hexstrike-ai_pulse_stabilisation_plan.md`).
Ce module est le cœur de `run_security_tool()` et `setup_mcp_server_standalone()`.
Il reste à 83% avec 121 miss et 258 branches — c'est le seul module critique non hardened.

---

## Pourquoi ce module en priorité

Le plan de stabilisation (Phase 2) dit explicitement :
> "Make the real Phase 3 execution path predictable and consistent."
> Primary focus: `mcp_core/server_setup.py` et `run_security_tool(...)`

Tant que ce module n'est pas à 90%+, la stabilisation Phase 2 n'est pas déclarée complète.
`intelligent_decision_engine.py` (43%) attend derrière — mais le plan dit de ne pas
approfondir l'intelligence avant que l'orchestration soit solide.

---

## Commandes de démarrage

```bash
cd ~/hexstrike-ai-community-edition
source hexstrike-env/bin/activate
git status --short
pytest tests/ -q 2>&1 | tail -3

# Voir les miss actuels de server_setup.py :
pytest tests/ --cov=mcp_core.server_setup --cov-report=term-missing -q \
  --ignore=tests/test_real_integration.py \
  --ignore=tests/test_server_setup_standalone.py \
  2>&1 | grep -A 50 "mcp_core/server_setup"
```

---

## Zones à cibler (ordre de priorité)

### 1. Branches de registration des DIRECT_TOOLS
`setup_mcp_server_standalone()` enregistre 114 tools en boucle.
Les branches non couvertes sont probablement :
- handler manquant / KeyError sur un tool
- tool avec `requires_confirmation=True` path alternatif
- import error sur un `*_direct.py` module

### 2. `run_security_tool()` paths résiduels
La Phase 2 des fixes a couvert les bad-flows principaux (invalid JSON, unknown tool, etc.)
mais des branches de `parameters` parsing et de cache interaction peuvent rester.

### 3. `setup_mcp_server_standalone()` edge cases
- server déjà initialisé (appel double)
- missing dependency au boot
- FastMCP registration errors

---

## Contraintes à respecter (non-négociables)

1. **`pytest tests/ -q` doit finir ≥ 2328 passed** avant de committer
2. **Ne pas modifier** `server_core/intelligence/` ni les managers V6
3. **Ne pas modifier** la logique de `run_security_tool()` — tests seulement
4. **Lazy singletons intacts** — ne pas instancier les managers au niveau module
5. **`--ignore=tests/test_real_integration.py --ignore=tests/test_server_setup_standalone.py`**
   pour la suite rapide (ces deux fichiers sont opt-in lents)

---

## Fichier de test existant à enrichir

```
tests/test_server_setup_standalone.py  ← déjà existant (slow, opt-in)
```

Vérifie d'abord ce qu'il couvre déjà :
```bash
pytest tests/test_server_setup_standalone.py --cov=mcp_core.server_setup \
  --cov-report=term-missing -q 2>&1 | tail -30
```

Si ce fichier est trop lent pour itérer, crée un fichier séparé :
```
tests/test_server_setup_unit.py  ← mocks lourds, pas de vrai server boot
```

---

## Résultat attendu en fin de session

| Métrique | Avant | Après |
|----------|-------|-------|
| `server_setup.py` coverage | 83% | ≥ 90% |
| Tests total | 2328 | ≥ 2328 |
| Coverage globale | 92% | ≥ 92% (stable ou +1%) |

---

## Handoff en fin de session

Crée `Projects_reports_docs/AGENT_HANDOFF_2026-05-12_S17.md` avec :
- modules couverts + % avant/après
- nombre de tests ajoutés
- techniques de mock utilisées
- modules restants < 90% (liste à jour)
- résultat `pytest tests/ -q | tail -3`
- commit hash

---

## Note sur `intelligent_decision_engine.py`

Ne pas attaquer ce module cette session. Il a 242 miss et 262 branches — c'est
une session entière dédiée. Quand `server_setup.py` sera à 90%+, ce sera la
prochaine priorité logique (Session 18).
