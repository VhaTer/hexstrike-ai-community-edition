# Dev-to-dev — Session 18 (2026-05-12)

Hey,

Belle session 17 — le closure patching sur `__closure__` pour injecter dans
`DIRECT_TOOLS` sans rebooter le serveur, c'est la meilleure technique qu'on
ait vue sur ce projet. Propre et rapide.

Pour la S18, `intelligent_decision_engine.py` à 43% t'attend.
J'ai lu tout le module avant d'écrire ça — voici ce que j'ai trouvé.

---

## État actuel

```bash
pytest tests/ --cov=server_core.intelligence.intelligent_decision_engine \
  --cov-report=term-missing -q \
  --ignore=tests/test_real_integration.py \
  --ignore=tests/test_server_setup_standalone.py \
  2>&1 | grep -A 80 "intelligent_decision_engine"
```

Lance ça en premier pour voir les miss exacts avant de planifier.

---

## Anatomie du module (ce que j'ai lu)

Le module fait ~700 lignes. Structure :

```
IntelligentDecisionEngine
├── __init__()                          ← 3 dicts énormes de config statique
├── analyze_target(target) → TargetProfile
│   ├── _determine_target_type()        ← 6 branches regex/pattern
│   ├── _resolve_domain()               ← socket.gethostbyname, try/except
│   ├── _detect_technologies()          ← heuristics string matching
│   ├── _detect_cms()                   ← 3 branches CMS
│   ├── _calculate_attack_surface()     ← scoring avec open_ports/subdomains
│   ├── _determine_risk_level()         ← 5 seuils float
│   └── _calculate_confidence()        ← 4 conditions
├── select_optimal_tools(profile, objective) → List[str]
│   ├── objective == "quick"            ← top 3 by _effective_score
│   ├── objective == "comprehensive"    ← filter > 0.7
│   ├── objective == "stealth"          ← fixed list
│   └── else / tech-specific appends
├── optimize_parameters(tool, profile, context)
│   ├── _use_advanced_optimizer True    ← path dominant → parameter_optimizer
│   ├── _use_advanced_optimizer False   ← 20 elif tool-specific
│   └── else → parameter_optimizer
├── _effective_score(tool, target_type_value) ← _tool_stats.blended_effectiveness
├── create_attack_chain(profile, objective) → AttackChain
│   ├── 5 TargetType branches
│   └── TargetType.UNKNOWN → bug_bounty fallbacks
└── enable/disable_advanced_optimization()
```

---

## Zones de miss probables et stratégie

### Zone A — `analyze_target` branches (haute valeur, tests simples)

`_determine_target_type` a 6 branches distinctes — toutes testables sans mock :

```python
# Ces cas sont probablement non couverts :
ide.analyze_target("http://example.com/api/v1")        # → API_ENDPOINT
ide.analyze_target("192.168.1.1")                      # → NETWORK_HOST
ide.analyze_target("example.exe")                      # → BINARY_FILE
ide.analyze_target("s3.amazonaws.com/bucket")          # → CLOUD_SERVICE
ide.analyze_target("notavaliddomain")                  # → UNKNOWN (fallback)
ide.analyze_target("http://wordpress-site.com/wp-content")  # → WEB + WordPress tech
```

`_resolve_domain` : le `except Exception: pass` → retourne `[]`.
Mock `socket.gethostbyname` pour couvrir les deux branches (succès + exception).

`_calculate_attack_surface` : les branches `open_ports` et `subdomains` sont
probablement non couvertes. Crée un `TargetProfile` avec des ports et subdomains
peuplés manuellement après construction.

### Zone B — `select_optimal_tools` objectives

4 branches d'objectif. Seul "comprehensive" est probablement couvert.
À tester :

```python
# "quick" → top 3 tools
tools = ide.select_optimal_tools(profile, objective="quick")
assert len(tools) == 3

# "stealth" → subset fixed
tools = ide.select_optimal_tools(profile, objective="stealth")
assert all(t in ["amass", "subfinder", "httpx", "nuclei"] for t in tools)

# "unknown_objective" → else branch
tools = ide.select_optimal_tools(profile, objective="totally_unknown")

# tech-specific appends (WordPress → wpscan, PHP → nikto)
```

### Zone C — `optimize_parameters` avec `_use_advanced_optimizer = False`

C'est le plus gros volume de miss. 20 `elif` tool-specific non couverts.
La stratégie : `ide.disable_advanced_optimization()` puis tester les tools.

```python
ide.disable_advanced_optimization()

# nmap web vs network
p = ide.optimize_parameters("nmap", web_profile)
assert p["ports"] == "80,443,8080,8443,8000,9000"

p = ide.optimize_parameters("nmap", web_profile, {"stealth": True})
assert "-T2" in p.get("additional_args", "")

# Chaque tool a 2-3 branches context. Couvre au moins :
# nmap, gobuster (3 tech branches), nuclei (tags), sqlmap, ffuf, hydra,
# rustscan (stealth/aggressive/default), masscan, ghidra, trivy
```

Le `else` final (tool inconnu) → `parameter_optimizer.optimize_parameters_advanced`.
Test avec `tool="unknown_tool_xyz"`.

### Zone D — `create_attack_chain` objectives

Les branches `TargetType.CLOUD_SERVICE` ont 5 sous-objectifs (aws, kubernetes,
containers, iac, else). Probablement zéro couverture ici.

```python
# CLOUD_SERVICE avec chaque objectif
for obj in ["aws", "kubernetes", "containers", "iac", "multi_cloud"]:
    profile = make_profile(TargetType.CLOUD_SERVICE)
    chain = ide.create_attack_chain(profile, objective=obj)
    assert chain is not None

# UNKNOWN type → bug_bounty_recon / hunting / high_impact / else
for obj in ["bug_bounty_recon", "bug_bounty_hunting", "bug_bounty_high_impact", "other"]:
    profile = make_profile(TargetType.UNKNOWN)
    chain = ide.create_attack_chain(profile, objective=obj)
```

---

## Setup helper (lis ça avant d'écrire les tests)

`TargetProfile` est dans `shared/target_profile.py`. Il a probablement des
attributs comme `open_ports`, `subdomains`, `technologies`, `cms_type`.
Vérifie la signature exacte avant de te lancer :

```bash
grep -n "def __init__\|open_ports\|subdomains\|technologies" \
  shared/target_profile.py | head -20
```

Ça t'évitera de construire des profiles incorrects.

---

## Fichier de test à créer

```
tests/test_intelligent_decision_engine.py
```

Pas de server boot nécessaire ici — le module n'a aucune dépendance MCP.
Tests synchrones purs. Sera rapide.

Structure suggérée :

```python
import pytest
from unittest.mock import patch, MagicMock
from server_core.intelligence.intelligent_decision_engine import IntelligentDecisionEngine
from shared.target_types import TargetType, TechnologyStack
from shared.target_profile import TargetProfile

@pytest.fixture
def ide():
    return IntelligentDecisionEngine()

class TestDetermineTargetType: ...
class TestAnalyzeTarget: ...
class TestSelectOptimalTools: ...
class TestOptimizeParameters: ...
class TestCreateAttackChain: ...
```

---

## Points de vigilance

**`_effective_score` et `_tool_stats`**
`_tool_stats = ToolStatsStore()` est un singleton module-level (pas lazy).
`blended_effectiveness(tool, baseline)` peut retourner le baseline si aucune
stat n'existe encore. Mock proprement :

```python
with patch.object(ide, '_tool_stats') as mock_stats:
    mock_stats.blended_effectiveness.return_value = 0.85
    ...
```

Ou mieux : patch au niveau module `patch("server_core.intelligence.intelligent_decision_engine._tool_stats")`.

**`_resolve_domain` fait un vrai DNS lookup**
Patch `socket.gethostbyname` systématiquement dans les tests `analyze_target`
pour éviter des tests flaky en offline.

**`_optimize_trivy_params` et `_optimize_checkov_params`**
Appellent `os.path.isdir()` — patch si nécessaire pour les branches directory.

---

## Objectif chiffré

| Métrique | Avant | Cible |
|----------|-------|-------|
| `intelligent_decision_engine.py` | 43% | ≥ 75% |
| Tests total | 2390 | ≥ 2390 |
| Coverage globale | 93% | ≥ 94% |

75% est réaliste en une session. Les miss résiduels seront probablement dans
les `_optimize_*` edge cases les plus spécifiques (angr, pacu, etc.) — pas
critiques pour la stabilisation.

---

## Handoff en fin de session

`Projects_reports_docs/AGENT_HANDOFF_2026-05-12_S18.md` avec :
- coverage avant/après par méthode (si tu peux l'obtenir avec `--cov-report=term-missing`)
- techniques utilisées
- miss résiduels documentés et justifiés
- `pytest tests/ -q | tail -3`
- commit hash

Bonne session.
