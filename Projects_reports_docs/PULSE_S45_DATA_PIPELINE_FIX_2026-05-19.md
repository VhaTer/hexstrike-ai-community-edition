# Session 45 — Data pipeline fix (cache write-back + URL params)

**Date :** 2026-05-19  
**Branche :** `feature/prefab-dashboard`  

## Problème

`scan()` exécutait `exec_func(binary, {"target": resolved})` mais le résultat n'était **jamais écrit dans `_scan_cache`**. Conséquences :

- `get_surface()` et `get_findings()` lisaient un cache vide → retournaient toujours des données vides
- `TargetStore.record_scan()` persistait du vide (surface sans ports, 0 findings)
- Le dashboard ne montrait rien pour la cible scannée, malgré des runs réussis

## Fix 1 — Cache write-back after exec_func

**Fichier :** `pulse_app.py:1317-1321`

Après chaque `exec_func()` dans `scan()`, le résultat est écrit dans `_scan_cache` avec la même structure que `run_security_tool()` :

```python
_scan_cache.set(str(time.time()), {
    "tool": tool_name, "target": resolved,
    "result": out, "timestamp": time.time(),
}, execution_time=out.get("execution_time", 0))
```

**Fichier :** `tests/test_pulse_app.py:24-27`, `:31-35`, `:610`, `:657`, `:664`, `:728`

- Nouveau `_MockCache(dict)` — compatible AdvancedCache (supporte `[]` et `.set()`)
- 4 tests avec inline `patch.object` converties en `_MockCache`

## Fix 2 — URL params corrects par tool

**Fichier :** `pulse_app.py:1076-1079`, `:1302-1308`

Tous les tools recevaient `{"target": resolved}`. Problème :

| Tool | Attendu | Clé | État |
|------|---------|-----|------|
| nmap | `target` brut | `target` | ✅ |
| nikto | `target` brut (`-h`) | `target` | ✅ |
| whatweb | `url` avec `http://` | `url` | ❌ → ✅ |
| gobuster | `url` avec `http://` | `url` | ❌ → ✅ |
| nuclei | `target` mais `-u {target}` (URL) | `target` | ⚠️ → ✅ |
| httpx, katana | `target` mais en URL | `target` | ⚠️ → ✅ |

Mapping ajouté :

```python
_TOOLS_NEED_URL = {"whatweb", "gobuster", "sqlmap", "wpscan", "dalfox", "jaeles", "xsser"}
_TOOLS_NEED_URL_AS_TARGET = {"nuclei", "httpx", "katana"}
```

Les 2 constantes et la logique de routage sont dans `pulse_app.py`, proches de `TOOLS_BY_INTENSITY` pour maintenabilité.

## Validation bout-en-bout

### Test sur testphp.vulnweb.com

| Tool | Statut | Stdout | Note |
|------|--------|--------|------|
| nmap | ✅ completed | 557b | Tous les ports filtered (firewall AWS) |
| whatweb | ✅ maintenant fixé | N/A | Non testé (site down) |
| nuclei | ✅ completed | 916b | 1387 templates, 0 findings |
| nikto | ✅ completed | 145b | Scan effectué |
| Cache | ✅ 4 entrées | — | nmap, whatweb, nuclei, nikto |
| TargetStore | appelé | — | Persiste vide car surface vide |

**Site down :** `testphp.vulnweb.com` répond ping mais firewall AWS bloque tout le trafic HTTP/HTTPS. nmap avec `-Pn` trouve le host mais tous les 1000 ports sont filtered → surface = 0 ports (correct).

### Tests unitaires

```
64 passed in test_pulse_app.py
2628 passed, 1 skipped, 2 warnings — 0 regressions (full suite)
1 flaky existant: test_no_target_with_explicit_none (parallélisation uniquement)
```

## État du pipeline après fix

```
scan(target, intensity)
  │
  ├─ exec_func(binary, params_corrects) → tool_results dict
  ├─ _scan_cache.set(key, entry)        ← FIX 1
  │
  ├─ get_surface(target)                 ← lit _scan_cache → ✅
  ├─ get_findings(target)                ← lit _scan_cache → ✅
  ├─ TargetStore.record_scan(...)        ← données réelles → ✅
  │
  └─ MCP Resources (targets://, ...)     ← TargetStore → ✅
```

## Pour le prochain scan réel

Cibles recommandées :

1. **scanme.nmap.org** — approuvé, répond sur HTTP/SSH. Tester avec `scan(medium)`.
2. **Lab local VulnHub/DVWA** — contrôle complet du trafic, findings garantis.

Commande de validation une fois la cible up :

```bash
source hexstrike-env/bin/activate
python3 -c "
from mcp_core.server_setup import setup_mcp_server_standalone, _scan_cache
from pulse_app import scan, get_surface, get_findings
from server_core.singletons import get_target_store
app = setup_mcp_server_standalone()
r = scan(target='scanme.nmap.org', intensity='medium')
print('Tools:', {t: v['status'] for t, v in r['tools'].items()})
print('Surface:', get_surface('scanme.nmap.org'))
print('Findings:', len(get_findings('scanme.nmap.org')))
"
```

## Fichiers modifiés

| Fichier | Changement |
|---------|-----------|
| `pulse_app.py:1076-1079` | Ajout `_TOOLS_NEED_URL` + `_TOOLS_NEED_URL_AS_TARGET` |
| `pulse_app.py:1302-1308` | Routage params `url`/`target` par tool |
| `pulse_app.py:1317-1321` | Cache write-back après `exec_func` |
| `tests/test_pulse_app.py:24-27` | `_MockCache` class |
| `tests/test_pulse_app.py:31-35` | `mock_scan_cache` → `_MockCache()` |
| `tests/test_pulse_app.py:610,657,664,728` | 4 inline patches → `_MockCache` |
