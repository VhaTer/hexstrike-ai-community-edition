# État du projet — 2026-05-14 (sync collaborateurs)

De : Claude
Pour : HexDevMaster + GPT/Codex
Validé par : Nexus

---

## État actuel — feature/pulse-dashboard

```
2580 passed, 1 skipped, 0 failed
Branch : feature/pulse-dashboard
Base   : master v0.8.0
```

---

## Ce qui vient d'être fixé (Claude + Nexus)

**`hexstrike_server.py` — KeyError 'timestamp'**

`_build_dashboard_response()` accédait à `v["timestamp"]`, `v["tool"]`,
`v["target"]` avec `[]` au lieu de `.get()`. Un test CLI peuplait
`_scan_cache` avec des entrées sans ces clés → 5 tests tombaient en suite
complète mais passaient en isolation (state leak).

Fix appliqué — `.get()` défensif sur toutes les clés :

```python
# Avant
"timestamp": v["timestamp"],   # KeyError si clé absente

# Après
"timestamp": v.get("timestamp", 0),  # défensif
```

Commit à faire avec le prochain batch de changements.

---

## État des fichiers non trackés

```
M  hexstrike.py                        ← CLI fixes P0 (Codex)
M  pulse_app.py                        ← dashboard Sessions A→C (HexDevMaster)
M  hexstrike_server.py                 ← fix KeyError (Claude, today)
M  server_core/operational_metrics.py  ← memory_total_gb (Session C)
?? export_dashboard.py                 ← décision pending (debug tool ?)
?? serve_pulse.py                      ← décision pending (debug tool ?)
?? tests/test_hexstrike_cli.py         ← CLI tests P0 (Codex)
?? tests/test_pulse_app.py             ← 53 tests dashboard (HexDevMaster)
```

---

## Pour HexDevMaster — Session D

Pulse Session D écrit dans :
`Projects_reports_docs/PULSE_SESSION_D_2026-05-14.md`

Priorités :
1. Renderer Prefab — tester `pulse_working.html` (bundled + data in head)
2. Layout 3+2+1 grid — code complet dans le Pulse Session D
3. Décision `export_dashboard.py` / `serve_pulse.py` avant commit

---

## Pour GPT/Codex — CLI hardening

Le travail P0 est en place (`hexstrike.py` modifié, `test_hexstrike_cli.py`
non tracké). Pulse CLI à écrire quand disponible.

Rappel du plan validé :

**P0 — fait** : fix `ctf` variables, fix `scan --json` stdout strict
**P1 — à faire** : `--no-color`, `--quiet`, `--verbose`, exit codes,
  confirmation tools sensibles (metasploit, sqlmap, hydra, etc.)
**P2 — après P1** : tables propres, badges, suggestions, UX operator-grade
**P3 — post-merge** : `hexstrike plan`, `hexstrike scope`,
  `hexstrike history`, `hexstrike intel`

P3 est le miroir CLI du dashboard — même data, deux interfaces.

---

## Règle non-négociable (rappel)

```
Dashboard / CLI : observe, plan, suggest.
Pas d'exécution automatique sans confirmation explicite.
```

Visible dans le header du dashboard : `no automatic execution`. 🔗
