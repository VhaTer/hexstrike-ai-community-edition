# Known Issues

## `cancel_scan()` — comptage de sécurité sur le mauvais registre (S91)

**Ouvert 2026-08-02** — `same_tool_running` compte uniquement dans `_async_scans`
(scan async), mais le kill cible `ProcessManager.list_active_processes()` qui
contient aussi les commandes lancées par `scan()` (chemin synchrone). Scénario :
`scan('target_A')` synchrone + `run_async_tool('nmap', 'target_B')` → cancel sur
l'async → `same_tool_running` = 0 → kill par préfixe `nmap ` → le process du scan
synchrone peut être tué à la place. Séverité modérée (cas spécifique : deux
invocations concurrentes du même tool, une sync + une async), pas de corruption
de données. Piste de fix : taguer les entrées `ProcessManager` avec le `scan_id`
d'origine pour un ciblage exact au lieu du best-effort par préfixe.
Sources : `pulse_app.py:1244-1256`, `CLARIFICATION_S91_CANCEL_SCAN_2026-08-02.md`.

## `cancel_scan()` — kill par préfixe `tool + " "` rate les binaires ≠ nom du tool (P2, S91)

**Ouvert 2026-08-02** — `nmap_advanced` (commande `nmap_advanced ...`) et
`metasploit` (binaire `msfconsole`) ne matchent pas le préfixe → subprocess non
tué. Le cancel logique fonctionne quand même (statut `cancelled` + résultat non
utilisable), le kill est un bonus d'arrêt réel, jamais un prérequis. Piste de
fix : mapping tool → binaire réel.

## ~~Refactor `_require()` → `mcp_core._helpers.require()` — 2/12~~

✅ **Closed 2026-06-26** — Déjà 100% fait. Tous les 14 fichiers importent depuis `mcp_core._helpers` :
`active_directory_direct`, `exploit_framework_direct`, `net_scan_direct`, `osint_direct`,
`password_cracking_direct`, `recon_direct`, `security_direct`, `smb_enum_direct`, `testssl_direct`,
`web_fuzz_direct`, `web_probe_direct`, `web_recon_direct`, `web_scan_direct`, `vuln_intel_direct`
(+ `misc_direct`, `wifi_direct` déjà migrés avant la doc). Zéro définition locale `_require()` restante.
