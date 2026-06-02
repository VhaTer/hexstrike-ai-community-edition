# Session 57 — Phase 1 Pipeline Fixes & RPI Validation

**Date** : 2026-05-28
**Tag** : `v0.10.1` (branch `master`)

## Résumé

4 bugs bloquants du pipeline MCP identifiés et corrigés. Le pipeline `scan()` complet est maintenant validé de bout en bout sur le RPI réel (DVWA) via opencode MCP.

## Bugs corrigés

### Bug #1 — `task=True` sur tous les tools (perte de connexion MCP)

**Fichier** : `mcp_core/server_setup.py`
**Lignes** : 1108 (run_security_tool), 1823-1829 (130 typed tools)

**Symptôme** : Les outils longs (nuclei, nikto) plantent après ~2s et le client MCP perd la connexion. Les outils rapides (nmap 6s) fonctionnent.

**Cause** : `@mcp.tool(task=True, timeout=None)` → fastmcp expose le tool via le protocole MCP task. opencode (client) ne supporte pas le task protocol → le tool call ne retourne jamais, le client timeout et ferme la connexion.

**Fix** : `task=True` retiré des deux endroits. `timeout=None` retiré aussi — fastmcp n'a pas de timeout par défaut, c'est le client (opencode 300s) qui contrôle.

```python
# AVANT
@mcp.tool(task=True, timeout=None)
# APRÈS
@mcp.tool()
```

### Bug #2 — Parser `get_findings()` ignoré les ANSI codes

**Fichier** : `pulse_app.py` (lignes 499-558)

**Symptôme** : `get_findings()` retourne `[]` alors que nuclei a trouvé 3 vulnérabilités (dvwa-default-login critical, dockerfile-hidden-disclosure medium, git-config medium).

**Cause** : L'output nuclei contient des codes ANSI (`\x1b[31m`). Le regex `\[(critical|high|medium|low|info)\]` ne matche pas `[\u001b[31mcritical\u001b[0m]`.

**Fix** : `_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")` + `_strip_ansi()` appliqué avant parsing.

```python
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)
```

### Bug #3 — Parser `get_surface()` inclus les ports "filtered"

**Fichier** : `pulse_app.py` (ligne 451)

**Symptôme** : `get_surface()` retourne des ports "filtered" comme s'ils étaient ouverts.

**Cause** : Le parser lisait `/tcp filtered mysql` → split → `parts[0]="3306/tcp"`, `parts[1]="filtered"`, mais le test était juste `len(parts) >= 2 and "/" in parts[0]` sans vérifier l'état.

**Fix** : Ajout de `and parts[1] == "open"`.

```python
# AVANT
if len(parts) >= 2 and "/" in parts[0]:
# APRÈS
if len(parts) >= 2 and "/" in parts[0] and parts[1] == "open":
```

### Bug #4 — `scan()` passe l'URL entière à nmap (0 hosts)

**Fichier** : `pulse_app.py` (lignes 1167-1168, 1466-1476)

**Symptôme** : `scan(http://192.168.1.165/DVWA/)` → nmap reçoit `nmap http://192.168.1.165/DVWA/` → "Unable to split netmask from target expression" → 0 hosts scannés → `get_surface()` montre 0 ports.

**Cause** : nmap n'était ni dans `_TOOLS_NEED_URL` (qui passe `url` au lieu de `target`) ni dans `_TOOLS_NEED_URL_AS_TARGET`. Il recevait l'URL brute.

**Fix** : Nouvelle catégorie `_TOOLS_NEED_HOST` + extraction hostname via `urlparse()` + `scan_type="-sTV"` (TCP connect place de `-sCV` SYN, bloqué par le firewall RPI).

```python
_TOOLS_NEED_HOST = {"nmap", "nmap_advanced"}
# ...
elif tool_name in _TOOLS_NEED_HOST:
    host = urlparse(resolved).hostname or resolved
    params = {"target": host, "scan_type": "-sTV"}
```

### Bug #5 (correction mineure) — `nikto -no-update` invalide

**Fichier** : `mcp_core/web_scan_direct.py` (ligne 56)

**Cause** : Le flag `-no-update` n'existe pas dans nikto 2.6.0. Le flag correct pour skip l'update check est `-nocheck`. Le LAN (via bbox.lan 192.168.1.254) bloquait l'update check en HTTP 403 → nikto freeze.

**Fix** : `-no-update` → `-nocheck`.

### Bug #6 — Cache invalidation des résultats nmap "0 hosts up"

**Fichier** : `pulse_app.py` (lignes 1457-1463)

**Symptôme** : Les résultats buggés du Bug #4 restent en cache → même après le fix, `scan()` skip nmap (trouvé en cache) → 0 ports.

**Fix** : Invalidation conditionnelle des entrées nmap dont le stdout commence par "Starting Nmap" et contient "0 hosts up".

## Pipeline MCP validé (RPI 192.168.1.165)

### scan(quick)

```json
{
  "ports": [{"port": 22, "service": "ssh"}, {"port": 80, "service": "http"}],
  "ports_count": 2,
  "technologies": ["Apache", "PHP"],
  "next_suggested_tool": {"tool": "gobuster", "reason": "Web server detected with tech — discover hidden paths"}
}
```

### nuclei (full URL)

```json
{
  "findings": [
    {"id": "dvwa-default-login", "severity": "critical", "url": "http://192.168.1.165/DVWA/index.php", "credentials": "admin/password"},
    {"id": "dockerfile-hidden-disclosure", "severity": "medium"},
    {"id": "git-config", "severity": "medium"}
  ]
}
```

### nikto

12 findings info (missing security headers, directory listing, config exposure).

### RPI network constraints

- `-sCV` (SYN scan) → **filtered** par le firewall RPI (iptables `--syn -j DROP`)
- `-sTV` (TCP connect) → **open** (2 ports)
- LAN Ethernet direct (plus hotspot Samsung)

## Fichiers modifiés

```
pulse_app.py                         | 4 bugs fixés (task=True, ANSI, filtered, hostname)
mcp_core/web_scan_direct.py          | 1 bug fixé (nikto -nocheck)
tests/test_pipeline_integration.py   | 2 tests ajustés (3306 filtered)
```

## Tests

```
test_pulse_app.py              : 64/64  ✅
test_pipeline_integration.py   : 22/22  ✅
Full suite (non-slow)          : 2779/2787 ✅ (8 pré-existants, server/CLI)
```

## Configuration

- `opencode.jsonc timeout` : 120000 → **300000** (match Pulse 300s)
- Claude Desktop MCP désactivé (conflit lock file avec opencode)
- MCP lock file : `/tmp/hexstrike_mcp.lock`, TTL 5s, fcntl+PID combiné

## État du projet

- Phase 1 (pipeline scan) : **✅ complète et validée sur cible réelle**
- MCP via opencode : **✅ stable** (tools longs, async, pas de task protocol)
- Prochaine étape : Phase 2 — intégration tests + couverture beartype restante + TargetStore MCP resources tests

## Récupération

Le tag `v0.10.1` existe. Les bugs listés ici sont tous sur `master` (branch locale). Aucun push effectué cette session.
