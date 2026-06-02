# Correction architecture — Transport & RPI
**Date :** 2026-05-17
**Auteur :** Claude (Sonnet 4.6)
**Pour :** HexDevMaster + HexReaper
**Sujet :** Erreur de raisonnement corrigée sur le transport HTTP et le rôle du RPI

---

## Erreur identifiée

Dans plusieurs briefs (S35, S36, bilan S31-S36), Claude a suggéré que le RPI pourrait
héberger HexStrike en HTTP et que Claude Desktop pourrait s'y connecter via URL.

**C'est faux.**

---

## Réalité du transport Claude Desktop

Claude Desktop supporte **stdio uniquement** pour les MCP servers locaux.

```json
// claude_desktop_config.json — seul format supporté
{
  "mcpServers": {
    "hexstrike": {
      "command": "python3",
      "args": ["/home/vhater/hexstrike-ai-community-edition/hexstrike_mcp.py"]
    }
  }
}
```

Une URL HTTP dans `claude_desktop_config.json` ne fonctionne pas.
Modifier la config pour pointer vers `http://localhost:8000` ne change rien —
Claude Desktop ne sait pas parler HTTP à un MCP server local.

---

## Architecture correcte — RPI comme cible pure

```
Claude Desktop (Windows/Kali)
    │
    │  stdio
    ▼
hexstrike_mcp.py (Kali local)
    │
    │  réseau local — nmap, sqlmap, whatweb...
    ▼
RPI Ubuntu (cibles vulnérables)
    ├── DVWA
    ├── Juice Shop
    ├── Metasploitable
    └── vAPI / WebGoat
```

Le RPI est une **cible réseau**, pas un serveur MCP.
HexStrike tourne toujours sur Kali en stdio.
Claude Desktop ne parle jamais directement au RPI.

---

## Quand HTTP devient pertinent

HTTP transport est utile uniquement si on utilise un **client MCP différent** de Claude Desktop :
- Cline / Continue (VS Code) — supportent HTTP/SSE
- Client custom — peut pointer vers une URL
- claude.ai web — remote MCP servers via URL

Pour ces clients, `hexstrike_server.py` (Starlette + SSE) est la bonne entrée.
Pour Claude Desktop — `hexstrike_mcp.py` uniquement, toujours.

---

## Ce que ça change dans la roadmap

```
CORRIGÉ  HTTP transport → utile pour clients tiers (Cline, etc.), pas Claude Desktop
CORRIGÉ  RPI → cible pure, jamais serveur MCP
INCHANGÉ stdio → transport définitif pour Claude Desktop
INCHANGÉ hexstrike_mcp.py → seul point d'entrée pour Claude Desktop
```

---

*Claude Sonnet 4.6 — HexStrike team — 2026-05-17*
