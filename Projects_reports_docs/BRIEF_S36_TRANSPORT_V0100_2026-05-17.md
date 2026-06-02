# S36 — Transport audit + Couche 1 (5 tools) + v0.10.0 push

**Date:** 2026-05-17
**Branch:** master (v0.10.0)
**Commits:** 16 ahead of origin, pushed

---

## 1. Transport audit stdio vs HTTP 🟢

| Aspect | Status |
|--------|--------|
| `--transport` CLI arg | **Not present** — 7 args only |
| `mcp.run(transport="http")` | **Not called** — uses default stdio |
| FastMCP HTTP/SSE support | **Available** in FastMCP 3.2.4 but not wired |
| `hexstrike_server.py` | Separate process (Flask), not MCP transport |

**Conclusion :** stdio only. HTTP/SSE transport exists in FastMCP but requires adding `--transport` arg + passing to `mcp.run()`. Not urgent — stdio works with Claude Desktop. À garder pour roadmap post-RPI.

## 2. Couche 1 — tools restants 🟢

Nouveau `_TOOL_COUCHE1` dict dans `_build_typed_tool_doc()` :

| Tool | Workflow hint | Example |
|---|---|---|
| nmap | FIRST recon tool. Before whatweb. | `nmap(target='scanme.nmap.org')` |
| whatweb | Web tech detection. After nmap. | `whatweb(url='http://...')` |
| sqlmap | SQLi testing. After findings. | `sqlmap(url='...', additional_args='--batch')` |
| gobuster | Dir brute force. After whatweb. | `gobuster(target='...', params='dir -w ...')` |
| nuclei | Vuln scanning. After surface. | `nuclei(target='http://...')` |
| nikto | Web vuln scanner. After whatweb. | `nikto(target='http://...')` |

`_build_typed_tool_doc()` génère maintenant un template riche pour **tous les 130+ tools** :
- Workflow: (quand utiliser ce tool)
- Args: Required / Optional avec defaults
- Returns: structure du dict
- Example: (pour les 6 tools clés)
- Note: wrapper info

`server_health` enrichi directement (description + docstring).

## 3. v0.10.0 tag + push origin 🟢

```bash
git tag v0.10.0
git push origin master --tags
```

16 commits pushed to origin. Tags `v0.9.0` and `v0.10.0` pushed.

## Test result

```
2577 passed, 1 skipped, 2 warnings — 0 regressions
98/98 server_setup_coverage tests (doc builder updated)
```

## Cycle S31-S36 — ligne d'arrivée 🏁

| Session | Livré |
|---|---|
| S31 | Lock file, rate limit panel, cache seed fix |
| S32 | Header redesign (Icon+Sparkline+Tooltip) |
| S33 | Errors & Failures panel |
| S34 | Cache seed cache-hit, Tool Perf, Cache, System Trends |
| S35 | Cold start Couche 1 (4 tools), Sessions, Confirmations, Network I/O |
| S36 | Transport audit, Couche 1 (6 typed tools), v0.10.0 push |

**Prochain milestone : RPI vuln lab — test end-to-end sur cibles réelles contrôlées.**
