# FastMCP Community — Questions techniques ouvertes
**Date :** 2026-05-16  
**Auteur :** Claude (Sonnet 4.6)  
**Pour :** HexReaper + HexDevMaster  
**Sujet :** Questions FastMCP non résolues après session live S30

---

## 1. Ce que la session live S30 a déjà résolu

La session live du 2026-05-16 a répondu à plusieurs questions par la pratique :

| Question | Statut |
|---|---|
| Prefab UI fiable dans Claude Desktop ? | ✅ **Contourné** — Claude-side HTML rendering validé en live |
| structuredContent cause 5-15s delay ? | ✅ **Expliqué** — c'était la double instance MCP, pas structuredContent |
| Pattern pour large dataset ? | ✅ **Notre réponse** — 8 tools légers (<105 tokens each) |
| ToolResult dual output ? | ⏳ Toujours ouvert |
| GenerativeUI() expérimental ? | ⏳ Toujours ouvert |

---

## 2. Questions techniques encore ouvertes

**Q1 — Cache seed key format**  
On seed `_scan_cache` au boot avec `seed:tool:target:timestamp` mais les appels live génèrent `session_id:tool:target:hash`. Aucun match possible — les 4 entrées seedées ne sont jamais consommées. À investiguer dans `_cache_key_for()` et `_seed_scan_cache()`.

**Q2 — BM25SearchTransform avec always_visible**  
On utilise `always_visible=["nmap", "whatweb", "sqlmap", "pulse_dashboard"]`. Question : est-ce que ces 4 tools consomment des tokens dans le context window de Claude même quand ils ne sont pas appelés ?

**Q3 — Double instance stdio**  
Claude Desktop respawn le process MCP automatiquement. Si le process est déjà lancé en terminal, deux instances se bloquent mutuellement sur stdio. Lock file suggéré comme fix — à confirmer comme bonne pratique.

---

## 3. Ce qui reste à faire côté code

Par ordre de priorité :

1. **Lock file anti double-instance** — 15min, bloquant pour tout le reste
2. **Fix cache seed key format** — les clés `seed:*` ne matchent pas les clés live
3. **Panel RateLimitDetector** — priorité S31, valeur red team immédiate

---

## 4. Contexte projet pour référence

Stack validée live S30 :
- FastMCP 3.2.4, stdio transport, Claude Desktop
- 130 tools directs, 8 tools dashboard légers
- Claude-side HTML rendering depuis données MCP réelles — zéro PrefabApp, zéro iframe, zéro CSP

---

*Claude Sonnet 4.6 — HexStrike team — 2026-05-16*
