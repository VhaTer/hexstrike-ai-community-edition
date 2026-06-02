La tension est réelle et structurelle. Voici les points de friction concrets.

---

## 1. Double chemin vers le même endpoint Flask

Pour un outil comme `httpx`, il existe **deux chemins MCP distincts** qui aboutissent au même `POST /api/tools/httpx` :

```
Typed:   httpx_probe() → safe_post("api/tools/httpx", data)
Gateway: run_tool("httpx", params) → tool_registry.get_tool("httpx")
                                   → safe_post(tool_def["endpoint"], parsed_params)
```

Le `tool_registry.TOOLS["httpx"]["endpoint"]` pointe vers `/api/tools/httpx`. [1](#2-0) 

La route Flask est donc frappée par deux chemins avec des **validations différentes** : la signature Python typée pour le chemin typed, et la boucle manuelle `for pname, spec in tool_def["params"].items()` pour le gateway. [2](#2-1) [3](#2-2) 

---

## 2. Cache mort sur le chemin générique

`command_executor.execute_command()` accepte un paramètre `cache=None` :

```python
def execute_command(command: str, use_cache: bool = True, cache=None):
    if use_cache and cache is not None:   # ← jamais True si cache non passé
        cached_result = cache.get(...)
``` [4](#2-3) 

La route `/api/command` appelle `execute_command(command, use_cache=use_cache)` sans passer l'instance `cache` globale du serveur. Le paramètre `use_cache=True` dans la requête n'a donc **aucun effet** — le cache est silencieusement bypassé sur ce chemin. [5](#2-4) 

---

## 3. Deux taxonomies parallèles non synchronisées

| Fichier | Granularité | Nb catégories |
|---|---|---|
| `tool_registry.py` | 8 catégories larges (`network_recon`, `web_vuln`…) | 8 |
| `mcp_core/tool_profiles.py` | ~20 profils fins (`net_scan`, `web_fuzz`, `smb_enum`…) | 20+ |

Ces deux systèmes coexistent sans référence croisée. Un outil peut être dans `web_recon` dans le registry et dans `web_probe` + `web_crawl` + `web_fuzz` dans les profils. Aucun mécanisme ne garantit la cohérence. [6](#2-5) [7](#2-6) 

---

## 4. TelemetryCollector instancié deux fois

`enhanced_command_executor.py` crée son propre singleton module-level :

```python
telemetry = TelemetryCollector()  # instance A
``` [8](#2-7) 

Le serveur Flask expose `telemetry.get_stats()` via `/health`, mais ce `telemetry` est une **instance différente** (importée ou recréée via `from server_core import *`). Si ce ne sont pas le même objet en mémoire, les stats exposées ne reflètent pas les exécutions réelles. [9](#2-8) 

---

## 5. Validation en triple couche sans contrat partagé

Pour un outil typé, la validation se fait à trois niveaux indépendants :

1. **FastMCP** — type checking sur la signature Python (`httpx_probe(target: str, ...)`)
2. **`run_tool` gateway** — boucle manuelle sur `tool_def["params"]`
3. **Flask route** — validation propre dans chaque handler serveur

Aucun schéma partagé (ex. Pydantic, JSON Schema) ne sert de source de vérité unique. Une divergence entre `tool_registry.TOOLS["httpx"]["params"]` et la signature de `httpx_probe()` passe silencieusement.

---

## Résumé des tensions

```mermaid
graph TD
    "run_tool (gateway)" -- "même endpoint" --> "POST /api/tools/httpx"
    "httpx_probe (typed)" -- "même endpoint" --> "POST /api/tools/httpx"
    "POST /api/command" -- "cache=None → cache mort" --> "EnhancedCommandExecutor"
    "tool_registry (8 cats)" -- "non synchronisé" --> "tool_profiles (20+ profils)"
    "TelemetryCollector A (executor)" -- "instance séparée ?" --> "TelemetryCollector B (server /health)"
```

La tension principale est que le système a **deux modes d'accès aux outils** (typed vs gateway) qui auraient dû être mutuellement exclusifs mais sont en réalité superposés, avec des contrats de validation différents et un état partagé (cache, telemetry) dont la cohérence n'est pas garantie.

### Citations

**File:** tool_registry.py (L155-163)
```python
    "httpx": {
        "desc": "HTTP probing and tech detection",
        "endpoint": "/api/tools/httpx",
        "method": "POST",
        "category": "web_recon",
        "params": {"target": {"required": True}},
        "optional": {"probe": True, "tech_detect": False, "status_code": False, "title": False, "additional_args": ""},
        "effectiveness": 0.85,
    },
```

**File:** tool_registry.py (L524-533)
```python
CATEGORIES = {
    "network_recon": "Network scanning, port discovery, service enumeration",
    "web_recon": "Web directory brute-forcing, crawling, tech detection",
    "web_vuln": "Web vulnerability scanning (SQLi, XSS, template-based)",
    "exploitation": "Exploit execution, payload generation, module search",
    "brute_force": "Password cracking and login brute-forcing",
    "osint": "Subdomain enumeration, DNS recon, URL harvesting",
    "binary": "Binary analysis, reverse engineering, ROP gadgets",
    "cloud": "Cloud security auditing (AWS, containers, Kubernetes)",
}
```

**File:** mcp_tools/gateway.py (L47-55)
```python
        # Validate required params
        for pname, spec in tool_def["params"].items():
            if spec.get("required") and pname not in parsed_params:
                return {"error": f"Missing required param: {pname}", "success": False}

        # Fill defaults for optional params
        for k, v in tool_def.get("optional", {}).items():
            if k not in parsed_params:
                parsed_params[k] = v
```

**File:** mcp_tools/web_probe/httpx.py (L5-10)
```python
def register_httpx_tool(mcp, hexstrike_client, logger):
    @mcp.tool()
    def httpx_probe(target: str, probe: bool = True, tech_detect: bool = False,
                   status_code: bool = False, content_length: bool = False,
                   title: bool = False, web_server: bool = False, threads: int = 50,
                   additional_args: str = "") -> Dict[str, Any]:
```

**File:** server_core/command_executor.py (L7-22)
```python
def execute_command(command: str, use_cache: bool = True, cache=None) -> Dict[str, Any]:
    """
    Execute a shell command with enhanced features

    Args:
        command: The command to execute
        use_cache: Whether to use caching for this command
        cache: Optional cache instance

    Returns:
        A dictionary containing the stdout, stderr, return code, and metadata
    """
    if use_cache and cache is not None:
        cached_result = cache.get(command, {})
        if cached_result:
            return cached_result
```

**File:** hexstrike_server.py (L690-693)
```python
        "cache_stats": cache.get_stats(),
        "telemetry": telemetry.get_stats(),
        "uptime": time.time() - telemetry.stats["start_time"]
    })
```

**File:** hexstrike_server.py (L703-724)
```python
@app.route("/api/command", methods=["POST"])
def generic_command():
    """Execute any command provided in the request with enhanced logging"""
    try:
        params = request.json
        command = params.get("command", "")
        use_cache = params.get("use_cache", True)

        if not command:
            logger.warning("⚠️  Command endpoint called without command parameter")
            return jsonify({
                "error": "Command parameter is required"
            }), 400

        result = execute_command(command, use_cache=use_cache)
        return jsonify(result)
    except Exception as e:
        logger.error(f"💥 Error in command endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": f"Server error: {str(e)}"
        }), 500
```

**File:** mcp_core/tool_profiles.py (L350-377)
```python
DEFAULT_PROFILE = [
    "credential_harvest",
    "memory_forensics",
    "net_scan",
    "net_lookup",
    "dns_enum",
    "smb_enum",
    "recon",
    "web_probe",
    "web_crawl",
    "web_fuzz",
    "web_scan",
    "vuln_scan",
    "exploit_framework",
    "password_cracking",
    "param_discovery",
    "url_recon",
    "data_processing",
    "error_handling",

    # System tools"
    "monitoring",
    "process_management",
    "visual"
]

# Full profile includes all available tool categories
FULL_PROFILE = list(TOOL_PROFILES.keys())
```

**File:** server_core/enhanced_command_executor.py (L12-14)
```python
# Global telemetry collector
from server_core.telemetry_collector import TelemetryCollector
telemetry = TelemetryCollector()
```
