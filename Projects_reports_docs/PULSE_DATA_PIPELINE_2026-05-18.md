# Pulse — Data Pipeline (v0.10.1)

```mermaid
flowchart TB
    subgraph EXEC["⚡ Execution Pipeline"]
        direction TB
        BIN["nmap / whatweb / nuclei / nikto / ..."]
        ECE["EnhancedCommandExecutor\nsubprocess.Popen → stdout/stderr threads"]
        NORM["command_executor._normalize_result()\n{success, output, error, returncode, timed_out, execution_time}"]
        EXEC_FUNC["*_direct.py exec_func()\nnet_scan_exec / web_probe_exec / ..."]
        RST["run_security_tool()"]
        PARAM["ParameterOptimizer\nstealth/normal/aggressive"]
        CONFIRM["_build_destructive_confirmation()"]
        CACHE_CHECK["_scan_cache.get(key)\n→ seed fallback if miss"]
        CACHE_WRITE["_scan_cache.set(key, {tool, target, result, ts})"]
        METRICS["_op_metrics.record()\nToolStatsStore.record()"]
        ERR["IntelligentErrorHandler\n(on failure)"]
        RL["RateLimitDetector"]
        FINAL["finalize() → dict"]

        BIN --> ECE --> NORM --> EXEC_FUNC
        EXEC_FUNC --> RST
        RST --> PARAM --> CONFIRM --> CACHE_CHECK
        CACHE_CHECK -->|miss| EXEC_FUNC
        CACHE_CHECK -->|hit| RL
        EXEC_FUNC --> CACHE_WRITE --> METRICS --> RL
        EXEC_FUNC -.->|on error| ERR -.-> FINAL
        RL --> FINAL
    end

    subgraph DASHBOARD["📊 Dashboard Pipeline"]
        direction TB
        COLLECT["_collect_dashboard_state(target=None)"]
        OVERVIEW["get_overview()\n_op_metrics.summary()\nresource_monitor"]
        SCOPE["get_scope()\n_scan_cache.values()\n→ active target"]
        SURFACE["get_surface(target)\n_cache_for_target()\nnmap: port/service parsing\nwhatweb: tech keywords"]
        FINDINGS["get_findings(target)\n_cache_for_target()\nnuclei: regex [severity] [id]\nnikto: prefix '+ /:'"]
        PLAN["get_plan(target)\nIntelligentDecisionEngine\nattack chain"]
        TARGET_RECORD["TargetStore.record_scan()\nsurface_data + findings + tools_used"]
        ACTIVE["get_active_tools()\nEnhancedProcessManager"]
        HISTORY["get_history(target)\n_scan_cache filtered by target"]
        ERRORS["get_errors_and_failures()\n_op_metrics.error_count_by_tool()\nerror_handler.get_error_statistics()"]
        CACHE["get_cache_status()\n_op_metrics.cache_summary()"]
        PERF["get_tool_performance()\n_op_metrics.success_rate_by_tool()"]
        RLDASH["get_rate_limit_status()\n_rate_limit_events"]
        TRENDS["get_system_trends()\nresource_monitor"]
        ASYNC["_async_scans iteration\nrunning / complete lists"]
        INTEL["get_tool_intelligence()\nToolStatsStore"]

        COLLECT --> OVERVIEW --> SCOPE --> SURFACE --> FINDINGS --> PLAN
        FINDINGS --> TARGET_RECORD
        COLLECT --> ACTIVE
        COLLECT --> HISTORY
        COLLECT --> ERRORS
        COLLECT --> CACHE
        COLLECT --> PERF
        COLLECT --> RLDASH
        COLLECT --> TRENDS
        COLLECT --> ASYNC
        COLLECT --> INTEL
    end

    subgraph STORE["💾 TargetStore (server_core/target_store.py)"]
        direction TB
        TS_JSON["<data_dir>/target_store.json"]
        TS_INMEM["_store: dict[str, dict]\nthreading.Lock"]
        TS_MERGE_SURFACE["_merge_surface()\ndedup ports by port#\ndedup techs/services by string"]
        TS_MERGE_FINDINGS["_merge_findings()\ndedup vulns by id"]
        TS_SAVE["_save_locked()\n.tmp → os.replace()\natomic write"]
        TS_GET["get_target(target)\ndeep copy + inject 'target' key"]
        TS_LIST["get_all_targets()\nsorted summary list"]
        TS_ADD["add_finding(target, type, data, tool)\nport / service / technology / vulnerability"]

        TARGET_RECORD --> TS_MERGE_SURFACE --> TS_MERGE_FINDINGS --> TS_INMEM
        TS_INMEM --> TS_SAVE
        TS_SAVE -.-> TS_JSON
        TS_JSON -.->|_load()| TS_INMEM
        TS_INMEM --> TS_GET
        TS_INMEM --> TS_LIST
    end

    subgraph RESOURCES["🔗 MCP Resources"]
        RES_LIST["targets://\n→ json of get_all_targets()"]
        RES_TARGET["target://{target}\n→ json of get_target()"]
        RES_FINDINGS["target://{target}/findings\n→ findings sub-dict"]
        RES_SESSIONS["target://{target}/sessions\n→ sessions array"]
        RES_HEALTH["health://server"]
        RES_SCAN["scan://{target}/latest\nscan://{target}/{tool_name}\nscan://cache/list"]
        RES_METRICS["metrics://tools\nerrors://statistics"]
    end

    subgraph OUTPUT["🖥️ Output Surface"]
        DIRECTION TB
        LIVE_DASH["get_live_dashboard()\n13 keys: overview, scope, surface,\nfindings, plan, active_tools,\nhistory, rate_limit, tool_health,\ntool_performance, cache_status,\nasync_scans, intelligence"]
        PREFAB["PrefabApp Dashboard\n10 panels: Header, Scope,\nSurface, Findings, Plan IDE,\nHistorique, Rate Limit,\nTool Health, Async Scans,\nIntelligence DataTable"]
        SCAN_EP["scan(target, intensity, objective)\nquick / medium / full\n→ unified entry point"]
        ASYNC_EP["run_async_tool(tool, target, params)\nget_scan_status(scan_id)\n→ background execution"]
        MCP_TOOLS["130+ typed tools (Couche 1)\neach with workflow hints\n→ plug-and-play"]

        FINAL --> CACHE_CHECK
        COLLECT --> LIVE_DASH
        COLLECT --> PREFAB
        TS_GET --> RES_TARGET
        TS_LIST --> RES_LIST
    end

    %% Cross-links
    CACHE_CHECK -.->|cache seed| EXEC_FUNC
    TARGET_RECORD --> TS_INMEM
    FINAL -.->|destructive confirmation| CONFIRM

    %% Styling
    classDef binary fill:#1a1a2e,stroke:#e94560,color:#fff
    classDef process fill:#16213e,stroke:#0f3460,color:#ddd
    classDef cache fill:#0f3460,stroke:#533483,color:#ddd
    classDef store fill:#2d1b69,stroke:#533483,color:#ddd
    classDef dash fill:#1b4332,stroke:#40916c,color:#ddd
    classDef output fill:#3d0c11,stroke:#e94560,color:#ddd
    classDef resource fill:#240046,stroke:#7b2cbf,color:#ddd

    class BIN,ECE,NORM,EXEC_FUNC binary
    class RST,PARAM,CONFIRM,CACHE_CHECK,CACHE_WRITE,METRICS,ERR,RL process
    class SURFACE,FINDINGS,PLAN,TARGET_RECORD,ACTIVE,HISTORY,ERRORS,CACHE,PERF,RLDASH,TRENDS,ASYNC,INTEL cache
    class TS_JSON,TS_INMEM,TS_MERGE_SURFACE,TS_MERGE_FINDINGS,TS_SAVE,TS_GET,TS_LIST,TS_ADD store
    class COLLECT,OVERVIEW,SCOPE dash
    class LIVE_DASH,PREFAB,SCAN_EP,ASYNC_EP,MCP_TOOLS output
    class RES_LIST,RES_TARGET,RES_FINDINGS,RES_SESSIONS,RES_HEALTH,RES_SCAN,RES_METRICS resource
```

## Data Flow Summary

| Step | Component | Input | Output |
|------|-----------|-------|--------|
| 1 | `EnhancedCommandExecutor` | binary name + params | `{stdout, stderr, return_code, success}` |
| 2 | `command_executor._normalize_result()` | raw executor output | `{success, output, error, returncode, execution_time}` |
| 3 | `*_direct.py exec_func()` | binary + normalized params | normalized dict per tool |
| 4 | `run_security_tool()` | tool name + target | finalized dict (cache + metrics + error handling) |
| 5 | `_scan_cache` | `{session}:{tool}:{target}[:hash]` | cached result or miss |
| 6 | `get_surface(target)` | nmap/whatweb stdout in cache | `{ports, technologies, risk_level}` |
| 7 | `get_findings(target)` | nuclei/nikto stdout in cache | `[{severity, finding, details}]` sorted by severity |
| 8 | `TargetStore.record_scan()` | surface + findings + tools | persisted JSON file + in-memory store |
| 9 | `_collect_dashboard_state()` | all above | flat dict (27+ keys) |
| 10 | `get_live_dashboard()` | dashboard state | 13-key JSON response |
| 11 | Prefab UI panels | dashboard state | visual dashboard in Claude Desktop |

## Key Thread Safety

| Structure | Lock | Notes |
|-----------|------|-------|
| `_scan_cache` | `threading.RLock` | LRU + TTL eviction under lock |
| `_op_metrics` | `threading.RLock` | all counters + aggregation |
| `TargetStore._store` | `threading.Lock` | atomic `.tmp` + `os.replace()` |
| `_async_scans` | `threading.Lock` | dict operations |
| `EnhancedCommandExecutor` | fresh per call | no shared state |

## Cache Flow

```
key = _cache_key_for(session, tool, target, params)
    → _scan_cache.get(key)
        → hit: return result directly (skip execution)
        → miss: _scan_cache.get(f"seed:{tool}:{target}")  [seed fallback]
            → hit: return seed result
            → miss: execute binary → _scan_cache.set(key, {tool, target, result, ts})
```

Adaptive TTL: `<10s exec` → 30min, `>10s` → 60min, `>60s` → 90min. Max 500 entries.
