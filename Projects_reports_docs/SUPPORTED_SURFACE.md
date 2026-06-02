# HexStrike AI-PULSE — Supported Surface (Stabilization Window)

Date: 2026-05-01
Branch: feature/attack-intelligence
HEAD: cae0d9a
Status: Phase 1 CLOSED

---

## Entrypoint

Active entrypoint: `hexstrike_server.py`
MCP runtime setup: `mcp_core/server_setup.py::setup_mcp_server_standalone()`
Transport: FastMCP HTTP (`mcp.run(transport="http")`)

`setup_mcp_server()` (Flask-era) is present but NOT part of the stabilization surface.

---

## Supported MCP Tools

### Generic execution
- `run_security_tool(tool_name, parameters)` — routes to any DIRECT_TOOLS entry

### Typed wrappers
Generated from DIRECT_TOOLS × tool_registry. 106 direct routes. All typed wrappers
call `run_security_tool()` internally — same execution path, same normalization.

### Named tools
- `get_tool_skill(tool_name)` — returns skill bundle for a tool
- `plan_attack(target, objective)` — IntelligentDecisionEngine attack chain

---

## Supported Direct Routes (106 total)

wifi: airmon_ng, airodump_ng, aireplay_ng, aircrack_ng, hcxdumptool, wifite, wifite2
recon: amass, subfinder, autorecon, theharvester, dnsenum, fierce, whois
net_scan: nmap, nmap_advanced, masscan, rustscan, arp_scan
web_scan: nikto, sqlmap, wpscan, dalfox, jaeles, xsser, zap
web_fuzz: gobuster, ffuf, feroxbuster, dirsearch, dirb, wfuzz, dotdotpwn
password: hydra, hashcat, john, medusa, patator, hashid, ophcrack
smb_enum: enum4linux, netexec, rpcclient, smbmap, nbtscan
exploit: metasploit, msfvenom, searchsploit, exploit_db
web_recon: katana, hakrawler, gau, waybackurls, httpx, wafw00f, arjun, paramspider, x8
security: prowler, trivy, kube_hunter, kube_bench, checkov, terrascan
misc: ropgadget, ropper, one_gadget, volatility, volatility3, gdb, radare2, strings,
      objdump, checksec, binwalk, ghidra, angr, xxd, mysql, sqlite, exiftool, foremost,
      steghide, hashpump, anew, uro, nuclei, responder
osint: sherlock, spiderfoot, sublist3r, parsero
web_probe: testssl, whatweb, commix, joomscan
vuln_intel: vulnx
active_directory: impacket, ldapdomaindump, adidnsdump, certipy, certipy_ad, mitm6,
                  pywerview, bloodhound, bloodhound_python

---

## Supported MCP Resources

| URI | Description |
|---|---|
| `health://server` | Server uptime, tool count, cache stats |
| `scan://{target}/latest` | Most recent scan result for a target |
| `scan://{target}/{tool_name}` | Cached result for a specific tool+target |
| `scan://cache/list` | All cached scan entries with timestamps |
| `metrics://tools` | Operational metrics — success rates, errors, timeouts, cache |

---

## Supported Workflow Prompts

| Prompt | Entry condition |
|---|---|
| `bug_bounty_recon` | WAF/CMS detected or explicit request |
| `wifi_attack_chain` | Wireless interface available |
| `ctf_web_challenge` | CTF web target |
| `smb_lateral_movement` | SMB/LDAP services detected |
| `cloud_security_audit` | Cloud provider detected |

---

## Supported Dashboard

FastMCP custom routes in `hexstrike_server.py`:
- `/dashboard`
- `/health`
- `/ping`
- `/web-dashboard`
- `/web-dashboard/stream`
- static file catch-all (`server_static/` when present)

---

## Destructive Tools — Confirmation Required

| Tool | Exception (no confirmation needed) |
|---|---|
| `aireplay_ng` | attack_mode=9 (injection test only) |
| `mdk4` | none |
| `responder` | analyze=True (passive mode) |
| `metasploit` | auxiliary/scanner/* and auxiliary/gather/* |
| `mitm6` | none |

---

## NOT Supported During Stabilization

The following are present in the codebase but explicitly out of scope for the
stabilization quality gate:

| Component | Status |
|---|---|
| `setup_mcp_server()` (Flask-era) | Legacy — not in active path |
| `server_api/` Flask blueprints | Legacy — not in active path |
| `mcp_core/hexstrike_client.py` | Legacy — references Flask server |
| Flask endpoint tests (`test_endpoints_exist.py`) | Excluded in `pytest.ini` |
| React/Vite dashboard | Long-term deferred |
| MCP Apps integration | Deferred |

These components are not deleted. They are not stabilized, not tested in the default
suite, and not part of the stable candidate criteria.

---

## Normalization Pipeline (runtime contract)

```
EnhancedCommandExecutor.execute()
    {stdout, stderr, return_code, success, timed_out, partial_results, execution_time, timestamp}
        ↓
_normalize_result()  [server_core/command_executor.py]
    canonical + legacy aliases
        ↓
*_direct.py module
        ↓
_normalize_tool_result()  [mcp_core/server_setup.py]
    guaranteed: success, output, error, returncode, timed_out, partial_results,
                execution_time, timestamp
        ↓
run_security_tool() → caller
```

`_normalize_result()` and `_normalize_tool_result()` serve different layers.
Do NOT merge them.

---

## Phase 1 Exit Criteria — Status

| Criterion | Status |
|---|---|
| Active entrypoint defined | ✅ `hexstrike_server.py` |
| Supported MCP tool exposure defined | ✅ 106 routes + typed wrappers |
| Supported workflow prompts defined | ✅ 5 prompts |
| Supported dashboard scope defined | ✅ FastMCP custom routes |
| Compatibility surface explicitly declared | ✅ Listed above as NOT supported |
| No ambiguous ownership | ✅ FastMCP standalone is the only active path |
| Architectural churn frozen | ✅ No new rewrites until stable candidate |

**Phase 1: CLOSED.**

---

## Phases 2 / 3 / 4 — Completion Status

### Phase 2 — Runtime Stabilization: CLOSED

Core principle: active execution path predictable and consistent.

| Item | Status |
|---|---|
| Result shape normalized (`_normalize_result` + `_normalize_tool_result`) | ✅ |
| `execute_command` cache API fixed (`_cache_get/_cache_set`) | ✅ |
| Timeout produces `success=False`, `timed_out=True`, `error` message | ✅ |
| `plan_attack` `chain` variable defined on all paths | ✅ |
| `TargetProfile.from_dict()` implemented | ✅ |
| `finalize()` closure — metrics + telemetry on ALL exit paths | ✅ |
| Typed wrappers and generic call share execution path | ✅ |
| Destructive confirmations enforced with documented exceptions | ✅ |
| `parameters` accepts `str \| Dict` — typed wrappers pass dicts | ✅ |

Exit criteria met:
- No known P0/P1 on active path ✅
- Supported tools callable without unexpected divergence ✅
- Destructive-tool behavior consistent with policy ✅

### Phase 3 — Workflow and Intelligence Hardening: CLOSED

Core principle: prompts and intelligence helpers are safe and coherent.

| Item | Status |
|---|---|
| All 26 workflow prompt tools verified in DIRECT_TOOLS | ✅ |
| Skill guidance maps to correct tool routing | ✅ |
| Cache-derived tech detection validated (`_detect_from_cache`) | ✅ |
| Session state restore validated for tech profile and rate-limit profile | ✅ |
| `plan_attack` session restore via `TargetProfile.from_dict()` | ✅ |
| `IntelligentDecisionEngine` advisory only — no silent execution changes | ✅ |

Exit criteria met:
- No critical workflow breakages on supported flows ✅
- Intelligence helpers do not create silent or unsafe execution changes ✅

### Phase 4 — Observability and Monitoring: CLOSED

Core principle: failures diagnosable without reading raw code.

| Telemetry field | Status |
|---|---|
| tool | ✅ |
| target (normalized) | ✅ |
| success | ✅ |
| duration | ✅ |
| timed_out | ✅ |
| cache_hit | ✅ |
| session_state | ✅ |
| confirmation (accepted/denied/skipped) | ✅ |
| opt_profile | ✅ |
| skill_injected | ✅ |
| prompt_suggested | ✅ |

| Operational view | Implementation |
|---|---|
| error count by tool | `OperationalMetricsStore.error_count_by_tool()` |
| timeout count by tool | `OperationalMetricsStore.timeout_count_by_tool()` |
| success rate by tool | `OperationalMetricsStore.success_rate_by_tool()` |
| slowest tools | `OperationalMetricsStore.slowest_tools()` |
| cache hit/miss ratio | `OperationalMetricsStore.cache_summary()` |
| confirmation events | `OperationalMetricsStore.confirmation_summary()` |
| prompt suggestion count | `OperationalMetricsStore._prompt_suggestions` |
| MCP operational view | `metrics://tools` resource |

Exit criteria met:
- Critical failures visible without deep manual digging ✅
- Operational debugging no longer requires reading raw code ✅

---

## Phase 5 Entry Conditions

All three prior phases closed above. Phase 5 (Regression Validation) may proceed.

Phase 5 scope:
- typed MCP tools
- `run_security_tool()`
- destructive confirmation paths
- cache write and cache read (including cache-hit short-circuit)
- session state restore paths
- prompt suggestion paths
- representative workflows
- `_normalize_tool_result()` coverage (currently untested)
- typed-wrapper vs generic-call equivalence test (currently missing)

Stable candidate requires:
- zero open P0
- zero open P1
- agreed handling for remaining P2 items
- 1323+ tests passing, 0 unexpected failures
