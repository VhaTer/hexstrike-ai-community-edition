# HexStrike AI-PULSE — Full Project Audit

> Generated: 2026-05-08  
> Branch: `feature/attack-intelligence`  
> Upstream: `CommonHuman-Lab/hexstrike-ai-community-edition`  
> Fork: `VhaTer/hexstrike-ai-community-edition`  

---

## Table of Contents

1. [README Inflation — Listed vs Shipped](#1-readme-inflation--listed-vs-shipped)
2. [Dead Code — Unreachable Modules & Functions](#2-dead-code--unreachable-modules--functions)
3. [Missing Community Infrastructure](#3-missing-community-infrastructure)
4. [Branch Health & Divergence](#4-branch-health--divergence)
5. [Dependency Health](#5-dependency-health)
6. [Test Suite Reliability](#6-test-suite-reliability)
7. [Code Quality Observations](#7-code-quality-observations)
8. [Architecture Concerns](#8-architecture-concerns)
9. [Stabilization Roadmap](#9-stabilization-roadmap)
10. [Quick Wins vs Deep Work](#10-quick-wins-vs-deep-work)

---

## 1. README Inflation — Listed vs Shipped

### 1.1 Tools in README but NOT in DIRECT_TOOLS (31 tools)

These tools are documented in the README as available but have no entry in the `DIRECT_TOOLS` routing table in `mcp_core/server_setup.py`. They cannot be invoked via `run_security_tool()`.

| # | Tool | README Section | Notes |
|---|---|---|---|
| 1 | enum4linux-ng | Network Reconnaissance | Handler exists in `smb_enum_direct.py` but not wired |
| 2 | hcxpcapngtool | WiFi | Handler exists in `wifi_direct.py` but not wired |
| 3 | EAPHammer | WiFi | Handler exists in `wifi_direct.py` but not wired |
| 4 | Bettercap | WiFi | Handler exists in `wifi_direct.py` but not wired |
| 5 | mdk4 | WiFi | Handler exists + destructive flag set — not wired |
| 6 | SSLScan | Web Application | No handler exists |
| 7 | SSLyze | Web Application | No handler exists |
| 8 | JWT-Tool | Web Application | `jwt_analyzer` handler exists in `misc_direct.py` but not wired |
| 9 | Burp Suite | Web Application | No handler exists |
| 10 | Evil-WinRM | Password Security | No handler exists |
| 11 | SharpHound | Active Directory | No handler exists |
| 12 | Kerbrute | Active Directory | No handler exists |
| 13 | CrackMapExec | Active Directory | README says CME; code has `netexec` (its successor) |
| 14 | Pwntools | Binary Analysis | Handler exists in `exploit_framework_direct.py` but not wired |
| 15 | Scout Suite | Cloud Security | Handler exists in `security_direct.py` but not wired |
| 16 | Pacu | Cloud Security | Handler exists in `security_direct.py` but not wired |
| 17 | Social-Analyzer | OSINT | No handler exists |
| 18 | Recon-ng | OSINT | No handler exists |
| 19 | Maltego | OSINT | No handler exists |
| 20 | Shodan | OSINT | No handler exists |
| 21 | Censys | OSINT | No handler exists |
| 22 | TruffleHog | OSINT | No handler exists |
| 23 | Aquatone | OSINT | No handler exists |
| 24 | Subjack | OSINT | No handler exists |
| 25 | Stegsolve | CTF & Forensics | No handler exists |
| 26 | Zsteg | CTF & Forensics | No handler exists |
| 27 | Scalpel | CTF & Forensics | No handler exists |
| 28 | PhotoRec | CTF & Forensics | No handler exists |
| 29 | Autopsy | CTF & Forensics | Handler exists in `misc_direct.py` but not wired |
| 30 | Sleuth Kit | CTF & Forensics | No handler exists |
| 31 | CyberChef | CTF & Forensics | No handler exists |

**Critical subset: 8 tools that have handler code but are not wired**

These are the easiest to fix — the implementation exists, just needs a `DIRECT_TOOLS` entry:

| Tool | Module | PR Work |
|---|---|---|
| `hcxpcapngtool` | `wifi_direct.py` | +1 line in DIRECT_TOOLS |
| `eaphammer` | `wifi_direct.py` | +1 line |
| `bettercap_wifi` | `wifi_direct.py` | +1 line |
| `mdk4` | `wifi_direct.py` | +1 line (also a destructive tool — broken promise) |
| `enum4linux-ng` | `smb_enum_direct.py` | +1 line |
| `pwntools` | `exploit_framework_direct.py` | +1 line |
| `scout_suite` / `pacu` | `security_direct.py` | +1 line each |
| `autopsy` | `misc_direct.py` | +1 line |
| `jwt_analyzer` | `misc_direct.py` | +1 line (listed as JWT-Tool in README) |

### 1.2 Tools in DIRECT_TOOLS but NOT in README (34 tools)

These tools are fully functional but have zero documentation in the README:

| Category | Tools |
|---|---|
| **Web Recon** (9) | `gau`, `waybackurls`, `arjun`, `paramspider`, `x8`, `jaeles`, `xsser`, `joomscan` |
| **Web Fuzz** (3) | `dirb`, `wfuzz`, `dotdotpwn` |
| **AD** (4) | `adidnsdump`, `certipy`, `certipy_ad`, `pywerview`, `bloodhound_python` |
| **Exploit** (3) | `msfvenom`, `searchsploit`, `exploit_db` |
| **Password** (1) | `ophcrack` |
| **Network** (1) | `nmap_advanced` |
| **OSINT** (2) | `sublist3r`, `parsero` |
| **Binary/Misc** (11) | `one_gadget`, `strings`, `objdump`, `angr`, `xxd`, `mysql`, `sqlite`, `hashpump`, `anew`, `uro` |
| **Vuln Intel** (1) | `vulnx` |
| **Web Probe** (1) | `commix` |

### 1.3 README Tool Count Claim

The README says **"150+ security tools"** but `DIRECT_TOOLS` has **106 entries**. After subtracting the 31 that don't work and adding the 34 that aren't documented, the real shipped count is 106. The "150+" figure includes:
- Tools listed in the aspirational section at the bottom of `requirements.txt` (commented out)
- Tools from the external tools list that have no code integration
- Duplicate counting of suites as individual tools

**Fix:** Change "150+" to "106" in the README tagline and feature list.

---

## 2. Dead Code — Unreachable Modules & Functions

### 2.1 `mcp_tools/` Package — 193 files, ~11,190 LOC

This is the largest dead code block. The entire `mcp_tools/` package tree contains Phase 2 Flask-era tool wrapper modules. Each defines `register_*_tools(mcp, client, logger)` functions that are:

- Imported transitively through `mcp_core/tool_profiles.py` → `from mcp_tools import *`
- **Never called** — `setup_mcp_server_standalone()` never references them
- Kept alive by 4 dead imports in `server_setup.py` (lines 16-21): `TOOL_PROFILES`, `DEFAULT_PROFILE`, `FULL_PROFILE`, `resolve_profile_dependencies`

**Breakdown by sub-package:**

| Package | Files | Status |
|---|---|---|
| `mcp_tools/wifi_pentest/` | 12 | All dead |
| `mcp_tools/active_directory/` | 7 | All dead |
| `mcp_tools/web_scan/` | 9 | All dead |
| `mcp_tools/web_fuzz/` | 7 | All dead |
| `mcp_tools/web_crawl/` | 2 | All dead |
| `mcp_tools/ops/` | 8 | All dead (including system_monitoring.py) |
| `mcp_tools/osint/` | 4 | All dead |
| `mcp_tools/exploit_framework/` | 6 | All dead |
| `mcp_tools/net_scan/` | 4 | All dead |
| `mcp_tools/recon/` | 4 | All dead |
| `mcp_tools/password_cracking/` | 8 | All dead |
| `mcp_tools/binary_analysis/` | 9 | All dead |
| `mcp_tools/binary_debug/` | 2 | All dead |
| `mcp_tools/cloud_*/` | 6 | All dead |
| `mcp_tools/container_scan/` | 3 | All dead |
| `mcp_tools/k8s_scan/` | 2 | All dead |
| `mcp_tools/iac_scan/` | 2 | All dead |
| `mcp_tools/smb_enum/` | 5 | All dead |
| `mcp_tools/api_*/` | 5 | All dead |
| `mcp_tools/memory_forensics/` | 2 | All dead |
| `mcp_tools/gadget_search/` | 3 | All dead |
| `mcp_tools/vuln_*/` | 2 | All dead |
| `mcp_tools/web_probe/` | 1 | Dead |
| `mcp_tools/waf_detect/` | 1 | Dead |
| `mcp_tools/dns_enum/` | 2 | Dead |
| `mcp_tools/url_recon/` | 2 | Dead |
| `mcp_tools/param_*/` | 4 | Dead |
| `mcp_tools/data_processing/` | 1 | Dead |
| `mcp_tools/url_filter/` | 1 | Dead |
| `mcp_tools/db_query/` | 3 | Dead |
| `mcp_tools/recon_bot/` | 1 | Dead |
| `mcp_tools/credential_harvest/` | 1 | Dead |
| `mcp_tools/crypto_attack/` | 1 | Dead |
| `mcp_tools/file_carving/` | 1 | Dead |
| `mcp_tools/stego_analysis/` | 1 | Dead |
| `mcp_tools/metadata_extract/` | 1 | Dead |
| `mcp_tools/net_lookup/` | 1 | Dead |
| `mcp_tools/error_handling/` | 2 | Dead |
| `mcp_tools/runtime_monitor/` | 1 | Dead |
| `mcp_tools/ai_assist/` | 1 | Dead |
| `mcp_tools/ai_payload/` | 1 | Dead |
| `mcp_tools/bugbounty_workflow/` | 1 | Dead |
| `mcp_tools/fingerprint/` | 1 | Dead |
| `mcp_tools/gateway.py` | 1 | Self-documented no-op shim |
| `mcp_tools/__init__.py` | 1 | Dead import hub |
| **Total** | **193** | **Entire tree dead** |

**Recommendation:** Either:
- Option A: Delete the tree and the dead imports. Cleanest, saves 11K LOC.
- Option B: Archive it to `archive/mcp_tools/` with a README explaining it's Phase 2 preserved for reference.

### 2.2 `register_system_monitoring_tools` — Orphaned Function

Defined in `mcp_tools/ops/system_monitoring.py` (line 6), referenced in `tool_profiles.py` (line 381) but **never called** from any production path. The `hexstrike_server.py` has its own inline health-check. This function is unreachable.

**Fix:** Delete the function or replace with a docstring pointing to the inline implementation.

### 2.3 24 Unwired Handler Functions in `*_direct.py` Modules

These handlers are implemented in the `_HANDLERS` dicts of `*_direct.py` modules but have no corresponding entry in `DIRECT_TOOLS`:

`wifi_direct.py` (4 unwired):
- `airbase_ng`, `airdecap_ng`, `hcxpcapngtool`, `eaphammer`, `bettercap_wifi`, `mdk4`

`smb_enum_direct.py` (1):
- `enum4linux-ng`

`exploit_framework_direct.py` (2):
- `pwntools`, `pwninit`

`security_direct.py` (4):
- `scout_suite`, `pacu`, `cloudmapper`, `docker_bench`

`misc_direct.py` (7+):
- `libc`, `autopsy`, `postgresql`, `api_schema_analyzer`, `graphql_scanner`, `jwt_analyzer`, `falco`, `qsreplace`, `api_fuzzer`, `bbot`

**Impact:** ~24 functions with working code that can never be reached.

**Fix:** Wire the important ones to DIRECT_TOOLS or delete the handlers.

### 2.4 Dead `mcp_core/` Modules

- `mcp_core/hexstrike_client.py` (88 lines) — `HexStrikeClient` class never instantiated in production. Makes HTTP calls to a Flask server that no longer exists.
- `mcp_core/args.py` (17 lines) — Defines 6 CLI args but only `--debug` is consumed. `--server`, `--timeout`, `--compact`, `--profile`, `--auth-token`, `--disable-ssl-verify` are dead.

### 2.5 Dead Test Files — 14 files, ~4,921 LOC

These test files test Phase 2 modules or dead code paths:

| Test File | LOC | Tests What |
|---|---|---|
| `test_active_directory_mcp_tools.py` | 528 | Phase 2 AD wrappers |
| `test_osint_mcp_tools.py` | 276 | Phase 2 OSINT wrappers |
| `test_wifi_mcp_tools.py` | 442 | Phase 2 WiFi wrappers |
| `test_gateway_expanded.py` | 433 | Phase 2 gateway shim |
| `test_gateway_phase4a.py` | 644 | Phase 2 routing (via gateway) |
| `test_gateway_template.py` | 340 | Phase 2 template |
| `test_wordlist_ops.py` | 493 | Dead ops module |
| `test_vulnerability_intelligence.py` | 435 | Dead ops module |
| `test_bug_bounty_recon.py` | 291 | Dead workflow module |
| `test_ai_payload_generation.py` | 391 | Dead ai_payload module |
| `test_ai_decision_engine.py` | 21 | Dead ai_assist module |
| `test_hexstrike_client_template.py` | 383 | Dead client class |
| `test_mcp_entry_template.py` | 184 | Dead entry path |
| `test_args_template.py` | 460 | Dead args |

**Recommendation:** Move these to `tests/archive/` rather than deleting — they may contain useful test patterns or edge cases for when those tools get rewired.

---

## 3. Missing Community Infrastructure

### 3.1 Present ✅

| Item | Location | Quality |
|---|---|---|
| README | `README.md` | Good — 283 lines, covers quick start, features, config |
| LICENSE | `LICENSE` | AGPLv3, 633 lines |
| CONTRIBUTING.md | `.github/CONTRIBUTING.md` | Good — AI-assistance policy, PR size limits |
| PR Template | `.github/pull_request_template.md` | Good — 52 lines, structured |
| .gitignore | `.gitignore` | Good — 47 lines, covers common patterns |
| .editorconfig | `.editorconfig` | Basic |
| Tests | `tests/` (48 files) | Large suite, 1091 tests |

### 3.2 Missing ❌

| Item | Severity | Why It Matters |
|---|---|---|
| **SECURITY.md** | **Critical** | A pentesting platform has no vulnerability disclosure policy. Users have no way to report security bugs responsibly. |
| **CI/CD workflow** | **High** | No automated tests on push/PR. Regressions go undetected. No lint gates. No coverage checks. |
| **CODE_OF_CONDUCT.md** | **Medium** | CONTRIBUTING.md mentions conduct but links no document. Standard Contributor Covenant expected. |
| **Issue templates** | **Medium** | No `.github/ISSUE_TEMPLATE/`. Bug reports and feature requests have no structure. |
| **Lint config** | **Medium** | `.ruff_cache/` exists but no `.ruff.toml`. No style enforcement anywhere. |
| **pyproject.toml** | **Low** | Uses `requirements.txt` + `pytest.ini` + `.coveragerc` instead of one modern PEP 621 file. |
| **CHANGELOG.md** | **Low** | No release history. Users can't see what changed between versions. |
| **Docker support** | **Low** | No Dockerfile or docker-compose for containerized deployment. |

---

## 4. Branch Health & Divergence

### 4.1 Current State

| Metric | Value |
|---|---|
| Active branch | `feature/attack-intelligence` |
| Upstream | `CommonHuman-Lab/hexstrike-ai-community-edition` |
| Origin | `VhaTer/hexstrike-ai-community-edition` |
| Ahead of upstream | 146 commits |
| Behind upstream | 118 commits |
| Recent commits | CTF engine, CVE intelligence, FastMCP 3.2.4 upgrade, Flask removal |
| Fork workflow | PRs to upstream `beta/x.x.x` branches |

### 4.2 Risks

1. **146 ahead + 118 behind = 264 commit drift.** This is a wide divergence. Merging or rebasing will have significant conflict surface. The feature branch has substantive engine work (CTF, CVE, bug bounty) that upstream doesn't have, and upstream has 118 commits that this branch lacks.

2. **No `beta/x.x.x` target branch visible.** The AGENTS.md says PRs target `beta/x.x.x` but no such branch exists locally and none appeared in `git branch -a`. Actual upstream target for PRs is unclear.

3. **Last upstream sync unknown.** No merge commit from upstream in the recent log. The drift may contain duplicate or conflicting work.

### 4.3 Recommendations

1. **Rebase onto upstream/master** to resolve divergence early, before the gap widens further.
2. **Identify the correct PR target branch** — clarify with upstream which `beta/` branch is active.
3. **Consider smaller, focused PRs** rather than one massive 264-commit merge.
4. **Add a CI workflow** that runs `pytest tests/ -q` to prevent regressions during the merge process.

---

## 5. Dependency Health

### 5.1 By the Numbers

| Metric | Value |
|---|---|
| Installed packages | 293 |
| Requirements.txt entries | 22 pinned + 3 commented out |
| Pinned with hard `==` | 1 (`bcrypt==4.0.1`) |
| Pinned with `>=` or `>=,<` | 21 |
| Commented out | 3 (`kube-hunter`, `ScoutSuite`, `shodan`) |

### 5.2 Known Issues

| Package | Issue |
|---|---|
| `fastmcp` | Requires `>=3.1.1`, but server code references `3.2.4` (installed version is 3.2.4). The minimum should be bumped to match actual behavior. |
| `chardet` | Pinned `<6.0.0` because `dirsearch` pulls 7.x which breaks `requests`. This is a fragile workaround. |
| `bcrypt==4.0.1` | Pinned for `passlib` compatibility with `pwntools`. Prevents security updates. |
| `psycopg2` | Commented out with note "not a happy camper" — suggests PostgreSQL integration is aspirational. |

### 5.3 Recommendations

1. Bump `fastmcp>=3.2.4` to match actual deployed version and avoid regression from older APIs.
2. Evaluate whether `bcrypt` pin is still needed — test with `bcrypt>=4.1.0`.
3. Document the `chardet` constraint more clearly with a version range that's been tested.
4. Remove commented-out dependencies or move them to a separate `requirements-extras.txt`.

---

## 6. Test Suite Reliability

### 6.1 Current State

| Metric | Value (from AGENTS.md) |
|---|---|
| Total tests | 1,091 |
| Pass rate | 99.2% |
| Coverage | 23% |
| Flaky tests | 0 |
| Ignored by default | 2 (`test_endpoints_exist.py`, `test_fastmcp_real_world.py`) |

**Note:** During this audit, the full test suite (`pytest tests/ -q`) timed out after 60 seconds — significantly longer than the ~13s advertised in AGENTS.md. This needs investigation.

### 6.2 Test Distribution

| Category | Files | Health |
|---|---|---|
| Active (Phase 3 direct) | ~30 files | Core of the suite |
| Dead (Phase 2 wrappers) | 14 files (see 2.5) | Wasteful — running dead code tests |
| Integration | 1 (`test_fastmcp_real_world.py`) | Ignored by default |
| Legacy Flask | 1 (`test_endpoints_exist.py`) | Ignored by default |

### 6.3 Recommendations

1. **Investigate the timeout.** Run `pytest tests/ -q --durations=10` to find slow tests. The 13s claim vs 60s+ reality suggests a regression or a hanging test.
2. **Move dead test files** to `tests/archive/` to clean up the suite and reduce noise.
3. **Increase coverage target to 40%** as a first milestone. The 23% is low even for a security tool project.
4. **Add `pytest-timeout`** to prevent any single test from hanging the suite.

---

## 7. Code Quality Observations

### 7.1 What's Good

- **Consistent error handling** in `*_direct.py` modules — all return `{"success": bool, "output": str, "error": str, ...}` dicts.
- **Clean FastMCP integration** — typed tool wrappers, resource URIs, prompt templates all follow FastMCP conventions.
- **Session isolation** — scan cache and tech profiles are per-session, preventing cross-session data leaks.
- **No Flask remnants** — Phase 3 cleanup was thorough; zero Flask imports remain.
- **Good use of type hints** across most modules.

### 7.2 What Needs Work

- **`DIRECT_TOOLS` is a local variable** inside `setup_mcp_server_standalone()`. It cannot be imported or tested in isolation. Refactoring it to a module-level constant would improve testability.
- **`_HANDLERS` dicts in `*_direct.py` are never validated against `DIRECT_TOOLS`.** A mismatch (like the 24 unwired handlers) silently goes unnoticed. A startup validation check would catch this.
- **Logging is inconsistent** — some modules use `logger.info()`, others use `print()`, some use `ctx.info()`. Should standardize on one pattern.
- **No pre-commit hooks** — no automated formatting, linting, or type-checking before commits.
- **`try/except pass` patterns** exist in several `*_direct.py` modules, silently swallowing errors.

### 7.3 Recommendations

1. **Refactor `DIRECT_TOOLS` to module-level** — makes it importable for testing and validation.
2. **Add startup validation** — assert every `_HANDLERS` key in `*_direct.py` has a matching `DIRECT_TOOLS` entry (and vice versa).
3. **Standardize logging** — `ctx.info()` for runtime messages, `logger` for setup messages, eliminate `print()`.
4. **Add pre-commit** with ruff + a basic type checker.
5. **Audit `try/except pass`** — every silent exception swallow reduces debuggability.

---

## 8. Architecture Concerns

### 8.1 Strengths

- **MCP-native design** — FastMCP 3.x with proper Resources, Prompts, tool typing. No Flask, no middleware translation layer.
- **Smart tool discovery** — BM25SearchTransform reduces LLM context bloat while keeping all 106 tools callable by name.
- **Clean separation** — `*_direct.py` modules are single-responsibility executors. Easy to add new tools.
- **CTF/CVE/Bug bounty engines** — properly isolated in their own modules, registered after the main tool chain.

### 8.2 Concerns

- **`search_tools` returns only 5 results** for broad queries like "all security tools". This limits the LLM's ability to discover tools it doesn't know about. Consider increasing the result limit or adding pagination.
- **No tool versioning** — `DIRECT_TOOLS` doesn't track which external tool version each entry was tested against. A tool could break silently when the system's version changes.
- **`run_security_tool` is a single-point-of-failure.** If it has a bug, all 106 tools break. Consider more granular testing of the dispatch logic.
- **`_scan_cache` is an in-memory dict** — lost on server restart. No persistent scan history. For a security tool that runs long scans, persistence would be valuable.
- **Configuration is scattered** — env vars (`HEXSTRIKE_HOST`, `HEXSTRIKE_PORT`, `HEXSTRIKE_DATA_DIR`), `hexstrike_server.py` constants, and `DIRECT_TOOLS` are all separate. No single config file.

---

## 9. Stabilization Roadmap

Priority-ordered actions to stabilize the project and prepare it for community contribution:

### Phase 1 — Truth in Advertising (1-2 days)
1. Reduce README tool count claim from "150+" to "106"
2. Wire the 8 implemented-but-unwired tools to `DIRECT_TOOLS` (+8 lines)
3. Add the 34 undocumented tools to the README
4. Fix the OpenCode config in README (`"type": "http"` → `"type": "remote"`)
5. Bump `fastmcp>=3.2.4` in `requirements.txt`

### Phase 2 — Dead Code Cleanup (2-3 days)
1. Delete or archive `mcp_tools/` tree (193 files, ~11K LOC)
2. Remove 4 dead imports in `server_setup.py` (lines 16-21)
3. Delete or archive 14 dead test files
4. Delete `mcp_core/hexstrike_client.py` (88 lines of dead code)
5. Clean up `mcp_core/args.py` — remove unused arguments

### Phase 3 — Community Infrastructure (2-3 days)
1. Add `SECURITY.md` with vulnerability disclosure policy
2. Add CI workflow (`.github/workflows/ci.yml`) running `pytest tests/ -q`
3. Add `CODE_OF_CONDUCT.md` (Contributor Covenant)
4. Add issue templates (bug report + feature request)
5. Add `.ruff.toml` for lint consistency

### Phase 4 — Stability & Quality (3-5 days)
1. Investigate test suite timeout (13s → 60s+ regression)
2. Refactor `DIRECT_TOOLS` to module-level constant
3. Add startup validation for handler/route consistency
4. Add `pytest-timeout` to prevent hanging tests
5. Fix `try/except pass` patterns

### Phase 5 — Git Hygiene (1 day)
1. Resolve divergence with upstream (146 ahead / 118 behind)
2. Identify correct PR target branch
3. Rebase and prepare focused PRs

---

## 10. Quick Wins vs Deep Work

### Quick Wins (done in minutes)

| Fix | Effort | Impact |
|---|---|---|
| Wire 8 implemented tools to DIRECT_TOOLS | 5 min each (40 min total) | 8 tools become usable |
| Fix README OpenCode config | 1 min | Wrong docs → correct docs |
| Bump fastmcp version in requirements.txt | 1 min | Prevents version mismatch |
| Add SECURITY.md | 10 min (template) | Closes critical gap |

### Medium Work (done in hours)

| Fix | Effort |
|---|---|
| Archive mcp_tools/ tree | 1 hour |
| Delete dead test files | 1 hour |
| Add CI workflow | 2 hours |
| Refactor DIRECT_TOOLS to module-level | 2 hours |
| Add startup validation | 2 hours |

### Deep Work (done in days)

| Fix | Effort |
|---|---|
| Sync with upstream (264-commit drift) | 2-3 days |
| Standardize logging | 1 day |
| Increase test coverage to 40% | 2-3 days |
| Add persistent scan cache | 2 days |

---

## Appendix A: File Inventory by Category

### Production Code (Phase 3, active)
```
hexstrike_server.py              # Entry point: HTTP/SSE server
hexstrike_mcp.py                 # Entry point: stdio MCP (Claude Desktop)
mcp_core/server_setup.py         # Main setup: tools, resources, prompts (1,177 lines)
mcp_core/                        # Core modules
  mcp_entry.py                   #   MCP entry point for stdio
  parameter_optimizer.py         #   Smart parameter tuning
  technology_detector.py         #   WAF/tech detection
  elicitation.py                 #   Confirmation gate
  tool_profiles.py               #   Profile resolution (mostly dead imports)
  ctf_engine.py                  #   CTF tools
  cve_engine.py                  #   CVE tools
  bugbounty_engine.py            #   Bug bounty tools
  *_direct.py (16 files)         #   Tool executors
server_core/                     # Server infrastructure
  operational_metrics.py         #   Metrics tracking
  rate_limit_detector.py         #   Rate limiting
  hexstrike_middleware.py        #   Middleware
  setup_logging.py               #   Logging setup
tool_registry.py                 # Tool definitions (162 entries)
```

### Production Code (Phase 2, dead)
```
mcp_tools/ (193 files, 51 packages)  # Entirely unreachable
mcp_core/hexstrike_client.py         # Dead client class
```

### Test Files
```
tests/ (48 files, ~20K LOC)
  - 30 active Phase 3 tests
  - 14 dead Phase 2 tests
  - 2 ignored-by-default tests
  - 1 conftest.py
  - 1 __init__.py
```

### Documentation & Config
```
README.md                   # Project README
AGENTS.md                   # OpenCode agent guide (gitignored)
hexstrike-ai-mcp.json       # Claude Desktop MCP config
requirements.txt            # Python dependencies
pytest.ini                  # Test configuration
.coveragerc                 # Coverage configuration
.editorconfig               # Editor settings
.gitignore                  # Git ignore rules
.gitattributes              # Git attributes
.github/                    # Community files
  CONTRIBUTING.md           # Contribution guide
  pull_request_template.md  # PR template
```

---

## Appendix B: Test File Categorization

### Active Phase 3 Tests (30 files)
```
test_bugbounty_workflow_phase3.py       test_plan_attack.py
test_command_executor_normalize.py       test_prompts.py
test_ctf_toolmanager.py                  test_recovery_executor.py
test_ctf_workflowmanager.py              test_run_security_tool_extra.py
test_fastmcp_context_regressions.py      test_security_phase2.py
test_fastmcp3_ctx_methods.py             test_server_setup_helpers.py
test_hexstrike_mcp.py                    test_server_setup_standalone.py
test_hexstrike_middleware.py             test_server_setup.py
test_hexstrike_server_routes.py          test_setup_logging_filters.py
test_intelligent_decision_engine_phase3.py test_skills_modernization.py
test_intelligent_decision_engine_phase4b.py test_technology_detector.py
test_metrics.py                          test_testssl_direct.py
test_misc_direct_phase4c.py              test_tool_registry_phase2.py
test_operational_metrics.py              test_vuln_intel_direct.py
test_parameter_optimizer.py              test_web_probe_direct.py
                                        test_web_scan_direct_phase3.py
```

### Dead Phase 2 Tests (14 files — move to archive)
```
test_active_directory_mcp_tools.py       test_gateway_template.py
test_osint_mcp_tools.py                  test_hexstrike_client_template.py
test_wifi_mcp_tools.py                   test_mcp_entry_template.py
test_gateway_expanded.py                 test_args_template.py
test_gateway_phase4a.py                  test_wordlist_ops.py
test_bug_bounty_recon.py                 test_vulnerability_intelligence.py
test_ai_payload_generation.py            test_ai_decision_engine.py
```

### Ignored by Default (2 files)
```
test_endpoints_exist.py                  # Legacy Flask endpoint tests
test_fastmcp_real_world.py               # Integration tests
```

---

## 11. Changes Applied (2026-05-08)

| # | Change | Files | Status |
|---|---|---|---|
| 1 | README OpenCode config `"type": "http"` → `"type": "remote"` | `README.md:82` | ✅ |
| 2 | `BM25SearchTransform` optimized: max_results 5→15, markdown serializer, pinned tools | `mcp_core/server_setup.py:29,30-32,496-502` | ✅ |
| 3 | Full audit report | `Projects_reports_docs/full_audit.md` | ✅ |
| 4 | Executive summary | `Projects_reports_docs/executive_summary.md` | ✅ |
| 5 | AGENTS.md (gitignored) | `AGENTS.md` | ✅ |

### Change 1: README OpenCode Config (2026-05-08)

**Before:**
```json
{
  "mcp": {
    "hexstrike-pulse": {
      "type": "http",
      "url": "http://127.0.0.1:8888/mcp",
      "enabled": true
    }
  }
}
```

**After:**
```json
{
  "mcp": {
    "hexstrike-pulse": {
      "type": "remote",
      "url": "http://127.0.0.1:8888/mcp",
      "enabled": true
    }
  }
}
```

**Why:** OpenCode only supports `"type": "local"` (stdio) and `"type": "remote"` (Streamable HTTP). The old `"type": "http"` was silently rejected. HexStrike uses `transport="http"` (Streamable HTTP) which is the new MCP 2025-03-26 standard.

### Change 2: BM25SearchTransform Optimization (2026-05-08)

**File:** `mcp_core/server_setup.py`

**Import change (line 29):**
```python
# Before
from fastmcp.server.transforms.search import BM25SearchTransform

# After
from fastmcp.server.transforms.search import BM25SearchTransform, serialize_tools_for_output_markdown
```

**Fallback vars (lines 30-32):**
```python
# Before
BM25SearchTransform = None

# After
BM25SearchTransform = None
serialize_tools_for_output_markdown = None
```

**Transform instantiation (lines 496-502):**
```python
# Before
transforms = [BM25SearchTransform()] if BM25SearchTransform else []

# After
transforms = [BM25SearchTransform(
    max_results=15,
    search_result_serializer=serialize_tools_for_output_markdown,
    always_visible=["nmap", "whatweb", "sqlmap"],
)] if BM25SearchTransform else []
```

**Impact:**

| Metric | Before | After |
|---|---|---|
| `tools/list` tools | 2 (search_tools, call_tool) | 5 (+ nmap, whatweb, sqlmap pinned) |
| `search_tools` format | JSON (~200 chars/tool) | Markdown (~115 chars/tool, -40%) |
| Max search results | 5 | 15 |
| Token cost per search (15 tools) | ~3,000 chars JSON | ~1,725 chars markdown |

**Tests:** 29/29 pass in `test_server_setup_standalone.py`.

---

*End of audit. Total scope: 209 dead files (~16,200 LOC), 31 unshipped tools, 34 undocumented tools, 7 missing community files, 264-commit branch drift, 60s+ test timeout regression. 2 changes applied (README + BM25SearchTransform).*
