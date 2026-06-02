# HexStrike Documentation Claims Audit

Date: 2026-04-30  
Docs fixed: 2026-05-01  
Scope: `Projects_reports_docs/*` compared against the current repository state and test results.

## Source Documents Reviewed

- `README.md`
- `README_TESTS.md`
- `EXECUTIVE_SUMMARY.md`
- `COVERAGE_ANALYSIS.md`
- `LOW_COVERAGE_ANALYSIS_REPORT.md`
- `hexstrike-fastmcp-knowloadge-base.md`
- `hexstrike-ai_pulse_stabilisation_plan.Md`

## Verification Commands

```bash
git branch --show-current
git rev-parse --short HEAD
hexstrike-env/bin/python3 -m pytest tests/ -q
hexstrike-env/bin/python3 -m pytest tests/ --cov --cov-report=term -q
hexstrike-env/bin/python3 -m coverage report
```

Important: the active `pytest.ini` excludes `tests/test_endpoints_exist.py` and `tests/test_fastmcp_real_world.py`.

## Current Ground Truth

| Fact | Current value | Evidence |
|---|---:|---|
| Current branch | `feature/attack-intelligence` | `git branch --show-current` |
| Current HEAD | `4cd532b` | `git rev-parse --short HEAD` |
| Configured pytest result | `1277 passed, 6 xfailed, 1 warning` | `pytest tests/ -q` |
| Configured pytest runtime | `371.80s` | `pytest tests/ -q` |
| Coverage result | `30%` | `coverage report`, total `15717` statements, `10258` missed |
| Tool registry size | `162` tools | `tool_registry.TOOLS` |
| Direct runtime routes | `106` | `DIRECT_TOOLS` in `mcp_core/server_setup.py` |
| Typed MCP direct wrappers | `105` | direct routes with registry definitions/aliases |
| Direct modules | `16` | `mcp_core/*_direct.py` |
| Workflow prompts | `5` | decorators in `mcp_core/prompts.py` |
| MCP resources | `4` | decorators in `mcp_core/server_setup.py` |
| Destructive gates | `5` tools | `_DESTRUCTIVE_TOOLS` |
| Skill mappings | `91` | `_TOOL_SKILL_MAP` |
| Skill directories | `13` | `skills/*` directories |
| FastMCP requirement | `fastmcp>=3.2.4` | `requirements.txt` |
| Runtime port default | `8888` | `HEXSTRIKE_PORT` default in `hexstrike_server.py` |

## Claims That Match Current Code

| Claim | Verdict | Notes |
|---|---|---|
| `150+ security tools` | Valid | Current registry has `162` tools. |
| FastMCP 3.x native server | Valid | Code imports FastMCP, runs HTTP transport, and requires `fastmcp>=3.2.4`. |
| Default MCP endpoint uses port `8888` | Valid | `HEXSTRIKE_PORT` defaults to `8888`; server runs HTTP transport. |
| 5 workflow prompts exist | Valid | `bug_bounty_recon`, `wifi_attack_chain`, `ctf_web_challenge`, `smb_lateral_movement`, `cloud_security_audit`. |
| 16 `*_direct.py` modules exist | Valid | Current `mcp_core` contains 16 direct modules. |
| 106 direct routes and 105 typed MCP tools | Valid | Current code has 106 routes; all except `wifite2` resolve to a registry definition or alias. |
| 91 tool-to-skill mappings | Valid | `_TOOL_SKILL_MAP` contains 91 mappings. |
| Safety gates for `aireplay-ng`, `metasploit`, `responder`, `mdk4`, `mitm6` | Valid with naming caveat | Runtime keys are `aireplay_ng`, `metasploit`, `responder`, `mdk4`, `mitm6`. |
| WAF detection can force stealth profile | Valid | `ParameterOptimizer` forces stealth when `TechProfile.has_waf` is true. |
| WordPress detection can inject relevant tuning | Valid | Gobuster/WPScan/Nuclei WordPress-specific branches exist and are tested. |
| Skill guidance before tool execution | Valid | `run_security_tool` reads mapped `SKILL.md` and emits guidance via `ctx.info`. |
| MCP resources for health and scan cache | Valid | 4 resources exist: `health://server`, `scan://{target}/latest`, `scan://{target}/{tool_name}`, `scan://cache/list`. |
| `plan_attack()` exists | Valid | Registered in `mcp_core/server_setup.py` and returns an advisory attack-chain dict. |

## Claims That Are Outdated Or False

| Document claim | Current fact | Verdict |
|---|---|---|
| `1,091` tests, `1,082` passing, `9` failing | `1277` passing, `6` xfailed, `0` failing in configured suite | False / outdated |
| `99.2%` pass rate | Configured suite exits cleanly with all non-xfailed tests passing | False / outdated |
| `23%` coverage, `6,105/8,151` lines | `30%` coverage, `15717` statements, `10258` missed | False / outdated |
| `38.23%` coverage, `4,664/12,199` lines | `30%` coverage under current pytest-cov configuration | False / outdated |
| Full suite runs in `13 seconds` | Measured `371.80s` without coverage, `364.92s` with coverage in this environment | False for current environment |
| `9 failing tests` in `test_gateway_phase4a.py` and `test_gateway_template.py` | Current configured suite has no failures | False / fixed |
| `1,091 tests across 32 files` | Current `tests/` has 40 top-level `test*.py` files; configured run reports 1277 passing tests | False / outdated |
| `1227 passed, 6 xfailed` | Current configured suite reports `1277 passed, 6 xfailed` | False / outdated |
| Branch is `refactor/fastmcp-modernization` | Current branch is `feature/attack-intelligence` | False for current checkout |
| HEAD is `9cb6e88` | Current HEAD is `4cd532b` | False for current checkout |
| `14 skill bundles` | Current `skills/` has 13 directories; `testssl` is not present as a skill directory | False / outdated |
| `TEST_ANALYSIS_REPORT.md`, `TEST_METRICS.md`, `TEST_DOCUMENTATION.md` exist in docs set | These files are referenced but absent from `Projects_reports_docs/` | False / broken links |
| Low-coverage report says 8/10 target modules have zero dedicated tests | Dedicated tests now exist for several target modules | False / outdated |

## Coverage Claims Rechecked

| Module / area | Doc claim | Current coverage | Verdict |
|---|---:|---:|---|
| `server_core/intelligence/cve_intelligence_manager.py` | `4%` | `4%` | Still accurate |
| `server_core/failure_recovery_system.py` | `11%` | `11%` | Still accurate |
| `server_core/rate_limit_detector.py` | `10%` | `36%` | Outdated |
| `server_core/recovery_executor.py` | `7%` / low | `69%` | Outdated |
| `mcp_tools/ai_payload/ai_payload_generation.py` | `7%` | `74%` | Outdated |
| `mcp_tools/ops/wordlist.py` | `5%` | `78%` | Outdated |
| `mcp_tools/ops/vulnerability_intelligence.py` | `3%` | `45%` | Outdated |
| `mcp_tools/bugbounty_workflow/bug_bounty_recon.py` | `4%` | `36%` | Outdated |
| `mcp_tools/ops/file_ops_and_payload_gen.py` | `5%` | `4%` | Still effectively low |
| `mcp_tools/api_audit/comprehensive_api_audit.py` | `8%` | `5%` | Still low, exact value outdated |
| `mcp_tools/web_framework/http_framework.py` | `8%` | `7%` | Still low |
| `server_core/workflows/ctf/toolManager.py` | `6%` | `66%` | Outdated |
| `mcp_tools/gateway.py` | `8.16%` / `52%` in different docs | `48%` | Docs conflict; current value differs |
| `server_core/intelligence/intelligent_decision_engine.py` | `43%` | `43%` | Accurate for server-core module |

## Document-By-Document Assessment

### `README.md`

Mostly aligned with the active architecture. Main feature claims are backed by code: 150+ tools, FastMCP, prompts, resources, skill guidance, parameter optimization, destructive gates, and `plan_attack()`.

Issues:
- The branding and examples are aspirational in places, but the named core mechanisms exist.
- Tool availability is registry-level availability, not proof that every external binary is installed on the host.

### `README_TESTS.md`

Not reliable as a current status document.

Issues:
- Test count, pass/fail count, coverage, runtime, failing-test analysis, and document index are outdated.
- Links to `TEST_ANALYSIS_REPORT.md`, `TEST_METRICS.md`, and `TEST_DOCUMENTATION.md` are broken in `Projects_reports_docs/`.
- The claim that the suite has 9 known failures is false for the configured suite.

### `EXECUTIVE_SUMMARY.md`

Not reliable as a current status document.

Issues:
- Same stale test metrics as `README_TESTS.md`.
- Low-coverage risks are partially still true, but several modules improved substantially.
- Recommendations to fix 9 failures are obsolete.

### `COVERAGE_ANALYSIS.md`

Internally conflicts with `README_TESTS.md` and `EXECUTIVE_SUMMARY.md`.

Issues:
- Claims `38.23%` overall coverage while other docs claim `23%`; current measured coverage is `30%`.
- Some module priorities are still directionally useful, but exact percentages are stale.
- Coverage total line counts do not match the current coverage configuration.

### `LOW_COVERAGE_ANALYSIS_REPORT.md`

Partially useful historically, but materially stale.

Still useful:
- LOC values for the 10 analyzed modules are still correct.
- Some modules remain genuinely low: CVE manager, file ops/payload gen, API audit, HTTP framework.

Outdated:
- Dedicated tests now exist for `ai_payload_generation`, `wordlist`, `recovery_executor`, `bug_bounty_recon`, `vulnerability_intelligence`, and `toolManager`.
- Current coverage for several target modules is much higher than claimed.

### `hexstrike-fastmcp-knowloadge-base.md`

Mostly accurate for architecture, but stale for checkout metadata.

Accurate:
- FastMCP 3.2.4 requirement, 16 direct modules, 106 direct routes, 105 typed tools, 91 skill mappings, 5 workflow prompts, 4 resources, destructive elicitation policy, and advisory `IntelligentDecisionEngine` policy.

Outdated:
- Branch, HEAD, test count, and skill bundle count.
- Current skill directory count is 13, not 14.

### `hexstrike-ai_pulse_stabilisation_plan.Md`

Conceptual plan, not a metrics report. No hard factual conflict found. Its warning not to claim feature completeness from historical reports is directly supported by this audit.

## Fix Status

The following docs were rewritten on 2026-05-01 to remove stale metrics, broken links, and contradictory coverage claims:

- `README.md`
- `README_TESTS.md`
- `EXECUTIVE_SUMMARY.md`
- `COVERAGE_ANALYSIS.md`
- `LOW_COVERAGE_ANALYSIS_REPORT.md`
- `hexstrike-fastmcp-knowloadge-base.md`

`hexstrike-ai_pulse_stabilisation_plan.Md` was left unchanged because it is a conceptual stabilization plan and did not contain conflicting hard metrics.

## Remaining Recommendations

1. Decide whether `Projects_reports_docs/` should remain ignored by Git.
2. Re-run pytest and coverage before publishing any future numeric status.
3. Keep historical claims only when explicitly marked as historical.
4. Keep the stabilization plan as the policy baseline; it correctly requires docs to match runtime facts before declaring stability.
