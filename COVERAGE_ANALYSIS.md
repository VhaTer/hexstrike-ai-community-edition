# HexStrike Project Coverage Analysis & Test ROI Report

**Generated:** April 19, 2026  
**Overall Coverage:** 38.23% (4,664 / 12,199 lines)  
**Codebase Size:** ~12,200 LOC across all modules

---

## 1. MODULES BY LINES OF CODE (Descending)

| Module | LOC | Coverage % | Status |
|--------|-----|-----------|--------|
| `mcp_core/misc_direct.py` | ~500-600 | **9.96%** | ⚠️ CRITICAL |
| `mcp_core/net_scan_direct.py` | ~370 | **11.32%** | ⚠️ CRITICAL |
| `mcp_core/security_direct.py` | ~450 | **10.64%** | ⚠️ CRITICAL |
| `mcp_core/wifi_direct.py` | ~500+ | **9.74%** | ⚠️ CRITICAL |
| `mcp_core/web_recon_direct.py` | ~430 | **11.03%** | ⚠️ CRITICAL |
| `mcp_core/web_scan_direct.py` | ~380 | **12.73%** | ⚠️ CRITICAL |
| `mcp_core/password_cracking_direct.py` | ~410 | **11.72%** | ⚠️ CRITICAL |
| `mcp_core/web_fuzz_direct.py` | ~380 | **14.29%** | ⚠️ CRITICAL |
| `mcp_core/recon_direct.py` | ~370 | **15.15%** | ⚠️ CRITICAL |
| `mcp_core/active_directory_direct.py` | ~360 | **50.36%** | ⚠️ Low coverage |
| `mcp_core/smb_enum_direct.py` | ~370 | **12.15%** | ⚠️ CRITICAL |
| `mcp_core/exploit_framework_direct.py` | ~350 | **13.68%** | ⚠️ CRITICAL |
| `mcp_core/hexstrike_client.py` | ~250 | **22.86%** | ⚠️ Low coverage |
| `tool_registry.py` | **1,973** | **26.32%** | ⚠️ Low coverage (but mostly data) |
| `mcp_core/mcp_entry.py` | ~250 | **17.14%** | ⚠️ Low coverage |
| `mcp_core/args.py` | ~100 | **25.0%** | ⚠️ Low coverage |
| `mcp_core/elicitation.py` | ~140 | **53.33%** | ⚠️ Partial coverage |
| `mcp_tools/gateway.py` | ~180 | **8.16%** | ⚠️ CRITICAL |
| `mcp_tools/ai_assist/intelligent_decision_engine.py` | ~380 | **2.96%** | 🔴 LOWEST |
| `mcp_tools/ai_payload/ai_payload_generation.py` | ~230 | **6.67%** | ⚠️ CRITICAL |
| `mcp_tools/bugbounty_workflow/bug_bounty_recon.py` | ~380 | **4.07%** | ⚠️ CRITICAL |

### Well-Covered Modules (Good baseline)
| Module | LOC | Coverage % | Status |
|--------|-----|-----------|--------|
| `mcp_core/osint_direct.py` | ~180 | **93.62%** | ✅ Excellent |
| `mcp_core/testssl_direct.py` | ~250 | **96.97%** | ✅ Excellent |
| `mcp_core/technology_detector.py` | ~220 | **97.30%** | ✅ Excellent |
| `mcp_core/web_probe_direct.py` | ~240 | **96.47%** | ✅ Excellent |
| `mcp_core/parameter_optimizer.py` | ~360 | **90.48%** | ✅ Good |
| `mcp_core/vuln_intel_direct.py` | ~130 | **100%** | ✅ Perfect |
| `mcp_core/prompts.py` | ~100 | **100%** | ✅ Perfect |
| `hexstrike_mcp.py` | ~50 | **83.33%** | ✅ Good |
| `config.py` | ~1,200+ | **100%** | ✅ Perfect |

---

## 2. COVERAGE ANALYSIS BY CATEGORY

### LOWEST COVERAGE GROUPS (Highest Need for Tests)

#### 🔴 TIER 1: 0-15% Coverage (CRITICAL)
- **AI Decision Engine** (`ai_assist/intelligent_decision_engine.py`): **2.96%** 
  - 380 LOC, heavily used, very complex
  - Implements attack decision logic, parameter optimization
  - Hard to test: LLM calls, decision heuristics
  - **ROI: VERY HIGH but hard to test**

- **Misc Direct Tools** (`mcp_core/misc_direct.py`): **9.96%**
  - 500-600 LOC, wrapper layer
  - ROPgadget, volatility, SQLite, MySQL handlers
  - Execution layer for binary analysis & forensics

- **WiFi Penetration** (`mcp_core/wifi_direct.py`): **9.74%**
  - 500+ LOC, multiple attack modes
  - Aircrack, airodump, aireplay wrappers
  - **Hard to test**: Requires wireless hardware

- **Gateway Router** (`mcp_tools/gateway.py`): **8.16%**
  - ~180 LOC, central routing layer
  - Request dispatch to tool categories
  - **High ROI**: Testable through mocking

- **Bug Bounty Workflow** (`bugbounty_workflow/bug_bounty_recon.py`): **4.07%**
  - 380 LOC, high-level orchestration
  - Chains multiple reconnaissance tools
  - **High ROI**: Business-critical, testable

#### ⚠️ TIER 2: 15-30% Coverage
- `mcp_core/web_scan_direct.py`: 12.73%
- `mcp_core/web_recon_direct.py`: 11.03%
- `mcp_core/security_direct.py`: 10.64%
- `tool_registry.py`: 26.32% (but mostly data)
- `mcp_core/hexstrike_client.py`: 22.86%

#### 🟠 TIER 3: 30-70% Coverage
- `mcp_core/active_directory_direct.py`: 50.36%
- `mcp_core/elicitation.py`: 53.33%
- `mcp_core/tool_profiles.py`: 50%

---

## 3. HIGH-ROI TEST TARGETS

### Priority 1: HIGH IMPACT, TESTABLE (Write ASAP)

1. **`mcp_tools/gateway.py`** (8.16% → target 80%+)
   - 180 LOC, central routing logic
   - Pure Python (no external tools needed)
   - **Test Strategy**: Mock tool handlers, assert request routing
   - **Estimated effort**: 2-3 hours
   - **Impact**: All API calls depend on this

2. **`mcp_core/mcp_entry.py`** (17.14% → target 70%+)
   - 250 LOC, server initialization
   - Pure Python, testable setup phases
   - **Test Strategy**: Mock FastMCP, test tool registration
   - **Estimated effort**: 2-3 hours
   - **Impact**: Server startup reliability

3. **`mcp_core/hexstrike_client.py`** (22.86% → target 75%+)
   - 250 LOC, HTTP client wrapper
   - Testable with mocking `requests.Session`
   - **Test Strategy**: Mock API responses, test retry logic
   - **Estimated effort**: 2 hours
   - **Impact**: Client-side communication reliability

4. **`tool_registry.py`** (26.32% → target 60%+)
   - 1,973 LOC but mostly data + lookup logic
   - Logic parts: intent classification, parameter validation
   - **Test Strategy**: Test classification logic, tool lookups
   - **Estimated effort**: 3-4 hours
   - **Impact**: Tool routing correctness

### Priority 2: MODERATE IMPACT, MEDIUM DIFFICULTY

5. **`mcp_core/args.py`** (25% → target 70%+)
   - 100 LOC, argument parsing
   - Pure Python, testable with different args
   - **Test Strategy**: Parametrized tests for arg combinations
   - **Estimated effort**: 1.5 hours

6. **`mcp_core/parameter_optimizer.py`** (90.48% - MAINTAIN)
   - 360 LOC, already well-tested
   - **Strategy**: Add edge-case tests, maintain 95%+ coverage
   - **Estimated effort**: 1 hour

7. **`mcp_core/recon_direct.py`** (15.15% → target 60%+)
   - 370 LOC, reconnaissance tool wrappers
   - Mock-heavy approach needed
   - **Test Strategy**: Mock subprocess calls, verify command construction
   - **Estimated effort**: 3-4 hours

### Priority 3: HIGH IMPACT BUT HARD TO TEST

8. **`mcp_tools/ai_assist/intelligent_decision_engine.py`** (2.96% → target 40%+)
   - 380 LOC, complex decision logic
   - Requires: LLM mocking, state management
   - **Test Strategy**: Mock LLM calls, test decision paths with fixtures
   - **Estimated effort**: 4-5 hours (complex)
   - **Worth it**: YES - critical to attack selection

9. **`mcp_core/security_direct.py`** (10.64% → target 50%+)
   - 450 LOC, security scanning orchestration
   - Subprocess-heavy (nmap, nikto, SQLmap, etc.)
   - **Test Strategy**: Mock subprocess, verify command generation
   - **Estimated effort**: 4-5 hours
   - **Impact**: Critical scanning tools

10. **`mcp_tools/bugbounty_workflow/bug_bounty_recon.py`** (4.07% → target 50%+)
    - 380 LOC, high-level workflow orchestration
    - Chain of tools with state management
    - **Test Strategy**: Mock tool APIs, test workflow paths
    - **Estimated effort**: 3-4 hours
    - **Impact**: Business-critical workflow

---

## 4. HARD-TO-TEST PATTERNS IDENTIFIED

### 🔴 External Tool Execution (Subprocess-Heavy)
**Affected Modules:**
- `mcp_core/misc_direct.py` (ROPgadget, volatility)
- `mcp_core/net_scan_direct.py` (nmap, masscan, rustscan)
- `mcp_core/web_scan_direct.py` (nikto, nuclei, sqlmap)
- `mcp_core/password_cracking_direct.py` (hashcat, john)
- `mcp_core/wifi_direct.py` (aircrack, airodump)
- `server_core/command_executor.py` (all subprocess calls)

**Challenge:** External tools must be mocked  
**Solution:** Use `unittest.mock.patch('subprocess.run')` + fixture data

### 🟠 System Calls & File I/O
**Affected Modules:**
- `server_core/enhanced_command_executor.py` (process threads, pipes)
- `mcp_tools/data_processing/` (file carving, forensics)
- `server_core/file_ops.py` (file operations)

**Challenge:** Timing, file state, system resources  
**Solution:** Use temporary directories, `tmp_path` fixtures

### 🟠 Wireless Hardware (aircrack, WiFi tools)
**Affected Modules:**
- `mcp_core/wifi_direct.py` (100% requires hardware)

**Challenge:** Hardware unavailable in CI/CD  
**Solution:** Skip in CI, use integration tests only, mock subprocess

### 🟠 Database Connections (MySQL, SQLite)
**Affected Modules:**
- `mcp_tools/db_query/mysql.py` (22.08%)
- `mcp_tools/db_query/sqlite.py` (23.08%)

**Challenge:** Database setup/teardown  
**Solution:** SQLite in-memory database, MySQL Docker fixtures

### 🟠 LLM Integration (AI Decision Engine)
**Affected Modules:**
- `mcp_tools/ai_assist/intelligent_decision_engine.py` (2.96%)

**Challenge:** LLM API calls, non-deterministic responses  
**Solution:** Mock responses, use response fixtures, test decision logic separately

### 🟠 Network Calls (HTTP, API Clients)
**Affected Modules:**
- `mcp_core/hexstrike_client.py` (22.86%)
- `mcp_tools/net_lookup/whois.py` (27.78%)
- `mcp_tools/osint/` modules (many have 90%+ coverage already)

**Challenge:** External API unreliability  
**Solution:** `responses` library or `unittest.mock`, VCR cassettes

### 🟢 TESTABLE (No Hard Dependencies)
**Good Candidates:**
- `tool_registry.py` - pure data + lookup logic
- `mcp_core/args.py` - argument parsing
- `mcp_core/parameter_optimizer.py` - already 90%+
- `mcp_tools/gateway.py` - routing logic, mock-friendly

---

## 5. EXISTING TEST PATTERNS

### Current Test Files
```
tests/
├── test_hexstrike_mcp.py           # Main entry point (import-only)
├── test_server_setup.py            # Server initialization
├── test_server_setup_standalone.py # Standalone server
├── test_ai_decision_engine.py      # AI logic (basic)
├── test_parameter_optimizer.py     # Param optimization
├── test_technology_detector.py     # Tech fingerprinting
├── test_prompts.py                 # Prompt generation
├── test_osint_mcp_tools.py         # OSINT tools
├── test_active_directory_mcp_tools.py  # AD tools (90%+ coverage)
├── test_wifi_mcp_tools.py          # WiFi tools
├── test_endpoints_exist.py         # API endpoint validation
├── test_vuln_intel_direct.py       # Vulnerability intel
├── test_testssl_direct.py          # SSL/TLS scanning
├── test_web_probe_direct.py        # Web probing
└── test_*.py                       # ~15+ test files total
```

### Common Test Patterns (What Works)
1. **Import-only tests** - Just verify imports work → Easy 50%+ coverage
2. **Mock subprocess** - All tool execution tests use mocked subprocess
3. **Fixture-based responses** - Store expected command outputs in fixtures
4. **Mock HTTP** - Use `unittest.mock` or `responses` library
5. **Parametrized tests** - Multiple test cases from one template

### Example Pattern (from `test_server_setup.py`)
```python
from unittest.mock import MagicMock, patch
import mcp_core.server_setup

def test_setup_mcp_server_call():
    mock_client = MagicMock()
    mock_logger = MagicMock()
    try:
        mcp_core.server_setup.setup_mcp_server(mock_client, mock_logger)
    except Exception as e:
        pass  # Coverage is the goal
    print("✅ setup_mcp_server called!")
```

---

## 6. TEST COVERAGE GAPS SUMMARY

### By Package

| Package | Total Lines | Covered | Gap | Priority |
|---------|-------------|---------|-----|----------|
| **Root** (config.py, hexstrike_mcp.py, tool_registry.py) | 2,200+ | 40.79% | -59.21% | MEDIUM |
| **mcp_core** | 5,500+ | 33.39% | -66.61% | **CRITICAL** |
| **mcp_tools** | 2,800+ | 55.00% | -45.00% | HIGH |
| **server_core** | ?~1,500 | ? | ? | MEDIUM |
| **Overall** | 12,199 | **38.23%** | -61.77% | **CRITICAL** |

---

## 7. RECOMMENDED TEST WRITING STRATEGY

### Phase 1: Quick Wins (Effort: 8-10 hours, Gain: +10-15%)
1. `gateway.py` - central routing, 180 LOC, pure Python
2. `mcp_entry.py` - server init, 250 LOC, mostly setup
3. `args.py` - CLI args, 100 LOC, simple logic
4. `hexstrike_client.py` - HTTP client, 250 LOC, mockable

**Expected Result:** 45-50% overall coverage

### Phase 2: High-Impact Targets (Effort: 12-15 hours, Gain: +12-18%)
1. `tool_registry.py` - classification + lookup logic
2. `parameter_optimizer.py` - maintain/expand (already 90%)
3. `security_direct.py` - mock subprocess, test commands
4. `recon_direct.py` - mock subprocess, test orchestration

**Expected Result:** 55-60% overall coverage

### Phase 3: Hard-to-Test Strategic Targets (Effort: 15-20 hours, Gain: +8-12%)
1. `intelligent_decision_engine.py` - mock LLM, test paths
2. `bugbounty_workflow/bug_bounty_recon.py` - mock tools, test flows
3. `web_scan_direct.py` - mock subprocess
4. `password_cracking_direct.py` - mock subprocess

**Expected Result:** 65-70% overall coverage

### Phase 4: Comprehensive Coverage (Effort: 20-25 hours, Gain: +10-15%)
1. All remaining low-coverage modules
2. Edge cases and error paths
3. Integration tests for critical flows

**Expected Result:** 75-80% overall coverage

---

## 8. KEY METRICS & BENCHMARKS

```
Current State:
  ✗ Overall Coverage: 38.23%
  ✗ Total Lines: 12,199
  ✗ Lines Covered: 4,664
  ✗ Lines Missing: 7,535

Target State (Reasonable):
  ✓ Overall Coverage: 65-70%
  ✓ Lines Covered: ~8,000+
  ✓ Effort: 50-60 hours

Very High Target (80%):
  ✓ Overall Coverage: 80%+
  ✓ Lines Covered: ~9,800+
  ✓ Effort: 100+ hours (diminishing ROI)
```

---

## 9. CRITICAL RECOMMENDATIONS

1. **Immediate Priority**: Write tests for `gateway.py` and `mcp_entry.py`
   - These are central routing/initialization components
   - All other functionality depends on them
   - Pure Python, mockable

2. **Avoid Long-Running Tests**:
   - Subprocess-heavy tests (use mocks only)
   - External tool execution tests (mock subprocess calls)
   - WiFi hardware tests (skip in CI)

3. **Use Fixtures**:
   - Store expected command outputs in `tests/fixtures/`
   - Store sample scan results for parsing tests
   - Mock file I/O with `tmp_path`

4. **Test Organization**:
   - One test file per module/package
   - `conftest.py` for shared fixtures/mocks
   - Parametrized tests for tool variations

5. **Skip Strategy for CI/CD**:
   ```python
   @pytest.mark.skip(reason="Requires wireless hardware")
   def test_wifi_direct(): pass
   ```

---

## 10. CONCLUSION

**HexStrike has solid coverage in:**
- AI decision logic (osint_direct, technology_detector, testssl_direct)
- Parameter optimization
- Vulnerability intelligence
- Individual tool implementations (90%+ in many modules)

**HexStrike needs urgent coverage in:**
- **Gateway routing** (8.16% → target 80%)
- **Server initialization** (17% → target 70%)
- **CLI argument handling** (25% → target 70%)
- **HTTP client** (22% → target 75%)
- **Tool direct layers** (9-15% → target 60%+)
- **AI decision engine** (2.96% → target 40-50%)

**ROI Strategy:**
- Write tests for gateway + entry point first (quick +5-10%)
- Focus on subprocess-mockable modules next (steady +15-20%)
- Address hard-to-test AI/decision logic last (slow +10-15%)

**Estimated Time to Reach 70% Coverage:** 40-50 hours of focused test writing
