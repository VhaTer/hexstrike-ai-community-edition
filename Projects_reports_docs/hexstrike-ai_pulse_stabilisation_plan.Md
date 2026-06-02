# HexStrike Stabilization Plan

Date: 2026-04-25
Scope: move the project toward a stable, predictable, maintainable release before any new large-scale refactor

## Objective

Reach a version that is:

- functionally reliable
- operationally observable
- coherent across runtime paths
- safe enough to evolve without reintroducing known instability

This phase is not about redesigning the whole architecture again.
It is about making the current architecture trustworthy enough to support future refactoring.

## Core Principle

Do not refactor large structures while core runtime behavior, tool calls, workflows, and monitoring are still unstable.

Target state:

- no critical runtime surprises
- no broken MCP tool calls on supported paths
- no ambiguous workflow behavior
- clear error reporting
- enough observability to diagnose failures quickly

## Stabilization Strategy

The stabilization phase should happen in this order:

1. freeze architectural churn
2. stabilize the active runtime path
3. harden workflows and intelligence behavior
4. improve observability and monitoring
5. run regression validation
6. declare a stable candidate

## Phase 1. Freeze The Direction

### Goal

Prevent the codebase from moving in too many directions at once.

### Allowed changes

- bug fixes
- runtime consistency fixes
- workflow fixes
- monitoring and logging improvements
- test additions
- small local cleanups that reduce risk immediately

### Deferred changes

- new broad architectural rewrites
- massive file moves
- new compatibility layers
- public API changes unless required to fix supported behavior
- speculative abstractions not needed for stability

### Required output

Define the supported surface for the stabilization window:

- active entrypoint
- supported MCP tool exposure
- supported workflow prompts
- supported dashboard scope
- supported compatibility surface, if any

### Exit criteria

- team agreement on what is officially supported during stabilization
- no ambiguous ownership between standalone runtime and transitional compatibility code

## Phase 2. Stabilize The Active Runtime

### Goal

Make the real Phase 3 execution path predictable and consistent.

### Primary focus

- `hexstrike_server.py`
- `mcp_core/server_setup.py`
- `mcp_core/*_direct.py`
- active typed MCP tool calls
- `run_security_tool(...)`

### Questions to close

- do typed tools and `run_security_tool(...)` behave consistently?
- are confirmations enforced correctly on the standalone path?
- are cache and context interactions deterministic?
- are errors and timeouts normalized?
- are supported tools exposed coherently across the active surface?

### Concrete work items

- standardize result and error shapes on the active path
- verify timeout behavior and failure propagation
- validate tool registration coverage against the intended active surface
- remove or document remaining route drift between standalone and compatibility layers
- verify state restoration and cache reads are safe and non-surprising

### Exit criteria

- no known P0/P1 runtime bugs on the active path
- supported tools callable without unexpected divergence
- destructive-tool behavior consistent with policy

## Phase 3. Harden Workflows And Intelligence

### Goal

Make intelligence and workflows useful, relevant, and safe before deep integration work continues.

### Scope

- prompts
- skills
- context/state usage
- `ParameterOptimizer`
- `TechnologyDetector`
- future `IntelligentDecisionEngine` integration planning

### Key principle

Do not deepen intelligence integration until the current orchestration behavior is trustworthy.

### Immediate work

- verify workflow prompts still match the active runtime surface
- ensure skill guidance does not contradict typed tool usage
- validate cache-derived tech detection behavior
- validate session state restoration behavior
- identify current false positives, weak assumptions, and workflow gaps

### IntelligentDecisionEngine policy

Before integrating it deeply into `run_security_tool(...)`, answer:

- is it advisory or authoritative?
- does it plan tool order, parameters, or both?
- where does human override live?
- where do confirmation boundaries apply for generated chains?
- should `TargetProfile` and `AttackChain` become runtime contracts?

### Exit criteria

- no critical workflow breakages on supported flows
- intelligence helpers do not create silent or unsafe execution changes
- IDE integration plan is explicit before implementation expands

## Phase 4. Observability And Monitoring

### Goal

Make failures diagnosable quickly.

### Minimum telemetry to capture

- tool name
- normalized target identity where safe
- execution start and end
- duration
- success or failure
- timeout state
- cancellation or user rejection
- confirmation requested / accepted / denied
- cache hit or miss
- session state restored or not
- optimizer profile and important overrides
- workflow prompt suggestions emitted
- typed tool vs generic tool invocation source

### Minimum operational views

- error count by tool
- timeout count by tool
- success rate by tool
- slowest tools
- cache hit/miss ratio
- workflow usage counts
- confirmation events
- dashboard/API failures if dashboard remains supported

### Required outcome

When something breaks, maintainers should be able to answer quickly:

- what failed
- where it failed
- for which tool or workflow
- whether the failure is deterministic
- whether context/cache/skills influenced the call

### Exit criteria

- critical failures are visible without deep manual digging
- operational debugging no longer depends on reading raw code first

## Phase 5. Regression Validation

### Goal

Prove the system is stable enough to tag a version candidate.

### Required validation areas

- typed MCP tools
- `run_security_tool(...)`
- destructive confirmation paths
- cache writes and cache reads
- session state restore paths
- prompt suggestion paths
- representative workflows
- representative dashboard behavior, if supported

### Testing policy

- prioritize integration and regression tests over adding more isolated unit tests
- test the real supported runtime surface first
- separate legacy/transitional tests from active-runtime quality gates

### Bug policy

Use a simple severity model:

- P0: crash, unsafe behavior, or broken core execution path
- P1: supported workflow/tool path broken or misleading
- P2: degraded behavior with workaround
- P3: non-critical polish/documentation issue

Stable candidate requires:

- zero open P0
- zero open P1
- agreed handling or deferral for remaining P2 issues

## Stable Candidate Criteria

A build can be considered stable when all of the following are true:

- supported runtime surface is explicitly defined
- no known P0/P1 issues remain
- active MCP tool calls are coherent and reproducible
- critical workflows pass regression checks
- monitoring is sufficient for operational diagnosis
- docs used by maintainers and assistants match the real supported behavior

## What Not To Do Before Stability

- do not launch another broad architecture rewrite
- do not migrate everything at once "for cleanliness"
- do not expand intelligence scope without validation hooks
- do not let transitional behavior silently define the product
- do not claim feature completeness based only on historical reports

## Recommended Execution Order

1. define supported stabilization surface
2. list open P0/P1 issues
3. fix active runtime inconsistencies
4. harden workflows and intelligence helpers
5. add operational telemetry and monitoring views
6. run regression pass on supported flows
7. tag a stable candidate
8. only then reopen the maintainability refactor track

## Final Position

The correct move is not to stop evolving the project.
It is to stop changing too many variables at the same time.

Stability first.
Then refactor from a system that has already proven what works.
