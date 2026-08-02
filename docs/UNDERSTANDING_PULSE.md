<p align="center">
  <img src="../assets/hexstrike-pulse-logo.png" alt="HexStrike Pulse" width="300">
</p>

# Understanding Pulse — a plain-language guide

This page explains what Pulse actually does for you day to day, without the code. If you're comfortable reading Python, see [`API.md`](API.md) instead — this page is for everyone else: teammates, reviewers, or anyone who just wants to know what's happening when they run a scan.

---

## How to open the dashboard

There are two separate ways to see dashboard data, and they're not the same thing — pick based on whether a human or an AI agent is looking.

### As a human, in a browser

With the server running (`./hexstrike-pulse`), open:

```
http://127.0.0.1:8888/dashboard
```

This is the visual version — panels, badges, tables — the one described in this page. It loads fresh data each time you open it (the underlying endpoint is recomputed on every request); reload the page to see the latest results. No setup beyond having the server up.

### As an AI agent, through MCP

If you're driving Pulse through Claude Desktop or another MCP client, the visual dashboard is available as a callable view — the agent can open it directly ("open the pulse dashboard") and it renders inline in the chat, same panels as the browser version.

But most of the time, an agent doesn't need the *visual* dashboard — it needs the *data* to reason over. That's a separate call, described in the cost section below.

---

## What the dashboard is showing you

Every panel answers one question about your current session. Think of it less like a report and more like a cockpit — the data is collected fresh every time you look at it. Here's a selection of the most useful ones (18 in total):

| Panel | Question it answers |
|---|---|
| **Header** | Is the server healthy? How much memory/CPU is it using? |
| **Scope** | Which target am I currently focused on? |
| **Surface** | What ports and technologies did we find on that target? |
| **Findings** | What vulnerabilities were detected, ranked by how promising they are? |
| **Plan** | Given what we know, what's the recommended attack sequence? |
| **History** | What have I already scanned, and when? |
| **Active tools** | Is anything still running right now? |
| **Cache status / Cache intelligence** | Is Pulse reusing recent results instead of re-scanning? |
| **System trends** | Is the server under load? |
| **Errors & tool performance** | Which tools are failing or timing out? |

The remaining panels (Sessions, Confirmations, Network I/O, Rate limit, Async scans, Missing tools) cover more specific situations — see the cost table further down for the full list.

You don't need to open every panel every time — the header and scope bar alone tell you "is the system healthy, and what am I working on."

---

## Why some scans finish instantly

The first time you scan a target, Pulse actually runs the tool (`nmap`, `whatweb`, etc.) and waits for it to finish. But if you (or the agent) scan the *same* target with the *same* tool again shortly after, Pulse skips running it again and hands back the previous result immediately. That's the cache.

What makes it smarter than a simple "remember for 30 minutes" cache: **it pays attention to how often each tool's cached results actually get reused.**

- If a tool's results are reused a lot (say, `nmap` on a target you keep coming back to), Pulse gradually **keeps its results around longer** — up to 2 hours.
- If a tool's results are almost never reused, Pulse **shortens how long it keeps them** — down to as little as 5 minutes — so you're not carrying around stale data for no reason.

You can watch this happen in the **Cache Intelligence** panel — it shows, per tool, how often results are being reused and how long they're currently being kept.

---

## The "Next tool" recommendation

At the bottom of the dashboard there's a small badge that says something like *"Next tool: sqlmap — SQL injection candidate found."* This is Pulse's built-in recommendation for what to try next, based on everything found so far.

It works in three tiers, tried in order:

1. **If a specific finding looks strong** (a real vulnerability with a good confidence score), it recommends the exact tool to exploit it.
2. **If not, it looks at keywords in the findings** — mentions of SQL injection, XSS, SMB vulnerabilities, SSL issues — and recommends accordingly.
3. **If there are no findings yet, it looks at what's open** — a web port suggests fingerprinting the technology, an SSH port suggests testing credentials, a database port suggests probing for weak authentication.

This recommendation exists so that whoever (or whatever agent) is driving Pulse has a sensible next step to follow instead of guessing — it's meant to encourage working through a target methodically rather than jumping around.

---

## What's happening behind the scenes

Two things run continuously underneath everything you see, and it helps to know what they're for even without touching any code.

**The process manager** keeps watch over the machine's resources — CPU, memory, and how much work is queued — and tracks the lifecycle of the tools that get launched (started, still running, stopped). Each scan runs up to 5 tools in parallel at most, a deliberately simple limit rather than something the process manager juggles dynamically. The **Active Tools** panel shows what this layer sees: current resource usage and worker state. Work still running in the background shows up in the **Async Scans** panel.

**The decision engine** is the planner behind the **Plan** panel. Once Pulse has scanned a target, the decision engine looks at everything it learned (open ports, detected technology, known vulnerabilities) and produces an ordered sequence of steps — each with an estimated success probability and time estimate — rather than a plain list of "things you could try."

Both of these exist once per running server, shared by every scan and every connected client — so results and recommendations stay consistent no matter how many tools you run in sequence.

---

## A note on caution

Everything above is designed to make Pulse *more efficient at doing what you told it to do* — it never decides to scan something on its own, and every recommendation is exactly that: a recommendation, not an automatic action. Nothing runs against a target unless you (or your connected agent) explicitly asks for it.

---

## The cost of asking for dashboard data

If an agent is driving Pulse, "opening the dashboard" and "reading the dashboard data" cost very different amounts of context. This matters because context is a limited, consumed resource for an LLM — every panel returned as text is tokens the agent has to read.

There are two ways an agent can pull data, and they trade off differently:

| Method | What it returns | When to use it |
|---|---|---|
| **Full dashboard** (all 18 panels at once) | Everything — overview, findings, plan, history, cache, errors, performance, trends, and more | You genuinely need the full picture, or you're not sure yet what matters |
| **Selected sections only** | Just the panels you ask for, or an auto-picked relevant subset | You know what you need — e.g. just the findings, or just the current scope |

The full pull is convenient but always costs roughly the same regardless of what's actually useful in the moment. Asking for specific sections costs only for what's requested — cheaper, but requires knowing what you're looking for.

**Approximate cost per section** (relative token weight, not an exact count — 18 panels total):

| Section | Relative cost | Section | Relative cost |
|---|---|---|---|
| Findings | Highest | Errors | Medium |
| Surface | High | Cache status | Medium |
| History / Plan | Medium-high | Overview / Intelligence / Trends / Sessions | Low |
| Tool performance | Medium | Active tools / Async scans / Confirmations / Network I/O / Rate limit / Missing tools | Lowest |

The practical takeaway: if an agent seems to be burning a lot of context just checking status between scans, it's worth switching from "give me everything" to "give me just the scope and overview" — the full picture is rarely needed on every single check-in, only when actually planning next steps.

