---
name: web-vuln
description: Web vulnerability scanning via template-based CVE matching, injection testing, and traversal detection. Use after endpoint mapping to find exploitable weaknesses.
---

# Web Vuln

## When to use

Use this skill in **Phase 3: Vulnerability Assessment** after recon has identified concrete endpoints:

- Identify known CVEs in detected frameworks/plugins (nuclei)
- Test injection vectors (SQLi, command injection, template injection)
- Detect path traversal, XXE, SSRF, or authentication bypass
- Validate parameter handling and input validation flaws

## Working Style

**Reduce false positives by targeting confirmed attack surface:**

1. **Broad Scan** — `nuclei` with severity filter (critical, high only on first pass)
2. **Targeted Injection** — `sqlmap` or `commix` only on identified params; avoid blind without evidence
3. **XSS/Traversal** — `dalfox` or `xsser` on forms; `dotdotpwn` on suspected traversal endpoints
4. **Parameter Fuzzing** — `arjun` output feeds refined fuzzing (minimize noise)

**Entry point:**

```python
nuclei(target="https://app.example.com", tags="cve,auth", severity="critical,high")
```

## Notes

- **Prerequisites:** Always run `web-recon` first; blind scanning produces noise and false positives
- **Effectiveness:** nuclei (0.93), sqlmap (0.85), dalfox (0.82); dalfox XSS has 15% FP rate
- **Targeted Over Blanket:** Use specific URLs/params, not full domain scans
- **Avoid:** Invasive payload injection on endpoints not confirmed via recon; test timeout management on long scans
- **Handoff:** Confirmed vulns → `exploitation` for payload crafting
