---
name: web-vuln
description: Web vulnerability assessment workflows for HexStrike tools. Use when a web target has already been identified and you need focused scanning for CVEs, injection, XSS, traversal, or framework-specific weaknesses.
---

# Web Vuln

## When to use

Use this skill after recon has produced concrete URLs, parameters, technologies, or candidate weaknesses.

## Working Style

1. start with `nuclei` or `nikto` for broad signal
2. move to targeted tools such as `sqlmap`, `dalfox`, `xsser`, `dotdotpwn`, or `commix`
3. reserve more invasive actions for URLs and parameters that justify them

Preferred entrypoint:

```python
nuclei(target="https://app.example.com", severity="critical,high")
```

Read `REFERENCE.md` before constructing targeted payload-heavy requests.

## Notes

- web recon should feed this skill, not the other way around
- use focused URLs and parameters whenever possible; generic blanket scans produce noise
- if the user is asking for exploitation or payload generation after a confirmed finding, switch to `exploitation`
