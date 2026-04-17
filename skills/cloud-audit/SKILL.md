---
name: cloud-audit
description: Cloud, container, and Kubernetes security workflows for HexStrike tools. Use when the target is a cloud account, image, cluster, or IaC project rather than a traditional host scan.
---

# Cloud Audit

## When to use

Use this skill for:

- cloud posture review
- container image or filesystem CVE scanning
- Kubernetes exposure assessment
- IaC security checks

## Working Style

1. identify the environment first: cloud account, image, cluster, or IaC repo
2. use the environment-specific tool instead of forcing one scanner onto everything
3. keep active cluster testing explicit

Preferred entrypoint:

```python
prowler(provider="aws", profile="default")
```

See `REFERENCE.md` for common cloud and container calls.

## Notes

- some tools require local credentials or cloud CLI configuration before use
- active Kubernetes probing should be treated as higher risk than passive config review
