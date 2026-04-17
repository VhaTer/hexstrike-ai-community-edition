# Reference

## Example Calls

```python
prowler(provider="aws", profile="default")
trivy(target="nginx:latest", scan_type="image", severity="HIGH,CRITICAL")
kube_hunter(additional_args="--remote 10.10.10.20")
kube_bench()
checkov(directory="/workspace/iac")
terrascan(iac_dir="/workspace/iac")
```

## Tool Guide

| Tool | Common parameters | Use |
|---|---|---|
| `prowler` | `provider`, `profile`, `region`, `checks` | cloud account posture |
| `trivy` | `target`, `scan_type`, `severity` | image or filesystem CVEs |
| `kube_hunter` | `additional_args` | cluster exposure testing |
| `kube_bench` | environment-specific | CIS-style Kubernetes checks |
| `checkov`, `terrascan` | `target` | IaC scanning |
