---
name: cloud-audit
description: Cloud infrastructure, container image, and Kubernetes security assessment via IAM review, secrets detection, and RBAC analysis.
---

# Cloud Audit

## When to use

Use this skill when target is cloud infrastructure rather than traditional network:

- Cloud account security posture (AWS, Azure, GCP, DO)
- Container image CVE and secrets scanning (Docker, OCI)
- Kubernetes cluster RBAC and exposure assessment
- Infrastructure-as-Code (Terraform, CloudFormation) security review
- S3/blob storage misconfiguration detection

## Working Style

**Environment-specific tools prevent wasted effort:**

1. **Identify Platform** — AWS, Azure, GCP, Kubernetes, Docker, etc.
2. **Cloud Account** — `prowler` (AWS/Azure/GCP compliance) or `pacu` (AWS exploitation path)
3. **Container Image** — `trivy` (fast CVE + secrets scan) or `grype` (detailed report)
4. **Kubernetes** — `kube-hunter` (passive), `kubectl` (active RBAC/secret enumeration) — passive only without authorization
5. **IaC Review** — `checkov` or `terrascan` on Terraform/CloudFormation before deployment

**Entry point:**

```python
prowler(provider="aws", profile="default", region="us-east-1")
```

## Notes

- **Effectiveness:** prowler (0.91), trivy (0.95), kube-hunter (0.85), checkov (0.89)
- **Credentials Required:** AWS CLI config, Azure CLI login, or Kubernetes kubeconfig
- **Detection:** Cloud audit is immediately logged in CloudTrail/Activity Log; assume blue team visibility
- **Active Cluster Testing:** `kubectl exec`, pod escape, and RBAC abuse require explicit scope confirmation
- **Secrets in Images:** Use `trivy` with registry scan; private registries require auth credentials
- **Avoid:** Credential harvesting from containers without incident response plan; active Kubernetes exploitation without blue team coordination
