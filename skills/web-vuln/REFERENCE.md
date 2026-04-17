# Reference

## Example Calls

```python
nuclei(target="https://app.example.com", severity="critical,high")
nikto(target="https://app.example.com")
sqlmap(url="https://app.example.com/item?id=1", additional_args="--batch --level=3 --risk=2")
dalfox(url="https://app.example.com/search?q=test")
dotdotpwn(target="app.example.com", additional_args="-m http -o unix")
zap(target="https://app.example.com", scan_type="baseline")
```

## Tool Guide

| Tool | Common parameters | Use |
|---|---|---|
| `nuclei` | `target`, `severity`, `tags` | broad template-based scanning |
| `nikto` | `target` | server misconfiguration and exposure checks |
| `sqlmap` | `url`, `data`, `additional_args` | SQL injection testing |
| `dalfox`, `xsser` | `url`, `additional_args` | XSS testing |
| `dotdotpwn` | `target`, `additional_args` | traversal testing |
| `commix` | `url`, `additional_args` | command injection follow-up |
| `jaeles`, `vulnx`, `zap` | target-specific | broader or framework-aware follow-up |
