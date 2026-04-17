# Reference

## Example Calls

```python
subfinder(domain="example.com")
amass(domain="example.com")
theharvester(domain="example.com", additional_args="-b all")
dnsenum(domain="example.com")
fierce(domain="example.com")
```

## Tool Guide

| Tool | Common parameters | Use |
|---|---|---|
| `subfinder` | `domain` | fast passive enumeration |
| `amass` | `domain` | deeper graph-heavy discovery |
| `theharvester` | `domain`, `source` | public source harvesting |
| `dnsenum` | `domain` | DNS and zone-transfer style checks |
| `fierce` | `domain` | DNS brute-force expansion |
