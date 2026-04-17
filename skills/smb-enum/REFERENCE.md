# Reference

## Example Calls

```python
nbtscan(target="10.10.10.10")
enum4linux(target="10.10.10.10")
smbmap(target="10.10.10.10")
rpcclient(target="10.10.10.10", username="guest", password="")
netexec(target="10.10.10.10", protocol="smb", username="user", password="Passw0rd!")
```

## Tool Guide

| Tool | Common parameters | Use |
|---|---|---|
| `nbtscan` | `target` | host naming and NetBIOS discovery |
| `enum4linux` | `target` | broad anonymous SMB enumeration |
| `smbmap` | `target`, auth fields | share visibility and access checks |
| `rpcclient` | `target`, `username`, `password` | RPC follow-up |
| `netexec` | `target`, `service`, auth fields | credential validation and targeted SMB ops |
