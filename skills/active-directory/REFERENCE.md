# Reference

## Example Calls

```python
ldapdomaindump(hostname="dc01.example.local", username="user@example.local", password="Passw0rd!")
adidnsdump(target="dc01.example.local", username="user@example.local", password="Passw0rd!")
certipy(action="find", target="dc01.example.local", username="user@example.local", password="Passw0rd!")
bloodhound(domain="example.local", username="user@example.local", password="Passw0rd!", dc_ip="10.10.10.1")
mitm6(interface="eth0", domain="example.local")
impacket(script="GetADUsers", target="example.local/", options={"dc-ip": "10.10.10.1", "u": "user@example.local", "p": "Passw0rd!"})
```

## Tool Guide

| Tool | Common parameters | Use |
|---|---|---|
| `ldapdomaindump` | `target`, auth fields | LDAP domain inventory |
| `adidnsdump` | `target`, auth fields | AD-integrated DNS data |
| `certipy`, `certipy_ad` | `target`, auth fields | AD CS enumeration and abuse-path prep |
| `bloodhound`, `bloodhound_python` | `target`, auth fields | graph collection |
| `pywerview`, `impacket` | target-specific | targeted AD follow-up |
| `mitm6` | `interface`, `domain` | explicit poisoning workflow |
