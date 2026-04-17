# Reference

## Prompt Calls

```python
bug_bounty_recon(target="example.com")
wifi_attack_chain(interface="wlan0", bssid="AA:BB:CC:DD:EE:FF", channel="6")
ctf_web_challenge(url="http://challenge.ctf.local:8080")
smb_lateral_movement(target="10.10.10.10")
cloud_security_audit(provider="aws", profile="default")
```

## Linked Skills

| Prompt | Skills |
|---|---|
| `bug_bounty_recon` | `subdomain-enum`, `nmap-recon`, `web-recon`, `web-vuln` |
| `wifi_attack_chain` | `wifi-pentest` |
| `ctf_web_challenge` | `web-recon`, `web-vuln` |
| `smb_lateral_movement` | `smb-enum`, `active-directory`, `exploitation` |
| `cloud_security_audit` | `cloud-audit` |

## Runtime Helpers

```python
get_tool_skill(tool_name="nmap")
nmap(target="10.10.10.10", ports="22,80,443", scan_type="-sCV")
run_security_tool(tool_name="nmap", parameters='{"target":"10.10.10.10","ports":"22,80,443","flags":"-sV -sC"}')
```
