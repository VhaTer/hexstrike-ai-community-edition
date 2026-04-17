# Reference

## Example Calls

```python
hashid(hash="8846f7eaee8fb117ad06bdd830b7586c")
john(hash_file="/tmp/hashes.txt", wordlist="/usr/share/wordlists/rockyou.txt")
hashcat(hash_file="/tmp/hashes.txt", hash_type="1000", attack_mode="0", wordlist="/usr/share/wordlists/rockyou.txt")
hydra(target="10.10.10.10", service="ssh", username="admin", password_file="/usr/share/wordlists/rockyou.txt")
medusa(target="10.10.10.10", module="ssh", username="root", password_file="/usr/share/wordlists/rockyou.txt")
```

## Tool Guide

| Tool | Common parameters | Use |
|---|---|---|
| `hashid` | `hash` | format identification |
| `john` | `hash_file`, `wordlist`, `format_type` | general offline cracking |
| `hashcat` | `hash_file`, `hash_type`, `attack_mode`, `wordlist`, `mask` | fast rule or mask attacks |
| `hydra`, `medusa`, `patator` | target, service-specific fields | online brute-force |
| `ophcrack` | `hash_file`, `tables_dir` | NTLM rainbow-table workflow |
