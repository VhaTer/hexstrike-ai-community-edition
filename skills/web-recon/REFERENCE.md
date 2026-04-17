# Reference

## Example Calls

```python
wafw00f(url="https://app.example.com")
httpx(target="app.example.com", probe=True, tech_detect=True, status_code=True, title=True)
ffuf(url="https://app.example.com/FUZZ", wordlist="/usr/share/wordlists/dirb/common.txt", match_codes="200,204,301,302,307,401,403")
katana(url="https://app.example.com")
arjun(url="https://app.example.com/api/search")
testssl(target="app.example.com:443", vulnerable=True, headers=True)
```

## Tool Guide

| Tool | Common parameters | Use |
|---|---|---|
| `wafw00f` | `url` | WAF detection |
| `httpx` | `target`, `probe`, `tech_detect` | probing and fingerprinting |
| `ffuf` | `url`, `wordlist`, `match_codes` | fast fuzzing and vhost discovery |
| `feroxbuster` | `url`, `wordlist`, `threads` | recursive content discovery |
| `gobuster` | `url`, `mode`, `wordlist`, `additional_args` | extension-aware enumeration |
| `katana` | `url` | crawler-driven expansion |
| `gau`, `waybackurls` | `domain` or `target` | historical URLs |
| `arjun`, `paramspider`, `x8` | `url` or `target` | parameter discovery |
| `whatweb`, `testssl`, `joomscan`, `wpscan` | target-specific | lightweight fingerprinting or CMS follow-up |
