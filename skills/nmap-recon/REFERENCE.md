# Reference

## Entry Points

Use `get_tool_skill(tool_name="nmap")` to fetch this bundle at runtime.

Prefer the typed MCP tools for execution:

```python
rustscan(target="10.10.10.10", additional_args="--range 1-65535")
nmap(target="10.10.10.10", ports="22,80,443", scan_type="-sCV", additional_args="-O")
nmap_advanced(target="10.10.10.10", ports="445", nse_scripts="smb-enum-shares,smb-vuln-*")
arp_scan(target="192.168.1.0/24")
```

Fallback:

```python
run_security_tool(tool_name="nmap", parameters='{"target":"10.10.10.10","scan_type":"-sV -sC","ports":"22,80,443","additional_args":"-O"}')
```

## Tool Guide

| Tool | Common parameters | Use |
|---|---|---|
| `rustscan` | `target`, `ports` | fast single-target discovery |
| `masscan` | `target`, `ports`, `rate` | high-speed range scanning |
| `nmap` | `target`, `scan_type`, `ports`, `additional_args` | version, default script, and OS detection |
| `nmap_advanced` | `target`, `ports`, `scripts` | targeted NSE follow-up |
| `arp_scan` | `target` | local subnet host discovery |
