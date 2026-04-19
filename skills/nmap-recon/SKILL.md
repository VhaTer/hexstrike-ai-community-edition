---
name: nmap-recon
description: Network reconnaissance via host discovery, port scanning, and service fingerprinting. Use during early reconnaissance phase to map attack surface and identify services.
---

# Nmap Recon

## When to use

Use this skill in **Phase 1: Reconnaissance** when the objective is network-layer intelligence:

- Identify live hosts on internal/external subnet (ARP, ICMP, TCP)
- Discover open ports and services (UDP/TCP)
- Enumerate service versions and OS fingerprints
- Apply NSE scripts for targeted service interrogation

## Working Style

**Staged scanning minimizes noise and false positives:**

1. **Discovery** — `arp_scan` (subnet), `masscan` (fast port sweep), or `rustscan` (dual-mode)
2. **Enumeration** — `nmap -sCV` on confirmed ports (versions + scripts)
3. **Enrichment** — `nmap_advanced` only on services requiring deeper interrogation (e.g., SMB, LDAP, HTTP)

**Entry point:**

```python
nmap(target="10.10.10.0/24", ports="22,80,443,445", scan_type="-sCV")
```

## Notes

- **Authorization:** Verify scope explicitly before scanning; network scans are immediately detectable
- **Stealth:** Timing templates `-T1` to `-T2` reduce detection; avoid `-T5` in monitored networks
- **Handoff:** Pass live hosts and open ports → `web-recon`, `smb-enum`, or `active-directory`
- **Avoid:** Full port scans before verifying target scope (noise, signatures logged)
