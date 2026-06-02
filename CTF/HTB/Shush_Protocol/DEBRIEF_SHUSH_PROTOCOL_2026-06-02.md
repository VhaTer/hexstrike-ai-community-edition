# Shush Protocol — HTB CTF Challenge

**Category**: Forensics / Custom Protocol  
**Points**: 4  
**Date**: 2026-06-02  
**Status**: ✅ Solved

## Challenge

PCAP file `traffic.pcap` — Modbus/TCP traffic (port 502) between 192.168.178.105 and 192.168.178.23.

## Approach

1. Downloaded ZIP via curl (Pulse `http_request` returned empty body for binary content)
2. Identified standard pcap format (magic 0xa1b2c3d4, link type 1 = Ethernet)
3. Parsed 110 packets manually: Modbus/TCP with custom function code `0x66`
4. Found flag in Pkt 35 — client writing to server with subfunction `0x10` and length prefix `0x2e`

## Key Observations

- **Protocol**: Modbus TCP (port 502) with a "shush" custom function `0x66`
- **Traffic pattern**: Client opens multiple connections, sends write requests, server responds with padding + `ff 01 00 01 00 01` acknowledgement
- **Repeated polling**: `00 66 00 47` pattern = keep-alive or register polling
- **Flag delivery**: Pkt 35 (`TID=12`) contains the flag in plain ASCII via custom subfunction `0x10` (write)

## Flag

```
HTB{50m371m35_cu570m_p2070c01_423_n07_3n0u9h7}
```

Decoded: "sometimes custom protocol 423 not enough"

## Tools Used

- Bash (curl, python3 for pcap parsing)
- Pulse http_request (failed binary capture — body empty)
- Pulse search_tools (tshark discovery)
- Pulse tshark (typed tool — offline analysis worked via `additional_args`)

## Bugs Found & Fixed (Session 72)

### ✅ Fixed: `http_request` binary body empty

**Root cause**: `misc_direct.py:_http_request()` lisait le fichier body avec `open(bf_path)` en mode texte. Pour un ZIP (binaire), Python échouait à décoder UTF-8 → `except Exception: pass` → body vide.

**Fix**: Lecture en `open(bf_path, "rb")`, tentative UTF-8, fallback hex dans nouveau champ `body_hex`. Le ZIP est maintenant retourné comme hex dans `body_hex`.

- Fichier: `mcp_core/misc_direct.py:757-794`
- Tests: `test_http_request.py` — 10 pass, 0 regressions

### ✅ Fixed: tshark `file` param manquant

**Root cause**: `tool_registry.py` définissait `interface` comme seul paramètre requis, et n'exposait pas `file` pour l'analyse offline PCAP. Le handler `_tshark()` supportait déjà `file` mais le typed tool ne le transmettait pas.

**Fix**: `params` vidé (plus rien de requis), `optional` enrichi avec `interface`, `file`, `filter`, `count`. Les params morts `capture_filter`, `display_filter`, `duration`, `output_file` supprimés.

- Fichier: `tool_registry.py:1362-1370`
- Prochaine connexion Pulse → typed tool tshark accepte `file=/path/to/pcap`

### Not Bug: `binary_block_parser` pcap

Parsing pcap complet avec `binary_block_parser` est théoriquement possible mais nécessiterait des centaines d'appels — pas rentable. Utiliser tshark directement est la bonne approche.
