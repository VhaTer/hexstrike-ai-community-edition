#!/usr/bin/env python3
"""Test HexStrike Pulse tools against real targets via MCP protocol.

Targets:
  - HTB machine (network/SMB/exploit)
  - testphp.vulnweb.com (web tools)
"""
import asyncio, json, sys, time
from fastmcp import FastMCP
from fastmcp.client import Client, StreamableHttpTransport

HTB_TARGET = "10.129.95.191"
WEB_TARGET = "http://testphp.vulnweb.com"
SERVER_URL = "http://127.0.0.1:8888/mcp"
TO = 300  # default tool timeout (seconds)

results = {"pass": 0, "fail": 0, "skip": 0}


async def test_tool(client: Client, name: str, args: dict, timeout: float = TO) -> dict:
    t0 = time.time()
    try:
        result = await asyncio.wait_for(
            client.call_tool(name, arguments=args),
            timeout=timeout,
        )
        elapsed = round(time.time() - t0, 1)
        return {"tool": name, "success": True, "elapsed": elapsed, "result": result}
    except asyncio.TimeoutError:
        elapsed = round(time.time() - t0, 1)
        return {"tool": name, "success": False, "elapsed": elapsed, "error": f"TIMEOUT after {elapsed}s"}
    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        return {"tool": name, "success": False, "elapsed": elapsed, "error": str(e)[:200]}


def report(r: dict):
    status = "PASS" if r["success"] else "FAIL"
    print(f"  [{status}] {r['tool']:25s} ({r['elapsed']:6.1f}s)")
    if not r["success"]:
        print(f"          Error: {r.get('error', 'unknown')[:200]}")
        results["fail"] += 1
    else:
        results["pass"] += 1
        ct = r.get("result", None)
        if ct and hasattr(ct, 'content') and ct.content:
            snippet = "".join(c.text[:200] for c in ct.content if hasattr(c, 'text') and c.text)
            if snippet:
                print(f"          -> {snippet[:200].replace(chr(10), ' ')}")


def print_header(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


async def run_phase(client, phase_name, tests):
    print_header(f"Phase: {phase_name}")
    for label, tool, args, to in tests:
        print(f"  {label}...")
        r = await test_tool(client, tool, args, timeout=to)
        report(r)


async def main():
    # Determine targets from args (default: both HTB and WEB)
    targets = set(sys.argv[1:]) if len(sys.argv) > 1 else {"htb", "web"}
    htb = "htb" in targets
    web = "web" in targets

    print(f"{'='*62}")
    print(f"  HexStrike Pulse — Integration Test Suite")
    if htb: print(f"  HTB target: {HTB_TARGET}")
    if web: print(f"  Web target: {WEB_TARGET}")
    print(f"{'='*62}\n")

    async with Client(StreamableHttpTransport(SERVER_URL)) as client:
        await client.initialize()
        print("  MCP session initialized\n")

        # =====================================================================
        # HTB TARGET: network recon, SMB, exploit search
        # =====================================================================
        if htb:
            target = HTB_TARGET
            web_url = f"http://{target}"

            # Phase 1: Fast recon
            await run_phase(client,
                "1a — Fast recon (HTB)", [
                    ("whatweb (web fingerprint)", "whatweb", {"url": web_url}, 60),
                    ("httpx (HTTP probe)", "httpx", {"target": target}, 60),
                    ("wafw00f (WAF detect)", "wafw00f", {"url": web_url}, 60),
                    ("whois (domain lookup)", "whois", {"target": target}, 60),
                    ("testssl (SSL check)", "testssl", {"target": target}, 60),
                ]
            )

            # Phase 1b: Port scan
            await run_phase(client,
                "1b — Port scan (HTB)", [
                    ("nmap -sV (service scan)", "nmap", {"target": target, "scan_type": "-sV", "additional_args": "-T4 -Pn --top-ports 100"}, 180),
                    ("masscan (fast port scan)", "masscan", {"target": target, "ports": "80,443,22,21,3306,8080,445,139", "rate": "500"}, 120),
                ]
            )

            # Phase 2: Web scanning (HTB)
            await run_phase(client,
                "2 — Web scan (HTB)", [
                    ("gobuster (dir bust)", "gobuster", {"url": web_url, "wordlist": "/usr/share/wordlists/dirb/common.txt"}, 180),
                    ("feroxbuster (recursive content)", "feroxbuster", {"url": web_url, "wordlist": "/usr/share/wordlists/dirb/common.txt", "threads": 5}, 180),
                    ("nikto (web vuln scan)", "nikto", {"target": web_url}, 180),
                    ("xsser (XSS scan)", "xsser", {"url": web_url}, 120),
                    ("ffuf (fuzzing)", "ffuf", {"url": f"{web_url}/FUZZ", "wordlist": "/usr/share/wordlists/dirb/small.txt"}, 240),
                    ("dirsearch (dir search)", "dirsearch", {"url": web_url}, 120),
                ]
            )

            # Phase 3: SMB recon (HTB)
            await run_phase(client,
                "3 — SMB recon (HTB)", [
                    ("nbtscan", "nbtscan", {"target": target}, 60),
                    ("enum4linux", "enum4linux", {"target": target}, 120),
                    ("smbmap", "smbmap", {"target": target}, 60),
                ]
            )

            # Phase 4a: Nuclei + Searchsploit (HTB)
            await run_phase(client,
                "4a — Vuln scan + Exploit search (HTB)", [
                    ("nuclei (vuln scanner)", "nuclei", {"target": web_url, "severity": "low"}, 300),
                    ("searchsploit (exploit search)", "searchsploit", {"query": "apache 2.4"}, 60),
                ]
            )

        # =====================================================================
        # WEB TARGET (testphp.vulnweb.com): full web tool suite
        # =====================================================================
        if web:
            # Phase 1w: Fast web recon
            await run_phase(client,
                "1w — Fast web recon (testphp)", [
                    ("whatweb (fingerprint)", "whatweb", {"url": WEB_TARGET}, 60),
                    ("httpx (HTTP probe)", "httpx", {"target": WEB_TARGET}, 60),
                    ("wafw00f (WAF detect)", "wafw00f", {"url": WEB_TARGET}, 60),
                ]
            )

            # Phase 2w: Web scanning
            await run_phase(client,
                "2w — Web scan (testphp)", [
                    ("gobuster (dir bust)", "gobuster", {"url": WEB_TARGET, "wordlist": "/usr/share/wordlists/dirb/small.txt"}, 120),
                    ("feroxbuster (recursive content)", "feroxbuster", {"url": WEB_TARGET, "wordlist": "/usr/share/wordlists/dirb/small.txt", "threads": 5}, 120),
                    ("nikto (web vuln scan)", "nikto", {"target": WEB_TARGET}, 180),
                    ("xsser (XSS scan)", "xsser", {"url": WEB_TARGET}, 120),
                    ("dirsearch (dir search)", "dirsearch", {"url": WEB_TARGET}, 120),
                    ("ffuf (fuzzing)", "ffuf", {"url": f"{WEB_TARGET}/FUZZ", "wordlist": "/usr/share/wordlists/dirb/small.txt"}, 240),
                ]
            )

            # Phase 3w: Vuln scanning
            await run_phase(client,
                "3w — Vuln scan (testphp)", [
                    ("nuclei (vuln scanner)", "nuclei", {"target": WEB_TARGET, "severity": "low"}, 300),
                    ("nuclei (critical vulns)", "nuclei", {"target": WEB_TARGET, "severity": "critical"}, 180),
                ]
            )

            # Phase 4w: Exploit search
            await run_phase(client,
                "4w — Exploit search", [
                    ("searchsploit (PHP exploits)", "searchsploit", {"query": "PHP"}, 60),
                    ("searchsploit (SQLi exploits)", "searchsploit", {"query": "SQL injection"}, 60),
                ]
            )

        # =====================================================================
        # PHASE 5: Cache key validation + plan_attack chain
        # =====================================================================
        # Validate that calling the same tool twice with same args uses cache
        if htb:
            print_header("Phase 5 — Cache key validation (HTB)")
            print("  whatweb (first call)...")
            r1 = await test_tool(client, "whatweb", {"url": f"http://{HTB_TARGET}"}, timeout=60)
            report(r1)
            print("  whatweb (second call — should be cached)...")
            r2 = await test_tool(client, "whatweb", {"url": f"http://{HTB_TARGET}"}, timeout=60)
            report(r2)
            if r1["success"] and r2["success"]:
                print(f"          First: {r1['elapsed']}s, Second: {r2['elapsed']}s — {'CACHED ' if r2['elapsed'] < r1['elapsed'] * 0.5 else 'not cached '}")
            print()

        # =====================================================================
        # RESULTS
        # =====================================================================
        total = results["pass"] + results["fail"] + results["skip"]
        print(f"{'='*62}")
        print(f"  FINAL RESULTS:  PASS {results['pass']}  FAIL {results['fail']}  SKIP {results['skip']}  TOTAL {total}")
        print(f"{'='*62}")
        if results["fail"] > 0:
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
