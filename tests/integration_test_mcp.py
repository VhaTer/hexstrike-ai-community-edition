#!/usr/bin/env python3
"""Test HexStrike Pulse tools against HTB machines via the MCP protocol.

Usage:
  python3 tests/integration_test_mcp.py --help
  python3 tests/integration_test_mcp.py --target oopsie --profile quick
  python3 tests/integration_test_mcp.py --target oopsie --profile standard
  python3 tests/integration_test_mcp.py --target oopsie --profile full

First, spawn an HTB machine, edit tests/htb_targets.json with its IP, then run.
"""
import asyncio, json, os, sys, time
from fastmcp.client import Client, StreamableHttpTransport

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "htb_targets.json")
SERVER_URL = "http://127.0.0.1:8888/mcp"
TO = 300

results = {"pass": 0, "fail": 0, "skip": 0}


def log(msg):
    print(f"  {msg}")


async def call_tool(client, name: str, args: dict, timeout: float = TO) -> dict:
    t0 = time.time()
    try:
        result = await asyncio.wait_for(
            client.call_tool(name, arguments=args), timeout=timeout,
        )
        return {"tool": name, "success": True, "elapsed": round(time.time() - t0, 1), "result": result}
    except asyncio.TimeoutError:
        return {"tool": name, "success": False, "elapsed": round(time.time() - t0, 1), "error": "TIMEOUT"}
    except Exception as e:
        return {"tool": name, "success": False, "elapsed": round(time.time() - t0, 1), "error": str(e)[:200]}


def report(r: dict):
    status = "PASS" if r["success"] else "FAIL"
    print(f"  [{status}] {r['tool']:25s} ({r['elapsed']:6.1f}s)", end="")
    if not r["success"]:
        print(f"  {r.get('error', '?')[:120]}")
        results["fail"] += 1
    else:
        results["pass"] += 1
        snippet = ""
        ct = r.get("result")
        if ct and hasattr(ct, "content") and ct.content:
            for c in ct.content:
                text = getattr(c, "text", "")
                if text:
                    snippet += text[:200]
        if snippet:
            print(f"  -> {snippet[:180].replace(chr(10), ' ')}")
        else:
            print()


def print_header(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def load_config():
    if not os.path.exists(CONFIG_PATH):
        log(f"❌ Config not found: {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return json.load(f)


def resolve_target(config: dict, name: str) -> tuple:
    for m in config.get("machines", []):
        if m["name"] == name:
            ip = m.get("ip", "").strip()
            if not ip:
                log(f"❌ Machine '{name}' has no IP set. Edit tests/htb_targets.json")
                sys.exit(1)
            return m, ip
    log(f"❌ Unknown target '{name}'. Available: {[m['name'] for m in config.get('machines', [])]}")
    sys.exit(1)


def print_usage(config: dict):
    machines = [m["name"] for m in config.get("machines", [])]
    profiles = list(config.get("test_profiles", {}).keys())
    print(f"Usage: python3 {sys.argv[0]} --target <name> [--profile <name>]")
    print(f"  Targets: {', '.join(machines)}")
    print(f"  Profiles: {', '.join(profiles)} (default: quick)")
    print(f"\n  Edit tests/htb_targets.json to set machine IPs before running.")
    sys.exit(0)


async def main():
    config = load_config()

    if "--help" in sys.argv or "-h" in sys.argv:
        print_usage(config)

    # Parse args
    target_name = None
    profile_name = "quick"
    args_iter = iter(sys.argv[1:])
    for arg in args_iter:
        if arg == "--target":
            target_name = next(args_iter)
        elif arg == "--profile":
            profile_name = next(args_iter)

    if not target_name:
        print_usage(config)

    if profile_name not in config.get("test_profiles", {}):
        log(f"❌ Unknown profile '{profile_name}'. Available: {list(config['test_profiles'].keys())}")
        sys.exit(1)

    machine, ip = resolve_target(config, target_name)
    profile = config["test_profiles"][profile_name]
    timeout = profile.get("timeout", 120)
    tools_requested = profile.get("tools", [])
    services = machine.get("services", [])
    machine_name = machine["name"]
    web_url = machine.get("web_url") or f"http://{ip}"

    print(f"{'='*62}")
    print(f"  HexStrike Pulse — Test: {target_name} @ {ip}")
    print(f"  Profile: {profile_name} | Timeout: {timeout}s")
    print(f"  Tools: {len(tools_requested)} requested")
    print(f"{'='*62}\n")

    async with Client(StreamableHttpTransport(SERVER_URL)) as client:
        await client.initialize()
        log("MCP session initialized\n")

        # Phase 0: Validate environment — skip missing tools
        print_header("Phase 0 — Validate environment")
        log("validate_environment...")
        env = await call_tool(client, "validate_environment", {})
        report(env)
        if not env["success"]:
            log("❌ validate_environment failed — cannot proceed")
            sys.exit(1)

        env_result = env.get("result")
        tool_status = {}
        if env_result and hasattr(env_result, "content") and env_result.content:
            for c in env_result.content:
                text = getattr(c, "text", "")
                if text:
                    try:
                        data = json.loads(text)
                        for t_name, t_info in data.get("tools", {}).items():
                            tool_status[t_name] = t_info.get("present", False)
                    except (json.JSONDecodeError, TypeError):
                        pass

        available = [t for t in tools_requested if tool_status.get(t, False)]
        missing = [t for t in tools_requested if not tool_status.get(t, False)]
        log(f"Available: {len(available)}  Missing: {len(missing)}")
        if missing:
            log(f"Skipping: {', '.join(missing)}")

        if not available:
            log("❌ No tools available — check your installation")
            results["skip"] = len(tools_requested)
            sys.exit(1)
        print()

        # Phase 1: Recon
        recon_tools = [t for t in available if t in ("whatweb", "httpx", "wafw00f", "whois", "testssl")]
        if recon_tools:
            print_header("Phase 1 — Reconnaissance")
            cmds = []
            if "whatweb" in recon_tools:
                cmds.append(("whatweb", {"url": web_url}, timeout))
            if "httpx" in recon_tools:
                cmds.append(("httpx", {"target": ip}, timeout))
            if "wafw00f" in recon_tools:
                cmds.append(("wafw00f", {"url": web_url}, timeout))
            if "whois" in recon_tools:
                cmds.append(("whois", {"target": ip}, timeout))
            if "testssl" in recon_tools:
                cmds.append(("testssl", {"target": ip}, timeout))
            for label, args, to in cmds:
                log(f"{label}...")
                r = await call_tool(client, label, args, timeout=to)
                report(r)

        # Phase 2: Port scan
        scan_tools = [t for t in available if t in ("nmap", "masscan", "rustscan", "arp_scan")]
        if scan_tools:
            print_header("Phase 2 — Port scanning")
            if "nmap" in scan_tools:
                log("nmap -sV (top 100 ports)...")
                r = await call_tool(client, "nmap", {
                    "target": ip, "scan_type": "-sV", "additional_args": "-T4 -Pn --top-ports 100",
                }, timeout=min(timeout * 2, 300))
                report(r)
            if "masscan" in scan_tools and profile_name != "quick":
                log("masscan...")
                r = await call_tool(client, "masscan", {
                    "target": ip, "ports": "80,443,22,21,3306,8080,445,139", "rate": "500",
                }, timeout=120)
                report(r)

        # Phase 3: Web scanning (only if http service or web_url)
        web_tools = [t for t in available if t in (
            "gobuster", "feroxbuster", "nikto", "xsser", "dirsearch", "ffuf", "nuclei",
        )]
        if web_tools and ("http" in services or profile_name != "quick"):
            print_header("Phase 3 — Web scanning")
            if "gobuster" in web_tools:
                log("gobuster...")
                r = await call_tool(client, "gobuster", {
                    "url": web_url, "wordlist": "/usr/share/wordlists/dirb/common.txt",
                }, timeout=timeout)
                report(r)
            if "feroxbuster" in web_tools and profile_name != "quick":
                log("feroxbuster...")
                r = await call_tool(client, "feroxbuster", {
                    "url": web_url, "wordlist": "/usr/share/wordlists/dirb/common.txt", "threads": 5,
                }, timeout=timeout)
                report(r)
            if "nikto" in web_tools:
                log("nikto...")
                r = await call_tool(client, "nikto", {"target": web_url}, timeout=timeout)
                report(r)
            if "dirsearch" in web_tools and profile_name != "quick":
                log("dirsearch...")
                r = await call_tool(client, "dirsearch", {"url": web_url}, timeout=timeout)
                report(r)
            if "ffuf" in web_tools and profile_name != "quick":
                log("ffuf (small wordlist)...")
                r = await call_tool(client, "ffuf", {
                    "url": f"{web_url}/FUZZ", "wordlist": "/usr/share/wordlists/dirb/small.txt",
                }, timeout=min(timeout * 2, 300))
                report(r)

        # Phase 4: Vulnerability scanning
        vuln_tools = [t for t in available if t in ("nuclei", "searchsploit")]
        if vuln_tools:
            print_header("Phase 4 — Vulnerability scanning")
            if "nuclei" in vuln_tools:
                sev = "critical" if profile_name == "quick" else "critical,high"
                nuc_timeout = min(timeout, 120)
                log(f"nuclei (severity={sev}, tags=cve, timeout={nuc_timeout}s)...")
                r = await call_tool(client, "nuclei", {
                    "target": web_url, "severity": sev, "tags": "cve", "timeout": nuc_timeout,
                }, timeout=nuc_timeout + 10)
                report(r)
            if "searchsploit" in vuln_tools:
                log("searchsploit...")
                r = await call_tool(client, "searchsploit", {"query": "apache"}, timeout=60)
                report(r)

        # Phase 5: SMB recon
        smb_tools = [t for t in available if t in ("nbtscan", "enum4linux", "smbmap")]
        if smb_tools and ("smb" in services or profile_name != "quick"):
            print_header("Phase 5 — SMB recon")
            if "nbtscan" in smb_tools:
                log("nbtscan...")
                r = await call_tool(client, "nbtscan", {"target": ip}, timeout=timeout)
                report(r)
            if "enum4linux" in smb_tools:
                log("enum4linux...")
                r = await call_tool(client, "enum4linux", {"target": ip}, timeout=timeout)
                report(r)
            if "smbmap" in smb_tools:
                log("smbmap...")
                r = await call_tool(client, "smbmap", {"target": ip}, timeout=timeout)
                report(r)

        # Results
        total = results["pass"] + results["fail"] + results["skip"]
        print(f"\n{'='*62}")
        print(f"  FINAL: PASS {results['pass']}  FAIL {results['fail']}  SKIP {results['skip']}  TOTAL {total}")
        print(f"{'='*62}")
        if results["fail"] > 0:
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
