# mcp_tools/recon/autorecon.py

from typing import Dict, Any
from fastmcp import Context

def register_autorecon_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def autorecon_scan(
        ctx: Context,
        target: str = "",
        target_file: str = "",
        ports: str = "",
        output_dir: str = "",
        max_scans: str = "",
        max_port_scans: str = "",
        heartbeat: str = "",
        timeout: str = "",
        target_timeout: str = "",
        config_file: str = "",
        global_file: str = "",
        plugins_dir: str = "",
        add_plugins_dir: str = "",
        tags: str = "",
        exclude_tags: str = "",
        port_scans: str = "",
        service_scans: str = "",
        reports: str = "",
        single_target: bool = False,
        only_scans_dir: bool = False,
        no_port_dirs: bool = False,
        nmap: str = "",
        nmap_append: str = "",
        proxychains: bool = False,
        disable_sanity_checks: bool = False,
        disable_keyboard_control: bool = False,
        force_services: str = "",
        accessible: bool = False,
        verbose: int = 0,
        curl_path: str = "",
        dirbuster_tool: str = "",
        dirbuster_wordlist: str = "",
        dirbuster_threads: str = "",
        dirbuster_ext: str = "",
        onesixtyone_community_strings: str = "",
        global_username_wordlist: str = "",
        global_password_wordlist: str = "",
        global_domain: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Full-featured AutoRecon scan with complete parameter control.

        Workflow position: comprehensive automated recon — orchestrates nmap,
        web enumeration, service-specific scripts, and report generation
        in a single automated workflow.

        Use autorecon_comprehensive for quick scans.
        Use this tool when you need fine-grained control over plugins,
        wordlists, concurrency, or proxychains.

        Key parameters:
        - target: single IP/hostname (mutually exclusive with target_file)
        - target_file: file with one target per line (for batch scanning)
        - ports: specific ports to scan (default: autorecon decides)
        - output_dir: results directory (default: ./results/<target>)
        - max_scans: max concurrent scans (default: 50)
        - timeout: global timeout in seconds
        - tags/exclude_tags: filter which plugins run (e.g. 'default', 'safe')
        - proxychains: route all traffic through proxychains
        - verbose: 0=quiet, 1=verbose, 2=very verbose, 3=debug

        Prerequisites: autorecon installed — it orchestrates many sub-tools
        (nmap, gobuster, nikto, etc.) which must also be installed.

        Output: structured results directory with:
            results/<target>/scans/     — raw tool output
            results/<target>/report/    — formatted reports
        """
        data = {
            "target": target, "target_file": target_file, "ports": ports,
            "output_dir": output_dir, "max_scans": max_scans,
            "max_port_scans": max_port_scans, "heartbeat": heartbeat,
            "timeout": timeout, "target_timeout": target_timeout,
            "config_file": config_file, "global_file": global_file,
            "plugins_dir": plugins_dir, "add_plugins_dir": add_plugins_dir,
            "tags": tags, "exclude_tags": exclude_tags,
            "port_scans": port_scans, "service_scans": service_scans,
            "reports": reports, "single_target": single_target,
            "only_scans_dir": only_scans_dir, "no_port_dirs": no_port_dirs,
            "nmap": nmap, "nmap_append": nmap_append,
            "proxychains": proxychains,
            "disable_sanity_checks": disable_sanity_checks,
            "disable_keyboard_control": disable_keyboard_control,
            "force_services": force_services, "accessible": accessible,
            "verbose": verbose, "curl_path": curl_path,
            "dirbuster_tool": dirbuster_tool,
            "dirbuster_wordlist": dirbuster_wordlist,
            "dirbuster_threads": dirbuster_threads,
            "dirbuster_ext": dirbuster_ext,
            "onesixtyone_community_strings": onesixtyone_community_strings,
            "global_username_wordlist": global_username_wordlist,
            "global_password_wordlist": global_password_wordlist,
            "global_domain": global_domain, "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting autorecon: {target or target_file}")
        result = hexstrike_client.safe_post("api/tools/autorecon", data)
        if result.get("success"):
            await ctx.info(f"✅ autorecon completed for {target or target_file}")
        else:
            await ctx.error(f"❌ autorecon failed for {target or target_file}")
        return result

    @mcp.tool()
    async def autorecon_comprehensive(
        ctx: Context,
        target: str,
        output_dir: str = "/tmp/autorecon",
        port_scans: str = "top-100-ports",
        service_scans: str = "default",
        heartbeat: int = 60,
        timeout: int = 300,
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Quick AutoRecon scan with sensible defaults — automated recon in one command.

        Workflow position: comprehensive recon after initial host discovery.
        Ideal for CTF targets or pentest scoping — runs nmap, web enum,
        and service-specific checks automatically.

        Use this for most cases. Use autorecon_scan for fine-grained control.

        Parameters:
        - target: IP address or hostname (e.g. '10.10.10.1')
        - output_dir: results directory (default: /tmp/autorecon)
        - port_scans: port scan profile ('top-100-ports', 'top-1000-ports', 'all-ports')
        - service_scans: service scan profile ('default', 'safe', 'thorough')
        - heartbeat: status update interval in seconds (default 60)
        - timeout: per-scan timeout in seconds (default 300)
        - additional_args: extra autorecon flags

        Prerequisites: autorecon + sub-tools installed (nmap, gobuster, nikto, etc.)

        Output: structured results in output_dir/<target>/
            scans/   — raw nmap XML, gobuster output, nikto reports
            report/  — formatted HTML/markdown report

        Typical CTF/pentest sequence:
            1. autorecon_comprehensive(target='10.10.10.1')  — full automated recon
            2. Review output_dir/10.10.10.1/report/
            3. Follow up with targeted tools based on discovered services
        """
        data = {
            "target": target,
            "output_dir": output_dir,
            "port_scans": port_scans,
            "service_scans": service_scans,
            "heartbeat": heartbeat,
            "timeout": timeout,
            "additional_args": additional_args
        }
        await ctx.info(f"🔄 Starting autorecon comprehensive: {target}")
        result = hexstrike_client.safe_post("api/tools/autorecon", data)
        if result.get("success"):
            await ctx.info(f"✅ autorecon completed for {target}")
        else:
            await ctx.error(f"❌ autorecon failed for {target}")
        return result
