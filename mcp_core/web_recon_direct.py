"""
mcp_core/web_recon_direct.py

Phase 2 — Direct execution layer for web recon tools.
Covers: web_crawl, url_recon, web_probe, waf_detect, param_discovery.

Usage:
    import mcp_core.web_recon_direct as _web_recon_direct
    result = await loop.run_in_executor(
        None, lambda: _web_recon_direct.web_recon_exec("katana", data)
    )
"""

import logging
from typing import Any, Dict
from pathlib import Path

logger = logging.getLogger(__name__)
import shutil

from server_core.command_executor import execute_command


def _require(data: dict, *keys: str) -> Dict[str, Any]:
    _HINTS = {"url": "use http://host[:port]", "target": "use an IP or hostname", "domain": "use a domain name like example.com"}
    for key in keys:
        if not data.get(key, ""):
            hint = _HINTS.get(key, "")
            msg = f"'{key}' is required" + (f" ({hint})" if hint else "")
            return {"success": False, "error": msg}
    return {}


# ---------------------------------------------------------------------------
# web_crawl/ handlers
# ---------------------------------------------------------------------------

def _katana(data: dict) -> dict:
    err = _require(data, "url")
    if err: return err
    url             = data["url"].strip()
    depth           = data.get("depth", 3)
    js_crawl        = data.get("js_crawl", True)
    form_extraction = data.get("form_extraction", True)
    output_format   = data.get("output_format", "json")
    additional_args = data.get("additional_args", "")

    command = f"katana -u {url} -d {depth}"
    if js_crawl:        command += " -jc"
    if form_extraction: command += " -fx"
    if output_format == "json": command += " -jsonl"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _hakrawler(data: dict) -> dict:
    err = _require(data, "url")
    if err: return err
    url             = data["url"].strip()
    depth           = data.get("depth", 2)
    forms           = data.get("forms", True)
    robots          = data.get("robots", True)
    sitemap         = data.get("sitemap", True)
    wayback         = data.get("wayback", False)
    additional_args = data.get("additional_args", "")

    command = f"echo '{url}' | hakrawler -d {depth}"
    if forms: command += " -s"
    if robots or sitemap or wayback: command += " -subs"
    command += " -u"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


# ---------------------------------------------------------------------------
# url_recon/ handlers
# ---------------------------------------------------------------------------

def _gau(data: dict) -> dict:
    err = _require(data, "domain")
    if err: return err
    domain          = data["domain"].strip()
    providers       = data.get("providers", "wayback,commoncrawl,otx,urlscan")
    include_subs    = data.get("include_subs", True)
    blacklist       = data.get("blacklist", "png,jpg,gif,jpeg,swf,woff,svg,pdf,css,ico")
    additional_args = data.get("additional_args", "")

    command = f"gau {domain}"
    if providers != "wayback,commoncrawl,otx,urlscan":
        command += f" --providers {providers}"
    if include_subs: command += " --subs"
    if blacklist:    command += f" --blacklist {blacklist}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _waybackurls(data: dict) -> dict:
    err = _require(data, "domain")
    if err: return err
    domain          = data["domain"].strip()
    get_versions    = data.get("get_versions", False)
    no_subs         = data.get("no_subs", False)
    additional_args = data.get("additional_args", "")

    command = f"waybackurls {domain}"
    if get_versions: command += " --get-versions"
    if no_subs:      command += " --no-subs"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


# ---------------------------------------------------------------------------
# web_probe/ handler
# ---------------------------------------------------------------------------

def _resolve_httpx() -> str:
    """Return projectdiscovery/httpx path, or empty string if unavailable."""
    # Try PATH first
    p = shutil.which("httpx")
    if p:
        try:
            import subprocess
            r = subprocess.run([p, "-version"], capture_output=True, text=True, timeout=5)
            if "projectdiscovery" in r.stdout.lower():
                return p
        except Exception:
            logger.debug("web_recon_direct: httpx binary check failed", exc_info=True)
    # Fall back to Go install path
    go_path = Path.home() / "go" / "bin" / "httpx"
    if go_path.exists():
        return str(go_path)
    return ""


def _httpx(data: dict) -> dict:
    err = _require(data, "target")
    if err: return err

    httpx_bin = _resolve_httpx()
    if not httpx_bin:
        return {
            "success": False,
            "error": "projectdiscovery/httpx not installed — install with: go install github.com/projectdiscovery/httpx/cmd/httpx@latest",
            "output": "",
            "returncode": -1,
        }

    target          = data["target"].strip()
    probe           = data.get("probe", True)
    tech_detect     = data.get("tech_detect", False)
    status_code     = data.get("status_code", False)
    content_length  = data.get("content_length", False)
    title           = data.get("title", False)
    web_server      = data.get("web_server", False)
    threads         = data.get("threads", 50)
    additional_args = data.get("additional_args", "")

    command = f"{httpx_bin} -u {target} -t {threads}"
    if probe:          command += " -probe"
    if tech_detect:    command += " -tech-detect"
    if status_code:    command += " -sc"
    if content_length: command += " -cl"
    if title:          command += " -title"
    if web_server:     command += " -web-server"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


# ---------------------------------------------------------------------------
# waf_detect/ handler
# ---------------------------------------------------------------------------

def _wafw00f(data: dict) -> dict:
    # Normalize: registry declares "url" but handler uses "target"
    if "target" not in data and "url" in data:
        data["target"] = data["url"]
    err = _require(data, "target")
    if err: return err
    target          = data["target"].strip()
    additional_args = data.get("additional_args", "")

    command = f"wafw00f {target}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


# ---------------------------------------------------------------------------
# param_discovery/ handlers
# ---------------------------------------------------------------------------

def _arjun(data: dict) -> dict:
    err = _require(data, "url")
    if err: return err
    url             = data["url"].strip()
    method          = data.get("method", "GET")
    wordlist        = data.get("wordlist", "")
    delay           = data.get("delay", 0)
    threads         = data.get("threads", 25)
    stable          = data.get("stable", False)
    additional_args = data.get("additional_args", "")

    command = f"arjun -u {url} -m {method} -t {threads}"
    if wordlist:    command += f" -w {wordlist}"
    if delay > 0:   command += f" -d {delay}"
    if stable:      command += " --stable"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _paramspider(data: dict) -> dict:
    err = _require(data, "domain")
    if err: return err
    domain          = data["domain"].strip()
    level           = data.get("level", 2)
    exclude         = data.get("exclude", "png,jpg,gif,jpeg,swf,woff,svg,pdf,css,ico")
    output          = data.get("output", "")
    additional_args = data.get("additional_args", "")

    command = f"paramspider -d {domain} -l {level}"
    if exclude: command += f" --exclude {exclude}"
    if output:  command += f" -o {output}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _x8(data: dict) -> dict:
    err = _require(data, "url")
    if err: return err
    url             = data["url"].strip()
    wordlist        = data.get("wordlist", "/usr/share/wordlists/x8/params.txt")
    method          = data.get("method", "GET")
    body            = data.get("body", "")
    headers         = data.get("headers", "")
    additional_args = data.get("additional_args", "")

    command = f"x8 -u {url} -w {wordlist} -X {method}"
    if body:    command += f" -b '{body}'"
    if headers: command += f" -H '{headers}'"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_HANDLERS = {
    # web_crawl
    "katana":      _katana,
    "hakrawler":   _hakrawler,
    # url_recon
    "gau":         _gau,
    "waybackurls": _waybackurls,
    # web_probe
    "httpx":       _httpx,
    # waf_detect
    "wafw00f":     _wafw00f,
    # param_discovery
    "arjun":       _arjun,
    "paramspider": _paramspider,
    "x8":          _x8,
}


def web_recon_exec(tool: str, data: dict) -> dict:
    """
    Execute a web recon tool directly — no Flask, no HTTP.

    Args:
        tool: tool name (e.g. "katana", "gau", "httpx", "arjun")
        data: parameter dict

    Returns:
        {"success": bool, "output": str, "returncode": int, ...}
    """
    handler = _HANDLERS.get(tool)
    if handler is None:
        return {"success": False, "error": f"Unknown web_recon tool: '{tool}'"}
    return handler(data)
