"""
mcp_core/security_direct.py

Phase 2 — Direct execution layer for cloud/container/k8s/iac security tools.
Covers: cloud_audit, cloud_exploit, cloud_visual, container_scan, k8s_scan, iac_scan.

Usage:
    import mcp_core.security_direct as _security_direct
    result = await loop.run_in_executor(
        None, lambda: _security_direct.security_exec("trivy", data)
    )
"""

import os
from pathlib import Path
from typing import Any, Dict
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
# cloud_audit/
# ---------------------------------------------------------------------------

def _prowler(data: dict) -> dict:
    provider        = data.get("provider", "aws")
    profile         = data.get("profile", "default")
    region          = data.get("region", "")
    checks          = data.get("checks", "")
    output_dir      = data.get("output_dir", "/tmp/prowler_output")
    output_format   = data.get("output_format", "json")
    additional_args = data.get("additional_args", "")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    command = f"prowler {provider}"
    if profile:         command += f" --profile {profile}"
    if region:          command += f" --region {region}"
    if checks:          command += f" --checks {checks}"
    command += f" --output-directory {output_dir} --output-format {output_format}"
    if additional_args: command += f" {additional_args}"

    result = execute_command(command)
    result["output_directory"] = output_dir
    return result


def _scout_suite(data: dict) -> dict:
    provider        = data.get("provider", "aws")
    profile         = data.get("profile", "default")
    report_dir      = data.get("report_dir", "/tmp/scout-suite")
    services        = data.get("services", "")
    exceptions      = data.get("exceptions", "")
    additional_args = data.get("additional_args", "")

    Path(report_dir).mkdir(parents=True, exist_ok=True)

    command = f"scout {provider}"
    if profile and provider == "aws": command += f" --profile {profile}"
    if services:    command += f" --services {services}"
    if exceptions:  command += f" --exceptions {exceptions}"
    command += f" --report-dir {report_dir}"
    if additional_args: command += f" {additional_args}"

    result = execute_command(command)
    result["report_directory"] = report_dir
    return result


# ---------------------------------------------------------------------------
# cloud_exploit/
# ---------------------------------------------------------------------------

def _pacu(data: dict) -> dict:
    session_name    = data.get("session_name", "hexstrike_session")
    modules         = data.get("modules", "")
    data_services   = data.get("data_services", "")
    regions         = data.get("regions", "")
    additional_args = data.get("additional_args", "")

    commands = [f"set_session {session_name}"]
    if data_services: commands.append(f"data {data_services}")
    if regions:       commands.append(f"set_regions {regions}")
    if modules:
        for module in modules.split(","):
            commands.append(f"run {module.strip()}")
    commands.append("exit")

    command_file = "/tmp/pacu_commands.txt"
    with open(command_file, "w") as f:
        f.write("\n".join(commands))

    command = f"pacu < {command_file}"
    if additional_args: command += f" {additional_args}"

    result = execute_command(command)
    try:
        os.remove(command_file)
    except Exception:
        pass
    return result


def _cloudmapper(data: dict) -> dict:
    action          = data.get("action", "collect")
    account         = data.get("account", "")
    config          = data.get("config", "config.json")
    additional_args = data.get("additional_args", "")

    if not account and action != "webserver":
        return {"success": False, "error": "account is required for most actions"}

    command = f"cloudmapper {action}"
    if account:         command += f" --account {account}"
    if config:          command += f" --config {config}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


# ---------------------------------------------------------------------------
# container_scan/
# ---------------------------------------------------------------------------

def _trivy(data: dict) -> dict:
    err = _require(data, "target")
    if err: return err
    target          = data["target"].strip()
    scan_type       = data.get("scan_type", "image")
    output_format   = data.get("output_format", "json")
    severity        = data.get("severity", "")
    output_file     = data.get("output_file", "")
    additional_args = data.get("additional_args", "")

    command = f"trivy {scan_type} {target}"
    if output_format: command += f" --format {output_format}"
    if severity:      command += f" --severity {severity}"
    if output_file:   command += f" --output {output_file}"
    if additional_args: command += f" {additional_args}"

    result = execute_command(command)
    if output_file: result["output_file"] = output_file
    return result


def _docker_bench(data: dict) -> dict:
    checks          = data.get("checks", "")
    exclude         = data.get("exclude", "")
    output_file     = data.get("output_file", "/tmp/docker-bench-results.json")
    additional_args = data.get("additional_args", "")

    command = "docker-bench-security"
    if checks:       command += f" -c {checks}"
    if exclude:      command += f" -e {exclude}"
    if output_file:  command += f" -l {output_file}"
    if additional_args: command += f" {additional_args}"

    result = execute_command(command)
    result["output_file"] = output_file
    return result


def _clair(data: dict) -> dict:
    err = _require(data, "image")
    if err: return err
    image           = data["image"].strip()
    config          = data.get("config", "/etc/clair/config.yaml")
    output_format   = data.get("output_format", "json")
    additional_args = data.get("additional_args", "")

    command = f"clairctl analyze {image}"
    if config:        command += f" --config {config}"
    if output_format: command += f" --format {output_format}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


# ---------------------------------------------------------------------------
# k8s_scan/
# ---------------------------------------------------------------------------

def _kube_hunter(data: dict) -> dict:
    target          = data.get("target", "")
    remote          = data.get("remote", "")
    cidr            = data.get("cidr", "")
    interface       = data.get("interface", "")
    active          = data.get("active", False)
    report          = data.get("report", "json")
    additional_args = data.get("additional_args", "")

    command = "kube-hunter"
    if target:     command += f" --remote {target}"
    elif remote:   command += f" --remote {remote}"
    elif cidr:     command += f" --cidr {cidr}"
    elif interface: command += f" --interface {interface}"
    else:          command += " --pod"

    if active:  command += " --active"
    if report:  command += f" --report {report}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _kube_bench(data: dict) -> dict:
    targets         = data.get("targets", "")
    version         = data.get("version", "")
    config_dir      = data.get("config_dir", "")
    output_format   = data.get("output_format", "json")
    additional_args = data.get("additional_args", "")

    command = "kube-bench"
    if targets:     command += f" --targets {targets}"
    if version:     command += f" --version {version}"
    if config_dir:  command += f" --config-dir {config_dir}"
    if output_format: command += f" --outputfile /tmp/kube-bench-results.{output_format} --json"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


# ---------------------------------------------------------------------------
# iac_scan/
# ---------------------------------------------------------------------------

def _checkov(data: dict) -> dict:
    directory       = data.get("directory", ".")
    framework       = data.get("framework", "")
    check           = data.get("check", "")
    skip_check      = data.get("skip_check", "")
    output_format   = data.get("output_format", "json")
    additional_args = data.get("additional_args", "")

    command = f"checkov -d {directory}"
    if framework:    command += f" --framework {framework}"
    if check:        command += f" --check {check}"
    if skip_check:   command += f" --skip-check {skip_check}"
    if output_format: command += f" --output {output_format}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _terrascan(data: dict) -> dict:
    scan_type       = data.get("scan_type", "all")
    iac_dir         = data.get("iac_dir", ".")
    policy_type     = data.get("policy_type", "")
    output_format   = data.get("output_format", "json")
    severity        = data.get("severity", "")
    additional_args = data.get("additional_args", "")

    command = f"terrascan scan -t {scan_type} -d {iac_dir}"
    if policy_type:   command += f" -p {policy_type}"
    if output_format: command += f" -o {output_format}"
    if severity:      command += f" --severity {severity}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_HANDLERS = {
    # cloud_audit
    "prowler":      _prowler,
    "scout-suite":  _scout_suite,
    # cloud_exploit / cloud_visual
    "pacu":         _pacu,
    "cloudmapper":  _cloudmapper,
    # container_scan
    "trivy":        _trivy,
    "docker-bench": _docker_bench,
    "clair":        _clair,
    # k8s_scan
    "kube-hunter":  _kube_hunter,
    "kube-bench":   _kube_bench,
    # iac_scan
    "checkov":      _checkov,
    "terrascan":    _terrascan,
}


def security_exec(tool: str, data: dict) -> dict:
    """
    Execute a cloud/container/k8s/iac security tool directly — no Flask, no HTTP.

    Args:
        tool: tool name (e.g. "trivy", "prowler", "kube-bench")
        data: parameter dict

    Returns:
        {"success": bool, "output": str, "returncode": int, ...}
    """
    handler = _HANDLERS.get(tool)
    if handler is None:
        return {"success": False, "error": f"Unknown security tool: '{tool}'"}
    return handler(data)
