"""
mcp_core/wifi_direct.py

Phase 1 — Direct execution layer for wifi_pentest tools.

Bypasses Flask/HTTP entirely. Each function extracts the pure logic from
server_api/wifi_pentest/ and calls execute_command() directly.

Usage:
    from mcp_core.wifi_direct import wifi_exec

    result = await wifi_exec("airmon_ng", {
        "interface": "wlan0",
        "action": "start",
    })

All functions return the same dict shape as before:
    {"success": bool, "output": str, "returncode": int, ...}
"""

import logging
import os
import tempfile
import subprocess
from typing import Any, Dict

from server_core.command_executor import execute_command
from mcp_core._helpers import require

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool implementations — pure Python, no Flask, no HTTP
# ---------------------------------------------------------------------------

def _airmon_ng(data: dict) -> dict:
    err = require(data, "interface", "action")
    if err:
        return err

    interface = data["interface"].strip()
    action    = data["action"].strip()
    channel   = data.get("channel", "")

    if action not in ("start", "stop", "check kill"):
        return {"success": False, "error": "action must be: start | stop | check kill"}

    if action == "check kill":
        command = "airmon-ng check kill"
    else:
        command = f"airmon-ng {action} {interface}"
        if channel:
            command += f" {channel}"

    return execute_command(command, use_cache=False)


def _airodump_ng(data: dict) -> dict:
    err = require(data, "interface")
    if err:
        return err

    interface = data["interface"].strip()
    prefix    = data.get("output_prefix", "capture")
    bssid     = data.get("bssid", "")
    channel   = data.get("channel", "")
    essid     = data.get("essid", "")

    command = f"airodump-ng {interface} --write {prefix} --output-format pcap,csv"
    if bssid:   command += f" --bssid {bssid}"
    if channel: command += f" --channel {channel}"
    if essid:   command += f" --essid {essid}"

    return execute_command(command, use_cache=False)


def _aireplay_ng(data: dict) -> dict:
    VALID_MODES = {0, 1, 2, 3, 4, 5, 6, 7, 9}
    err = require(data, "interface")
    if err:
        return err

    interface = data["interface"].strip()
    try:
        mode = int(data.get("attack_mode"))
    except (TypeError, ValueError):
        return {"success": False, "error": "attack_mode must be an integer"}

    if mode not in VALID_MODES:
        return {"success": False, "error": f"attack_mode {mode} invalid. Valid: {sorted(VALID_MODES)}"}

    bssid      = data.get("bssid", "")
    client_mac = data.get("client_mac", "")
    count      = data.get("count", 0)

    command = f"aireplay-ng -{mode}"
    if count:      command += f" {count}"
    if bssid:      command += f" -a {bssid}"
    if client_mac: command += f" -c {client_mac}"
    command += f" {interface}"

    return execute_command(command, use_cache=False)


def _aircrack_ng(data: dict) -> dict:
    capture_files = data.get("capture_files", [])
    wordlist      = data.get("wordlist", "")
    bssid         = data.get("bssid", "")

    if not capture_files:
        return {"success": False, "error": "capture_files is required"}
    if not wordlist:
        return {"success": False, "error": "wordlist is required"}

    command = f"aircrack-ng {' '.join(capture_files)} -w {wordlist}"
    if bssid:
        command += f" -b {bssid}"

    return execute_command(command, use_cache=False)


def _airbase_ng(data: dict) -> dict:
    err = require(data, "interface", "essid")
    if err:
        return err

    interface = data["interface"].strip()
    essid     = data["essid"].strip()
    channel   = data.get("channel", 6)
    bssid     = data.get("bssid", "")
    wpa_mode  = data.get("wpa_mode", "")

    command = f"airbase-ng -e {essid} -c {channel}"
    if bssid: command += f" -a {bssid}"
    if wpa_mode == "wpa2": command += " -z 4"
    elif wpa_mode == "wpa": command += " -z 2"
    command += f" {interface}"

    return execute_command(command, use_cache=False)


def _airdecap_ng(data: dict) -> dict:
    err = require(data, "capture_file")
    if err:
        return err

    cap_file = data["capture_file"].strip()
    password = data.get("password", "")
    wep_key  = data.get("wep_key", "")
    bssid    = data.get("bssid", "")
    essid    = data.get("essid", "")

    command = "airdecap-ng"
    if wep_key:    command += f" -w {wep_key}"
    elif password: command += f" -p {password}"
    if bssid:      command += f" -b {bssid}"
    if essid:      command += f" -e {essid}"
    command += f" {cap_file}"

    return execute_command(command, use_cache=False)


def _hcxdumptool(data: dict) -> dict:
    err = require(data, "interface")
    if err:
        return err

    interface    = data["interface"].strip()
    output_file  = data.get("output_file", "pmkid_capture.pcapng")
    target_bssid = data.get("target_bssid", "").strip()
    duration     = int(data.get("duration", 1))

    bpf_file = None
    command  = f"hcxdumptool -i {interface} -w {output_file} --tot={duration}"

    if target_bssid:
        try:
            bssid_hex = target_bssid.replace(":", "").lower()
            bpf_expr  = f"wlan addr3 {bssid_hex}"
            bpf_result = subprocess.run(
                ["hcxdumptool", f"--bpfc={bpf_expr}"],
                capture_output=True, text=True
            )
            if bpf_result.returncode == 0 and bpf_result.stdout.strip():
                bpf_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".bpf", delete=False)
                bpf_tmp.write(bpf_result.stdout)
                bpf_tmp.close()
                bpf_file = bpf_tmp.name
                command += f" --bpf={bpf_file}"
        except Exception as e:
            logger.warning(f"⚠️ BPF setup error: {e}, running without filter")

    if data.get("additional_args"):
        command += f" {data['additional_args']}"

    try:
        return execute_command(command, use_cache=False)
    finally:
        if bpf_file and os.path.exists(bpf_file):
            os.unlink(bpf_file)


def _hcxpcapngtool(data: dict) -> dict:
    err = require(data, "input_file", "output_file")
    if err:
        return err

    inp = data["input_file"].strip()
    out = data["output_file"].strip()

    command = f"hcxpcapngtool -o {out} {inp}"
    return execute_command(command, use_cache=False)


def _wifite2(data: dict) -> dict:
    err = require(data, "interface")
    if err:
        return err

    interface = data["interface"].strip()
    command   = f"wifite -i {interface} --kill --no-wps"

    if data.get("target_essid"):           command += f" --essid {data['target_essid']}"
    if data.get("target_bssid"):           command += f" --bssid {data['target_bssid']}"
    if data.get("attack_wps"):             command  = command.replace("--no-wps", "--wps")
    if data.get("attack_pmkid", True):     command += " --pmkid"
    if data.get("attack_handshake", True): command += " --wpa"
    if data.get("wordlist"):               command += f" --dict {data['wordlist']}"

    return execute_command(command, use_cache=False)


def _eaphammer(data: dict) -> dict:
    err = require(data, "interface", "essid")
    if err:
        return err

    interface   = data["interface"].strip()
    essid       = data["essid"].strip()
    channel     = data.get("channel", 6)
    auth_mode   = data.get("auth_mode", "wpa-eap")
    attack_type = data.get("attack_type", "creds")
    negotiate   = data.get("negotiate", "gtc-downgrade")
    cert_path   = data.get("cert_path", "")

    command = f"eaphammer -i {interface} --essid {essid} --channel {channel} --{auth_mode}"
    if attack_type == "creds":            command += " --creds"
    elif attack_type == "captive-portal": command += " --captive-portal"
    if negotiate in ("gtc-downgrade", "weakest"):
        command += f" --negotiate {negotiate}"
    if cert_path: command += f" --cert-dir {cert_path}"

    return execute_command(command, use_cache=False)


def _bettercap_wifi(data: dict) -> dict:
    interface = data.get("interface", "").strip()
    mode      = data.get("mode", "recon")
    bssid     = data.get("target_bssid", "")
    caplet    = data.get("caplet", "")

    if not interface:
        return {"success": False, "error": "interface is required"}
    if mode in ("deauth", "evil-twin") and not bssid and not caplet:
        return {"success": False, "error": f"target_bssid required for mode '{mode}'"}

    tmp_path = None
    if not caplet:
        lines = [f"set wifi.interface {interface}"]
        if data.get("channel_hop", True): lines.append("set wifi.hop.period 1")
        if mode == "recon":
            lines += ["wifi.recon on", "sleep 30", "wifi.show", "wifi.recon off"]
        elif mode == "deauth":
            target = "ff:ff:ff:ff:ff:ff" if data.get("deauth_all") else bssid
            lines += [f"wifi.recon {bssid}", f"wifi.deauth {target}"]
        elif mode == "evil-twin":
            lines += [f"wifi.ap.ssid {bssid}", "wifi.ap on"]
        content = "\n".join(lines)
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".cap", delete=False)
        tmp.write(content)
        tmp.close()
        caplet = tmp.name
        tmp_path = tmp.name

    command = f"bettercap -iface {interface} -caplet {caplet}"
    try:
        return execute_command(command, use_cache=False)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _mdk4(data: dict) -> dict:
    VALID_MODES = {"b", "d", "a", "m", "e", "p"}
    interface = data.get("interface", "").strip()
    mode      = data.get("attack_mode", "").strip()

    if not interface:
        return {"success": False, "error": "interface is required"}
    if mode not in VALID_MODES:
        return {"success": False, "error": f"Invalid mode '{mode}'. Valid: {sorted(VALID_MODES)}"}

    bssid   = data.get("target_bssid", "")
    ssid_wl = data.get("ssid_wordlist", "")
    rate    = int(data.get("burst_rate", 50))

    command = f"mdk4 {interface} {mode}"
    if mode == "b" and ssid_wl:              command += f" -f {ssid_wl}"
    if mode in ("d", "a", "m", "e") and bssid: command += f" -B {bssid}"
    if rate != 50:                           command += f" -s {rate}"

    return execute_command(command, use_cache=False)


def _tcpdump(data: dict) -> dict:
    err = require(data, "interface")
    if err: return err
    interface   = data["interface"].strip()
    count       = data.get("count", 10)
    filter_expr = data.get("filter", "")
    output_file = data.get("output_file", "")
    extra       = data.get("additional_args", "").strip()
    command = f"tcpdump -i {interface} -nn"
    if count > 0: command += f" -c {count}"
    if output_file: command += f" -w {output_file}"
    if filter_expr: command += f" \"{filter_expr}\""
    if extra:      command += f" {extra}"
    return execute_command(command)


def _tshark(data: dict) -> dict:
    interface = data.get("interface", "").strip()
    file_path = data.get("file", "").strip()
    if not interface and not file_path:
        return {"success": False, "error": "'interface' or 'file' is required"}
    filter_expr = data.get("filter", "")
    count       = data.get("count", 50)
    extra       = data.get("additional_args", "").strip()
    command = "tshark"
    if interface: command += f" -i {interface}"
    if file_path: command += f" -r {file_path}"
    if count > 0: command += f" -c {count}"
    if filter_expr: command += f" -Y \"{filter_expr}\""
    if extra:      command += f" {extra}"
    return execute_command(command)


# ---------------------------------------------------------------------------
# Dispatch table — maps tool name → handler function
# ---------------------------------------------------------------------------

_HANDLERS = {
    "airmon_ng":      _airmon_ng,
    "airodump_ng":    _airodump_ng,
    "aireplay_ng":    _aireplay_ng,
    "aircrack_ng":    _aircrack_ng,
    "airbase_ng":     _airbase_ng,
    "airdecap_ng":    _airdecap_ng,
    "hcxdumptool":    _hcxdumptool,
    "hcxpcapngtool":  _hcxpcapngtool,
    "wifite2":        _wifite2,
    "eaphammer":      _eaphammer,
    "bettercap_wifi": _bettercap_wifi,
    "bettercap":      _bettercap_wifi,
    "tcpdump":        _tcpdump,
    "tshark":         _tshark,
    "mdk4":           _mdk4,
}


def wifi_exec(tool: str, data: dict) -> dict:
    """
    Execute a wifi_pentest tool directly — no Flask, no HTTP.

    Args:
        tool: tool name (e.g. "airmon_ng")
        data: parameter dict (same shape as the old safe_post payload)

    Returns:
        {"success": bool, "output": str, "returncode": int, ...}
    """
    handler = _HANDLERS.get(tool)
    if handler is None:
        return {"success": False, "error": f"Unknown wifi tool: '{tool}'"}
    return handler(data)
