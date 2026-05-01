"""
mcp_core/active_directory_direct.py

Phase 2 — Direct execution layer for Active Directory tools.
Covers: impacket, ldapdomaindump, adidnsdump, certipy_ad, mitm6, pywerview,
        bloodhound_ce_python

Usage:
    import mcp_core.active_directory_direct as _ad_direct
    result = await loop.run_in_executor(
        None, lambda: _ad_direct.ad_exec("ldapdomaindump", data)
    )
"""

import shlex
from server_core.command_executor import execute_command

# ---------------------------------------------------------------------------
# Validated Impacket scripts available on Kali
# ---------------------------------------------------------------------------
IMPACKET_SCRIPTS = {
    "DumpNTLMInfo", "Get-GPPPassword", "GetADComputers", "GetADUsers",
    "GetLAPSPassword", "GetNPUsers", "GetUserSPNs", "addcomputer", "atexec",
    "changepasswd", "dacledit", "dcomexec", "describeTicket", "dpapi",
    "esentutl", "findDelegation", "getArch", "getPac", "getST", "getTGT",
    "goldenPac", "keylistattack", "lookupsid", "mimikatz", "mssqlclient",
    "mssqlinstance", "net", "ntlmrelayx", "owneredit", "psexec", "raiseChild",
    "rbcd", "rdp_check", "reg", "rpcclient", "smbclient", "smbexec",
    "smbserver", "ticketConverter", "ticketer", "wmiquery",
}


def _require(data: dict, *keys: str) -> dict:
    for key in keys:
        if not data.get(key, ""):
            return {"success": False, "error": f"'{key}' is required"}
    return {}


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _impacket(data: dict) -> dict:
    """Generic impacket-{script} dispatcher."""
    err = _require(data, "script", "target")
    if err:
        return err

    script = data["script"].strip()
    if script not in IMPACKET_SCRIPTS:
        return {
            "success": False,
            "error": f"Unsupported Impacket script: '{script}'. "
                     f"Supported: {sorted(IMPACKET_SCRIPTS)}",
        }

    target  = data["target"].strip()
    options = data.get("options", {})
    extra   = data.get("extra_args", "").strip()

    binary = f"impacket-{script}"
    argv   = [binary]

    # Append options dict as CLI flags
    for key, value in options.items():
        flag = f"-{key}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif value not in (None, ""):
            argv.extend([flag, str(value)])

    argv.append(target)

    if extra:
        argv.extend(shlex.split(extra))

    command = shlex.join(argv)
    return execute_command(command)


def _ldapdomaindump(data: dict) -> dict:
    err = _require(data, "hostname")
    if err:
        return err

    hostname = data["hostname"].strip()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    authtype = data.get("authtype", "NTLM").strip()

    command = f"ldapdomaindump {hostname} --authtype {authtype}"
    if username and password:
        command += f" --user {username} --password {password}"

    return execute_command(command)


def _adidnsdump(data: dict) -> dict:
    err = _require(data, "target")
    if err:
        return err

    target   = data["target"].strip()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    zone     = data.get("zone", "").strip()
    extra    = data.get("additional_args", "").strip()

    command = f"adidnsdump {target}"
    if username:
        command += f" -u {username}"
    if password:
        command += f" -p {password}"
    if zone:
        command += f" --zone {zone}"
    if extra:
        command += f" {extra}"

    return execute_command(command, use_cache=True)


def _certipy_ad(data: dict) -> dict:
    err = _require(data, "action")
    if err:
        return err

    action   = data["action"].strip()          # find, req, auth, shadow, relay...
    target   = data.get("target", "").strip()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    dc_ip    = data.get("dc_ip", "").strip()
    extra    = data.get("additional_args", "").strip()

    command = f"certipy {action}"
    if target:
        command += f" -target {target}"
    if username:
        command += f" -u {username}"
    if password:
        command += f" -p {password}"
    if dc_ip:
        command += f" -dc-ip {dc_ip}"
    if extra:
        command += f" {extra}"

    return execute_command(command)


def _mitm6(data: dict) -> dict:
    err = _require(data, "interface")
    if err:
        return err

    interface = data["interface"].strip()
    domain    = data.get("domain", "").strip()
    extra     = data.get("additional_args", "").strip()

    command = f"mitm6 -i {interface}"
    if domain:
        command += f" -d {domain}"
    if extra:
        command += f" {extra}"

    return execute_command(command)


def _pywerview(data: dict) -> dict:
    err = _require(data, "request", "target")
    if err:
        return err

    request_type = data["request"].strip()   # get-netuser, get-netgroup, etc.
    target       = data["target"].strip()
    username     = data.get("username", "").strip()
    password     = data.get("password", "").strip()
    dc_ip        = data.get("dc_ip", "").strip()
    extra        = data.get("additional_args", "").strip()

    command = f"pywerview {request_type} -t {target}"
    if username:
        command += f" -u {username}"
    if password:
        command += f" -p {password}"
    if dc_ip:
        command += f" --dc-ip {dc_ip}"
    if extra:
        command += f" {extra}"

    return execute_command(command, use_cache=True)


def _bloodhound(data: dict) -> dict:
    err = _require(data, "domain", "username", "password", "dc_ip")
    if err:
        return err

    domain   = data["domain"].strip()
    username = data["username"].strip()
    password = data["password"].strip()
    dc_ip    = data["dc_ip"].strip()
    extra    = data.get("additional_args", "").strip()

    command = (
        f"bloodhound-python -d {domain} -u {username} -p {password} "
        f"--dc {dc_ip} -c All --zip"
    )
    if extra:
        command += f" {extra}"

    return execute_command(command)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_HANDLERS = {
    "impacket":          _impacket,
    "ldapdomaindump":    _ldapdomaindump,
    "adidnsdump":        _adidnsdump,
    "certipy_ad":        _certipy_ad,
    "certipy":           _certipy_ad,
    "mitm6":             _mitm6,
    "pywerview":         _pywerview,
    "bloodhound":        _bloodhound,
    "bloodhound_python": _bloodhound,
}


def ad_exec(tool: str, data: dict) -> dict:
    """Execute an Active Directory tool directly """
    handler = _HANDLERS.get(tool)
    if handler is None:
        return {
            "success": False,
            "error": f"Unknown AD tool: '{tool}'. "
                     f"Available: {sorted(_HANDLERS.keys())}",
        }
    return handler(data)
