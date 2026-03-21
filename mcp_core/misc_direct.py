"""
mcp_core/misc_direct.py

Phase 2 — Direct execution layer for miscellaneous tools.
Covers: gadget_search, memory_forensics, binary_debug, db_query, api_scan.

Note: api_scan tools (api_schema_analyzer, graphql_scanner, jwt_analyzer)
contain complex Python logic — they are wrapped here as pure Python calls,
bypassing Flask/HTTP while keeping their internal logic intact.

Usage:
    import mcp_core.misc_direct as _misc_direct
    result = await loop.run_in_executor(
        None, lambda: _misc_direct.misc_exec("volatility", data)
    )
"""

import base64
import json
import logging
import os
import sqlite3
from typing import Any, Dict

import pymysql

from server_core.command_executor import execute_command

logger = logging.getLogger(__name__)


def _require(data: dict, *keys: str) -> Dict[str, Any]:
    for key in keys:
        if not data.get(key, ""):
            return {"success": False, "error": f"'{key}' is required"}
    return {}


# ---------------------------------------------------------------------------
# gadget_search/
# ---------------------------------------------------------------------------

def _ropgadget(data: dict) -> dict:
    err = _require(data, "binary")
    if err: return err
    binary          = data["binary"].strip()
    gadget_type     = data.get("gadget_type", "")
    additional_args = data.get("additional_args", "")

    command = f"ROPgadget --binary {binary}"
    if gadget_type:     command += f" --only '{gadget_type}'"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _ropper(data: dict) -> dict:
    err = _require(data, "binary")
    if err: return err
    binary          = data["binary"].strip()
    search          = data.get("search", "")
    arch            = data.get("arch", "")
    additional_args = data.get("additional_args", "")

    command = f"ropper --file {binary}"
    if search:          command += f" --search '{search}'"
    if arch:            command += f" --arch {arch}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _one_gadget(data: dict) -> dict:
    err = _require(data, "libc")
    if err: return err
    libc            = data["libc"].strip()
    level           = data.get("level", 0)
    additional_args = data.get("additional_args", "")

    command = f"one_gadget {libc}"
    if level:           command += f" -l {level}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


# ---------------------------------------------------------------------------
# memory_forensics/
# ---------------------------------------------------------------------------

def _volatility(data: dict) -> dict:
    memory_file     = data.get("memory_file", "").strip()
    plugin          = data.get("plugin", "").strip()
    if not memory_file: return {"success": False, "error": "memory_file is required"}
    if not plugin:      return {"success": False, "error": "plugin is required"}

    profile         = data.get("profile", "")
    additional_args = data.get("additional_args", "")

    command = f"volatility -f {memory_file}"
    if profile: command += f" --profile={profile}"
    command += f" {plugin}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


def _volatility3(data: dict) -> dict:
    memory_file     = data.get("memory_file", "").strip()
    plugin          = data.get("plugin", "").strip()
    if not memory_file: return {"success": False, "error": "memory_file is required"}
    if not plugin:      return {"success": False, "error": "plugin is required"}

    output_file     = data.get("output_file", "")
    additional_args = data.get("additional_args", "")

    command = f"vol -f {memory_file} {plugin}"
    if output_file:     command += f" -o {output_file}"
    if additional_args: command += f" {additional_args}"

    return execute_command(command)


# ---------------------------------------------------------------------------
# binary_debug/
# ---------------------------------------------------------------------------

def _gdb(data: dict) -> dict:
    err = _require(data, "binary")
    if err: return err
    binary          = data["binary"].strip()
    commands        = data.get("commands", "")
    script_file     = data.get("script_file", "")
    additional_args = data.get("additional_args", "")

    command = f"gdb {binary}"
    if script_file: command += f" -x {script_file}"
    if commands:
        temp = "/tmp/gdb_commands.txt"
        with open(temp, "w") as f:
            f.write(commands)
        command += f" -x {temp}"
    if additional_args: command += f" {additional_args}"
    command += " -batch"

    result = execute_command(command)
    if commands:
        try: os.remove("/tmp/gdb_commands.txt")
        except Exception: pass
    return result


def _radare2(data: dict) -> dict:
    err = _require(data, "binary")
    if err: return err
    binary          = data["binary"].strip()
    commands        = data.get("commands", "")
    additional_args = data.get("additional_args", "")

    if commands:
        temp = "/tmp/r2_commands.txt"
        with open(temp, "w") as f:
            f.write(commands)
        command = f"r2 -i {temp} -q {binary}"
    else:
        command = f"r2 -q {binary}"
    if additional_args: command += f" {additional_args}"

    result = execute_command(command)
    if commands:
        try: os.remove("/tmp/r2_commands.txt")
        except Exception: pass
    return result


# ---------------------------------------------------------------------------
# db_query/ — direct Python DB connections (no CLI needed)
# ---------------------------------------------------------------------------

def _mysql(data: dict) -> dict:
    host     = data.get("host")
    user     = data.get("user")
    password = data.get("password", "")
    database = data.get("database")
    query    = data.get("query")

    if not all([host, user, database, query]):
        return {"success": False, "error": "host, user, database, query are required"}

    try:
        conn = pymysql.connect(
            host=host, user=user, password=password,
            database=database, cursorclass=pymysql.cursors.DictCursor
        )
        with conn.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()
        conn.close()
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _sqlite(data: dict) -> dict:
    db_path = data.get("db_path")
    query   = data.get("query")

    if not db_path or not query:
        return {"success": False, "error": "db_path and query are required"}

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(query)
        result = cur.fetchall()
        columns = [desc[0] for desc in cur.description] if cur.description else []
        cur.close()
        conn.close()
        return {"success": True, "columns": columns, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _postgresql(data: dict) -> dict:
    # psycopg2 is commented out in upstream — return graceful error
    return {"success": False, "error": "PostgreSQL support requires psycopg2 — install with: pip install psycopg2-binary"}


# ---------------------------------------------------------------------------
# api_scan/ — complex Python logic, no simple CLI delegation
# ---------------------------------------------------------------------------

def _api_schema_analyzer(data: dict) -> dict:
    schema_url  = data.get("schema_url", "").strip()
    schema_type = data.get("schema_type", "openapi")

    if not schema_url:
        return {"success": False, "error": "schema_url is required"}

    # Fetch schema
    result = execute_command(f"curl -s '{schema_url}'", use_cache=True)
    if not result.get("success"):
        return {"success": False, "error": "Failed to fetch API schema"}

    schema_content = result.get("stdout", "")
    analysis = {
        "schema_url": schema_url,
        "schema_type": schema_type,
        "endpoints_found": [],
        "security_issues": [],
        "recommendations": []
    }

    try:
        schema_data = json.loads(schema_content)
        if schema_type.lower() in ["openapi", "swagger"]:
            for path, methods in schema_data.get("paths", {}).items():
                for method, details in methods.items():
                    if isinstance(details, dict):
                        endpoint_info = {
                            "path": path, "method": method.upper(),
                            "summary": details.get("summary", ""),
                            "parameters": details.get("parameters", []),
                            "security": details.get("security", [])
                        }
                        analysis["endpoints_found"].append(endpoint_info)
                        if not endpoint_info["security"]:
                            analysis["security_issues"].append({
                                "endpoint": f"{method.upper()} {path}",
                                "issue": "no_authentication",
                                "severity": "MEDIUM",
                                "description": "Endpoint has no authentication requirements"
                            })
                        for param in endpoint_info["parameters"]:
                            pname = param.get("name", "").lower()
                            if any(s in pname for s in ["password", "token", "key", "secret"]):
                                analysis["security_issues"].append({
                                    "endpoint": f"{method.upper()} {path}",
                                    "issue": "sensitive_parameter",
                                    "severity": "HIGH",
                                    "description": f"Sensitive parameter: {pname}"
                                })
        if analysis["security_issues"]:
            analysis["recommendations"] = [
                "Implement authentication for all endpoints",
                "Use HTTPS for all API communications",
                "Validate and sanitize all input parameters",
                "Implement rate limiting"
            ]
    except json.JSONDecodeError:
        analysis["security_issues"].append({
            "endpoint": "schema", "issue": "invalid_json",
            "severity": "HIGH", "description": "Schema is not valid JSON"
        })

    return {"success": True, "schema_analysis_results": analysis}


def _graphql_scanner(data: dict) -> dict:
    endpoint    = data.get("endpoint", "").strip()
    if not endpoint:
        return {"success": False, "error": "endpoint is required"}

    introspection = data.get("introspection", True)
    query_depth   = data.get("query_depth", 10)

    results = {"endpoint": endpoint, "tests_performed": [], "vulnerabilities": [], "recommendations": []}

    if introspection:
        clean_query = '{ __schema { types { name fields { name type { name } } } } }'
        cmd = f"curl -s -X POST -H 'Content-Type: application/json' -d '{{\"query\":\"{clean_query}\"}}' '{endpoint}'"
        r = execute_command(cmd, use_cache=False)
        results["tests_performed"].append("introspection_query")
        if "data" in r.get("stdout", ""):
            results["vulnerabilities"].append({
                "type": "introspection_enabled", "severity": "MEDIUM",
                "description": "GraphQL introspection is enabled"
            })

    deep = "{ " * query_depth + "field" + " }" * query_depth
    cmd = f"curl -s -X POST -H 'Content-Type: application/json' -d '{{\"query\":\"{deep}\"}}' {endpoint}"
    r = execute_command(cmd, use_cache=False)
    results["tests_performed"].append("query_depth_analysis")
    if "error" not in r.get("stdout", "").lower():
        results["vulnerabilities"].append({
            "type": "no_query_depth_limit", "severity": "HIGH",
            "description": f"No query depth limiting (tested depth: {query_depth})"
        })

    if results["vulnerabilities"]:
        results["recommendations"] = [
            "Disable introspection in production",
            "Implement query depth limiting",
            "Add rate limiting for batch queries"
        ]

    return {"success": True, "graphql_scan_results": results}


def _jwt_analyzer(data: dict) -> dict:
    jwt_token  = data.get("jwt_token", "").strip()
    target_url = data.get("target_url", "")

    if not jwt_token:
        return {"success": False, "error": "jwt_token is required"}

    results = {
        "token": jwt_token[:50] + "..." if len(jwt_token) > 50 else jwt_token,
        "vulnerabilities": [], "token_info": {}, "attack_vectors": []
    }

    try:
        parts = jwt_token.split('.')
        if len(parts) >= 2:
            header = json.loads(base64.b64decode(parts[0] + '=' * (4 - len(parts[0]) % 4)))
            payload = json.loads(base64.b64decode(parts[1] + '=' * (4 - len(parts[1]) % 4)))
            algorithm = header.get("alg", "").lower()
            results["token_info"] = {"header": header, "payload": payload, "algorithm": algorithm}

            if algorithm == "none":
                results["vulnerabilities"].append({
                    "type": "none_algorithm", "severity": "CRITICAL",
                    "description": "JWT uses 'none' algorithm — no signature verification"
                })
            if algorithm in ["hs256", "hs384", "hs512"]:
                results["attack_vectors"].append("hmac_key_confusion")
                results["vulnerabilities"].append({
                    "type": "hmac_algorithm", "severity": "MEDIUM",
                    "description": "HMAC algorithm — vulnerable to key confusion attacks"
                })
            if not payload.get("exp"):
                results["vulnerabilities"].append({
                    "type": "no_expiration", "severity": "HIGH",
                    "description": "JWT token has no expiration time"
                })
    except Exception as e:
        results["vulnerabilities"].append({
            "type": "malformed_token", "severity": "HIGH",
            "description": f"Token decoding failed: {str(e)}"
        })

    if target_url:
        parts = jwt_token.split('.')
        if len(parts) >= 2:
            none_header = base64.b64encode(b'{"alg":"none","typ":"JWT"}').decode().rstrip('=')
            none_token = f"{none_header}.{parts[1]}."
            cmd = f"curl -s -H 'Authorization: Bearer {none_token}' '{target_url}'"
            r = execute_command(cmd, use_cache=False)
            if "200" in r.get("stdout", "") or "success" in r.get("stdout", "").lower():
                results["vulnerabilities"].append({
                    "type": "none_algorithm_accepted", "severity": "CRITICAL",
                    "description": "Server accepts tokens with 'none' algorithm"
                })

    return {"success": True, "jwt_analysis_results": results}


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_HANDLERS = {
    # gadget_search
    "ropgadget":           _ropgadget,
    "ropper":              _ropper,
    "one_gadget":          _one_gadget,
    # memory_forensics
    "volatility":          _volatility,
    "volatility3":         _volatility3,
    # binary_debug
    "gdb":                 _gdb,
    "radare2":             _radare2,
    # db_query
    "mysql":               _mysql,
    "sqlite":              _sqlite,
    "postgresql":          _postgresql,
    # api_scan
    "api_schema_analyzer": _api_schema_analyzer,
    "graphql_scanner":     _graphql_scanner,
    "jwt_analyzer":        _jwt_analyzer,
}


def misc_exec(tool: str, data: dict) -> dict:
    """
    Execute a miscellaneous tool directly — no Flask, no HTTP.

    Args:
        tool: tool name (e.g. "ropgadget", "volatility", "mysql")
        data: parameter dict

    Returns:
        {"success": bool, ...}
    """
    handler = _HANDLERS.get(tool)
    if handler is None:
        return {"success": False, "error": f"Unknown misc tool: '{tool}'"}
    return handler(data)
