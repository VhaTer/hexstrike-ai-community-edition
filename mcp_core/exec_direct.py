"""
mcp_core/exec_direct.py

Direct execution layer for arbitrary code execution.
Primitives: execute_code

Supported languages: python, bash, node.
Sandbox parameter reserved for future isolation.
"""

import subprocess
import sys
from typing import Any, Dict

_LANGUAGE_MAP: dict[str, tuple[str, str]] = {
    "python": (sys.executable, "-c"),
    "bash": ("bash", "-c"),
    "sh": ("bash", "-c"),
    "node": ("node", "-e"),
    "nodejs": ("node", "-e"),
}


def _validate_language(lang: str) -> tuple[str, str] | None:
    return _LANGUAGE_MAP.get(lang.lower().strip())


def _execute_code(data: dict) -> dict:
    code = data.get("code", "")
    language = data.get("language", "python")
    timeout_val = int(data.get("timeout", 30))
    _ = data.get("sandbox", False)  # reserved for future isolation

    if not code:
        return {"success": False, "error": "code is required"}

    entry = _validate_language(language)
    if entry is None:
        return {
            "success": False,
            "error": f"unsupported language: {language} (supported: {', '.join(sorted(_LANGUAGE_MAP))})",
        }

    interpreter, flag = entry
    try:
        result = subprocess.run(
            [interpreter, flag, code],
            capture_output=True,
            text=True,
            timeout=timeout_val,
        )
        return {
            "success": True,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "language": language,
            "timeout": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout_val}s",
            "exit_code": -1,
            "language": language,
            "timeout": True,
        }
    except FileNotFoundError:
        return {"success": False, "error": f"interpreter not found: {interpreter} (for language: {language})"}
    except Exception as e:
        return {"success": False, "error": str(e)}


_HANDLERS = {
    "execute_code": _execute_code,
}


def exec_direct(tool: str, data: dict) -> dict:
    handler = _HANDLERS.get(tool)
    if handler is None:
        return {"success": False, "error": f"Unknown exec tool: '{tool}'"}
    return handler(data)
