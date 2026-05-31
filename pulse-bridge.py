#!/usr/bin/env python3
"""
Pulse stdio→HTTP bridge for Claude Desktop.

Reads JSON-RPC from stdin, proxies to the running Pulse HTTP server,
writes JSON-RPC responses to stdout.

No heavy imports — urllib + json only (stdlib).
"""

import json
import sys
import urllib.request
import urllib.error
import http.client

PULSE_URL = "http://localhost:8888/mcp"

_session_id: str | None = None
_connection: http.client.HTTPConnection | None = None


def _get_conn() -> http.client.HTTPConnection:
    global _connection
    if _connection is None:
        _connection = http.client.HTTPConnection("localhost", 8888, timeout=300)
    return _connection


def _rpc(body: bytes) -> tuple[bytes, dict[str, str]]:
    conn = _get_conn()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Content-Length": str(len(body)),
    }
    if _session_id:
        headers["MCP-Session-ID"] = _session_id

    try:
        conn.request("POST", "/mcp", body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        resp_headers = dict(resp.getheaders())
        return data, resp_headers
    except Exception:
        # Connection might be stale — reset and retry once
        global _connection
        _connection = None
        conn = _get_conn()
        conn.request("POST", "/mcp", body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        return data, dict(resp.getheaders())


def _parse_sse(data: bytes) -> list[dict]:
    """Parse SSE event stream, extract all JSON data payloads."""
    results = []
    text = data.decode()
    for event in text.split("\n\n"):
        event = event.strip()
        if not event:
            continue
        # Collect all data: lines in this event
        data_lines = []
        for line in event.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                data_lines.append(line[6:])
            elif line.startswith("data:"):  # empty data value
                data_lines.append("")
        if data_lines:
            try:
                results.append(json.loads("".join(data_lines)))
            except json.JSONDecodeError:
                pass
    return results


def _send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    global _session_id

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Notifications (no id) — forward but do not send response
        is_notification = req.get("id") is None
        req_id = req.get("id")

        body = json.dumps(req).encode()
        try:
            resp_data, resp_headers = _rpc(body)
        except Exception as e:
            if not is_notification:
                _send({
                    "jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32000, "message": f"Bridge error: {e}"},
                })
            continue

        # Capture session ID from response header
        sid = resp_headers.get("mcp-session-id")
        if sid:
            _session_id = sid

        payloads = _parse_sse(resp_data)
        for p in payloads:
            _send(p)

        # Fallback: empty response but never for notifications
        if not payloads and not is_notification:
            _send({"jsonrpc": "2.0", "id": req_id, "result": {}})


if __name__ == "__main__":
    main()
