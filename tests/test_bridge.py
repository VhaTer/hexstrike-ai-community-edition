"""Comprehensive unit tests for pulse.server.bridge — 128 lines, 0% coverage target."""

import json
import http.client
from unittest.mock import MagicMock, patch, call

import pytest


# ===========================================================================
# _parse_sse
# ===========================================================================


class TestParseSSE:
    def test_valid_sse_events(self):
        from pulse.server.bridge import _parse_sse

        data = b"data: {\"id\":1}\n\ndata: {\"id\":2}\n\n"
        result = _parse_sse(data)
        assert result == [{"id": 1}, {"id": 2}]

    def test_multi_line_data(self):
        from pulse.server.bridge import _parse_sse

        data = b"data: {\"id\":\ndata: 1}\n\n"
        result = _parse_sse(data)
        assert result == [{"id": 1}]

    def test_empty_event_skipped(self):
        from pulse.server.bridge import _parse_sse

        data = b"data: {\"id\":1}\n\n\n\n\ndata: {\"id\":2}\n\n"
        result = _parse_sse(data)
        assert result == [{"id": 1}, {"id": 2}]

    def test_keepalive_event_skipped(self):
        from pulse.server.bridge import _parse_sse

        data = b": keepalive\n\ndata: {\"id\":1}\n\n"
        result = _parse_sse(data)
        assert result == [{"id": 1}]

    def test_json_decode_error_skipped(self):
        from pulse.server.bridge import _parse_sse

        data = b"data: not-json\n\ndata: {\"id\":1}\n\n"
        result = _parse_sse(data)
        assert result == [{"id": 1}]

    def test_mixed_content(self):
        from pulse.server.bridge import _parse_sse

        data = b"event: foo\ndata: {\"id\":1}\n\ndata: {\"id\":2}\n\n: keepalive\n\n"
        result = _parse_sse(data)
        assert result == [{"id": 1}, {"id": 2}]

    def test_no_data_lines(self):
        from pulse.server.bridge import _parse_sse

        data = b"event: foo\n\n: keepalive\n\n"
        result = _parse_sse(data)
        assert result == []

    def test_empty_data_value(self):
        from pulse.server.bridge import _parse_sse

        data = b"data:\n\n"
        result = _parse_sse(data)
        assert result == []

    def test_empty_bytes(self):
        from pulse.server.bridge import _parse_sse

        result = _parse_sse(b"")
        assert result == []


# ===========================================================================
# _send
# ===========================================================================


class TestSend:
    def test_sends_json_to_stdout(self, capsys):
        from pulse.server.bridge import _send

        _send({"jsonrpc": "2.0", "id": 1, "result": "ok"})
        captured = capsys.readouterr()
        assert json.loads(captured.out) == {"jsonrpc": "2.0", "id": 1, "result": "ok"}
        assert captured.out.endswith("\n")

    def test_sends_with_unicode(self, capsys):
        from pulse.server.bridge import _send

        _send({"result": "héllo"})
        captured = capsys.readouterr()
        assert '"héllo"' in captured.out

    def test_null_sends_none(self, capsys):
        from pulse.server.bridge import _send

        _send({"jsonrpc": "2.0", "id": None, "result": {}})
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["id"] is None

    def test_flush_called(self, capsys):
        from pulse.server.bridge import _send

        with patch("sys.stdout.flush") as mock_flush:
            _send({"ok": True})
        mock_flush.assert_called_once()


# ===========================================================================
# _get_conn
# ===========================================================================


class TestGetConn:
    def test_returns_http_connection(self):
        import pulse.server.bridge as bridge_mod
        bridge_mod._connection = None

        from pulse.server.bridge import _get_conn

        conn = _get_conn()
        assert isinstance(conn, http.client.HTTPConnection)
        assert conn.host == "localhost"
        assert conn.port == 8888

    def test_singleton(self):
        import pulse.server.bridge as bridge_mod
        bridge_mod._connection = None

        from pulse.server.bridge import _get_conn

        conn1 = _get_conn()
        conn2 = _get_conn()
        assert conn1 is conn2


# ===========================================================================
# _rpc
# ===========================================================================


class TestRpc:
    def test_successful_request(self):
        import pulse.server.bridge as bridge_mod
        bridge_mod._connection = None
        bridge_mod._session_id = None

        from pulse.server.bridge import _rpc

        mock_conn = MagicMock()
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"result":"ok"}'
        mock_resp.getheaders.return_value = [("content-type", "application/json")]
        mock_conn.getresponse.return_value = mock_resp

        body = json.dumps({"id": 1}).encode()

        with patch.object(bridge_mod, "_get_conn", return_value=mock_conn):
            data, headers = _rpc(body)

        assert data == b'{"result":"ok"}'
        assert headers == {"content-type": "application/json"}
        mock_conn.request.assert_called_once_with(
            "POST", "/mcp",
            body=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Content-Length": str(len(body)),
            },
        )

    def test_stale_connection_retry(self):
        import pulse.server.bridge as bridge_mod
        bridge_mod._connection = None
        bridge_mod._session_id = None

        from pulse.server.bridge import _rpc

        stale_conn = MagicMock()
        stale_conn.request.side_effect = ConnectionError("stale")
        fresh_conn = MagicMock()
        fresh_resp = MagicMock()
        fresh_resp.read.return_value = b'{"result":"ok"}'
        fresh_resp.getheaders.return_value = []
        fresh_conn.getresponse.return_value = fresh_resp

        call_count = [0]

        def get_conn_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                return stale_conn
            return fresh_conn

        with patch.object(bridge_mod, "_get_conn", side_effect=get_conn_side_effect):
            data, headers = _rpc(json.dumps({"id": 1}).encode())

        assert data == b'{"result":"ok"}'

    def test_includes_session_id(self):
        import pulse.server.bridge as bridge_mod
        bridge_mod._connection = None
        bridge_mod._session_id = "sess-123"

        from pulse.server.bridge import _rpc

        mock_conn = MagicMock()
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"{}"
        mock_resp.getheaders.return_value = []
        mock_conn.getresponse.return_value = mock_resp

        with patch.object(bridge_mod, "_get_conn", return_value=mock_conn):
            _rpc(json.dumps({"id": 1}).encode())

        call_headers = mock_conn.request.call_args[1]["headers"]
        assert call_headers["MCP-Session-ID"] == "sess-123"


# ===========================================================================
# main()
# ===========================================================================


class TestMain:
    def test_successful_request_roundtrip(self):
        import pulse.server.bridge as bridge_mod
        bridge_mod._session_id = None
        bridge_mod._connection = None

        from pulse.server.bridge import main

        req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        sse_response = b"data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":\"pong\"}\n\n"

        with (
            patch("sys.stdin", [req]),
            patch.object(bridge_mod, "_rpc", return_value=(sse_response, {})),
            patch.object(bridge_mod, "_send") as mock_send,
        ):
            main()

        mock_send.assert_called_once_with(
            {"jsonrpc": "2.0", "id": 1, "result": "pong"}
        )

    def test_notification_no_response(self):
        import pulse.server.bridge as bridge_mod
        bridge_mod._session_id = None
        bridge_mod._connection = None

        from pulse.server.bridge import main

        req = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})

        with (
            patch("sys.stdin", [req]),
            patch.object(bridge_mod, "_rpc", return_value=(b"{}", {})),
            patch.object(bridge_mod, "_send") as mock_send,
        ):
            main()

        mock_send.assert_not_called()

    def test_notification_still_sends_sse_payloads(self):
        import pulse.server.bridge as bridge_mod
        bridge_mod._session_id = None
        bridge_mod._connection = None

        from pulse.server.bridge import main

        req = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
        sse_response = b"data: {\"result\":\"unexpected\"}\n\n"

        with (
            patch("sys.stdin", [req]),
            patch.object(bridge_mod, "_rpc", return_value=(sse_response, {})),
            patch.object(bridge_mod, "_send") as mock_send,
        ):
            main()

        mock_send.assert_called_once_with({"result": "unexpected"})

    def test_captures_session_id(self):
        import pulse.server.bridge as bridge_mod
        bridge_mod._session_id = None
        bridge_mod._connection = None

        from pulse.server.bridge import main

        req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        sse_response = b"data: {\"result\":{}}\n\n"

        with (
            patch("sys.stdin", [req]),
            patch.object(
                bridge_mod, "_rpc",
                return_value=(sse_response, {"mcp-session-id": "sess-abc"}),
            ),
            patch.object(bridge_mod, "_send"),
        ):
            main()

        assert bridge_mod._session_id == "sess-abc"

    def test_fallback_empty_result(self):
        import pulse.server.bridge as bridge_mod
        bridge_mod._session_id = None
        bridge_mod._connection = None

        from pulse.server.bridge import main

        req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"})

        with (
            patch("sys.stdin", [req]),
            patch.object(bridge_mod, "_rpc", return_value=(b"", {})),
            patch.object(bridge_mod, "_send") as mock_send,
        ):
            main()

        mock_send.assert_called_once_with(
            {"jsonrpc": "2.0", "id": 1, "result": {}}
        )

    def test_no_fallback_for_notification(self):
        import pulse.server.bridge as bridge_mod
        bridge_mod._session_id = None
        bridge_mod._connection = None

        from pulse.server.bridge import main

        req = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})

        with (
            patch("sys.stdin", [req]),
            patch.object(bridge_mod, "_rpc", return_value=(b"", {})),
            patch.object(bridge_mod, "_send") as mock_send,
        ):
            main()

        mock_send.assert_not_called()

    def test_rpc_error_sends_error_response(self):
        import pulse.server.bridge as bridge_mod
        bridge_mod._session_id = None
        bridge_mod._connection = None

        from pulse.server.bridge import main

        req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"})

        with (
            patch("sys.stdin", [req]),
            patch.object(bridge_mod, "_rpc", side_effect=RuntimeError("timeout")),
            patch.object(bridge_mod, "_send") as mock_send,
        ):
            main()

        mock_send.assert_called_once()
        sent = mock_send.call_args[0][0]
        assert sent["id"] == 1
        assert sent["error"]["code"] == -32000
        assert "timeout" in sent["error"]["message"]

    def test_rpc_error_notification_no_error_response(self):
        import pulse.server.bridge as bridge_mod
        bridge_mod._session_id = None
        bridge_mod._connection = None

        from pulse.server.bridge import main

        req = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})

        with (
            patch("sys.stdin", [req]),
            patch.object(bridge_mod, "_rpc", side_effect=RuntimeError("timeout")),
            patch.object(bridge_mod, "_send") as mock_send,
        ):
            main()

        mock_send.assert_not_called()

    def test_skips_empty_lines(self):
        import pulse.server.bridge as bridge_mod
        bridge_mod._session_id = None
        bridge_mod._connection = None

        from pulse.server.bridge import main

        req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        sse_response = b"data: {\"result\":{}}\n\n"

        with (
            patch("sys.stdin", ["", "  ", req]),
            patch.object(bridge_mod, "_rpc", return_value=(sse_response, {})),
            patch.object(bridge_mod, "_send") as mock_send,
        ):
            main()

        mock_send.assert_called_once()

    def test_skips_invalid_json(self):
        import pulse.server.bridge as bridge_mod
        bridge_mod._session_id = None
        bridge_mod._connection = None

        from pulse.server.bridge import main

        req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        sse_response = b"data: {\"result\":{}}\n\n"

        with (
            patch("sys.stdin", ["not-json", req]),
            patch.object(bridge_mod, "_rpc", return_value=(sse_response, {})),
            patch.object(bridge_mod, "_send") as mock_send,
        ):
            main()

        mock_send.assert_called_once()

    def test_multiple_payloads_from_sse(self):
        import pulse.server.bridge as bridge_mod
        bridge_mod._session_id = None
        bridge_mod._connection = None

        from pulse.server.bridge import main

        req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        sse_response = (
            b"data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"tools\":[]}}\n\n"
        )

        with (
            patch("sys.stdin", [req]),
            patch.object(bridge_mod, "_rpc", return_value=(sse_response, {})),
            patch.object(bridge_mod, "_send") as mock_send,
        ):
            main()

        mock_send.assert_called_once()

    def test_multiple_requests(self):
        import pulse.server.bridge as bridge_mod
        bridge_mod._session_id = None
        bridge_mod._connection = None

        from pulse.server.bridge import main

        req1 = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        req2 = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"})
        sse = b"data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\n"

        with (
            patch("sys.stdin", [req1, req2]),
            patch.object(bridge_mod, "_rpc", return_value=(sse, {})),
            patch.object(bridge_mod, "_send") as mock_send,
        ):
            main()

        assert mock_send.call_count == 2
