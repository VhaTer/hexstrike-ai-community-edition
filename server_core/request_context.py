"""Request-scoped context — correlates log lines across middleware <-> tool lifecycle.

Usage (middleware):
    from server_core.request_context import generate_request_id, set_request_id
    rid = generate_request_id()
    set_request_id(rid)

Usage (tool execution / finalize):
    from server_core.request_context import get_request_id
    rid = get_request_id()
"""

from contextvars import ContextVar
import uuid

_request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def generate_request_id() -> str:
    """Return a short unique request identifier (12 hex chars)."""
    return uuid.uuid4().hex[:12]


def set_request_id(request_id: str) -> None:
    """Set the request_id for the current asynchronous context."""
    _request_id_var.set(request_id)


def get_request_id() -> str:
    """Retrieve the current request_id (empty string if not set)."""
    return _request_id_var.get()
