"""Compatibility shim: cgi module removed in Python 3.13 (PEP 594).

Provides cgi.escape() and other commonly used cgi functions for
legacy tools like xsser that still import cgi.
"""
import html
import warnings

def escape(s, quote=True):
    return html.escape(s, quote=quote)

def parse_qs(qs, keep_blank_values=False, strict_parsing=False, encoding='utf-8', errors='replace'):
    from urllib.parse import parse_qs as _parse_qs
    return _parse_qs(qs, keep_blank_values=keep_blank_values, strict_parsing=strict_parsing, encoding=encoding, errors=errors)

def parse_qsl(qs, keep_blank_values=False, strict_parsing=False, encoding='utf-8', errors='replace'):
    from urllib.parse import parse_qsl as _parse_qsl
    return _parse_qsl(qs, keep_blank_values=keep_blank_values, strict_parsing=strict_parsing, encoding=encoding, errors=errors)

class FieldStorage:
    """Stub — FieldStorage requires a web environment. Raises if instantiated."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("cgi.FieldStorage is not available in Python 3.13")
