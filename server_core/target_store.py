"""
server_core/target_store.py

Persistent per-target findings and session history.

Stores discovered ports, services, technologies, vulnerabilities, and session
metadata for each scanned target.  Data is written to a JSON file in the
standard HexStrike data directory so it survives server restarts.

Design:
  - KISS: plain JSON file, one dict of {target: target_data}
  - Thread-safe via a single lock
  - Atomic writes (write to .tmp then os.replace) to avoid corruption
  - Idempotent: safe to call record_scan() repeatedly
"""

import copy
import json
import logging
import os
import threading
import time
from typing import Any, Optional

import server_core.config_core as config_core
from beartype import beartype

logger = logging.getLogger(__name__)

STORE_FILE_NAME = "target_store.json"


class TargetStore:
    """
    Persistent per-target scan findings and metadata.

    Public API:
        get_target(target)          — full target profile or None
        get_all_targets()           — list of all known targets with summary
        record_scan(target, surface_data, findings, session_id=None, tools_used=None)
        add_finding(target, finding_type, data, tool)
    """

    def __init__(self, data_dir: Optional[str] = None) -> None:
        self._data_dir = data_dir or config_core.resolve_data_dir()
        self._store_path = os.path.join(self._data_dir, STORE_FILE_NAME)
        self._lock = threading.Lock()
        self._store: dict[str, dict[str, Any]] = {}
        config_core.ensure_data_dir()
        self._load()

    # ── Public API ────────────────────────────────────────────────────

    @beartype
    def get_target(self, target: str) -> Optional[dict[str, Any]]:
        """Return full target profile (with injected target key), or None."""
        with self._lock:
            raw = self._store.get(target)
            if raw:
                result = copy.deepcopy(raw)
                result["target"] = target
                return result
            return None

    @beartype
    def get_all_targets(self) -> list[dict[str, Any]]:
        """Return a summary list of all known targets."""
        with self._lock:
            return [
                {
                    "target": t,
                    "first_seen":  d.get("first_seen"),
                    "last_seen":   d.get("last_seen"),
                    "scan_count":  d.get("scan_count", 0),
                    "tools_count": len(d.get("tools_used", [])),
                    "findings": {
                        "ports": len(d.get("findings", {}).get("ports", [])),
                        "services": len(d.get("findings", {}).get("services", [])),
                        "technologies": len(d.get("findings", {}).get("technologies", [])),
                        "vulnerabilities": len(d.get("findings", {}).get("vulnerabilities", [])),
                    },
                }
                for t, d in sorted(self._store.items())
            ]

    @beartype
    def record_scan(
        self,
        target: str,
        surface_data: Optional[dict[str, Any]] = None,
        findings: Optional[list[dict[str, Any]]] = None,
        session_id: Optional[str] = None,
        tools_used: Optional[list[str]] = None,
    ) -> None:
        """Record a scan result for a target. Creates or updates the profile."""
        now = time.time()
        with self._lock:
            entry = self._store.setdefault(target, {
                "first_seen": now,
                "last_seen":  now,
                "scan_count": 0,
                "tools_used": [],
                "findings": {
                    "ports": [],
                    "services": [],
                    "technologies": [],
                    "vulnerabilities": [],
                },
                "sessions": [],
                "last_tool_runs": [],
            })
            entry["last_seen"] = now
            entry["scan_count"] = entry.get("scan_count", 0) + 1

            if tools_used:
                existing = set(entry.get("tools_used", []))
                for t in tools_used:
                    if t not in existing:
                        entry["tools_used"].append(t)
                        existing.add(t)

            if surface_data:
                findings_dict = entry.setdefault("findings", {})
                self._merge_surface(findings_dict, surface_data)

            if findings:
                findings_dict = entry.setdefault("findings", {})
                self._merge_findings(findings_dict, findings)

            if session_id:
                sessions = entry.setdefault("sessions", [])
                sessions.append({
                    "session_id": session_id,
                    "ts": now,
                    "tools": tools_used or [],
                    "findings_count": len(findings or []),
                })

            self._save_locked()

    @beartype
    def add_finding(
        self,
        target: str,
        finding_type: str,
        data: Any,
        tool: str = "",
    ) -> None:
        """Add a single finding (port, service, vuln, tech) to a target profile."""
        now = time.time()
        with self._lock:
            entry = self._store.get(target)
            if not entry:
                return
            findings_dict = entry.setdefault("findings", {})
            if finding_type == "port":
                ports = findings_dict.setdefault("ports", [])
                if data not in ports:
                    ports.append(data)
            elif finding_type == "service":
                services = findings_dict.setdefault("services", [])
                if data not in services:
                    services.append(data)
            elif finding_type == "technology":
                techs = findings_dict.setdefault("technologies", [])
                if data not in techs:
                    techs.append(data)
            elif finding_type == "vulnerability":
                vulns = findings_dict.setdefault("vulnerabilities", [])
                vulns.append({**data, "tool": tool, "ts": now})
            runs = entry.setdefault("last_tool_runs", [])
            runs.append({"tool": tool, "ts": now, "finding_type": finding_type})
            entry["last_seen"] = now
            self._save_locked()

    # ── Internal ──────────────────────────────────────────────────────

    def _merge_surface(self, findings_dict: dict, surface: dict) -> None:
        for p in surface.get("ports", []):
            port_entry = {"port": p.get("port"), "service": p.get("service", ""), "product": p.get("product", "")}
            existing_ports = findings_dict.setdefault("ports", [])
            if not any(e["port"] == port_entry["port"] for e in existing_ports):
                existing_ports.append(port_entry)
        for t in surface.get("technologies", []):
            techs = findings_dict.setdefault("technologies", [])
            if t not in techs:
                techs.append(t)
        for s in surface.get("services", []):
            svcs = findings_dict.setdefault("services", [])
            if s not in svcs:
                svcs.append(s)

    def _merge_findings(self, findings_dict: dict, findings: list) -> None:
        vulns = findings_dict.setdefault("vulnerabilities", [])
        existing_ids = {v.get("id") for v in vulns}
        for f in findings:
            fid = f.get("id", f.get("finding", ""))
            if fid and fid not in existing_ids:
                vulns.append({
                    "id": fid,
                    "severity": f.get("severity", "info"),
                    "tool": f.get("tool", ""),
                    "details": f.get("details", ""),
                    "ts": time.time(),
                })
                existing_ids.add(fid)

    def _load(self) -> None:
        if not os.path.exists(self._store_path):
            self._store = {}
            return
        try:
            with open(self._store_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                self._store = raw
            else:
                self._store = {}
            logger.debug("target_store: loaded %d targets from %s", len(self._store), self._store_path)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning("target_store: could not load %s (%s) — starting fresh", self._store_path, exc)
            self._store = {}

    def _save_locked(self) -> None:
        tmp = self._store_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._store, f, indent=2)
            os.replace(tmp, self._store_path)
        except OSError as exc:
            logger.error("target_store: failed to save %s: %s", self._store_path, exc)
