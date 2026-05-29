"""
tests/test_telemetry_collector.py

Unit tests for server_core/telemetry_collector.py (TelemetryCollector).

Covers:
  - Constructor initialises stats with zero defaults
  - record_execution(success=True) increments counters
  - record_execution(success=False) increments failure counter
  - get_system_metrics() returns expected keys
  - get_stats() returns formatted output including uptime/success rate
  - Edge: zero executions returns 0% success rate
"""

import time
import pytest
from server_core.telemetry_collector import TelemetryCollector


def test_constructor_defaults():
    c = TelemetryCollector()
    assert c.stats["commands_executed"] == 0
    assert c.stats["successful_commands"] == 0
    assert c.stats["failed_commands"] == 0
    assert c.stats["total_execution_time"] == 0.0
    assert c.stats["start_time"] > 0


def test_record_execution_success():
    c = TelemetryCollector()
    c.record_execution(success=True, execution_time=1.5)
    assert c.stats["commands_executed"] == 1
    assert c.stats["successful_commands"] == 1
    assert c.stats["failed_commands"] == 0
    assert c.stats["total_execution_time"] == 1.5


def test_record_execution_failure():
    c = TelemetryCollector()
    c.record_execution(success=False, execution_time=0.3)
    assert c.stats["commands_executed"] == 1
    assert c.stats["successful_commands"] == 0
    assert c.stats["failed_commands"] == 1
    assert c.stats["total_execution_time"] == 0.3


def test_record_execution_mixed():
    c = TelemetryCollector()
    c.record_execution(success=True, execution_time=1.0)
    c.record_execution(success=False, execution_time=2.0)
    c.record_execution(success=True, execution_time=3.0)
    assert c.stats["commands_executed"] == 3
    assert c.stats["successful_commands"] == 2
    assert c.stats["failed_commands"] == 1
    assert c.stats["total_execution_time"] == 6.0


def test_get_system_metrics_keys():
    c = TelemetryCollector()
    metrics = c.get_system_metrics()
    assert "cpu_percent" in metrics
    assert "memory_percent" in metrics
    assert "disk_usage" in metrics
    assert "network_io" in metrics


def test_get_stats_zero_executions():
    c = TelemetryCollector()
    stats = c.get_stats()
    assert stats["commands_executed"] == 0
    assert stats["success_rate"] == "0.0%"
    assert stats["average_execution_time"] == "0.00s"
    assert "uptime_seconds" in stats
    assert "system_metrics" in stats


def test_get_stats_after_executions():
    c = TelemetryCollector()
    c.record_execution(success=True, execution_time=2.0)
    c.record_execution(success=False, execution_time=1.0)
    stats = c.get_stats()
    assert stats["commands_executed"] == 2
    assert stats["success_rate"] == "50.0%"
    assert stats["average_execution_time"] == "1.50s"


def test_get_stats_uptime_increases():
    c = TelemetryCollector()
    t0 = c.get_stats()["uptime_seconds"]
    time.sleep(0.01)
    t1 = c.get_stats()["uptime_seconds"]
    assert t1 > t0
