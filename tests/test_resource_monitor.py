import pytest
from unittest.mock import patch, MagicMock
from server_core.resource_monitor import ResourceMonitor


class TestResourceMonitor:
    def test_get_current_usage(self):
        with patch("server_core.resource_monitor.psutil") as mock_psutil:
            mock_psutil.cpu_percent.return_value = 42.0
            mock_psutil.virtual_memory.return_value.percent = 60.0
            mock_psutil.virtual_memory.return_value.available = 8 * 1024**3
            mock_psutil.disk_usage.return_value.percent = 50.0
            mock_psutil.disk_usage.return_value.free = 100 * 1024**3
            mock_psutil.net_io_counters.return_value.bytes_sent = 1000
            mock_psutil.net_io_counters.return_value.bytes_recv = 2000

            rm = ResourceMonitor()
            usage = rm.get_current_usage()
            assert usage["cpu_percent"] == 42.0
            assert len(rm.usage_history) == 1

    def test_get_current_usage_history_limit(self):
        with patch("server_core.resource_monitor.psutil") as mock_psutil:
            mock_psutil.cpu_percent.return_value = 10.0
            mock_psutil.virtual_memory.return_value.percent = 50.0
            mock_psutil.virtual_memory.return_value.available = 8 * 1024**3
            mock_psutil.disk_usage.return_value.percent = 30.0
            mock_psutil.disk_usage.return_value.free = 100 * 1024**3
            mock_psutil.net_io_counters.return_value.bytes_sent = 0
            mock_psutil.net_io_counters.return_value.bytes_recv = 0

            rm = ResourceMonitor(history_size=2)
            rm.get_current_usage()
            rm.get_current_usage()
            rm.get_current_usage()
            assert len(rm.usage_history) == 2

    def test_get_current_usage_exception(self):
        with patch("server_core.resource_monitor.psutil") as mock_psutil:
            mock_psutil.cpu_percent.side_effect = RuntimeError("oops")
            rm = ResourceMonitor()
            usage = rm.get_current_usage()
            assert usage["cpu_percent"] == 0
            assert usage["memory_percent"] == 0

    def test_get_process_usage(self):
        with patch("server_core.resource_monitor.psutil") as mock_psutil:
            mock_proc = MagicMock()
            mock_proc.cpu_percent.return_value = 5.0
            mock_proc.memory_percent.return_value = 1.0
            mock_proc.memory_info.return_value.rss = 50 * 1024**2
            mock_proc.num_threads.return_value = 4
            mock_proc.status.return_value = "running"
            mock_psutil.Process.return_value = mock_proc

            rm = ResourceMonitor()
            info = rm.get_process_usage(123)
            assert info["cpu_percent"] == 5.0
            assert info["num_threads"] == 4

    def test_get_process_usage_not_found(self):
        with patch("server_core.resource_monitor.psutil") as mock_psutil:
            class FakeNoSuchProcess(Exception):
                pass
            mock_psutil.NoSuchProcess = FakeNoSuchProcess
            mock_psutil.AccessDenied = Exception
            mock_psutil.Process.side_effect = FakeNoSuchProcess(999)
            rm = ResourceMonitor()
            info = rm.get_process_usage(999)
            assert info == {}

    def test_get_usage_trends(self):
        with patch("server_core.resource_monitor.psutil") as mock_psutil:
            mock_psutil.cpu_percent.return_value = 30.0
            mock_psutil.virtual_memory.return_value.percent = 50.0
            mock_psutil.virtual_memory.return_value.available = 8 * 1024**3
            mock_psutil.disk_usage.return_value.percent = 20.0
            mock_psutil.disk_usage.return_value.free = 100 * 1024**3
            mock_psutil.net_io_counters.return_value.bytes_sent = 0
            mock_psutil.net_io_counters.return_value.bytes_recv = 0

            rm = ResourceMonitor()
            rm.get_current_usage()
            rm.get_current_usage()
            trends = rm.get_usage_trends()
            assert trends["cpu_avg_10"] == 30.0
            assert trends["measurements"] == 2

    def test_get_usage_trends_not_enough_data(self):
        rm = ResourceMonitor()
        trends = rm.get_usage_trends()
        assert trends == {}
