import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from server_core.enhanced_process_manager import EnhancedProcessManager


@pytest.fixture
def manager():
    with patch("server_core.enhanced_process_manager.ResourceMonitor"), \
         patch("server_core.enhanced_process_manager.ProcessPool"), \
         patch("server_core.enhanced_process_manager.AdvancedCache"), \
         patch("server_core.enhanced_process_manager.PerformanceDashboard"), \
         patch("server_core.enhanced_process_manager.threading.Thread"):
        mgr = EnhancedProcessManager()
        yield mgr


class TestEnhancedProcessManager:
    def test_init_sets_defaults(self, manager):
        assert manager.auto_scaling_enabled is True
        assert manager.resource_thresholds["cpu_high"] == 85.0

    def test_execute_command_async_returns_task_id(self, manager):
        task_id = manager.execute_command_async("echo hello")
        assert task_id.startswith("cmd_")

    def test_execute_command_async_with_cache(self, manager):
        manager.cache.get.return_value = {"cached": True}
        result = manager.execute_command_async("echo hello", {"use_cache": True})
        assert result == {"cached": True}
        manager.cache.get.assert_called_once()

    def test_execute_command_async_no_cache(self, manager):
        manager.cache.get.return_value = None
        task_id = manager.execute_command_async("echo hello", {"use_cache": True})
        assert task_id.startswith("cmd_")
        manager.process_pool.submit_task.assert_called_once()

    def test_execute_command_async_cache_disabled(self, manager):
        manager.cache.get.return_value = {"cached": True}
        task_id = manager.execute_command_async("echo hello", {"use_cache": False})
        assert task_id.startswith("cmd_")
        manager.process_pool.submit_task.assert_called_once()

    def test_get_task_result(self, manager):
        manager.get_task_result("cmd_123")
        manager.process_pool.get_task_result.assert_called_with("cmd_123")

    def test_terminate_process_gracefully_not_found(self, manager):
        with manager.registry_lock:
            pass  # registry is empty
        result = manager.terminate_process_gracefully(99999)
        assert result is False

    def test_get_comprehensive_stats_structure(self, manager):
        stats = manager.get_comprehensive_stats()
        assert "process_pool" in stats
        assert "cache" in stats
        assert "resource_usage" in stats
        assert "active_processes" in stats
        assert "auto_scaling_enabled" in stats
        assert "resource_thresholds" in stats

    def test_auto_scaling_enabled_flag(self, manager):
        assert manager.auto_scaling_enabled is True


class TestExecuteCommandInternal:
    def test_successful_execution(self, manager):
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.pid = 12345
        mock_process.communicate.return_value = ("output", "")

        with patch("subprocess.Popen", return_value=mock_process), \
             patch.object(manager.resource_monitor, 'get_current_usage',
                         return_value={"cpu_percent": 50, "memory_percent": 50}), \
             patch.object(manager.resource_monitor, 'get_process_usage',
                         return_value={"cpu": 10}):
            result = manager._execute_command_internal("echo test", {})

        assert result["success"] is True
        assert result["stdout"] == "output"

    def test_failed_execution(self, manager):
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.pid = 12346
        mock_process.communicate.return_value = ("", "error")

        with patch("subprocess.Popen", return_value=mock_process), \
             patch.object(manager.resource_monitor, 'get_current_usage',
                         return_value={"cpu_percent": 50, "memory_percent": 50}), \
             patch.object(manager.resource_monitor, 'get_process_usage',
                         return_value={"cpu": 10}):
            result = manager._execute_command_internal("false", {})

        assert result["success"] is False
        assert result["return_code"] == 1

    def test_execution_exception(self, manager):
        with patch("subprocess.Popen", side_effect=Exception("boom")), \
             patch.object(manager.resource_monitor, 'get_current_usage',
                         return_value={"cpu_percent": 50, "memory_percent": 50}):
            result = manager._execute_command_internal("crash", {})

        assert result["success"] is False
        assert "boom" in result["error"]

    def test_nice_prefix_on_high_cpu(self, manager):
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.pid = 12347
        mock_process.communicate.return_value = ("", "")

        with patch("subprocess.Popen", return_value=mock_process) as mock_popen, \
             patch.object(manager.resource_monitor, 'get_current_usage',
                         return_value={"cpu_percent": 90, "memory_percent": 50}), \
             patch.object(manager.resource_monitor, 'get_process_usage',
                         return_value={"cpu": 10}):
            manager._execute_command_internal("heavy_command", {})

        cmd_arg = mock_popen.call_args[0][0]
        assert cmd_arg.startswith("nice -n 10")

    def test_nice_not_added_when_already_nice(self, manager):
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.pid = 12348
        mock_process.communicate.return_value = ("", "")

        with patch("subprocess.Popen", return_value=mock_process) as mock_popen, \
             patch.object(manager.resource_monitor, 'get_current_usage',
                         return_value={"cpu_percent": 90, "memory_percent": 50}), \
             patch.object(manager.resource_monitor, 'get_process_usage',
                         return_value={"cpu": 10}):
            manager._execute_command_internal("nice -n 5 already_nice", {})

        cmd_arg = mock_popen.call_args[0][0]
        assert cmd_arg == "nice -n 5 already_nice"

    def test_cache_hit_in_execute_command_internal(self, manager):
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.pid = 12349
        mock_process.communicate.return_value = ("output", "")

        with patch("subprocess.Popen", return_value=mock_process), \
             patch.object(manager.resource_monitor, 'get_current_usage',
                         return_value={"cpu_percent": 50, "memory_percent": 50}), \
             patch.object(manager.resource_monitor, 'get_process_usage',
                         return_value={"cpu": 10}):
            result = manager._execute_command_internal("echo test", {"cache_result": True, "cache_ttl": 3600})

        assert result["success"] is True
        assert result["stdout"] == "output"
        manager.cache.set.assert_called_once()
        call_args = manager.cache.set.call_args[0]
        assert call_args[0].startswith("cmd_result_")
        assert call_args[1] is result
        assert call_args[2] == 3600

    def test_cache_disabled_in_execute_command_internal(self, manager):
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.pid = 12350
        mock_process.communicate.return_value = ("output", "")

        with patch("subprocess.Popen", return_value=mock_process), \
             patch.object(manager.resource_monitor, 'get_current_usage',
                         return_value={"cpu_percent": 50, "memory_percent": 50}), \
             patch.object(manager.resource_monitor, 'get_process_usage',
                         return_value={"cpu": 10}):
            result = manager._execute_command_internal("echo test", {"cache_result": False})

        assert result["success"] is True
        assert result["stdout"] == "output"
        manager.cache.set.assert_not_called()

    def test_execution_communicate_fails(self, manager):
        mock_process = MagicMock()
        mock_process.pid = 12351
        mock_process.communicate.side_effect = RuntimeError("communicate failed")

        with patch("subprocess.Popen", return_value=mock_process), \
             patch.object(manager.resource_monitor, 'get_current_usage',
                         return_value={"cpu_percent": 50, "memory_percent": 50}):
            result = manager._execute_command_internal("test", {})

        assert result["success"] is False
        assert "communicate" in result["error"]

    def test_execution_pid_not_in_registry_cleanup(self, manager):
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.pid = 12352
        mock_process.communicate.return_value = ("output", "")

        with patch("subprocess.Popen", return_value=mock_process), \
             patch.object(manager.resource_monitor, 'get_current_usage',
                         return_value={"cpu_percent": 50, "memory_percent": 50}), \
             patch.object(manager.resource_monitor, 'get_process_usage',
                         return_value={"cpu": 10}):
            class FailingDict(dict):
                def __setitem__(self, key, value):
                    raise RuntimeError("registry write failed")
            manager.process_registry = FailingDict()
            result = manager._execute_command_internal("test", {})

        assert result["success"] is False
        assert "registry" in result["error"]


class TestTerminateProcess:
    def test_terminate_gracefully(self, manager):
        mock_process = MagicMock()
        mock_process.pid = 100
        with manager.registry_lock:
            manager.process_registry[100] = {
                "command": "test", "process": mock_process,
                "start_time": 0, "context": {}, "status": "running"
            }
        result = manager.terminate_process_gracefully(100)
        assert result is True
        mock_process.terminate.assert_called_once()

    def test_force_kill_on_timeout(self, manager):
        mock_process = MagicMock()
        mock_process.pid = 101
        mock_process.wait.side_effect = __import__('subprocess').TimeoutExpired("cmd", 30)

        with manager.registry_lock:
            manager.process_registry[101] = {
                "command": "test", "process": mock_process,
                "start_time": 0, "context": {}, "status": "running"
            }
        result = manager.terminate_process_gracefully(101, timeout=1)
        assert result is True
        mock_process.kill.assert_called_once()

    def test_terminate_process_exception(self, manager):
        mock_process = MagicMock()
        mock_process.terminate.side_effect = Exception("terminate failed")
        with manager.registry_lock:
            manager.process_registry[200] = {
                "command": "test", "process": mock_process,
                "start_time": 0, "context": {}, "status": "running"
            }
        result = manager.terminate_process_gracefully(200)
        assert result is False


class TestAutoScale:
    def test_auto_scale_down_high_cpu(self, manager):
        manager.process_pool.get_pool_stats.return_value = {"active_workers": 10}
        manager.process_pool.min_workers = 4
        manager._auto_scale_based_on_resources({"cpu_percent": 90, "memory_percent": 50})
        manager.process_pool._scale_down.assert_called_once_with(1)

    def test_auto_scale_down_high_memory(self, manager):
        manager.process_pool.get_pool_stats.return_value = {"active_workers": 10}
        manager.process_pool.min_workers = 4
        manager._auto_scale_based_on_resources({"cpu_percent": 50, "memory_percent": 95})
        manager.process_pool._scale_down.assert_called_once_with(1)

    def test_auto_scale_no_down_when_at_min(self, manager):
        manager.process_pool.get_pool_stats.return_value = {"active_workers": 4}
        manager.process_pool.min_workers = 4
        manager._auto_scale_based_on_resources({"cpu_percent": 90, "memory_percent": 50})
        manager.process_pool._scale_down.assert_not_called()

    def test_auto_scale_up(self, manager):
        manager.process_pool.get_pool_stats.return_value = {"active_workers": 4, "queue_size": 5}
        manager.process_pool.min_workers = 2
        manager.process_pool.max_workers = 32
        manager._auto_scale_based_on_resources({"cpu_percent": 40, "memory_percent": 50})
        manager.process_pool._scale_up.assert_called_once_with(1)

    def test_auto_scale_no_up_when_at_max(self, manager):
        manager.process_pool.get_pool_stats.return_value = {"active_workers": 32, "queue_size": 5}
        manager.process_pool.max_workers = 32
        manager._auto_scale_based_on_resources({"cpu_percent": 40, "memory_percent": 50})
        manager.process_pool._scale_up.assert_not_called()

    def test_auto_scale_no_action(self, manager):
        manager.process_pool.get_pool_stats.return_value = {"active_workers": 4, "queue_size": 1}
        manager._auto_scale_based_on_resources({"cpu_percent": 50, "memory_percent": 50})
        manager.process_pool._scale_down.assert_not_called()
        manager.process_pool._scale_up.assert_not_called()


class TestMonitorSystem:
    def test_monitor_system_normal(self, manager):
        manager.resource_monitor.get_current_usage.return_value = {"cpu_percent": 50, "memory_percent": 50}
        with patch.object(manager, '_auto_scale_based_on_resources') as mock_auto_scale, \
             patch("server_core.enhanced_process_manager.time.sleep", side_effect=[None, KeyboardInterrupt]):
            with pytest.raises(KeyboardInterrupt):
                manager._monitor_system()
        mock_auto_scale.assert_called_once()
        manager.performance_dashboard.update_system_metrics.assert_called_once()

    def test_monitor_system_auto_scaling_disabled(self, manager):
        manager.auto_scaling_enabled = False
        manager.resource_monitor.get_current_usage.return_value = {"cpu_percent": 50, "memory_percent": 50}
        with patch.object(manager, '_auto_scale_based_on_resources') as mock_auto_scale, \
             patch("server_core.enhanced_process_manager.time.sleep", side_effect=[None, KeyboardInterrupt]):
            with pytest.raises(KeyboardInterrupt):
                manager._monitor_system()
        mock_auto_scale.assert_not_called()
        manager.performance_dashboard.update_system_metrics.assert_called_once()

    def test_monitor_system_exception(self, manager):
        with patch.object(manager, '_auto_scale_based_on_resources') as mock_auto_scale, \
             patch("server_core.enhanced_process_manager.time.sleep", side_effect=[Exception("monitor error"), KeyboardInterrupt]):
            with pytest.raises(KeyboardInterrupt):
                manager._monitor_system()
        mock_auto_scale.assert_not_called()
        manager.performance_dashboard.update_system_metrics.assert_not_called()
