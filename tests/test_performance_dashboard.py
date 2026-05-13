import time
from server_core.performance_dashboard import PerformanceDashboard


class TestPerformanceDashboard:
    def test_record_execution(self):
        d = PerformanceDashboard()
        d.record_execution("nmap -sV target", {"success": True, "execution_time": 1.5, "return_code": 0})
        assert len(d.execution_history) == 1
        assert d.execution_history[0]["success"] is True

    def test_record_execution_max_history(self):
        d = PerformanceDashboard()
        d.max_history = 2
        for i in range(5):
            d.record_execution(f"cmd{i}", {"success": True, "execution_time": 0.1, "return_code": 0})
        assert len(d.execution_history) == 2

    def test_update_system_metrics(self):
        d = PerformanceDashboard()
        d.update_system_metrics({"cpu": 50})
        assert len(d.system_metrics) == 1

    def test_update_system_metrics_max_history(self):
        d = PerformanceDashboard()
        d.max_history = 1
        d.update_system_metrics({"cpu": 50})
        d.update_system_metrics({"cpu": 60})
        assert len(d.system_metrics) == 1

    def test_get_summary_empty(self):
        d = PerformanceDashboard()
        summary = d.get_summary()
        assert summary["executions"] == 0

    def test_get_summary_with_data(self):
        d = PerformanceDashboard()
        d.record_execution("cmd", {"success": True, "execution_time": 1.0, "return_code": 0})
        d.record_execution("cmd", {"success": False, "execution_time": 2.0, "return_code": 1})
        summary = d.get_summary()
        assert summary["total_executions"] == 2
        assert summary["success_rate"] == 50.0
        assert summary["avg_execution_time"] == 1.5
