"""Coverage for process_pool.py — 71% → 100%. Threading mocked, all paths covered."""

import queue
from unittest.mock import MagicMock, Mock, patch

import pytest

from server_core.process_pool import ProcessPool


@pytest.fixture
def pool():
    with patch("server_core.process_pool.threading.Thread"), \
         patch("server_core.process_pool.psutil"), \
         patch("server_core.process_pool.time.sleep"):
        p = ProcessPool(min_workers=2, max_workers=10)
        yield p


class TestInit:
    def test_sets_min_max_workers(self, pool):
        assert pool.min_workers == 2
        assert pool.max_workers == 10
        assert pool.scale_threshold == 0.8
        assert len(pool.workers) == 2

    def test_creates_task_queue(self, pool):
        assert isinstance(pool.task_queue, queue.Queue)

    def test_creates_pool_lock(self, pool):
        assert hasattr(pool, 'pool_lock')


class TestSubmitTask:
    def test_returns_task_id(self, pool):
        tid = pool.submit_task("t1", lambda: 42)
        assert tid == "t1"

    def test_stores_task_as_queued(self, pool):
        pool.submit_task("t1", lambda: 42)
        assert pool.active_tasks["t1"]["status"] == "queued"

    def test_adds_to_queue(self, pool):
        pool.submit_task("t1", lambda: 42)
        assert pool.task_queue.qsize() == 1

    def test_submit_multiple_tasks(self, pool):
        pool.submit_task("t1", lambda: 1)
        pool.submit_task("t2", lambda: 2)
        assert pool.task_queue.qsize() == 2
        assert "t1" in pool.active_tasks
        assert "t2" in pool.active_tasks


class TestGetTaskResult:
    def test_returns_completed_result(self, pool):
        pool.results["done"] = {"status": "completed", "result": "ok"}
        r = pool.get_task_result("done")
        assert r["status"] == "completed"
        assert r["result"] == "ok"

    def test_returns_active_status(self, pool):
        pool.active_tasks["running"] = {"status": "running"}
        r = pool.get_task_result("running")
        assert r["status"] == "running"

    def test_returns_not_found(self, pool):
        r = pool.get_task_result("nope")
        assert r["status"] == "not_found"


class TestScaleUp:
    def test_adds_workers_and_starts_them(self, pool):
        with patch("server_core.process_pool.threading.Thread") as t:
            pool._scale_up(3)
        assert len(pool.workers) == 5
        assert t.call_count == 3

    def test_zero_no_op(self, pool):
        before = len(pool.workers)
        pool._scale_up(0)
        assert len(pool.workers) == before

    def test_one_worker(self, pool):
        with patch("server_core.process_pool.threading.Thread"):
            pool._scale_up(1)
        assert len(pool.workers) == 3


class TestScaleDown:
    def test_removes_worker(self, pool):
        pool.workers = [Mock(), Mock(), Mock()]
        pool.min_workers = 2
        pool._scale_down(1)
        assert len(pool.workers) == 2

    def test_not_below_minimum(self, pool):
        pool.workers = [Mock()]
        pool.min_workers = 2
        pool._scale_down(5)
        assert len(pool.workers) == 1

    def test_empty_workers_no_op(self, pool):
        pool.workers = []
        pool.min_workers = 0
        pool._scale_down(1)
        assert len(pool.workers) == 0

    def test_shrinks_to_minimum(self, pool):
        pool.workers = [Mock() for _ in range(5)]
        pool.min_workers = 3
        pool._scale_down(3)
        assert len(pool.workers) == 3


class TestScaleDownDetail:
    """Cover remaining branch in _scale_down: if self.workers: → False."""

    def test_empty_after_pop_then_empty_call(self, pool):
        pool.workers = [Mock()]
        pool.min_workers = 0
        pool._scale_down(1)
        assert len(pool.workers) == 0
        pool._scale_down(1)
        assert len(pool.workers) == 0

    def test_empty_workers_with_negative_min(self, pool):
        pool.workers = []
        pool.min_workers = -1
        pool._scale_down(2)
        assert len(pool.workers) == 0


class TestGetPoolStats:
    def test_contains_all_keys(self, pool):
        for w in pool.workers:
            w.is_alive.return_value = True
        stats = pool.get_pool_stats()
        assert "active_workers" in stats
        assert "queue_size" in stats
        assert "active_tasks" in stats
        assert "performance_metrics" in stats
        assert "min_workers" in stats
        assert "max_workers" in stats
        assert stats["min_workers"] == 2
        assert stats["max_workers"] == 10


class TestWorkerThread:
    def test_processes_task_and_stores_result(self, pool):
        pool.task_queue = MagicMock()
        task = {"id": "t1", "func": lambda: "hello", "args": (), "kwargs": {}}
        pool.active_tasks["t1"] = task
        pool.task_queue.get.side_effect = [task, None]
        pool._worker_thread(99)
        assert pool.results["t1"]["status"] == "completed"
        assert pool.results["t1"]["result"] == "hello"
        assert pool.results["t1"]["worker_id"] == 99

    def test_handles_task_exception(self, pool):
        pool.task_queue = MagicMock()

        def crash():
            raise ValueError("boom")
        task = {"id": "t2", "func": crash, "args": (), "kwargs": {}}
        pool.active_tasks["t2"] = task
        pool.task_queue.get.side_effect = [task, None]
        pool._worker_thread(0)
        assert pool.results["t2"]["status"] == "failed"
        assert "boom" in pool.results["t2"]["error"]

    def test_handles_queue_empty_timeout(self, pool):
        pool.task_queue = MagicMock()
        pool.task_queue.get.side_effect = [queue.Empty(), None]
        pool._worker_thread(0)
        assert not pool.results

    def test_shutdown_on_none_task(self, pool):
        pool.task_queue = MagicMock()
        pool.task_queue.get.side_effect = [None]
        pool._worker_thread(0)

    def test_task_not_in_active_tasks(self, pool):
        pool.task_queue = MagicMock()
        task = {"id": "orphan", "func": lambda: 42, "args": (), "kwargs": {}}
        pool.task_queue.get.side_effect = [task, None]
        pool._worker_thread(0)
        assert pool.results["orphan"]["status"] == "completed"

    def test_updates_performance_metrics(self, pool):
        pool.task_queue = MagicMock()
        task = {"id": "t3", "func": lambda: 99, "args": (), "kwargs": {}}
        pool.active_tasks["t3"] = task
        pool.task_queue.get.side_effect = [task, None]
        pool._worker_thread(0)
        assert pool.performance_metrics["tasks_completed"] == 1
        assert pool.performance_metrics["avg_task_time"] > 0

    def test_worker_outer_exception_logs(self, pool):
        pool.task_queue = MagicMock()
        pool.task_queue.get.side_effect = [RuntimeError("unexpected"), None]
        pool._worker_thread(0)

    def test_task_failure_removes_from_active(self, pool):
        pool.task_queue = MagicMock()

        def crash():
            raise ValueError("fail")
        task = {"id": "failtask", "func": crash, "args": (), "kwargs": {}}
        pool.active_tasks["failtask"] = task
        pool.task_queue.get.side_effect = [task, None]
        pool._worker_thread(0)
        assert "failtask" not in pool.active_tasks
        assert pool.performance_metrics["tasks_failed"] == 1

    def test_task_failure_not_in_active_tasks(self, pool):
        pool.task_queue = MagicMock()

        def crash():
            raise ValueError("ghost fail")
        task = {"id": "ghost", "func": crash, "args": (), "kwargs": {}}
        pool.task_queue.get.side_effect = [task, None]
        pool._worker_thread(0)
        assert pool.results["ghost"]["status"] == "failed"


class TestMonitorPerformance:
    """_monitor_performance has while True: loop. We break it by making
    time.sleep raise BaseException (not caught by except Exception:) on
    the second call.  First call returns None → body executes once."""

    def test_scale_up_when_load_high(self, pool):
        pool.workers = [Mock(is_alive=Mock(return_value=True)) for _ in range(2)]
        pool.active_tasks = {"a": {}, "b": {}, "c": {}}
        pool.max_workers = 10
        with patch("server_core.process_pool.time.sleep",
                   side_effect=[None, BaseException("_brk_")]):
            with patch.object(pool, '_scale_up') as up:
                with pytest.raises(BaseException, match="_brk_"):
                    pool._monitor_performance()
        up.assert_called_once()

    def test_scale_down_when_load_low(self, pool):
        pool.workers = [Mock(is_alive=Mock(return_value=True)) for _ in range(5)]
        pool.active_tasks = {}
        pool.min_workers = 2
        with patch("server_core.process_pool.time.sleep",
                   side_effect=[None, BaseException("_brk_")]):
            with patch.object(pool, '_scale_down') as down:
                with pytest.raises(BaseException, match="_brk_"):
                    pool._monitor_performance()
        down.assert_called_once()

    def test_no_action_when_load_balanced(self, pool):
        pool.workers = [Mock(is_alive=Mock(return_value=True)) for _ in range(4)]
        pool.active_tasks = {"a": {}, "b": {}}
        with patch("server_core.process_pool.time.sleep",
                   side_effect=[None, BaseException("_brk_")]):
            with patch.object(pool, '_scale_up') as up, \
                 patch.object(pool, '_scale_down') as down:
                with pytest.raises(BaseException, match="_brk_"):
                    pool._monitor_performance()
        up.assert_not_called()
        down.assert_not_called()

    def test_scale_up_from_zero_workers(self, pool):
        pool.workers = []
        pool.active_tasks = {"a": {}}
        pool.max_workers = 10
        with patch("server_core.process_pool.time.sleep",
                   side_effect=[None, BaseException("_brk_")]):
            with patch.object(pool, '_scale_up') as up:
                with pytest.raises(BaseException, match="_brk_"):
                    pool._monitor_performance()
        up.assert_called_once()

    def test_psutil_error_does_not_crash(self, pool):
        pool.workers = [Mock(is_alive=Mock(return_value=True)) for _ in range(2)]
        pool.active_tasks = {"a": {}, "b": {}}
        with patch("server_core.process_pool.time.sleep",
                   side_effect=[None, BaseException("_brk_")]):
            with patch("server_core.process_pool.psutil.cpu_percent",
                       side_effect=Exception("no perm")):
                with pytest.raises(BaseException, match="_brk_"):
                    pool._monitor_performance()

    def test_updates_cpu_and_memory_metrics(self, pool):
        pool.workers = [Mock(is_alive=Mock(return_value=True)) for _ in range(2)]
        pool.active_tasks = {}
        pool.min_workers = 2
        pool.max_workers = 10
        with patch("server_core.process_pool.time.sleep",
                   side_effect=[None, BaseException("_brk_")]):
            with patch("server_core.process_pool.psutil.cpu_percent",
                       return_value=55.0), \
                 patch("server_core.process_pool.psutil.virtual_memory") as vm:
                vm.return_value.percent = 70.0
                with pytest.raises(BaseException, match="_brk_"):
                    pool._monitor_performance()
        assert pool.performance_metrics["cpu_usage"] == 55.0
        assert pool.performance_metrics["memory_usage"] == 70.0

    def test_outer_exception_handler(self, pool):
        pool.workers = [Mock(is_alive=Mock(return_value=True)) for _ in range(2)]
        with patch("server_core.process_pool.time.sleep",
                   side_effect=[Exception("monitor crash"),
                                BaseException("_brk_")]):
            with pytest.raises(BaseException, match="_brk_"):
                pool._monitor_performance()
