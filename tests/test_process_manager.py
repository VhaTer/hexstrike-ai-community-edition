"""Coverage for process_manager.py — 42% → 100%."""

import os
import signal
from unittest.mock import Mock, patch, MagicMock, PropertyMock

import pytest

from server_core.process_manager import ProcessManager, active_processes, process_lock


@pytest.fixture(autouse=True)
def clear_processes():
    with process_lock:
        active_processes.clear()
    yield
    with process_lock:
        active_processes.clear()


def make_process_obj(alive=True, poll_result=None):
    p = MagicMock()
    p.poll.return_value = poll_result
    type(p).poll = MagicMock(return_value=poll_result)
    return p


class TestProcessManager:
    def test_register_process(self):
        proc = make_process_obj(alive=True)
        ProcessManager.register_process(123, "nmap -sV target", proc)
        with process_lock:
            assert 123 in active_processes
            assert active_processes[123]["status"] == "running"

    def test_update_progress(self):
        proc = make_process_obj(alive=True)
        ProcessManager.register_process(123, "nmap", proc)
        ProcessManager.update_process_progress(123, 0.5, "scanning...", 1024)
        with process_lock:
            assert active_processes[123]["progress"] == 0.5
            assert active_processes[123]["last_output"] == "scanning..."
            assert active_processes[123]["bytes_processed"] == 1024
            assert active_processes[123]["eta"] > 0

    def test_update_progress_zero_skip_eta(self):
        proc = make_process_obj(alive=True)
        ProcessManager.register_process(123, "nmap", proc)
        ProcessManager.update_process_progress(123, 0.0, "", 0)
        with process_lock:
            assert active_processes[123]["progress"] == 0.0
            assert active_processes[123]["eta"] == 0

    def test_update_progress_unknown_pid(self):
        ProcessManager.update_process_progress(999, 0.5)
        with process_lock:
            assert 999 not in active_processes

    def test_terminate_process(self):
        proc = make_process_obj(poll_result=None)
        ProcessManager.register_process(123, "nmap", proc)
        result = ProcessManager.terminate_process(123)
        assert result is True
        proc.terminate.assert_called_once()

    def test_terminate_process_not_running(self):
        proc = make_process_obj(poll_result=0)
        ProcessManager.register_process(123, "nmap", proc)
        result = ProcessManager.terminate_process(123)
        assert result is False

    def test_terminate_stops_after_terminate(self):
        proc = make_process_obj(poll_result=None)
        proc.poll.side_effect = [None, 0]
        ProcessManager.register_process(123, "nmap", proc)
        result = ProcessManager.terminate_process(123)
        assert result is True
        proc.terminate.assert_called_once()
        proc.kill.assert_not_called()

    def test_terminate_process_unknown_pid(self):
        result = ProcessManager.terminate_process(999)
        assert result is False

    def test_terminate_needs_kill(self):
        proc = make_process_obj(poll_result=None)
        # First poll returns None (running), then after terminate still returns None
        proc.poll.side_effect = [None, None]
        ProcessManager.register_process(123, "nmap", proc)
        result = ProcessManager.terminate_process(123)
        assert result is True
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()

    def test_terminate_exception(self):
        proc = make_process_obj(poll_result=None)
        proc.terminate.side_effect = Exception("access denied")
        ProcessManager.register_process(123, "nmap", proc)
        result = ProcessManager.terminate_process(123)
        assert result is False

    def test_cleanup_process(self):
        proc = make_process_obj(alive=True)
        ProcessManager.register_process(123, "nmap", proc)
        info = ProcessManager.cleanup_process(123)
        assert info is not None
        with process_lock:
            assert 123 not in active_processes

    def test_cleanup_unknown_pid(self):
        result = ProcessManager.cleanup_process(999)
        assert result is None

    def test_get_process_status(self):
        proc = make_process_obj(alive=True)
        ProcessManager.register_process(123, "nmap", proc)
        status = ProcessManager.get_process_status(123)
        assert status is not None
        assert status["pid"] == 123

    def test_get_process_status_unknown(self):
        assert ProcessManager.get_process_status(999) is None

    def test_list_active_processes(self):
        proc = make_process_obj(alive=True)
        ProcessManager.register_process(123, "nmap", proc)
        proc2 = make_process_obj(alive=True)
        ProcessManager.register_process(456, "sqlmap", proc2)
        processes = ProcessManager.list_active_processes()
        assert len(processes) == 2
        assert 123 in processes
        assert 456 in processes

    def test_pause_process(self):
        proc = make_process_obj(poll_result=None)
        ProcessManager.register_process(123, "nmap", proc)
        with patch("os.kill") as mock_kill:
            result = ProcessManager.pause_process(123)
        assert result is True
        mock_kill.assert_called_once_with(123, signal.SIGSTOP)
        with process_lock:
            assert active_processes[123]["status"] == "paused"

    def test_pause_not_running(self):
        proc = make_process_obj(poll_result=0)
        ProcessManager.register_process(123, "nmap", proc)
        result = ProcessManager.pause_process(123)
        assert result is False

    def test_pause_unknown_pid(self):
        result = ProcessManager.pause_process(999)
        assert result is False

    def test_pause_exception(self):
        proc = make_process_obj(poll_result=None)
        ProcessManager.register_process(123, "nmap", proc)
        with patch("os.kill", side_effect=Exception("permission denied")):
            result = ProcessManager.pause_process(123)
        assert result is False

    def test_resume_process(self):
        proc = make_process_obj(poll_result=None)
        ProcessManager.register_process(123, "nmap", proc)
        with patch("os.kill") as mock_kill:
            result = ProcessManager.resume_process(123)
        assert result is True
        mock_kill.assert_called_once_with(123, signal.SIGCONT)
        with process_lock:
            assert active_processes[123]["status"] == "running"

    def test_resume_not_running(self):
        proc = make_process_obj(poll_result=0)
        ProcessManager.register_process(123, "nmap", proc)
        result = ProcessManager.resume_process(123)
        assert result is False

    def test_resume_unknown_pid(self):
        result = ProcessManager.resume_process(999)
        assert result is False

    def test_resume_exception(self):
        proc = make_process_obj(poll_result=None)
        ProcessManager.register_process(123, "nmap", proc)
        with patch("os.kill", side_effect=Exception("permission denied")):
            result = ProcessManager.resume_process(123)
        assert result is False
