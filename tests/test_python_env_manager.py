import pytest
from unittest.mock import patch, MagicMock
from server_core.python_env_manager import PythonEnvironmentManager


@pytest.fixture
def manager(tmp_path):
    with patch("server_core.python_env_manager.venv.create"):
        mgr = PythonEnvironmentManager(base_dir=str(tmp_path / "envs"))
        yield mgr


class TestPythonEnvironmentManager:
    def test_create_venv(self, manager, tmp_path):
        env_path = manager.create_venv("test_env")
        assert str(env_path) == str(tmp_path / "envs" / "test_env")

    def test_create_venv_already_exists(self, manager, tmp_path):
        env_dir = tmp_path / "envs" / "existing_env"
        env_dir.mkdir(parents=True)
        with patch("server_core.python_env_manager.venv.create") as mock_venv:
            env_path = manager.create_venv("existing_env")
            mock_venv.assert_not_called()
            assert str(env_path) == str(env_dir)

    def test_create_venv_twice_does_not_fail(self, manager):
        env_path1 = manager.create_venv("test_env")
        env_path2 = manager.create_venv("test_env")
        assert env_path1 == env_path2

    @patch("server_core.python_env_manager.subprocess.run")
    def test_install_package_success(self, mock_run, manager):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = manager.install_package("test_env", "requests")
        assert result is True
        mock_run.assert_called_once()

    @patch("server_core.python_env_manager.subprocess.run")
    def test_install_package_failure(self, mock_run, manager):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        result = manager.install_package("test_env", "nonexistent")
        assert result is False

    @patch("server_core.python_env_manager.subprocess.run")
    def test_install_package_exception(self, mock_run, manager):
        mock_run.side_effect = Exception("network error")
        result = manager.install_package("test_env", "requests")
        assert result is False

    def test_get_python_path(self, manager):
        py_path = manager.get_python_path("test_env")
        assert "bin/python" in py_path
        assert "test_env" in py_path

    def test_base_dir_creation(self, tmp_path):
        base = tmp_path / "custom_envs"
        with patch("server_core.python_env_manager.venv.create"):
            mgr = PythonEnvironmentManager(base_dir=str(base))
            assert base.exists()

    def test_global_env_manager_singleton(self):
        from server_core.python_env_manager import env_manager
        assert isinstance(env_manager, PythonEnvironmentManager)
        assert str(env_manager.base_dir) == "/tmp/hexstrike_envs"
