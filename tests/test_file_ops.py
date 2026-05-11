import pytest
from server_core.file_ops import FileOperationsManager


@pytest.fixture
def mgr(tmp_path):
    return FileOperationsManager(base_dir=str(tmp_path / "files"))


class TestFileOperationsManager:
    def test_create_file(self, mgr):
        result = mgr.create_file("hello.txt", "Hello World")
        assert result["success"] is True
        assert result["size"] == 11

    def test_create_file_in_subdir(self, mgr):
        result = mgr.create_file("sub/dir/test.txt", "content")
        assert result["success"] is True

    def test_create_binary_file(self, mgr):
        result = mgr.create_file("data.bin", "\x00\x01\x02", binary=True)
        assert result["success"] is True

    def test_create_binary_file_bytes_content(self, mgr):
        result = mgr.create_file("data.bin", b"\x00\x01\x02", binary=True)
        assert result["success"] is True

    def test_create_file_too_large(self, mgr):
        big = "x" * (100 * 1024 * 1024 + 1)
        result = mgr.create_file("big.txt", big)
        assert result["success"] is False
        assert "exceeds" in result["error"]

    def test_modify_file(self, mgr):
        mgr.create_file("edit.txt", "original")
        result = mgr.modify_file("edit.txt", " modified")
        assert result["success"] is True

    def test_modify_file_append(self, mgr):
        mgr.create_file("append.txt", "line1")
        result = mgr.modify_file("append.txt", "\nline2", append=True)
        assert result["success"] is True

    def test_modify_nonexistent(self, mgr):
        result = mgr.modify_file("nonexistent.txt", "content")
        assert result["success"] is False
        assert "not exist" in result["error"]

    def test_modify_file_path_traversal(self, mgr):
        result = mgr.modify_file("../../etc/passwd", "hack")
        assert result["success"] is False

    def test_delete_file(self, mgr):
        mgr.create_file("todelete.txt", "content")
        result = mgr.delete_file("todelete.txt")
        assert result["success"] is True

    def test_delete_nonexistent(self, mgr):
        result = mgr.delete_file("nonexistent.txt")
        assert result["success"] is False

    def test_delete_directory(self, mgr):
        mgr.create_file("mydir/a.txt", "content")
        result = mgr.delete_file("mydir")
        assert result["success"] is True

    def test_delete_file_path_traversal(self, mgr):
        result = mgr.delete_file("../../etc/passwd")
        assert result["success"] is False

    def test_list_files(self, mgr):
        mgr.create_file("a.txt", "aaa")
        mgr.create_file("b.txt", "bbb")
        result = mgr.list_files(".")
        assert result["success"] is True
        assert len(result["files"]) == 2

    def test_list_files_with_subdirs(self, mgr):
        mgr.create_file("subdir/a.txt", "content")
        mgr.create_file("file.txt", "content")
        result = mgr.list_files(".")
        files_by_name = {f["name"]: f for f in result["files"]}
        assert files_by_name["subdir"]["type"] == "directory"
        assert files_by_name["file.txt"]["type"] == "file"

    def test_list_nonexistent_dir(self, mgr):
        result = mgr.list_files("nonexistent")
        assert result["success"] is False

    def test_list_files_path_traversal(self, mgr):
        result = mgr.list_files("../../etc")
        assert result["success"] is False

    def test_path_traversal_prevention(self, mgr):
        result = mgr.create_file("../../etc/passwd", "hack")
        assert result["success"] is False

    def test_safe_path_valid(self, mgr):
        path = mgr._safe_path("test.txt")
        assert "test.txt" in str(path)

    def test_safe_path_traversal(self, mgr):
        with pytest.raises(ValueError, match="Path traversal"):
            mgr._safe_path("../../etc/passwd")
