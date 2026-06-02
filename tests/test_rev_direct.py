"""Tests for mcp_core/rev_direct.py."""

import os
import tempfile
import pytest

from mcp_core.rev_direct import compute_entropy_map, _shannon_entropy


class TestShannonEntropy:
    def test_zero_length(self):
        assert _shannon_entropy(b"") == 0.0

    def test_same_byte(self):
        assert _shannon_entropy(b"\x00" * 100) == 0.0

    def test_half_half(self):
        ent = _shannon_entropy(bytes([0x00] * 128 + [0xFF] * 128))
        assert ent == 1.0  # log2(2) = 1

    def test_max_entropy(self):
        ent = _shannon_entropy(bytes(range(256)))
        assert 7.9 < ent < 8.1  # ~8.0 for uniform 256-symbol distribution

    def test_single_byte(self):
        assert _shannon_entropy(b"A") == 0.0


class TestComputeEntropyMap:
    def _make_binary(self, size=256):
        return os.urandom(size)

    def test_ok(self):
        data = self._make_binary(512)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(data)
            path = f.name
        try:
            r = compute_entropy_map(path, block_size=128)
            assert r["success"] is True
            assert r["file_size"] == 512
            assert r["block_size"] == 128
            assert r["total_blocks"] == 4
            assert len(r["blocks"]) == 4
            for b in r["blocks"]:
                assert "block" in b
                assert "offset" in b
                assert "size" in b
                assert "entropy" in b
                assert "suspicious" in b
        finally:
            os.unlink(path)

    def test_block_size_1(self):
        data = bytes(range(10))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(data)
            path = f.name
        try:
            r = compute_entropy_map(path, block_size=1)
            assert r["success"] is True
            assert r["total_blocks"] == 10
            assert all(b["size"] == 1 for b in r["blocks"])
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        r = compute_entropy_map("/nonexistent/file_xyz123.bin")
        assert r["success"] is False
        assert "not found" in r["error"]

    def test_empty_block_size(self):
        import tempfile as _tf
        with _tf.NamedTemporaryFile(delete=False, suffix=".bin") as _f:
            _f.write(b"x")
            _p = _f.name
        try:
            r = compute_entropy_map(_p, block_size=0)
            assert r["success"] is False
            assert "block_size must be >= 1" in r["error"]
        finally:
            os.unlink(_p)

    def test_high_entropy_threshold(self):
        data = os.urandom(4096)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(data)
            path = f.name
        try:
            r = compute_entropy_map(path, block_size=256, threshold=6.0)
            assert r["high_entropy_blocks"] > 0
        finally:
            os.unlink(path)

    def test_zero_entropy_threshold_high(self):
        data = b"\x00" * 512
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(data)
            path = f.name
        try:
            r = compute_entropy_map(path, block_size=256, threshold=7.0)
            assert r["high_entropy_blocks"] == 0
        finally:
            os.unlink(path)

    def test_permission_error(self):
        r = compute_entropy_map("/root/secret.bin")
        assert r["success"] is False
        assert "error" in r

    def test_partial_last_block(self):
        data = bytes(list(range(256)) * 2)[:300]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(data)
            path = f.name
        try:
            r = compute_entropy_map(path, block_size=128)
            assert r["total_blocks"] == 3
            assert r["blocks"][-1]["size"] == 44
        finally:
            os.unlink(path)
