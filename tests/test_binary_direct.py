"""Tests for mcp_core/binary_direct.py."""

import json
import os
import pytest

from mcp_core.binary_direct import (
    binary_varint,
    binary_block_parser,
    binary_rle_decode,
    binary_signal_decode,
    binary_exec,
    _parse_varints,
    _to_bytes,
)


class TestToBytes:
    def test_basic_hex(self):
        assert _to_bytes("48656c6c6f") == b"Hello"

    def test_with_0x_prefix(self):
        assert _to_bytes("0x48656c6c6f") == b"Hello"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Empty hex"):
            _to_bytes("")

    def test_invalid_hex_raises(self):
        with pytest.raises(ValueError):
            _to_bytes("ZZZZ")


class TestParseVarints:
    def test_single_byte(self):
        assert _parse_varints(b"\x01") == [1]

    def test_two_byte(self):
        assert _parse_varints(b"\xff\x7f") == [16383]

    def test_three_byte(self):
        assert _parse_varints(b"\xff\xff\x7f") == [2097151]

    def test_zero(self):
        assert _parse_varints(b"\x00") == [0]

    def test_multiple_values(self):
        assert _parse_varints(b"\x01\x02\x03") == [1, 2, 3]

    def test_known_sequence(self):
        """Recreate the .sal idle block pattern: 72 ff 7f 01 00 00"""
        result = _parse_varints(bytes.fromhex("72ff7f010000"))
        assert result == [114, 16383, 1, 0, 0]

    def test_empty(self):
        assert _parse_varints(b"") == []


class TestBinaryVarint:
    def test_single_value(self):
        r = binary_varint(data_hex="01")
        assert r["success"] is True
        assert r["values"] == [1]
        assert r["count"] == 1

    def test_multi_value(self):
        r = binary_varint(data_hex="010203")
        assert r["values"] == [1, 2, 3]

    def test_signed_positive(self):
        r = binary_varint(data_hex="01", encoding="leb128_signed")
        assert r["values"] == [1]

    def test_signed_negative(self):
        r = binary_varint(data_hex="7f", encoding="leb128_signed")
        assert r["values"] == [-1]

    def test_sal_idle_pattern(self):
        r = binary_varint(data_hex="72ff7f010000")
        assert r["values"] == [114, 16383, 1, 0, 0]

    def test_invalid_hex(self):
        r = binary_varint(data_hex="ZZ")
        assert r["success"] is False

    def test_empty_hex_returns_empty(self):
        r = binary_varint(data_hex="")
        assert r["success"] is True
        assert r["values"] == []

    def test_unknown_encoding(self):
        r = binary_varint(data_hex="01", encoding="unknown")
        assert r["success"] is True  # default to unsigned
        assert r["values"] == [1]


class TestBinaryBlockParser:
    def test_single_block_uint32(self):
        data = "deadbeef"  # 4 bytes
        fields = [{"name": "magic", "offset": 0, "size": 4, "type": "uint32"}]
        r = binary_block_parser(data_hex=data, block_size=4, fields=fields)
        assert r["success"] is True
        assert r["block_count"] == 1
        assert r["blocks"][0]["magic"] == 0xEFBEADDE

    def test_multiple_blocks(self):
        data = "0100000002000000"  # 8 bytes, 2 uint32 blocks
        fields = [{"name": "val", "offset": 0, "size": 4, "type": "uint32"}]
        r = binary_block_parser(data_hex=data, block_size=4, fields=fields)
        assert r["block_count"] == 2
        assert r["blocks"][0]["val"] == 1
        assert r["blocks"][1]["val"] == 2

    def test_ascii_field(self):
        data = "48656c6c6f"  # "Hello"
        fields = [{"name": "msg", "offset": 0, "size": 5, "type": "ascii"}]
        r = binary_block_parser(data_hex=data, block_size=5, fields=fields)
        assert r["blocks"][0]["msg"] == "Hello"

    def test_uint8_field(self):
        data = "010203"
        fields = [{"name": "a", "offset": 0, "size": 1, "type": "uint8"},
                  {"name": "b", "offset": 1, "size": 1, "type": "uint8"},
                  {"name": "c", "offset": 2, "size": 1, "type": "uint8"}]
        r = binary_block_parser(data_hex=data, block_size=3, fields=fields)
        assert r["blocks"][0]["a"] == 1
        assert r["blocks"][0]["b"] == 2
        assert r["blocks"][0]["c"] == 3

    def test_bytes_field(self):
        data = "deadbeef"
        fields = [{"name": "raw", "offset": 0, "size": 4, "type": "bytes"}]
        r = binary_block_parser(data_hex=data, block_size=4, fields=fields)
        assert r["blocks"][0]["raw"] == "deadbeef"

    def test_invalid_block_size(self):
        r = binary_block_parser(data_hex="00", block_size=0)
        assert r["success"] is False

    def test_block_size_larger_than_data(self):
        r = binary_block_parser(data_hex="00", block_size=10)
        assert r["success"] is False

    def test_invalid_hex(self):
        r = binary_block_parser(data_hex="ZZ", block_size=1)
        assert r["success"] is False

    def test_field_beyond_block(self):
        data = "0001"
        fields = [{"name": "x", "offset": 0, "size": 10, "type": "uint8"}]
        r = binary_block_parser(data_hex=data, block_size=2, fields=fields)
        assert r["blocks"][0]["x"] is None


class TestBinaryRleDecode:
    def test_value_count(self):
        """[1, 3, 0, 2] -> 1,1,1,0,0"""
        data = "0103"  # varints: [1, 3]  (two single-byte varints)
        # Actually we need hex of varint bytes
        # varint(1) = 01, varint(3) = 03
        r = binary_rle_decode(data_hex="01030002", format="value_count")
        # varints: [1, 3, 0, 2] -> [1, 1, 1, 0, 0]
        assert r["success"] is True
        assert r["signal_values"] == [1, 1, 1, 0, 0]
        assert r["sample_count"] == 5

    def test_toggle_count(self):
        """[2, 3, 1] with initial_state=0 -> 0,0,1,1,1,0"""
        data = "020301"  # varints: [2, 3, 1]
        r = binary_rle_decode(data_hex=data, format="toggle_count", initial_state=0)
        assert r["signal_values"] == [0, 0, 1, 1, 1, 0]

    def test_toggle_count_initial_1(self):
        """[2, 3] with initial_state=1 -> 1,1,0,0,0"""
        data = "0203"
        r = binary_rle_decode(data_hex=data, format="toggle_count", initial_state=1)
        assert r["signal_values"] == [1, 1, 0, 0, 0]

    def test_pulse_width(self):
        """[2, 3, 1] with initial_state=0 -> 0,0,1,1,1,0"""
        data = "020301"
        r = binary_rle_decode(data_hex=data, format="pulse_width", initial_state=0)
        assert r["signal_values"] == [0, 0, 1, 1, 1, 0]

    def test_sal_idle_block(self):
        """[114, 16383, 1, 0, 0] toggle_count initial_state=3(?) -> idle pattern"""
        # Actually idle state is HIGH (1) for UART
        data = "72ff7f010000"
        r = binary_rle_decode(data_hex=data, format="toggle_count", initial_state=1)
        assert r["success"] is True
        assert r["sample_count"] == 114 + 16383 + 1 + 0 + 0

    def test_empty_values(self):
        r = binary_rle_decode(data_hex="")
        assert r["success"] is True
        assert r["signal_values"] == []

    def test_unknown_format(self):
        r = binary_rle_decode(data_hex="01", format="unknown")
        assert r["success"] is False

    def test_truncation(self):
        """Generate >100k samples -> truncated"""
        # 3 varints each encoding a large value
        # varint(50000) = a0 8c 03
        # Actually let's use a simpler approach: one huge run
        # varint(200000) = 80 a2 0c
        data = "80a20c"  # varint(200000) = 200000 > 100000
        r = binary_rle_decode(data_hex=data, format="toggle_count", initial_state=0)
        assert r["truncated"] is True
        assert len(r["signal_values"]) == 100000  # MAX_SAMPLES

    def test_invalid_hex(self):
        r = binary_rle_decode(data_hex="ZZ")
        assert r["success"] is False


class TestBinarySignalDecodeUartFromSamples:
    """UART decode from raw sample bytes."""

    def _make_uart_bytes(self, text: str, baud: int, sample_rate: int) -> bytes:
        """Generate UART 8N1 samples for text."""
        bit_period = max(1, int(sample_rate / baud))
        samples = bytearray()
        for byte_val in text.encode("ascii"):
            # Start bit (LOW)
            samples.extend([0] * bit_period)
            # 8 data bits, LSB first
            for bit in range(8):
                value = (byte_val >> bit) & 1
                samples.extend([value] * bit_period)
            # Stop bit (HIGH)
            samples.extend([1] * bit_period)
        # Idle (HIGH) before first byte
        preamble = bytearray([1] * (bit_period * 4))
        return bytes(preamble + samples)

    def test_decode_simple(self):
        samples = self._make_uart_bytes("AB", 9600, 100000)
        r = binary_signal_decode(
            data_hex=samples.hex(),
            protocol="uart",
            baud=9600,
            options={"sample_rate": 100000},
        )
        assert r["success"] is True
        assert r["protocol"] == "uart"
        assert r["baud"] == 9600
        assert r["frames_count"] >= 2
        assert "A" in r["decoded"]
        assert "B" in r["decoded"]

    def test_decode_hello(self):
        samples = self._make_uart_bytes("Hello", 115200, 1000000)
        r = binary_signal_decode(
            data_hex=samples.hex(),
            protocol="uart",
            baud=115200,
            options={"sample_rate": 1000000},
        )
        assert r["success"] is True
        assert r["frames_count"] >= 5

    def test_decode_with_rle_values(self):
        """RLE-encoded UART signal."""
        text = "OK"
        sample_rate = 100000
        baud = 9600
        samples = self._make_uart_bytes(text, baud, sample_rate)

        # Convert to RLE (toggle_count format)
        rle = []
        if len(samples) > 0:
            current = samples[0]
            count = 1
            for b in samples[1:]:
                if b == current:
                    count += 1
                else:
                    rle.append(count)
                    current = b
                    count = 1
            rle.append(count)

        r = binary_signal_decode(
            rle_values=json.dumps(rle),
            protocol="uart",
            baud=baud,
            options={"sample_rate": sample_rate},
        )
        assert r["success"] is True
        assert r["frames_count"] >= 2

    def test_invalid_hex(self):
        r = binary_signal_decode(data_hex="ZZ")
        assert r["success"] is False

    def test_no_input(self):
        r = binary_signal_decode()
        assert r["success"] is False

    def test_invalid_rle_values(self):
        r = binary_signal_decode(rle_values="not json")
        assert r["success"] is False


class TestBinarySignalDecodeSigrok:
    def test_sigrok_not_available_i2c(self):
        """Tests that sigrok fallback handles gracefully."""
        r = binary_signal_decode(
            data_hex="0001",
            protocol="i2c",
            baud=100000,
            options={"sample_rate": 100000},
        )
        assert "success" in r


class TestBinaryExec:
    def test_binary_varint(self):
        r = binary_exec("binary_varint", {"data_hex": "01"})
        assert r["success"] is True
        assert r["values"] == [1]

    def test_binary_block_parser(self):
        r = binary_exec("binary_block_parser", {
            "data_hex": "deadbeef",
            "block_size": 4,
            "fields": [{"name": "magic", "offset": 0, "size": 4, "type": "uint32"}],
        })
        assert r["success"] is True
        assert r["blocks"][0]["magic"] == 0xEFBEADDE

    def test_binary_rle_decode(self):
        r = binary_exec("binary_rle_decode", {"data_hex": "020301", "format": "toggle_count"})
        assert r["success"] is True
        assert r["signal_values"] == [0, 0, 1, 1, 1, 0]

    def test_binary_signal_decode(self):
        r = binary_exec("binary_signal_decode", {"data_hex": ""})
        assert r["success"] is False  # no input

    def test_unknown_tool(self):
        r = binary_exec("nonexistent", {})
        assert r["success"] is False
        assert "Unknown" in r["error"]
