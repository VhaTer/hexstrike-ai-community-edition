"""
mcp_core/binary_direct.py

Binary analysis primitives — varint, block parser, RLE signal decode, UART decode.
stdlib only; sigrok-cli optional for binary_signal_decode (I2C/SPI).

Usage:
    from mcp_core.binary_direct import binary_varint, binary_block_parser
    result = binary_varint("7201ff7f...")
"""

import json
import struct
from typing import Any


# ── helpers ──────────────────────────────────────────────────────────

def _to_bytes(data_hex: str) -> bytes:
    stripped = data_hex.strip()
    if not stripped:
        raise ValueError("Empty hex data")
    # strip optional 0x prefix
    if stripped.startswith("0x") or stripped.startswith("0X"):
        stripped = stripped[2:]
    return bytes.fromhex(stripped)


def _parse_varints(data: bytes) -> list[int]:
    """Parse LEB128 unsigned varints from a byte sequence."""
    values = []
    i = 0
    while i < len(data):
        value = 0
        shift = 0
        while i < len(data):
            byte = data[i]
            i += 1
            value |= (byte & 0x7F) << shift
            shift += 7
            if not (byte & 0x80):
                break
        values.append(value)
    return values


def _parse_varints_signed(data: bytes) -> list[int]:
    """Parse LEB128 signed varints from a byte sequence."""
    values = []
    i = 0
    while i < len(data):
        value = 0
        shift = 0
        while i < len(data):
            byte = data[i]
            i += 1
            value |= (byte & 0x7F) << shift
            shift += 7
            if not (byte & 0x80):
                if byte & 0x40:
                    value |= -(1 << shift)
                break
        values.append(value)
    return values


# ── varint decode ────────────────────────────────────────────────────

def binary_varint(data_hex: str = "", encoding: str = "leb128_unsigned") -> dict[str, Any]:
    """Decode a sequence of LEB128/varint values from hex-encoded bytes.

    Args:
        data_hex:  Hex string of varint-encoded bytes (e.g. "7201ff7f01")
        encoding:  "leb128_unsigned" (default), "leb128_signed", or "uleb128"

    Returns:
        dict with success, values (list of ints), and count.
    """
    if not data_hex.strip():
        return {"success": True, "values": [], "count": 0}
    try:
        data = _to_bytes(data_hex)
    except Exception as e:
        return {"success": False, "error": f"Invalid hex: {e}"}

    if not data:
        return {"success": True, "values": [], "count": 0}

    try:
        if encoding == "leb128_signed":
            values = _parse_varints_signed(data)
        else:
            values = _parse_varints(data)
    except Exception as e:
        return {"success": False, "error": f"Varint decode error: {e}"}

    return {"success": True, "values": values, "count": len(values)}


# ── block parser ─────────────────────────────────────────────────────

def binary_block_parser(
    data_hex: str = "",
    block_size: int = 0,
    fields: list[dict] | None = None,
) -> dict[str, Any]:
    """Split hex-encoded binary data into fixed-size blocks and parse fields.

    Each field dict:
      {"name": "...", "offset": int, "size": int, "type": "uint8"|"uint16"|"uint32"|"int32"|"bytes"|"ascii"}

    Args:
        data_hex:   Hex string of raw binary data
        block_size: Number of bytes per block
        fields:     List of field definitions to parse within each block

    Returns:
        dict with success, blocks (list of parsed blocks), block_count.
    """
    if fields is None:
        fields = []
    try:
        data = _to_bytes(data_hex)
    except Exception as e:
        return {"success": False, "error": f"Invalid hex: {e}"}

    if block_size <= 0:
        return {"success": False, "error": "block_size must be > 0"}

    if block_size > len(data):
        return {"success": False, "error": f"block_size {block_size} > data length {len(data)}"}

    blocks = []
    for idx in range(0, len(data), block_size):
        chunk = data[idx : idx + block_size]
        parsed = {"block": len(blocks), "offset": idx}
        for f in fields:
            name = f.get("name", f"field_{f.get('offset', 0)}")
            offset = f.get("offset", 0)
            size = f.get("size", 1)
            ftype = f.get("type", "bytes")
            if offset + size > len(chunk):
                parsed[name] = None
                continue
            raw = chunk[offset : offset + size]
            if ftype == "uint8":
                parsed[name] = raw[0]
            elif ftype == "uint16":
                parsed[name] = struct.unpack("<H", raw)[0]
            elif ftype == "uint32":
                parsed[name] = struct.unpack("<I", raw)[0]
            elif ftype == "uint64":
                parsed[name] = struct.unpack("<Q", raw)[0]
            elif ftype == "int32":
                parsed[name] = struct.unpack("<i", raw)[0]
            elif ftype == "int64":
                parsed[name] = struct.unpack("<q", raw)[0]
            elif ftype == "ascii":
                parsed[name] = raw.decode("ascii", errors="replace").rstrip("\x00")
            elif ftype == "bytes":
                parsed[name] = raw.hex()
            else:
                parsed[name] = None
        blocks.append(parsed)

    return {"success": True, "blocks": blocks, "block_count": len(blocks)}


# ── RLE decode ───────────────────────────────────────────────────────

def binary_rle_decode(
    data_hex: str = "",
    format: str = "value_count",
    initial_state: int = 0,
) -> dict[str, Any]:
    """Reconstruct a digital signal from RLE-encoded binary data.

    Parses the hex data as varints (LEB128 unsigned) first, then
    interprets them as run-length encoded runs.

    Format variants:
      "value_count":  [val, count, val, count, ...] — explicit value for each run
      "toggle_count": [run1, run2, ...] — state toggles each run (Saleae-style)
      "pulse_width":  [width1, width2, ...] — alternating high/low pulse widths

    Args:
        data_hex:      Hex string of varint-encoded run lengths
        format:        One of "value_count", "toggle_count", "pulse_width"
        initial_state: Starting state (0 or 1) for toggle_count / pulse_width

    Returns:
        dict with signal_values (truncated to 100k), sample_count, truncated bool.
    """
    if not data_hex.strip():
        return {"success": True, "signal_values": [], "sample_count": 0, "truncated": False}
    try:
        data = _to_bytes(data_hex)
    except Exception as e:
        return {"success": False, "error": f"Invalid hex: {e}"}

    values = _parse_varints(data)

    MAX_SAMPLES = 100_000
    signal: list[int] = []
    state = initial_state
    total_samples = 0

    try:
        if format == "value_count":
            i = 0
            while i + 1 < len(values):
                val = values[i]
                count = values[i + 1]
                i += 2
                remaining = min(count, MAX_SAMPLES - len(signal))
                signal.extend([val] * remaining)
                total_samples += count
                if len(signal) >= MAX_SAMPLES:
                    break
        elif format == "toggle_count":
            for count in values:
                remaining = min(count, MAX_SAMPLES - len(signal))
                signal.extend([state] * remaining)
                total_samples += count
                state = 1 - state
                if len(signal) >= MAX_SAMPLES:
                    break
        elif format == "pulse_width":
            for width in values:
                remaining = min(width, MAX_SAMPLES - len(signal))
                signal.extend([state] * remaining)
                total_samples += width
                state = 1 - state
                if len(signal) >= MAX_SAMPLES:
                    break
        else:
            return {"success": False, "error": f"Unknown RLE format: '{format}'"}
    except Exception as e:
        return {"success": False, "error": f"RLE decode error: {e}"}

    truncated = total_samples > MAX_SAMPLES
    remaining = 0
    if truncated:
        remaining = total_samples - MAX_SAMPLES

    return {
        "success": True,
        "signal_values": signal,
        "sample_count": total_samples,
        "truncated": truncated,
        "truncated_count": remaining if truncated else 0,
    }


# ── Signal decode (UART) ─────────────────────────────────────────────

def _run_state_at(values: list[int], pos: int, initial_state: int) -> int:
    """Return signal state at sample position pos given RLE values."""
    cum = 0
    for i, run_len in enumerate(values):
        if cum + run_len > pos:
            return initial_state if i % 2 == 0 else (1 - initial_state)
        cum += run_len
    return initial_state if len(values) % 2 == 0 else (1 - initial_state)


def _decode_uart_from_rle(
    values: list[int],
    sample_rate: int,
    baud: int,
    initial_state: int = 1,
    invert: bool = False,
) -> dict[str, Any]:
    """Decode UART 8N1 frames from RLE-encoded digital signal (toggle_count).

    Walks through RLE runs, finds falling edges (start bits), samples
    8 data bits at the correct baud-rate interval, collects stop bit.

    Returns dict with bytes, frames, decoded text, and frame_count.
    """
    if invert:
        initial_state = 1 - initial_state
    if initial_state != 1:
        # Inverted or active-low UART — swap sample logic
        pass

    bit_period = max(1, int(sample_rate / baud))
    half_bit = bit_period // 2

    # Build cumulative position index for binary search
    positions: list[int] = []
    cum = 0
    for v in values:
        positions.append(cum)
        cum += v
    total = cum

    def sample_at(pos: int) -> int:
        if pos >= total:
            return initial_state if len(values) % 2 == 0 else (1 - initial_state)
        lo, hi = 0, len(positions)
        while lo < hi:
            mid = (lo + hi) // 2
            if positions[mid] <= pos:
                lo = mid + 1
            else:
                hi = mid
        run_idx = lo - 1
        if run_idx < 0:
            return initial_state
        return initial_state if run_idx % 2 == 0 else (1 - initial_state)

    bytes_out: list[int] = []
    frames_out: list[dict[str, Any]] = []

    # Walk through runs looking for falling edges
    pos = 0
    prev_state = initial_state

    for i, run_len in enumerate(values):
        current_state = initial_state if i % 2 == 0 else (1 - initial_state)

        # Check if edge from HIGH to LOW at start of this run
        if prev_state == 1 and current_state == 0 and run_len >= 1:
            falling_edge = pos
            # Start bit confirmed — decode 8 data bits
            byte_val = 0
            valid = True
            for bit in range(8):
                sample_pos = falling_edge + half_bit + (bit + 1) * bit_period
                b = sample_at(sample_pos)
                byte_val |= (b << bit)
            # Stop bit (should be HIGH)
            stop_pos = falling_edge + half_bit + 9 * bit_period
            stop_bit = sample_at(stop_pos) if stop_pos < total else 1

            if invert:
                byte_val = (~byte_val) & 0xFF

            char = chr(byte_val) if 32 <= byte_val <= 126 else f"\\x{byte_val:02x}"
            bytes_out.append(byte_val)
            frames_out.append({
                "byte": byte_val,
                "char": char,
                "ascii": char,
                "start_sample": falling_edge,
                "stop_bit": stop_bit,
            })

            # Advance past this frame (skip runs within the frame)
            frame_end = falling_edge + 10 * bit_period
            # Find position in RLE that covers frame_end
            while pos < frame_end and i < len(values):
                run_consumed = min(values[i], frame_end - pos)
                pos += run_consumed
                values[i] -= run_consumed
                if values[i] <= 0:
                    i += 1
                    if i < len(values):
                        current_state = initial_state if i % 2 == 0 else (1 - initial_state)
                        prev_state = current_state
                    continue
                break
            continue

        pos += run_len
        prev_state = current_state

    text = "".join(f.get("char", "") for f in frames_out)
    return {
        "bytes": bytes_out,
        "frames_count": len(frames_out),
        "frames": frames_out,
        "decoded": text,
    }


def _decode_uart_from_samples(
    data: bytes,
    sample_rate: int,
    baud: int,
    invert: bool = False,
) -> dict[str, Any]:
    """Decode UART 8N1 from raw sample bytes (each byte = 0x00 or 0x01)."""
    bit_period = max(1, int(sample_rate / baud))
    half_bit = bit_period // 2

    bytes_out: list[int] = []
    frames_out: list[dict[str, Any]] = []
    i = 0
    n = len(data)

    while i < n:
        # Scan for falling edge (high→low)
        if i + 1 < n and data[i] > 0 and data[i + 1] == 0:
            falling_edge = i + 1  # start bit begins here
            byte_val = 0
            for bit in range(8):
                sample_pos = falling_edge + half_bit + (bit + 1) * bit_period
                if sample_pos < n:
                    b = data[sample_pos]
                    byte_val |= (b << bit)
                else:
                    byte_val = 0
                    break
            if invert:
                byte_val = (~byte_val) & 0xFF
            char = chr(byte_val) if 32 <= byte_val <= 126 else f"\\x{byte_val:02x}"
            bytes_out.append(byte_val)
            frames_out.append({
                "byte": byte_val,
                "char": char,
                "ascii": char,
                "start_sample": falling_edge,
            })
            i += 10 * bit_period  # skip frame
        else:
            i += 1

    text = "".join(f.get("char", "") for f in frames_out)
    return {
        "bytes": bytes_out,
        "frames_count": len(frames_out),
        "frames": frames_out,
        "decoded": text,
    }


def binary_signal_decode(
    data_hex: str = "",
    rle_values: str = "",
    protocol: str = "uart",
    baud: int = 9600,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decode a digital signal (UART/I2C/SPI) from hex sample data or RLE values.

    Primary use: UART 8N1 decode from RLE-encoded signal (Saleae .sal files).
    Pass rle_values as a JSON array of run lengths (toggle_count format).

    For sigrok-supported protocols (I2C, SPI), falls back to sigrok-cli
    if installed and pure Python is not available.

    Args:
        data_hex:   Hex-encoded raw signal (each byte = one sample, 0x00 or 0x01)
        rle_values: JSON array of RLE run lengths (alt to data_hex)
        protocol:   "uart" (default), "i2c", "spi"
        baud:       Baud rate for UART (default 9600)
        options:    Dict with sample_rate (required for timing),
                    rle_format ("toggle_count" by default),
                    initial_state, invert, parity, stop_bits

    Returns:
        dict with success, decoded text, bytes, frames, frame_count.
    """
    opts = options or {}
    sample_rate = opts.get("sample_rate", 1_000_000)
    rle_format = opts.get("rle_format", "toggle_count")
    initial_state = opts.get("initial_state", 1)
    invert = opts.get("invert", False)

    if protocol != "uart":
        # Attempt sigrok fallback for I2C/SPI
        return _signal_decode_sigrok(
            data_hex=data_hex,
            rle_values=rle_values,
            protocol=protocol,
            baud=baud,
            options=opts,
        )

    # UART decode
    if rle_values:
        try:
            run_lengths = json.loads(rle_values)
            if not isinstance(run_lengths, list):
                return {"success": False, "error": "rle_values must be a JSON array"}
            return {
                "success": True,
                "protocol": "uart",
                "baud": baud,
                **_decode_uart_from_rle(
                    run_lengths, sample_rate, baud,
                    initial_state=initial_state, invert=invert,
                ),
            }
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid rle_values JSON: {e}"}

    if data_hex:
        try:
            data = _to_bytes(data_hex)
        except Exception as e:
            return {"success": False, "error": f"Invalid hex: {e}"}
        if not all(b in (0, 1) for b in data):
            return {"success": False, "error": "Sample data must contain only 0x00 and 0x01 bytes"}
        return {
            "success": True,
            "protocol": "uart",
            "baud": baud,
            **_decode_uart_from_samples(data, sample_rate, baud, invert=invert),
        }

    return {"success": False, "error": "Provide data_hex or rle_values"}


def _signal_decode_sigrok(
    data_hex: str = "",
    rle_values: str = "",
    protocol: str = "i2c",
    baud: int = 100000,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fallback to sigrok-cli for I2C/SPI decode when pure Python isn't available."""
    import subprocess
    import tempfile

    opts = options or {}
    sample_rate = opts.get("sample_rate", 1_000_000)

    # Build input data
    input_data = b""
    if rle_values:
        try:
            values = json.loads(rle_values)
            # Expand RLE to raw samples (truncated to 1M for sigrok)
            state = opts.get("initial_state", 1)
            expanded = bytearray()
            for v in values:
                chunk = min(v, 1_000_000 - len(expanded))
                expanded.extend([state] * chunk)
                state = 1 - state
                if len(expanded) >= 1_000_000:
                    break
            input_data = bytes(expanded)
        except (json.JSONDecodeError, Exception) as e:
            return {"success": False, "error": f"RLE expansion failed: {e}"}
    elif data_hex:
        try:
            input_data = _to_bytes(data_hex)
        except Exception as e:
            return {"success": False, "error": f"Invalid hex: {e}"}

    if not input_data:
        return {"success": False, "error": "No input data"}

    # Write to temp file for sigrok
    import os
    fd, tmp_path = tempfile.mkstemp(suffix=".bin")
    try:
        os.write(fd, input_data)
        os.close(fd)

        decoder_proto = {
            "uart": "uart:baudrate=%d" % baud,
            "i2c": "i2c",
            "spi": "spi",
        }.get(protocol, "uart")

        # sigrok needs a samplerate file or CLI arg
        cmd = [
            "sigrok-cli",
            "-I", "binary:samplerate=%d" % sample_rate,
            "-i", tmp_path,
            "-P", decoder_proto,
            "-A", "%s=%s" % (protocol, "rx-data" if protocol == "uart" else "data"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = proc.stdout or proc.stderr or ""
        return {
            "success": True,
            "protocol": protocol,
            "baud": baud,
            "decoded": output.strip(),
            "frames_count": len(output.strip().splitlines()),
        }
    except FileNotFoundError:
        return {"success": False, "error": "sigrok-cli not installed. Install with: apt install sigrok-cli"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "sigrok-cli timed out"}
    except Exception as e:
        return {"success": False, "error": f"sigrok error: {e}"}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── dispatch ─────────────────────────────────────────────────────────

_HANDLERS: dict[str, Any] = {
    "binary_varint": binary_varint,
    "binary_block_parser": binary_block_parser,
    "binary_rle_decode": binary_rle_decode,
    "binary_signal_decode": binary_signal_decode,
}


def binary_exec(tool: str, data: dict) -> dict:
    """Dispatch execution to the appropriate binary analysis function.

    Args:
        tool: Tool name (e.g. "binary_varint")
        data: Dict of parameters passed to the handler

    Returns:
        Handler result dict
    """
    handler = _HANDLERS.get(tool)
    if handler is None:
        return {"success": False, "error": f"Unknown binary tool: '{tool}'"}
    return handler(**data)
