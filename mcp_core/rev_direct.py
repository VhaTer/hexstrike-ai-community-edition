"""
mcp_core/rev_direct.py

Direct execution layer for reversing/analysis primitives.
Plain functions testable without MCP server.

Usage:
    from mcp_core.rev_direct import compute_entropy_map
    result = compute_entropy_map("/path/to/binary", block_size=256)
"""

import math
import os
from typing import Any


def _shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    entropy = 0.0
    length = len(data)
    for count in freq:
        if count == 0:
            continue
        p = count / length
        entropy -= p * math.log2(p)
    return round(entropy, 4)


_SUSPICIOUS_ENTROPY_THRESHOLD = 7.0


def compute_entropy_map(
    file_path: str,
    block_size: int = 256,
    threshold: float = _SUSPICIOUS_ENTROPY_THRESHOLD,
) -> dict[str, Any]:
    if not os.path.isfile(file_path):
        return {"success": False, "error": f"File not found: {file_path}"}
    if block_size < 1:
        return {"success": False, "error": "block_size must be >= 1"}

    try:
        with open(file_path, "rb") as f:
            data = f.read()
    except PermissionError:
        return {"success": False, "error": f"Permission denied: {file_path}"}
    except OSError as e:
        return {"success": False, "error": str(e)}

    total_len = len(data)
    blocks = []
    high_entropy_count = 0

    for i in range(0, total_len, block_size):
        chunk = data[i : i + block_size]
        ent = _shannon_entropy(chunk)
        offset = i
        block_idx = i // block_size
        is_high = ent >= threshold
        if is_high:
            high_entropy_count += 1
        blocks.append({
            "block": block_idx,
            "offset": offset,
            "size": len(chunk),
            "entropy": ent,
            "suspicious": is_high,
        })

    return {
        "success": True,
        "file_path": file_path,
        "file_size": total_len,
        "block_size": block_size,
        "threshold": threshold,
        "total_blocks": len(blocks),
        "high_entropy_blocks": high_entropy_count,
        "blocks": blocks,
    }
