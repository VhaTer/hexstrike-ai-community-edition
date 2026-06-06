"""
mcp_core/crypto_direct.py

Direct execution layer for cryptanalysis primitives.
Plain functions testable without MCP server.

Usage:
    from mcp_core.crypto_direct import xor_crack, z3_solve, rsa_attack
"""

import math
import os
import re
import struct
import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────

def _decode_input(data: str = "", file_path: str = "") -> bytes:
    if file_path:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(file_path)
        with open(file_path, "rb") as f:
            return f.read()
    data = data.strip()
    if not data:
        raise ValueError("No input provided — supply data or file_path")
    hex_re = re.compile(r"^[0-9a-fA-F]+$")
    if hex_re.match(data):
        return bytes.fromhex(data)
    if len(data) > 4 and len(data) % 4 == 0:
        b64_re = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")
        if b64_re.match(data):
            import base64
            try:
                return base64.b64decode(data)
            except Exception:
                logger.debug("crypto_direct: base64 decode failed", exc_info=True)
    # fallback: raw string as bytes
    return data.encode("latin-1")


_ENGLISH_FREQ = {
    ord(" "): 0.130, ord("e"): 0.127, ord("t"): 0.091, ord("a"): 0.082,
    ord("o"): 0.075, ord("i"): 0.070, ord("n"): 0.067, ord("s"): 0.063,
    ord("h"): 0.061, ord("r"): 0.060, ord("d"): 0.043, ord("l"): 0.040,
    ord("c"): 0.028, ord("u"): 0.028, ord("m"): 0.024, ord("w"): 0.024,
    ord("f"): 0.022, ord("g"): 0.020, ord("y"): 0.020, ord("p"): 0.019,
    ord("b"): 0.015, ord("v"): 0.010, ord("k"): 0.008, ord("j"): 0.002,
    ord("x"): 0.002, ord("q"): 0.001, ord("z"): 0.001,
}


def _score_text(text: bytes) -> float:
    score = 0.0
    for b in text:
        # treat uppercase same as lowercase for frequency
        k = b
        if 65 <= k <= 90:
            k += 32
        if k in _ENGLISH_FREQ:
            score += _ENGLISH_FREQ[k]
        elif 32 <= b <= 126:
            score += 0.01
        else:
            score -= 0.10
    return score


def _hamming_distance(a: bytes, b: bytes) -> int:
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b))


# Complete English letter frequency (including space) for chi-squared
_ENGLISH_EXPECTED = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    # 32 = space
    0.130, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    # digits 0-9
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    # uppercase A-Z
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    # lowercase a-z
    0.08167, 0.01492, 0.02782, 0.04253, 0.12702, 0.02228, 0.02015, 0.06094,
    0.06966, 0.00153, 0.00772, 0.04025, 0.02406, 0.06749, 0.07507, 0.01929,
    0.00095, 0.05987, 0.06327, 0.09056, 0.02758, 0.00978, 0.02360, 0.00150,
    0.01974, 0.00074,
    # rest
    0.0, 0.0, 0.0, 0.0, 0.0,
]


def _chi_squared(observed: list[float], expected: list[float]) -> float:
    """Lower = better match to expected distribution."""
    score = 0.0
    for o, e in zip(observed, expected):
        if e > 0:
            score += (o - e) ** 2 / e
    return score


def _best_single_byte_xor(chunk: bytes) -> tuple[int, float]:
    """Find key byte that minimizes chi-squared vs English letter frequencies."""
    length = len(chunk)
    if length == 0:
        return (0, 0.0)

    best_key = 0
    best_score = float("inf")

    for key in range(256):
        decrypted = bytes(b ^ key for b in chunk)
        freq = [0.0] * 256
        for b in decrypted:
            freq[b] += 1.0
        freq = [f / length for f in freq]
        cs = _chi_squared(freq, _ENGLISH_EXPECTED)
        if cs < best_score:
            best_score = cs
            best_key = key

    return best_key, best_score


# ── xor_crack ────────────────────────────────────────────────────────


def _xor_single(data: bytes, key: int) -> bytes:
    return bytes(b ^ key for b in data)


def _xor_repeating(data: bytes, key: bytes) -> bytes:
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


def xor_crack(
    data: str = "",
    file_path: str = "",
    key_length_range: str = "1-20",
    top_n: int = 5,
    known_plaintext: str = "",
) -> dict[str, Any]:
    """Attempt to crack XOR-encrypted data.

    Tries single-byte XOR (all 256 keys) and multi-byte repeating-key XOR
    (Hamming-distance key-length detection + frequency scoring).

    Args:
        data: Hex, base64, or raw string input
        file_path: Load bytes from file instead of data string
        key_length_range: Range of key lengths to try, e.g. "1-20"
        top_n: Number of top results to return
        known_plaintext: If provided, also reports if it appears in decrypted output

    Returns:
        dict with success, results (list of {key, plaintext, score, method}),
        and matched_plaintext if found
    """
    try:
        ciphertext = _decode_input(data, file_path)
        if len(ciphertext) < 2:
            return {"success": False, "error": "Ciphertext too short (< 2 bytes)"}
    except (FileNotFoundError, PermissionError, OSError) as e:
        return {"success": False, "error": str(e)}
    except ValueError as e:
        return {"success": False, "error": str(e)}

    try:
        parts = key_length_range.split("-")
        min_len = max(1, int(parts[0]))
        max_len = min(40, int(parts[1]) if len(parts) > 1 else int(parts[0]))
    except (ValueError, IndexError):
        min_len, max_len = 1, 20

    results = []
    found_plaintext = ""

    # single-byte XOR: try all 256
    for key in range(256):
        plain = _xor_single(ciphertext, key)
        score = _score_text(plain)
        results.append((score, bytes([key]), plain, "xor_single"))

    # multi-byte repeating-key: only try short keys (2-6)
    for klen in range(2, min(7, max_len + 1)):
        if len(ciphertext) < klen * 3:
            continue
        key = bytearray()
        for pos in range(klen):
            chunk = ciphertext[pos::klen]
            best_k = max(range(256), key=lambda k: _score_text(_xor_single(chunk, k)))
            key.append(best_k)
        key_bytes = bytes(key)
        plain = _xor_repeating(ciphertext, key_bytes)
        score = _score_text(plain)
        score -= 0.04 * (klen - 1)  # length penalty
        results.append((score, key_bytes, plain, f"xor_repeat({klen})"))

    # known-plaintext key derivation
    if known_plaintext and len(known_plaintext) >= 2:
        kp_bytes = known_plaintext.encode("latin-1")
        derived_key = bytes(
            ciphertext[i] ^ kp_bytes[i]
            for i in range(min(len(kp_bytes), len(ciphertext)))
        )
        # try as single-byte
        if all(b == derived_key[0] for b in derived_key):
            plain = _xor_single(ciphertext, derived_key[0])
            score = _score_text(plain)
            results.append((score + 5.0, derived_key[:1], plain, "xor_single(derived)"))
        # try as multi-byte at full length
        if 2 <= len(derived_key) <= max_len:
            plain = _xor_repeating(ciphertext, derived_key)
            score = _score_text(plain)
            results.append((score + 5.0, derived_key, plain, f"xor_repeat({len(derived_key)})(derived)"))
        # try shorter prefixes (in case key is shorter than known)
        for trunc in range(2, len(derived_key)):
            k = derived_key[:trunc]
            plain = _xor_repeating(ciphertext, k)
            score = _score_text(plain)
            results.append((score + 5.0, k, plain, f"xor_repeat({trunc})(derived)"))

    results.sort(key=lambda x: x[0], reverse=True)
    top = results[:top_n]

    if known_plaintext:
        kp_lower = known_plaintext.lower()
        for _, _, plain, _ in results:
            if kp_lower in plain.decode("latin-1", errors="replace").lower():
                found_plaintext = known_plaintext
                break

    return {
        "success": True,
        "input_length": len(ciphertext),
        "key_length_range": f"{min_len}-{max_len}",
        "results": [
            {
                "key": r[1].hex() if len(r[1]) > 1 else str(r[1][0]),
                "key_bytes": list(r[1]),
                "plaintext": r[2].decode("latin-1", errors="replace"),
                "score": round(r[0], 4),
                "method": r[3],
            }
            for r in top
        ],
        "matched_known_plaintext": found_plaintext if found_plaintext else "",
    }


# ── z3_solve ─────────────────────────────────────────────────────────


def z3_solve(
    constraints: str = "",
    file_path: str = "",
    variables: str = "",
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Solve SMT constraints using Z3 (SMT-LIB2 syntax).

    Accepts raw SMT-LIB2 constraints as input string or from file.
    Optionally declare variables in a comma-separated list (e.g. "x, y, z")
    as BitVec(64) — useful when you only have a formula without explicit
    variable declarations.
    """
    if not constraints and not file_path:
        return {"success": False, "error": "No constraints — supply data or file_path"}

    if file_path:
        try:
            with open(file_path) as f:
                constraints = f.read()
        except (FileNotFoundError, PermissionError, OSError) as e:
            return {"success": False, "error": str(e)}

    constraints = constraints.strip()
    if not constraints:
        return {"success": False, "error": "Empty constraints"}

    try:
        import subprocess
        import tempfile

        smt2_lines = ["(set-option :timeout {})".format(timeout_seconds * 1000)]

        if variables:
            for var in variables.split(","):
                var = var.strip()
                if var:
                    smt2_lines.append(f"(declare-const {var} (_ BitVec 64))")

        smt2_lines.append(constraints)
        if not constraints.strip().startswith("(check-sat"):
            smt2_lines.append("(check-sat)")
        if not constraints.strip().startswith("(get-model") and "(get-model" not in constraints:
            smt2_lines.append("(get-model)")

        smt2_input = "\n".join(smt2_lines)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".smt2", delete=False
        ) as f:
            f.write(smt2_input)
            f.flush()
            tmp_path = f.name

        try:
            r = subprocess.run(
                ["z3", tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout_seconds + 5,
            )
            stdout = r.stdout.strip()
            stderr = r.stderr.strip()

            sat = stdout.startswith("sat")
            unsat = stdout.startswith("unsat")
            unknown = stdout.startswith("unknown")

            model = ""
            if sat:
                model = stdout[3:].strip()

            return {
                "success": True,
                "sat": sat,
                "unsat": unsat,
                "unknown": unknown,
                "output": stdout,
                "stderr": stderr,
                "model": model,
                "variables_declared": variables or "",
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "Z3 solver not found — install with: sudo apt install z3",
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Z3 timed out after {timeout_seconds}s",
            }
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    except Exception as e:
        return {"success": False, "error": str(e)}


# ── rsa_attack ───────────────────────────────────────────────────────


def _egcd(a: int, b: int):
    if b == 0:
        return a, 1, 0
    g, x1, y1 = _egcd(b, a % b)
    return g, y1, x1 - (a // b) * y1


def _modinv(a: int, m: int) -> int:
    g, x, _ = _egcd(a, m)
    if g != 1:
        raise ValueError("No modular inverse")
    return x % m


def _is_perfect_square(n: int) -> bool:
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n


def _factor_fermat(n: int) -> tuple[int, int] | None:
    a = math.isqrt(n)
    if a * a < n:
        a += 1
    for _ in range(1000000):
        b2 = a * a - n
        if _is_perfect_square(b2):
            return a - int(math.isqrt(b2)), a + int(math.isqrt(b2))
        a += 1
    return None


def _factor_small_primes(n: int, limit: int = 10000) -> tuple[int, int] | None:
    # Trial division up to limit
    for p in range(2, limit + 1):
        if n % p == 0:
            q = n // p
            return p, q
    return None


def rsa_attack(
    n: int,
    e: int,
    ciphertext: int = 0,
    attacks: str = "all",
    factordb: bool = False,
) -> dict[str, Any]:
    """Attempt to break RSA given public key parameters.

    Attack strategies:
      - small_primes: trial division up to 10k (fast)
      - fermat: Fermat factorization (close primes)
      - wiener: Wiener's attack (small d)
      - all: try each in order until one works

    Args:
        n: RSA modulus
        e: Public exponent
        ciphertext: Optional ciphertext to decrypt
        attacks: Comma-separated list ("small_primes,fermat,wiener") or "all"
        factordb: If True, query FactorDB API for known factorization

    Returns:
        dict with success, method used, p, q, d, plaintext (if ciphertext provided)
    """
    try:
        n = int(n)
        e = int(e)
    except (TypeError, ValueError):
        return {"success": False, "error": "n and e must be integers"}

    if n < 2:
        return {"success": False, "error": "n must be >= 2"}

    if attacks == "all":
        attack_list = ["small_primes", "fermat", "wiener"]
    else:
        attack_list = [a.strip() for a in attacks.split(",")]

    if factordb:
        attack_list = ["factordb"] + attack_list

    p, q = None, None
    method_used = ""

    for attack in attack_list:
        if attack == "small_primes":
            res = _factor_small_primes(n)
            if res:
                p, q = res
                method_used = "small_primes"
                break

        elif attack == "fermat":
            res = _factor_fermat(n)
            if res:
                p, q = res
                method_used = "fermat"
                break

        elif attack == "wiener":
            try:
                res = _wiener_attack(n, e)
                if res:
                    d = res
                    method_used = "wiener"
                    # recover p,q from d: not directly possible, skip p,q
                    break
            except Exception:
                continue

        elif attack == "factordb":
            try:
                p, q = _factordb_lookup(n)
                if p and q:
                    method_used = "factordb"
                    break
            except Exception:
                continue

    if not p and not q and method_used != "wiener":
        return {
            "success": False,
            "error": "No factorization found — try larger primes or different attack",
            "attacks_tried": attack_list,
        }

    result: dict[str, Any] = {
        "success": True,
        "method": method_used,
        "n": n,
        "e": e,
    }

    if method_used == "wiener":
        d = res
        result["d"] = d
    elif p and q:
        result["p"] = p
        result["q"] = q
        phi = (p - 1) * (q - 1)
        d = _modinv(e, phi)
        result["d"] = d
        result["phi"] = phi

    # Decrypt if ciphertext provided
    if ciphertext and "d" in result:
        try:
            ct = int(ciphertext)
            pt = pow(ct, result["d"], n)
            result["plaintext_int"] = pt
            # Try to decode as bytes
            try:
                pt_bytes = pt.to_bytes((pt.bit_length() + 7) // 8, "big")
                result["plaintext_hex"] = pt_bytes.hex()
                result["plaintext_ascii"] = pt_bytes.decode(
                    "latin-1", errors="replace"
                )
            except Exception:
                logger.debug("crypto_direct: plaintext decode failed", exc_info=True)
        except (TypeError, ValueError):
            logger.debug("crypto_direct: type/value error in RSA attack", exc_info=True)

    return result


def _wiener_attack(n: int, e: int) -> int | None:
    """Wiener's attack — finds d when d < N^0.25."""
    # Continued fraction expansion of e/n
    frac = _continued_fraction(e, n)
    for k, d in _convergents(frac):
        if k == 0:
            continue
        if (e * d - 1) % k != 0:
            continue
        phi = (e * d - 1) // k
        # Check discriminant: p+q = n - phi + 1
        s = n - phi + 1
        # Check if discriminant is a perfect square
        disc = s * s - 4 * n
        if disc < 0:
            continue
        if _is_perfect_square(disc):
            return d
    return None


def _continued_fraction(num: int, den: int) -> list[int]:
    cf = []
    while den:
        a = num // den
        cf.append(a)
        num, den = den, num - a * den
    return cf


def _convergents(cf: list[int]):
    n0, n1 = 0, 1
    d0, d1 = 1, 0
    for a in cf:
        n2 = a * n1 + n0
        d2 = a * d1 + d0
        yield n2, d2
        n0, n1 = n1, n2
        d0, d1 = d1, d2


def _factordb_lookup(n: int) -> tuple[int, int] | None:
    import urllib.request
    import urllib.parse

    url = f"http://factordb.com/api?query={n}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "HexStrike/0.11"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        import json
        j = json.loads(data)
        if j.get("status") in ("FF", "PP"):
            factors = j.get("factors", [])
            if len(factors) >= 2:
                p, q = int(factors[0][0]), int(factors[1][0])
                if p * q == n:
                    return p, q
        return None
    except Exception:
        return None
