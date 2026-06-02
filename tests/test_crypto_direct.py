"""Tests for mcp_core/crypto_direct.py."""

import os
import math
import tempfile
import pytest

from mcp_core.crypto_direct import (
    xor_crack,
    z3_solve,
    rsa_attack,
    _xor_single,
    _xor_repeating,
    _score_text,
    _hamming_distance,
    _modinv,
    _factor_small_primes,
    _factor_fermat,
    _decode_input,
)


class TestDecodeInput:
    def test_hex(self):
        assert _decode_input(data="48656c6c6f") == b"Hello"

    def test_base64(self):
        assert _decode_input(data="SGVsbG8=") == b"Hello"

    def test_raw(self):
        # Strings that aren't valid hex or base64 fall back to raw latin-1
        assert _decode_input(data="Hello!") == b"Hello!"

    def test_file(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(b"file_data")
            path = f.name
        try:
            assert _decode_input(file_path=path) == b"file_data"
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            _decode_input(file_path="/nonexistent/xyz")

    def test_empty(self):
        with pytest.raises(ValueError, match="No input"):
            _decode_input(data="")


class TestScoreText:
    def test_english_scores_high(self):
        text = b"the quick brown fox jumps over the lazy dog"
        assert _score_text(text) > 0

    def test_random_scores_low(self):
        text = os.urandom(50)
        assert _score_text(text) < 0

    def test_empty(self):
        assert _score_text(b"") == 0.0

    def test_uppercase_treated_like_lowercase(self):
        lower = _score_text(b"hello")
        upper = _score_text(b"HELLO")
        assert lower == upper

    def test_mixed_case(self):
        lower = _score_text(b"Hello World! This is a test.")
        upper = _score_text(b"HELLO WORLD! THIS IS A TEST.")
        assert abs(lower - upper) < 1.0


class TestHammingDistance:
    def test_identical(self):
        assert _hamming_distance(b"hello", b"hello") == 0

    def test_one_bit(self):
        assert _hamming_distance(b"\x00", b"\x01") == 1

    def test_known_value(self):
        # "this is a test" vs "wokka wokka!!!"
        a = b"this is a test"
        b = b"wokka wokka!!!"
        assert _hamming_distance(a, b) == 37


class TestXorSingleRepeating:
    def test_single(self):
        assert _xor_single(b"ABC", 0xFF) == bytes(b ^ 0xFF for b in b"ABC")

    def test_repeating(self):
        assert _xor_repeating(b"ABCD", b"XY") == b"\x19\x1b\x1b\x1d"

    def test_repeating_roundtrip(self):
        plain = b"Hello World! Test message."
        key = b"KEY"
        cipher = _xor_repeating(plain, key)
        assert _xor_repeating(cipher, key) == plain


class TestXorCrack:
    def test_single_byte_english(self):
        plain = b"the quick brown fox jumps over the lazy dog the quick brown fox"
        cipher = bytes(b ^ 0x7B for b in plain)
        r = xor_crack(data=cipher.hex(), top_n=3)
        assert r["success"] is True
        assert r["results"][0]["key"] == "123"  # 0x7B
        assert "the quick" in r["results"][0]["plaintext"]

    def test_single_byte_flag_like(self):
        """flag-like text needs known_plaintext"""
        flag = b"flag{cr4ck3d_th3_x0r_c1ph3r}"
        cipher = bytes(b ^ 0x37 for b in flag)
        r = xor_crack(data=cipher.hex(), known_plaintext="flag{", top_n=3)
        assert r["success"] is True
        # derived key should find it
        assert r["results"][0]["method"].endswith("(derived)")
        assert "flag{" in r["results"][0]["plaintext"]

    def test_repeating_key(self):
        plain = b"Hello World! This is a test message for XOR cracking."
        key = b"KEY"
        cipher = bytes(b ^ key[i % 3] for i, b in enumerate(plain))
        r = xor_crack(data=cipher.hex(), top_n=5)
        assert r["success"] is True
        # correct key should be in results
        keys = [res["key"] for res in r["results"]]
        assert "4b4559" in keys  # "KEY" hex

    def test_repeating_key_with_known(self):
        plain = b"CTF{x0r_r3p34t_k3y_w1ll_b3_cr4ck3d}"
        key = b"KEY"
        cipher = bytes(b ^ key[i % 3] for i, b in enumerate(plain))
        r = xor_crack(data=cipher.hex(), known_plaintext="CTF", top_n=5)
        assert r["success"] is True
        assert r["matched_known_plaintext"] == "CTF"
        assert any(res["method"].endswith("(derived)") for res in r["results"])

    def test_known_plaintext_match_no_known_param(self):
        plain = b"flag{this_is_a_test_flag}"
        cipher = bytes(b ^ 0x42 for b in plain)
        r = xor_crack(data=cipher.hex(), top_n=5, known_plaintext="flag{")
        assert r["matched_known_plaintext"] == "flag{"

    def test_key_length_range(self):
        plain = b"test " * 20
        key = b"AB"
        cipher = bytes(b ^ key[i % 2] for i, b in enumerate(plain))
        r = xor_crack(data=cipher.hex(), key_length_range="2-2", top_n=3)
        assert r["key_length_range"] == "2-2"

    def test_bad_key_length_range(self):
        plain = b"test message here " * 5
        cipher = bytes(b ^ 0x12 for b in plain)
        r = xor_crack(data=cipher.hex(), key_length_range="invalid")
        assert r["success"] is True  # falls back to defaults

    def test_file_input(self):
        plain = b"Hello from file. XOR test data here."
        cipher = bytes(b ^ 0x55 for b in plain)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(cipher)
            path = f.name
        try:
            r = xor_crack(file_path=path, top_n=3)
            assert r["success"] is True
            assert r["results"][0]["key"] == "85"  # 0x55
        finally:
            os.unlink(path)

    def test_too_short(self):
        r = xor_crack(data="aa")
        assert r["success"] is False

    def test_file_and_data_conflict_file_wins(self):
        """file_path takes priority over data"""
        r = xor_crack(data="deadbeef", file_path="/nonexistent/xyz")
        assert r["success"] is False  # file not found

    def test_base64_input(self):
        import base64
        plain = b"testing base64 xor input string for decoding"
        cipher = bytes(b ^ 0x33 for b in plain)
        b64 = base64.b64encode(cipher).decode()
        r = xor_crack(data=b64, top_n=3, known_plaintext="testing")
        assert r["success"] is True
        assert r["matched_known_plaintext"] == "testing"


class TestZ3Solve:
    def test_simple_sat(self):
        r = z3_solve(
            variables="x, y",
            constraints="(assert (= x #x000000000000002A))\n(assert (= y #x000000000000000B))\n(assert (= (bvadd x y) #x0000000000000035))",
            timeout_seconds=10,
        )
        assert r["success"] is True
        assert r["sat"] is True

    def test_unsat(self):
        r = z3_solve(
            variables="x",
            constraints="(assert (= x #x0000000000000005))\n(assert (= x #x000000000000000A))",
            timeout_seconds=10,
        )
        assert r["unsat"] is True

    def test_no_constraints(self):
        r = z3_solve()
        assert r["success"] is False
        assert "No constraints" in r["error"]

    def test_file_input(self):
        smt = ("(declare-const x (_ BitVec 8))\n"
               "(assert (= x #x2A))\n"
               "(check-sat)\n(get-model)")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".smt2", mode="w") as f:
            f.write(smt)
            path = f.name
        try:
            r = z3_solve(file_path=path, timeout_seconds=10)
            assert r["sat"] is True
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        r = z3_solve(file_path="/nonexistent.smt2")
        assert r["success"] is False

    def test_timeout(self):
        r = z3_solve(
            constraints="(declare-const x (_ BitVec 64))\n"
                        "(assert (= x x))\n"
                        "(check-sat)\n(get-model)",
            timeout_seconds=1,
        )
        assert r["success"] is True  # simple constraint should still work
        assert r["sat"] is True


class TestRsaAttack:
    def test_small_primes(self):
        n = 991 * 997  # both primes
        r = rsa_attack(n=n, e=65537)
        assert r["success"] is True
        assert r["method"] == "small_primes"
        assert r.get("p") and r.get("q")
        assert r["p"] * r["q"] == n

    def test_fermat(self):
        p = 1000000007
        q = 1000000021
        n = p * q
        r = rsa_attack(n=n, e=65537)
        assert r["success"] is True
        assert r["method"] == "fermat"

    def test_decrypt_with_ciphertext(self):
        p, q = 61, 53
        n = p * q
        e = 17
        # d = modinv(17, (p-1)*(q-1) = 3120) = 2753
        r = rsa_attack(n=n, e=e, ciphertext=855)
        assert r["success"] is True
        # 855^2753 mod 3233 = 123
        assert r.get("plaintext_int") is not None

    def test_too_large_n(self):
        # product of two 5-digit primes: trial division up to 10k won't help
        n = 100019 * 100129  # both > 10000
        r = rsa_attack(n=n, e=65537, attacks="small_primes")
        assert r["success"] is False

    def test_small_n(self):
        r = rsa_attack(n=1, e=3)
        assert r["success"] is False

    def test_non_integer(self):
        r = rsa_attack(n="abc", e=3)
        assert r["success"] is False

    def test_wiener_attack(self):
        """Wiener's attack: d < N^0.25. Choose p,q s.t. phi % 3 != 0."""
        p = 1013  # p mod 3 = 2 → (p-1) mod 3 = 1
        q = 1019  # q mod 3 = 2 → (q-1) mod 3 = 1
        n = p * q
        phi = (p - 1) * (q - 1)  # = 1012*1018 = 1030216, phi % 3 = 1
        # d=3: find e = (k*phi + 1) / 3. Need (k*phi + 1) % 3 == 0.
        # phi%3=1, so k*1+1 ≡ 0 mod 3 → k ≡ 2 mod 3. k=2:
        e = (2 * phi + 1) // 3
        r = rsa_attack(n=n, e=e, attacks="wiener")
        assert r["success"] is True
        assert r["d"] == 3

    def test_fermat_fails_on_product_of_distant_primes(self):
        n = 1000000007 * 9999991  # ~10^6 apart
        r = rsa_attack(n=n, e=65537, attacks="fermat")
        assert r["success"] is False
