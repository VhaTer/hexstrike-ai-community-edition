"""Unit tests for server_core/rate_limit_detector.py."""

import pytest
from server_core.rate_limit_detector import RateLimitDetector


@pytest.fixture
def detector():
    return RateLimitDetector()


class TestDetectRateLimiting:
    def test_http_429_detected(self, detector):
        result = detector.detect_rate_limiting("", 429)
        assert result["detected"] is True
        assert result["confidence"] >= 0.8
        assert "HTTP 429 status" in result["indicators"]

    def test_text_indicator_detected(self, detector):
        result = detector.detect_rate_limiting("rate limit exceeded", 200)
        assert result["detected"] is True
        assert result["confidence"] >= 0.2

    def test_too_many_requests_text(self, detector):
        result = detector.detect_rate_limiting("too many requests, slow down", 200)
        assert result["detected"] is True

    def test_quota_exceeded_text(self, detector):
        result = detector.detect_rate_limiting("api quota exceeded", 200)
        assert result["detected"] is True

    def test_no_rate_limit(self, detector):
        result = detector.detect_rate_limiting("OK", 200)
        assert result["detected"] is False
        assert result["confidence"] == 0.0
        assert result["indicators"] == []

    def test_header_detection(self, detector):
        headers = {"X-RateLimit-Remaining": "0", "Retry-After": "60"}
        result = detector.detect_rate_limiting("OK", 200, headers=headers)
        assert result["detected"] is True
        assert "Header: X-RateLimit-Remaining" in result["indicators"]

    def test_multiple_indicators_increase_confidence(self, detector):
        result = detector.detect_rate_limiting(
            "rate limit exceeded too many requests", 429
        )
        assert result["confidence"] >= 0.8

    def test_response_text_case_insensitive(self, detector):
        result = detector.detect_rate_limiting("RATE LIMIT", 200)
        assert result["detected"] is True

    def test_min_confidence_429_only(self, detector):
        result = detector.detect_rate_limiting("", 429)
        assert result["confidence"] == 0.8

    def test_max_confidence_capped(self, detector):
        headers = {"X-RateLimit": "true", "Retry-After": "30"}
        result = detector.detect_rate_limiting(
            "rate limit exceeded too many requests throttle", 429,
            headers=headers,
        )
        assert result["confidence"] == 1.0

    def test_empty_text_and_no_headers(self, detector):
        result = detector.detect_rate_limiting("", 200)
        assert result["detected"] is False
        assert result["confidence"] == 0.0


class TestRecommendTimingProfile:
    def test_stealth_at_high_confidence(self, detector):
        assert detector._recommend_timing_profile(0.9) == "stealth"
        assert detector._recommend_timing_profile(0.8) == "stealth"

    def test_conservative_at_medium_confidence(self, detector):
        assert detector._recommend_timing_profile(0.7) == "conservative"
        assert detector._recommend_timing_profile(0.5) == "conservative"

    def test_normal_at_low_confidence(self, detector):
        assert detector._recommend_timing_profile(0.4) == "normal"
        assert detector._recommend_timing_profile(0.2) == "normal"

    def test_aggressive_at_very_low_confidence(self, detector):
        assert detector._recommend_timing_profile(0.1) == "aggressive"
        assert detector._recommend_timing_profile(0.0) == "aggressive"


class TestAdjustTiming:
    def test_adjusts_threads_delay_timeout(self, detector):
        result = detector.adjust_timing(
            {"threads": 50, "delay": 0.1, "timeout": 5}, "stealth"
        )
        assert result["threads"] == 5
        assert result["delay"] == 2.0
        assert result["timeout"] == 30

    def test_unknown_profile_falls_back_to_normal(self, detector):
        result = detector.adjust_timing(
            {"threads": 100}, "bogus_profile"
        )
        assert result["threads"] == 20

    def test_modifies_additional_args_threads(self, detector):
        result = detector.adjust_timing(
            {"additional_args": "-t 50"}, "conservative"
        )
        assert "-t 10" in result["additional_args"]

    def test_modifies_additional_args_delay(self, detector):
        result = detector.adjust_timing(
            {"additional_args": ""}, "stealth"
        )
        assert "--delay 2.0" in result["additional_args"]

    def test_aggressive_has_small_delay(self, detector):
        result = detector.adjust_timing(
            {"additional_args": ""}, "aggressive"
        )
        assert "--delay 0.1" in result["additional_args"]

    def test_preserves_other_params(self, detector):
        result = detector.adjust_timing(
            {"url": "http://example.com", "threads": 50}, "normal"
        )
        assert result["url"] == "http://example.com"
        assert result["threads"] == 20

    def test_detect_and_adjust_integration(self, detector):
        detect_result = detector.detect_rate_limiting("rate limit", 429)
        assert detect_result["detected"] is True
        profile = detect_result["recommended_profile"]
        adjusted = detector.adjust_timing({"threads": 50, "delay": 0.1}, profile)
        assert adjusted["threads"] <= 5
