"""
tests/test_technology_detector.py

Unit tests for mcp_core/technology_detector.py
Covers: TechProfile, TechnologyDetector.detect()
"""

import pytest
from mcp_core.technology_detector import TechnologyDetector, TechProfile


@pytest.fixture
def detector():
    return TechnologyDetector()


# ---------------------------------------------------------------------------
# TechProfile
# ---------------------------------------------------------------------------

class TestTechProfile:
    def test_is_waf_true(self):
        p = TechProfile(security=["cloudflare"])
        assert p.is_waf is True

    def test_is_waf_false(self):
        p = TechProfile()
        assert p.is_waf is False

    def test_is_wordpress(self):
        p = TechProfile(cms=["wordpress"])
        assert p.is_wordpress is True

    def test_is_php(self):
        p = TechProfile(languages=["php"])
        assert p.is_php is True

    def test_summary_empty(self):
        p = TechProfile()
        assert p.summary() == "unknown stack"

    def test_summary_full(self):
        p = TechProfile(
            web_servers=["apache"],
            cms=["wordpress"],
            languages=["php"],
            security=["cloudflare"],
        )
        s = p.summary()
        assert "apache" in s
        assert "wordpress" in s
        assert "php" in s
        assert "cloudflare" in s

    def test_to_dict(self):
        p = TechProfile(web_servers=["nginx"])
        d = p.to_dict()
        assert d["web_servers"] == ["nginx"]
        assert "frameworks" in d


# ---------------------------------------------------------------------------
# Header detection
# ---------------------------------------------------------------------------

class TestHeaderDetection:
    def test_detect_apache(self, detector):
        p = detector.detect(headers={"Server": "Apache/2.4.51 (Ubuntu)"})
        assert "apache" in p.web_servers

    def test_detect_nginx(self, detector):
        p = detector.detect(headers={"Server": "nginx/1.18.0"})
        assert "nginx" in p.web_servers

    def test_detect_iis(self, detector):
        p = detector.detect(headers={"Server": "Microsoft-IIS/10.0"})
        assert "iis" in p.web_servers

    def test_detect_php(self, detector):
        p = detector.detect(headers={"X-Powered-By": "PHP/8.0.1"})
        assert "php" in p.languages

    def test_detect_dotnet(self, detector):
        p = detector.detect(headers={"X-Powered-By": "ASP.NET", "X-AspNet-Version": "4.0"})
        assert "dotnet" in p.languages

    def test_detect_cloudflare(self, detector):
        p = detector.detect(headers={"CF-RAY": "abc123", "Server": "cloudflare"})
        assert "cloudflare" in p.security
        assert p.is_waf is True

    def test_detect_incapsula(self, detector):
        p = detector.detect(headers={"X-CDN": "Incapsula"})
        assert "incapsula" in p.security

    def test_detect_nodejs(self, detector):
        p = detector.detect(headers={"X-Powered-By": "node"})
        assert "nodejs" in p.languages

    def test_empty_headers(self, detector):
        p = detector.detect(headers={})
        assert p.web_servers == []
        assert p.is_waf is False


# ---------------------------------------------------------------------------
# Content detection
# ---------------------------------------------------------------------------

class TestContentDetection:
    def test_detect_wordpress_content(self, detector):
        content = '<link rel="stylesheet" href="/wp-content/themes/twentytwenty/style.css">'
        p = detector.detect(content=content)
        assert "wordpress" in p.cms

    def test_detect_drupal_content(self, detector):
        content = '<meta name="Generator" content="Drupal 9">'
        p = detector.detect(content=content)
        assert "drupal" in p.cms

    def test_detect_django_content(self, detector):
        content = '<input type="hidden" name="csrfmiddlewaretoken" value="abc">'
        p = detector.detect(content=content)
        assert "django" in p.frameworks

    def test_detect_react_content(self, detector):
        content = '<div id="root" data-reactroot="">'
        p = detector.detect(content=content)
        assert "react" in p.frameworks

    def test_detect_laravel_content(self, detector):
        content = 'document.cookie = "laravel_session=abc"'
        p = detector.detect(content=content)
        assert "laravel" in p.frameworks

    def test_content_truncation(self, detector):
        # Should not raise even with huge content
        big_content = "wordpress " * 2_000_000  # > 10MB
        p = detector.detect(content=big_content)
        assert "wordpress" in p.cms

    def test_empty_content(self, detector):
        p = detector.detect(content="")
        assert p.cms == []


# ---------------------------------------------------------------------------
# Port detection
# ---------------------------------------------------------------------------

class TestPortDetection:
    def test_detect_mysql_port(self, detector):
        p = detector.detect(ports=[3306])
        assert "mysql" in p.services
        assert "mysql" in p.databases

    def test_detect_postgresql_port(self, detector):
        p = detector.detect(ports=[5432])
        assert "postgresql" in p.services
        assert "postgresql" in p.databases

    def test_detect_mongodb_port(self, detector):
        p = detector.detect(ports=[27017])
        assert "mongodb" in p.services
        assert "mongodb" in p.databases

    def test_detect_redis_port(self, detector):
        p = detector.detect(ports=[6379])
        assert "redis" in p.services

    def test_detect_rdp_port(self, detector):
        p = detector.detect(ports=[3389])
        assert "rdp" in p.services

    def test_detect_smb_port(self, detector):
        p = detector.detect(ports=[445])
        assert "smb" in p.services

    def test_multiple_ports(self, detector):
        p = detector.detect(ports=[80, 443, 3306, 5432])
        assert "http" in p.services
        assert "https" in p.services
        assert "mysql" in p.databases
        assert "postgresql" in p.databases

    def test_unknown_port(self, detector):
        p = detector.detect(ports=[12345])
        assert p.services == []

    def test_empty_ports(self, detector):
        p = detector.detect(ports=[])
        assert p.services == []


# ---------------------------------------------------------------------------
# Combined detection
# ---------------------------------------------------------------------------

class TestCombinedDetection:
    def test_wordpress_full_stack(self, detector):
        p = detector.detect(
            headers={"Server": "Apache/2.4", "X-Powered-By": "PHP/8.0"},
            content='<link href="/wp-content/themes/style.css">',
            ports=[80, 443, 3306],
        )
        assert "apache" in p.web_servers
        assert "php" in p.languages
        assert "wordpress" in p.cms
        assert "mysql" in p.databases

    def test_no_duplicates(self, detector):
        # Detected from both headers and content — should not duplicate
        p = detector.detect(
            headers={"Server": "Apache"},
            content="Apache server detected",
        )
        assert p.web_servers.count("apache") == 1

    def test_none_inputs(self, detector):
        p = detector.detect(headers=None, content="", ports=None)
        assert isinstance(p, TechProfile)
