"""Unit tests for mcp_core/technology_detector.py."""

import pytest
from mcp_core.technology_detector import TechnologyDetector, TechProfile


@pytest.fixture
def detector():
    return TechnologyDetector()


class TestTechProfile:
    def test_default_empty(self):
        p = TechProfile()
        assert p.web_servers == []
        assert p.frameworks == []
        assert p.cms == []
        assert not p.is_waf
        assert not p.is_wordpress
        assert not p.is_php

    def test_is_waf(self):
        p = TechProfile(security=["cloudflare"])
        assert p.is_waf is True

    def test_is_wordpress(self):
        p = TechProfile(cms=["wordpress"])
        assert p.is_wordpress is True

    def test_is_php(self):
        p = TechProfile(languages=["php"])
        assert p.is_php is True

    def test_summary_full(self):
        p = TechProfile(
            web_servers=["nginx"],
            cms=["wordpress"],
            languages=["php"],
            security=["cloudflare"],
            frameworks=["laravel"],
        )
        s = p.summary()
        assert "server=nginx" in s
        assert "cms=wordpress" in s
        assert "lang=php" in s
        assert "waf=cloudflare" in s
        assert "framework=laravel" in s

    def test_summary_empty(self):
        p = TechProfile()
        assert p.summary() == "unknown stack"

    def test_to_dict(self):
        p = TechProfile(web_servers=["apache"], cms=["drupal"])
        d = p.to_dict()
        assert d["web_servers"] == ["apache"]
        assert d["cms"] == ["drupal"]


class TestTechnologyDetector:
    def test_detect_apache_via_header(self, detector):
        profile = detector.detect(headers={"Server": "Apache/2.4.51"})
        assert "apache" in profile.web_servers

    def test_detect_nginx_via_header(self, detector):
        profile = detector.detect(headers={"Server": "nginx/1.18.0"})
        assert "nginx" in profile.web_servers

    def test_detect_wordpress_via_content(self, detector):
        profile = detector.detect(content="<meta name='generator' content='WordPress 6.0'>")
        assert "wordpress" in profile.cms

    def test_detect_php_via_header(self, detector):
        profile = detector.detect(headers={"X-Powered-By": "PHP/8.0"})
        assert "php" in profile.languages

    def test_detect_cloudflare_via_header(self, detector):
        profile = detector.detect(headers={"CF-RAY": "abc123"})
        assert "cloudflare" in profile.security

    def test_detect_ports_mysql(self, detector):
        profile = detector.detect(ports=[3306])
        assert "mysql" in profile.services
        assert "mysql" in profile.databases

    def test_detect_ports_ssh(self, detector):
        profile = detector.detect(ports=[22])
        assert "ssh" in profile.services

    def test_detect_ports_http_https(self, detector):
        profile = detector.detect(ports=[80, 443])
        assert "http" in profile.services
        assert "https" in profile.services

    def test_detect_multiple_signals(self, detector):
        profile = detector.detect(
            headers={"Server": "Apache/2.4.51", "X-Powered-By": "PHP/8.0"},
            content="wp-content",
            ports=[80, 3306],
        )
        assert "apache" in profile.web_servers
        assert "php" in profile.languages
        assert "wordpress" in profile.cms
        assert "http" in profile.services
        assert "mysql" in profile.services

    def test_detect_empty_inputs(self, detector):
        profile = detector.detect()
        assert profile.web_servers == []
        assert profile.services == []

    def test_detect_no_match(self, detector):
        profile = detector.detect(
            headers={"X-Custom": "something-unknown"},
            content="some random text without any signatures",
        )
        assert profile.web_servers == []
        assert profile.cms == []

    def test_detect_content_truncation(self, detector):
        large = "x" * (10 * 1024 * 1024 + 1)
        content = large + "Apache"
        profile = detector.detect(content=content)
        assert "apache" not in profile.web_servers

    def test_detect_rails_via_header(self, detector):
        profile = detector.detect(headers={"X-Runtime": "0.01"})
        assert "rails" in profile.frameworks

    def test_detect_django_via_content(self, detector):
        profile = detector.detect(content='csrfmiddlewaretoken')
        assert "django" in profile.frameworks

    def test_detect_react(self, detector):
        profile = detector.detect(content='__REACT')
        assert "react" in profile.frameworks

    def test_detect_asp_net(self, detector):
        profile = detector.detect(headers={"X-AspNet-Version": "4.0"})
        assert "asp_net" in profile.frameworks

    def test_detect_elasticsearch_port(self, detector):
        profile = detector.detect(ports=[9200])
        assert "elasticsearch" in profile.services
        assert "elasticsearch" in profile.databases

    def test_detect_docker_port(self, detector):
        profile = detector.detect(ports=[2375])
        assert "docker" in profile.services

    def test_detect_iis(self, detector):
        profile = detector.detect(headers={"Server": "Microsoft-IIS/10.0"})
        assert "iis" in profile.web_servers

    def test_detect_tomcat(self, detector):
        profile = detector.detect(headers={"Server": "Apache-Coyote/1.1"})
        assert "tomcat" in profile.web_servers

    def test_detect_express(self, detector):
        profile = detector.detect(headers={"X-Powered-By": "Express"})
        assert "express" in profile.frameworks

    def test_detect_drupal(self, detector):
        profile = detector.detect(content="Drupal 9")
        assert "drupal" in profile.cms
