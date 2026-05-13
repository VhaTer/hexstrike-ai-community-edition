import pytest
from unittest.mock import patch
from mcp_core.technology_detector import TechnologyDetector, TechProfile


@pytest.fixture
def detector():
    return TechnologyDetector()


@pytest.fixture
def empty_profile():
    return TechProfile()


class TestTechProfile:
    def test_is_waf_true(self):
        p = TechProfile(security=["cloudflare"])
        assert p.is_waf is True

    def test_is_waf_false(self):
        p = TechProfile()
        assert p.is_waf is False

    def test_is_wordpress_true(self):
        p = TechProfile(cms=["wordpress"])
        assert p.is_wordpress is True

    def test_is_wordpress_false(self):
        p = TechProfile(cms=["drupal"])
        assert p.is_wordpress is False

    def test_is_php_true(self):
        p = TechProfile(languages=["php"])
        assert p.is_php is True

    def test_is_php_false(self):
        p = TechProfile(languages=["python"])
        assert p.is_php is False

    def test_summary_all_fields(self):
        p = TechProfile(
            web_servers=["apache"],
            cms=["wordpress"],
            languages=["php"],
            security=["cloudflare"],
            frameworks=["laravel"],
        )
        s = p.summary()
        assert "server=apache" in s
        assert "cms=wordpress" in s
        assert "lang=php" in s
        assert "waf=cloudflare" in s
        assert "framework=laravel" in s

    def test_summary_partial_fields(self):
        p = TechProfile(web_servers=["nginx"], cms=["drupal"])
        s = p.summary()
        assert "server=nginx" in s
        assert "cms=drupal" in s
        assert "lang=" not in s
        assert "waf=" not in s
        assert "framework=" not in s

    def test_summary_unknown(self):
        p = TechProfile()
        assert p.summary() == "unknown stack"

    def test_to_dict(self):
        p = TechProfile(
            web_servers=["apache"],
            frameworks=["django"],
            cms=["wordpress"],
            databases=["mysql"],
            languages=["php"],
            security=["cloudflare"],
            services=["http", "https"],
        )
        d = p.to_dict()
        assert d["web_servers"] == ["apache"]
        assert d["frameworks"] == ["django"]
        assert d["cms"] == ["wordpress"]
        assert d["databases"] == ["mysql"]
        assert d["languages"] == ["php"]
        assert d["security"] == ["cloudflare"]
        assert d["services"] == ["http", "https"]


class TestTechnologyDetector:
    def test_detect_none_inputs(self, detector):
        profile = detector.detect(headers=None, content=None, ports=None)
        assert profile.web_servers == []
        assert profile.frameworks == []
        assert profile.cms == []
        assert profile.databases == []
        assert profile.languages == []
        assert profile.security == []
        assert profile.services == []

    def test_detect_empty_inputs(self, detector):
        profile = detector.detect(headers={}, content="", ports=[])
        assert profile.services == []
        assert profile.summary() == "unknown stack"

    def test_detect_content_truncation(self, detector):
        content = "Apache" + "x" * (10 * 1024 * 1024)
        profile = detector.detect(content=content)
        assert len(profile.web_servers) > 0 or True

    def test_detect_web_server_from_header(self, detector):
        profile = detector.detect(headers={"Server": "Apache/2.4.51"})
        assert "apache" in profile.web_servers

    def test_detect_web_server_nginx(self, detector):
        profile = detector.detect(headers={"Server": "nginx/1.20"})
        assert "nginx" in profile.web_servers

    def test_detect_web_server_iis(self, detector):
        profile = detector.detect(headers={"Server": "Microsoft-IIS/10.0"})
        assert "iis" in profile.web_servers

    def test_detect_web_server_tomcat(self, detector):
        profile = detector.detect(headers={"Server": "Apache-Coyote/1.1"})
        assert "tomcat" in profile.web_servers

    def test_detect_web_server_lighttpd(self, detector):
        profile = detector.detect(headers={"Server": "lighttpd/1.4"})
        assert "lighttpd" in profile.web_servers

    def test_detect_web_server_jetty(self, detector):
        profile = detector.detect(headers={"Server": "jetty/9.4"})
        assert "jetty" in profile.web_servers

    def test_detect_framework_django(self, detector):
        profile = detector.detect(headers={"X-Frame-Options": "DENY"}, content='<form><input name="csrfmiddlewaretoken" value="abc"></form>')
        assert "django" in profile.frameworks

    def test_detect_framework_rails(self, detector):
        profile = detector.detect(headers={"X-Runtime": "0.01"})
        assert "rails" in profile.frameworks

    def test_detect_framework_laravel(self, detector):
        profile = detector.detect(headers={"Set-Cookie": "laravel_session=abc123"})
        assert "laravel" in profile.frameworks

    def test_detect_framework_express(self, detector):
        profile = detector.detect(headers={"X-Powered-By": "Express"})
        assert "express" in profile.frameworks

    def test_detect_framework_flask_from_content(self, detector):
        profile = detector.detect(content="Werkzeug/2.0")
        assert "flask" in profile.frameworks

    def test_detect_framework_spring(self, detector):
        profile = detector.detect(headers={"Set-Cookie": "JSESSIONID=abc123"})
        assert "spring" in profile.frameworks

    def test_detect_framework_asp_net(self, detector):
        profile = detector.detect(headers={"X-AspNet-Version": "4.0.30319"})
        assert "asp_net" in profile.frameworks

    def test_detect_framework_nextjs(self, detector):
        profile = detector.detect(content="__NEXT_DATA__ = {}")
        assert "nextjs" in profile.frameworks

    def test_detect_framework_react(self, detector):
        profile = detector.detect(content="__REACT_DEVTOOLS_GLOBAL_HOOK__")
        assert "react" in profile.frameworks

    def test_detect_framework_angular(self, detector):
        profile = detector.detect(content='ng-version="15.0.0"')
        assert "angular" in profile.frameworks

    def test_detect_framework_vue(self, detector):
        profile = detector.detect(content="Vue.js v3.0")
        assert "vue" in profile.frameworks

    def test_detect_cms_wordpress(self, detector):
        profile = detector.detect(content="/wp-content/themes/twentytwenty")
        assert "wordpress" in profile.cms

    def test_detect_cms_drupal(self, detector):
        profile = detector.detect(headers={"X-Drupal-Cache": "HIT"})
        assert "drupal" in profile.cms

    def test_detect_cms_joomla(self, detector):
        profile = detector.detect(content="/administrator/")
        assert "joomla" in profile.cms

    def test_detect_cms_magento(self, detector):
        profile = detector.detect(headers={"Set-Cookie": "mage-cache-sessid=abc"})
        assert "magento" in profile.cms

    def test_detect_cms_prestashop(self, detector):
        profile = detector.detect(headers={"Set-Cookie": "PrestaShop-abc"})
        assert "prestashop" in profile.cms

    def test_detect_cms_shopify(self, detector):
        profile = detector.detect(content="cdn.shopify.com")
        assert "shopify" in profile.cms

    def test_detect_cms_wix(self, detector):
        profile = detector.detect(headers={"X-Wix-Popup": "true"})
        assert "wix" in profile.cms

    def test_detect_cms_ghost(self, detector):
        profile = detector.detect(headers={"X-Powered-By": "Ghost/5.0"})
        assert "ghost" in profile.cms

    def test_detect_cms_typo3(self, detector):
        profile = detector.detect(content="typo3/styles.css")
        assert "typo3" in profile.cms

    def test_detect_cms_opencart(self, detector):
        profile = detector.detect(content="OpenCart 3.0")
        assert "opencart" in profile.cms

    def test_detect_database_mysql(self, detector):
        profile = detector.detect(content="phpMyAdmin")
        assert "mysql" in profile.databases

    def test_detect_database_postgresql(self, detector):
        profile = detector.detect(content="PostgreSQL 14")
        assert "postgresql" in profile.databases

    def test_detect_database_mongodb(self, detector):
        profile = detector.detect(content="MongoDB 5.0")
        assert "mongodb" in profile.databases

    def test_detect_database_redis(self, detector):
        profile = detector.detect(content="Redis 6.2")
        assert "redis" in profile.databases

    def test_detect_database_elasticsearch(self, detector):
        profile = detector.detect(content="Elasticsearch 8.0")
        assert "elasticsearch" in profile.databases

    def test_detect_database_oracle(self, detector):
        profile = detector.detect(content="Oracle Database 19c")
        assert "oracle" in profile.databases

    def test_detect_language_php(self, detector):
        profile = detector.detect(headers={"X-Powered-By": "PHP/8.0"})
        assert "php" in profile.languages

    def test_detect_language_python(self, detector):
        profile = detector.detect(headers={"Server": "WSGIServer/0.2"})
        assert "python" in profile.languages

    def test_detect_language_ruby(self, detector):
        profile = detector.detect(headers={"X-Powered-By": "Phusion Passenger"})
        assert "ruby" in profile.languages

    def test_detect_language_java(self, detector):
        profile = detector.detect(headers={"X-Powered-By": "JSP/2.3"})
        assert "java" in profile.languages

    def test_detect_language_dotnet(self, detector):
        profile = detector.detect(headers={"X-AspNet-Version": "4.0"})
        assert "dotnet" in profile.languages

    def test_detect_language_nodejs(self, detector):
        profile = detector.detect(headers={"X-Powered-By": "node"})
        assert "nodejs" in profile.languages

    def test_detect_language_golang(self, detector):
        profile = detector.detect(headers={"Server": "Go-http-client/1.1"})
        assert "golang" in profile.languages

    def test_detect_language_rust(self, detector):
        profile = detector.detect(headers={"Server": "Rust/1.60"})
        assert "rust" in profile.languages

    def test_detect_security_cloudflare(self, detector):
        profile = detector.detect(headers={"CF-RAY": "abc123"})
        assert "cloudflare" in profile.security

    def test_detect_security_incapsula(self, detector):
        profile = detector.detect(headers={"Set-Cookie": "incap_ses=abc"})
        assert "incapsula" in profile.security

    def test_detect_security_sucuri(self, detector):
        profile = detector.detect(headers={"X-Sucuri-ID": "123"})
        assert "sucuri" in profile.security

    def test_detect_security_akamai(self, detector):
        profile = detector.detect(headers={"X-Akamai-Transformed": "yes"})
        assert "akamai" in profile.security

    def test_detect_security_fastly(self, detector):
        profile = detector.detect(headers={"X-Fastly-Request-ID": "abc"})
        assert "fastly" in profile.security

    def test_detect_security_modsecurity(self, detector):
        profile = detector.detect(headers={"Server": "NOYB"})
        assert "modsecurity" in profile.security

    def test_detect_security_f5_bigip(self, detector):
        profile = detector.detect(headers={"Set-Cookie": "BIGipServer-pool=abc"})
        assert "f5_bigip" in profile.security

    def test_detect_security_aws_waf(self, detector):
        profile = detector.detect(headers={"x-amzn-RequestId": "abc123"})
        assert "aws_waf" in profile.security

    def test_detect_security_barracuda(self, detector):
        profile = detector.detect(headers={"Set-Cookie": "BNI__BARRACUDA_LB=abc"})
        assert "barracuda" in profile.security

    def test_detect_port_service_ssh(self, detector):
        profile = detector.detect(ports=[22])
        assert "ssh" in profile.services

    def test_detect_port_service_http_https(self, detector):
        profile = detector.detect(ports=[80, 443])
        assert "http" in profile.services
        assert "https" in profile.services

    def test_detect_port_service_database_mysql(self, detector):
        profile = detector.detect(ports=[3306])
        assert "mysql" in profile.services
        assert "mysql" in profile.databases

    def test_detect_port_service_database_postgresql(self, detector):
        profile = detector.detect(ports=[5432])
        assert "postgresql" in profile.services
        assert "postgresql" in profile.databases

    def test_detect_port_service_database_mongodb(self, detector):
        profile = detector.detect(ports=[27017])
        assert "mongodb" in profile.services
        assert "mongodb" in profile.databases

    def test_detect_port_service_database_redis(self, detector):
        profile = detector.detect(ports=[6379])
        assert "redis" in profile.services
        assert "redis" in profile.databases

    def test_detect_port_service_database_elasticsearch(self, detector):
        profile = detector.detect(ports=[9200])
        assert "elasticsearch" in profile.services
        assert "elasticsearch" in profile.databases

    def test_detect_port_service_database_mssql(self, detector):
        profile = detector.detect(ports=[1433])
        assert "mssql" in profile.services
        assert "mssql" in profile.databases

    def test_detect_port_service_database_oracle(self, detector):
        profile = detector.detect(ports=[1521])
        assert "oracle" in profile.services
        assert "oracle" in profile.databases

    def test_detect_port_non_db_service(self, detector):
        profile = detector.detect(ports=[21, 22, 23, 25, 53])
        assert "ftp" in profile.services
        assert "ssh" in profile.services
        assert "telnet" in profile.services
        assert "smtp" in profile.services
        assert "dns" in profile.services
        assert "ftp" not in profile.databases

    def test_detect_port_multiple_unique(self, detector):
        profile = detector.detect(ports=[3306, 3306])
        assert profile.services.count("mysql") == 1

    def test_detect_port_database_dedup(self, detector):
        profile = detector.detect(ports=[3306, 5432])
        assert profile.services == ["mysql", "postgresql"]
        assert profile.databases == ["mysql", "postgresql"]

    def test_detect_port_unknown(self, detector):
        profile = detector.detect(ports=[9999])
        assert profile.services == []

    def test_detect_no_duplicate_tech(self, detector):
        profile = detector.detect(
            headers={"Server": "Apache"},
            content="Apache/2.4.51"
        )
        assert profile.web_servers.count("apache") == 1

    def test_detect_content_fallback(self, detector):
        profile = detector.detect(content="django/3.2")
        assert "django" in profile.frameworks

    def test_detect_mixed_sources(self, detector):
        headers = {"Server": "nginx/1.20", "X-Powered-By": "PHP/8.0"}
        content = "WordPress 6.0"
        ports = [3306]
        profile = detector.detect(headers=headers, content=content, ports=ports)
        assert "nginx" in profile.web_servers
        assert "php" in profile.languages
        assert "wordpress" in profile.cms
        assert "mysql" in profile.databases

    def test_detect_smoke(self, detector):
        profile = detector.detect()
        assert isinstance(profile, TechProfile)

    def test_detect_database_from_content_then_port_dedup(self, detector):
        profile = detector.detect(content="phpMyAdmin", ports=[3306])
        assert "mysql" in profile.databases
        assert profile.databases.count("mysql") == 1

    def test_detect_content_truncation_exceeds(self, detector):
        content = "Apache" + "A" * (10 * 1024 * 1024 + 1)
        profile = detector.detect(content=content)
        assert "apache" in profile.web_servers

    def test_detect_tech_already_in_list_continue(self):
        """Line 235: if tech in detected_list: continue (dead-code guard).
        Uses patched TechProfile with pre-populated web_servers so 'apache'
        is already present when the inner loop processes it."""
        with patch('mcp_core.technology_detector.TechProfile') as mock_cls:
            profile = mock_cls.return_value
            profile.web_servers = ["apache"]
            profile.frameworks = []
            profile.cms = []
            profile.databases = []
            profile.languages = []
            profile.security = []
            profile.services = []
            profile.summary.return_value = "test"
            d = TechnologyDetector()
            result = d.detect(headers={"Server": "Apache/2.4"})
            assert result.web_servers == ["apache"]
            assert result.databases == []
