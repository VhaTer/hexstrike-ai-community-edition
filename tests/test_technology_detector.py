import pytest
from server_core.technology_detector import TechnologyDetector


@pytest.fixture
def detector():
    return TechnologyDetector()


class TestTechnologyDetector:
    def test_empty_inputs(self, detector):
        result = detector.detect_technologies("example.com")
        assert result["web_servers"] == []
        assert result["frameworks"] == []
        assert result["cms"] == []
        assert result["databases"] == []
        assert result["languages"] == []
        assert result["security"] == []
        assert result["services"] == []

    def test_detect_from_headers(self, detector):
        headers = {"Server": "Apache/2.4.49", "X-Powered-By": "PHP/7.4"}
        result = detector.detect_technologies("example.com", headers=headers)
        assert "apache" in result["web_servers"]
        assert "php" in result["languages"]

    def test_detect_from_content(self, detector):
        content = '<html><head><title>WordPress</title><link rel="stylesheet" href="/wp-content/themes/..."></head></html>'
        result = detector.detect_technologies("example.com", content=content)
        assert "wordpress" in result["cms"]

    def test_detect_from_ports(self, detector):
        ports = [22, 80, 443, 3306]
        result = detector.detect_technologies("example.com", ports=ports)
        assert "ssh" in result["services"]
        assert "http" in result["services"]
        assert "https" in result["services"]
        assert "mysql" in result["services"]

    def test_detect_all_sources(self, detector):
        headers = {"Server": "nginx"}
        content = "Powered by Django"
        ports = [5432]
        result = detector.detect_technologies("example.com", headers=headers, content=content, ports=ports)
        assert "nginx" in result["web_servers"]
        assert "django" in result["frameworks"]
        assert "postgresql" in result["services"]

    def test_no_duplicates(self, detector):
        headers = {"Server": "Apache", "X-Generator": "Apache"}
        result = detector.detect_technologies("example.com", headers=headers)
        assert result["web_servers"].count("apache") == 1

    def test_security_detection(self, detector):
        headers = {"X-CF-Ray": "somevalue"}
        result = detector.detect_technologies("example.com", headers=headers)
        assert "waf" in result["security"]



    def test_port_service_mapping(self, detector):
        assert detector.port_services[21] == "ftp"
        assert detector.port_services[443] == "https"
        assert detector.port_services[3306] == "mysql"
        assert 27017 in detector.port_services

    def test_detection_patterns_structure(self, detector):
        assert "web_servers" in detector.detection_patterns
        assert "frameworks" in detector.detection_patterns
        assert "cms" in detector.detection_patterns
        assert "databases" in detector.detection_patterns
        assert "languages" in detector.detection_patterns
        assert "security" in detector.detection_patterns
