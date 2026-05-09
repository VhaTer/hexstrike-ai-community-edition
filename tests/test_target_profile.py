"""Unit tests for shared/target_profile.py — TargetProfile dataclass."""

import pytest
from shared.target_profile import TargetProfile
from shared.target_types import TargetType, TechnologyStack


class TestTargetProfile:
    def test_default_construction(self):
        p = TargetProfile(target="test.example")
        assert p.target == "test.example"
        assert p.target_type == TargetType.UNKNOWN
        assert p.open_ports == []
        assert p.services == {}
        assert p.cms_type is None
        assert p.attack_surface_score == 0.0
        assert p.risk_level == "unknown"

    def test_to_dict_roundtrip(self):
        p = TargetProfile(
            target="10.0.0.1",
            target_type=TargetType.NETWORK_HOST,
            ip_addresses=["10.0.0.1"],
            open_ports=[22, 80, 443],
            services={22: "ssh", 80: "http", 443: "https"},
            technologies=[TechnologyStack.APACHE, TechnologyStack.PHP],
            cms_type="wordpress",
            cloud_provider="aws",
            security_headers={"X-Frame-Options": "DENY"},
            ssl_info={"version": "TLSv1.3"},
            subdomains=["admin.example.com"],
            endpoints=["/api", "/admin"],
            attack_surface_score=5.5,
            risk_level="medium",
            confidence_score=0.75,
        )
        d = p.to_dict()
        assert d["target"] == "10.0.0.1"
        assert d["target_type"] == "network_host"
        assert 22 in d["open_ports"]
        assert d["services"][80] == "http"
        assert "apache" in d["technologies"]
        assert d["cms_type"] == "wordpress"
        assert d["cloud_provider"] == "aws"

    def test_from_dict_full(self):
        data = {
            "target": "10.0.0.1",
            "target_type": "network_host",
            "ip_addresses": ["10.0.0.1"],
            "open_ports": [22, 80, 443],
            "services": {"22": "ssh", "80": "http"},
            "technologies": ["apache", "php"],
            "cms_type": "wordpress",
            "cloud_provider": "aws",
            "security_headers": {"X-Frame-Options": "DENY"},
            "ssl_info": {"version": "TLSv1.3"},
            "subdomains": ["admin.example.com"],
            "endpoints": ["/api"],
            "attack_surface_score": "5.5",
            "risk_level": "medium",
            "confidence_score": "0.75",
        }
        p = TargetProfile.from_dict(data)
        assert p.target == "10.0.0.1"
        assert p.target_type == TargetType.NETWORK_HOST
        assert 443 in p.open_ports
        assert p.services["22"] == "ssh"
        assert TechnologyStack.APACHE in p.technologies
        assert TechnologyStack.PHP in p.technologies
        assert p.cms_type == "wordpress"
        assert p.cloud_provider == "aws"
        assert p.security_headers["X-Frame-Options"] == "DENY"
        assert p.ssl_info["version"] == "TLSv1.3"
        assert p.subdomains == ["admin.example.com"]
        assert p.endpoints == ["/api"]
        assert p.attack_surface_score == 5.5
        assert p.risk_level == "medium"
        assert p.confidence_score == 0.75

    def test_from_dict_empty(self):
        p = TargetProfile.from_dict({})
        assert p.target == ""
        assert p.target_type == TargetType.UNKNOWN
        assert p.open_ports == []
        assert p.services == {}
        assert p.technologies == []
        assert p.cms_type is None
        assert p.attack_surface_score == 0.0

    def test_from_dict_invalid_enum_falls_back(self):
        data = {"target": "x", "target_type": "not_a_real_type"}
        p = TargetProfile.from_dict(data)
        assert p.target_type == TargetType.UNKNOWN

    def test_from_dict_invalid_tech_skipped(self):
        data = {"target": "x", "technologies": ["apache", "bogus_tech", "php"]}
        p = TargetProfile.from_dict(data)
        assert TechnologyStack.APACHE in p.technologies
        assert TechnologyStack.PHP in p.technologies
        assert len(p.technologies) == 2

    def test_to_dict_after_modification(self):
        p = TargetProfile(target="example.com")
        p.open_ports.append(8080)
        p.services[8080] = "http-proxy"
        d = p.to_dict()
        assert 8080 in d["open_ports"]
        assert d["services"][8080] == "http-proxy"
