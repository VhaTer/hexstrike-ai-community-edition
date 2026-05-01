from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from shared.target_types import TargetType, TechnologyStack

@dataclass
class TargetProfile:
    """Comprehensive target analysis profile for intelligent decision making"""
    target: Any
    target_type: TargetType = TargetType.UNKNOWN
    ip_addresses: List[str] = field(default_factory=list)
    open_ports: List[int] = field(default_factory=list)
    services: Dict[int, str] = field(default_factory=dict)
    technologies: List[TechnologyStack] = field(default_factory=list)
    cms_type: Optional[str] = None
    cloud_provider: Optional[str] = None
    security_headers: Dict[str, str] = field(default_factory=dict)
    ssl_info: Dict[str, Any] = field(default_factory=dict)
    subdomains: List[str] = field(default_factory=list)
    endpoints: List[str] = field(default_factory=list)
    attack_surface_score: float = 0.0
    risk_level: str = "unknown"
    confidence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert TargetProfile to dictionary for JSON serialization"""
        return {
            "target": self.target,
            "target_type": self.target_type.value,
            "ip_addresses": self.ip_addresses,
            "open_ports": self.open_ports,
            "services": self.services,
            "technologies": [tech.value for tech in self.technologies],
            "cms_type": self.cms_type,
            "cloud_provider": self.cloud_provider,
            "security_headers": self.security_headers,
            "ssl_info": self.ssl_info,
            "subdomains": self.subdomains,
            "endpoints": self.endpoints,
            "attack_surface_score": self.attack_surface_score,
            "risk_level": self.risk_level,
            "confidence_score": self.confidence_score
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetProfile":
        """Reconstruct a TargetProfile from a serialized dictionary."""
        from shared.target_types import TargetType, TechnologyStack
        profile = cls(target=data.get("target", ""))
        try:
            profile.target_type = TargetType(data.get("target_type", TargetType.UNKNOWN.value))
        except ValueError:
            profile.target_type = TargetType.UNKNOWN
        profile.ip_addresses = data.get("ip_addresses", [])
        profile.open_ports = data.get("open_ports", [])
        profile.services = data.get("services", {})
        techs = []
        for t in data.get("technologies", []):
            try:
                techs.append(TechnologyStack(t))
            except ValueError:
                pass
        profile.technologies = techs
        profile.cms_type = data.get("cms_type")
        profile.cloud_provider = data.get("cloud_provider")
        profile.security_headers = data.get("security_headers", {})
        profile.ssl_info = data.get("ssl_info", {})
        profile.subdomains = data.get("subdomains", [])
        profile.endpoints = data.get("endpoints", [])
        profile.attack_surface_score = float(data.get("attack_surface_score", 0.0))
        profile.risk_level = data.get("risk_level", "unknown")
        profile.confidence_score = float(data.get("confidence_score", 0.0))
        return profile
