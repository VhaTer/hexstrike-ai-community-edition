"""
mcp_core/technology_detector.py

Port of V6 TechnologyDetector class to FastMCP 3.x Phase 3.

Original V6 class: hexstrike_server.py#L4223-L4303
Part of the IntelligentDecisionEngine subsystem.

Detects technology stack from HTTP headers, HTML content, and open ports.
Results feed into ParameterOptimizer to automatically tune tool parameters.

Usage:
    from mcp_core.technology_detector import TechnologyDetector

    detector = TechnologyDetector()
    profile = detector.detect(
        headers={"Server": "Apache/2.4.51", "X-Powered-By": "PHP/8.0"},
        content="<meta name='generator' content='WordPress 6.0'>",
        ports=[80, 443, 3306]
    )
    # profile.web_servers = ["apache"]
    # profile.languages   = ["php"]
    # profile.cms         = ["wordpress"]
    # profile.is_waf      = True/False
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection profile — result of TechnologyDetector.detect()
# ---------------------------------------------------------------------------

@dataclass
class TechProfile:
    """
    Technology profile for a scanned target.
    Mirrors V6 TechnologyDetector output structure.
    """
    web_servers:  List[str] = field(default_factory=list)
    frameworks:   List[str] = field(default_factory=list)
    cms:          List[str] = field(default_factory=list)
    databases:    List[str] = field(default_factory=list)
    languages:    List[str] = field(default_factory=list)
    security:     List[str] = field(default_factory=list)   # WAFs, CDNs
    services:     List[str] = field(default_factory=list)   # From ports

    @property
    def is_waf(self) -> bool:
        """True if a WAF or CDN was detected."""
        return bool(self.security)

    @property
    def is_wordpress(self) -> bool:
        return "wordpress" in self.cms

    @property
    def is_php(self) -> bool:
        return "php" in self.languages

    def summary(self) -> str:
        """One-line summary for ctx.info() logging."""
        parts = []
        if self.web_servers:
            parts.append(f"server={','.join(self.web_servers)}")
        if self.cms:
            parts.append(f"cms={','.join(self.cms)}")
        if self.languages:
            parts.append(f"lang={','.join(self.languages)}")
        if self.security:
            parts.append(f"waf={','.join(self.security)}")
        if self.frameworks:
            parts.append(f"framework={','.join(self.frameworks)}")
        return " | ".join(parts) if parts else "unknown stack"

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "web_servers": self.web_servers,
            "frameworks":  self.frameworks,
            "cms":         self.cms,
            "databases":   self.databases,
            "languages":   self.languages,
            "security":    self.security,
            "services":    self.services,
        }


# ---------------------------------------------------------------------------
# TechnologyDetector — ported from V6
# ---------------------------------------------------------------------------

class TechnologyDetector:
    """
    Detects technology stack from HTTP headers, HTML content, and open ports.

    Ported from V6 hexstrike_server.py#L4223-L4343.
    Uses signature-based matching (header patterns + content patterns + port map).
    Thread-safe: detect() is stateless (creates new TechProfile per call).
    """

    # Signature patterns — mirrors V6 detection_patterns dict
    _PATTERNS: Dict[str, Dict[str, List[str]]] = {
        "web_servers": {
            "apache":  ["Apache", "apache"],
            "nginx":   ["nginx", "Nginx"],
            "iis":     ["IIS", "Microsoft-IIS"],
            "tomcat":  ["Tomcat", "Apache-Coyote"],
            "lighttpd":["lighttpd"],
            "caddy":   ["Caddy"],
        },
        "frameworks": {
            "django":    ["Django", "csrfmiddlewaretoken", "django"],
            "rails":     ["Ruby on Rails", "X-Runtime", "rails"],
            "laravel":   ["Laravel", "laravel_session"],
            "express":   ["Express", "X-Powered-By: Express"],
            "flask":     ["Werkzeug", "flask"],
            "spring":    ["Spring", "JSESSIONID"],
            "asp_net":   ["ASP.NET", "X-AspNet-Version", "X-Powered-By: ASP.NET"],
            "nextjs":    ["__NEXT_DATA__", "next.js", "_next/"],
            "react":     ["react", "React", "__REACT"],
            "angular":   ["ng-version", "angular", "ng-app"],
            "vue":       ["Vue.js", "vue", "__vue__"],
        },
        "cms": {
            "wordpress": ["WordPress", "wp-content", "wp-json", "wp-admin", "wp-includes"],
            "drupal":    ["Drupal", "drupal", "X-Generator: Drupal"],
            "joomla":    ["Joomla", "joomla", "/components/com_"],
            "magento":   ["Magento", "Mage.Cookies", "mage-"],
            "prestashop":["PrestaShop", "prestashop"],
            "shopify":   ["Shopify", "shopify", "cdn.shopify.com"],
            "wix":       ["wix.com", "X-Wix-"],
            "ghost":     ["Ghost", "ghost/"],
            "typo3":     ["TYPO3", "typo3"],
        },
        "databases": {
            "mysql":      ["mysql", "MySQL"],
            "postgresql": ["PostgreSQL", "postgres"],
            "mongodb":    ["MongoDB", "mongo"],
            "redis":      ["Redis", "redis"],
            "elasticsearch": ["Elasticsearch", "elastic"],
        },
        "languages": {
            "php":    ["PHP", "X-Powered-By: PHP", ".php"],
            "python": ["Python", "X-Powered-By: Python", "WSGIServer"],
            "ruby":   ["Ruby", "X-Powered-By: Ruby", "Phusion Passenger"],
            "java":   ["Java", "JSP", "JSESSIONID", ".jsp"],
            "dotnet": [".NET", "ASP.NET", "X-AspNet-Version"],
            "nodejs": ["Node.js", "X-Powered-By: node"],
            "golang": ["Go ", "Go-http-client", "gorilla"],
        },
        "security": {
            "cloudflare": ["cloudflare", "CF-RAY", "__cfduid", "cf-cache-status"],
            "incapsula":  ["incap_ses", "visid_incap", "X-CDN: Incapsula"],
            "sucuri":     ["Sucuri", "X-Sucuri-ID", "sucuri"],
            "akamai":     ["AkamaiGHost", "X-Akamai-"],
            "fastly":     ["Fastly", "X-Fastly-"],
            "modsecurity":["Mod_Security", "NOYB"],
            "f5_bigip":   ["BigIP", "F5", "BIGipServer"],
            "aws_waf":    ["x-amzn-RequestId", "AWS WAF"],
            "barracuda":  ["barra_counter_session", "BNI__BARRACUDA_LB"],
        },
    }

    # Port → service mapping — from V6 port_services dict
    _PORT_SERVICES: Dict[int, str] = {
        21:    "ftp",
        22:    "ssh",
        23:    "telnet",
        25:    "smtp",
        53:    "dns",
        80:    "http",
        443:   "https",
        445:   "smb",
        1433:  "mssql",
        1521:  "oracle",
        2375:  "docker",
        3306:  "mysql",
        3389:  "rdp",
        5432:  "postgresql",
        5900:  "vnc",
        6379:  "redis",
        7001:  "weblogic",
        8080:  "http-alt",
        8443:  "https-alt",
        8888:  "jupyter",
        9200:  "elasticsearch",
        27017: "mongodb",
        28017: "mongodb-web",
    }

    def detect(
        self,
        headers:  Dict[str, str] | None = None,
        content:  str = "",
        ports:    List[int] | None = None,
    ) -> TechProfile:
        """
        Detect technology stack from available signals.

        Args:
            headers: HTTP response headers dict
            content: HTML/response body content
            ports:   List of open ports (from nmap/rustscan results)

        Returns:
            TechProfile with detected technologies by category
        """
        profile = TechProfile()
        headers  = headers or {}
        content  = content or ""
        ports    = ports or []

        # Sanitize inputs (V6 security consideration)
        MAX_CONTENT = 10 * 1024 * 1024  # 10 MB cap
        if len(content) > MAX_CONTENT:
            content = content[:MAX_CONTENT]
            logger.debug("TechnologyDetector: content truncated to 10MB")

        content_lower = content.lower()

        # --- Header + content detection ---
        for category, tech_map in self._PATTERNS.items():
            detected_list: List[str] = getattr(profile, category)
            for tech, signatures in tech_map.items():
                if tech in detected_list:
                    continue
                # Check headers
                for h_name, h_value in headers.items():
                    h_combined = f"{h_name}: {h_value}".lower()
                    if any(sig.lower() in h_combined for sig in signatures):
                        detected_list.append(tech)
                        break
                # Check content (only if not already found via headers)
                if tech not in detected_list:
                    if any(sig.lower() in content_lower for sig in signatures):
                        detected_list.append(tech)

        # --- Port-based service detection ---
        for port in ports:
            service = self._PORT_SERVICES.get(int(port))
            if service and service not in profile.services:
                profile.services.append(service)
                # Map port services to database category
                if service in ("mysql", "postgresql", "mongodb", "redis", "elasticsearch", "mssql", "oracle"):
                    if service not in profile.databases:
                        profile.databases.append(service)

        if profile.web_servers or profile.cms or profile.security:
            logger.debug(f"TechnologyDetector: {profile.summary()}")

        return profile
