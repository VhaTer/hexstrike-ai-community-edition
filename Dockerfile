FROM kalilinux/kali-rolling  
  
ENV DEBIAN_FRONTEND=noninteractive  
ENV HEXSTRIKE_PORT=8888  
  
# Install system dependencies and build tools  
RUN apt-get update && apt-get install -y --no-install-recommends \  
    build-essential \  
    python3-dev \  
    python3 python3-pip python3-venv \  
    git curl wget sudo gnupg2 ca-certificates \  
    && rm -rf /var/lib/apt/lists/*  
  
# Network & Reconnaissance Tools (25+ tools)  
RUN apt-get update && apt-get install -y --no-install-recommends \  
    nmap masscan rustscan autorecon amass subfinder nuclei dnsenum \  
    responder netexec enum4linux-ng enum4linux smbmap rpcclient \  
    nbtscan arp-scan fierce theharvester \  
    && rm -rf /var/lib/apt/lists/*  
  
# Web Application Security Tools (40+ tools)  
RUN apt-get update && apt-get install -y --no-install-recommends \  
    gobuster dirb ffuf feroxbuster dirsearch nikto sqlmap wpscan \  
    katana httpx dalfox jaeles hakrawler gau waybackurls wafw00f \  
    arjun paramspider x8 wfuzz dotdotpwn xsser \  
    && rm -rf /var/lib/apt/lists/*  
  
# Authentication & Password Security Tools (12+ tools)  
RUN apt-get update && apt-get install -y --no-install-recommends \  
    hydra john hashcat medusa patator hash-identifier ophcrack \  
    evil-winrm crackmapexec \  
    && rm -rf /var/lib/apt/lists/*  
  
# Binary Analysis & Reverse Engineering Tools (25+ tools)  
RUN apt-get update && apt-get install -y --no-install-recommends \  
    gdb radare2 binwalk ropgadget checksec strings objdump \  
    readelf xxd hexdump file ltrace strace \  
    && rm -rf /var/lib/apt/lists/*  
  
# Cloud & Container Security Tools (20+ tools)  
RUN apt-get update && apt-get install -y --no-install-recommends \  
    prowler scout-suite trivy kube-hunter kube-bench \  
    docker-bench-security checkov terrascan falco clair \  
    awscli azure-cli gcloud kubectl helm \  
    && rm -rf /var/lib/apt/lists/*  
  
# CTF & Forensics Tools (20+ tools)  
RUN apt-get update && apt-get install -y --no-install-recommends \  
    volatility3 foremost steghide exiftool photorec testdisk \  
    scalpel bulk-extractor sleuthkit autopsy stegsolve zsteg outguess \  
    && rm -rf /var/lib/apt/lists/*  
  
# OSINT & Intelligence Tools (20+ tools)  
RUN apt-get update && apt-get install -y --no-install-recommends \  
    sherlock recon-ng maltego spiderfoot shodan-cli censys-cli \  
    whois dig nslookup host dnsrecon sublist3r \  
    && rm -rf /var/lib/apt/lists/*  
  
# Browser Requirements  
RUN apt-get update && apt-get install -y --no-install-recommends \  
    chromium chromium-driver \  
    && rm -rf /var/lib/apt/lists/*  
  
# Create virtual environment  
RUN python3 -m venv /opt/hexstrike-venv  
ENV PATH="/opt/hexstrike-venv/bin:$PATH"  
  
WORKDIR /app  
  
# Copy and install Python dependencies  
COPY requirements.txt .  
RUN pip3 install --no-cache-dir -r requirements.txt  
  
# Copy application code  
COPY . .  
  
# Create chromium symlink for AI agent compatibility  
RUN ln -s /usr/bin/chromium /usr/bin/google-chrome  
  
EXPOSE 8888  
  
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \  
  CMD curl -f http://localhost:8888/health || exit 1  
  
CMD ["python3", "hexstrike_server.py", "--port", "8888"]
