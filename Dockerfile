FROM kalilinux/kali-rolling

ENV DEBIAN_FRONTEND=noninteractive
ENV HEXSTRIKE_PORT=8888
ENV GOPATH=/go
ENV PATH="$GOPATH/bin:/opt/hexstrike-venv/bin:$PATH"

# -----------------------------
# Base system + build tools
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3 python3-pip python3-venv git curl wget sudo \
    gnupg2 ca-certificates golang gcc libc6-dev pkg-config unzip \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Kali security tools (CLI only)
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    nmap masscan rustscan amass subfinder autorecon dnsenum theharvester \
    arp-scan nbtscan responder enum4linux smbmap rpcclient netexec \
    gobuster dirb ffuf feroxbuster dirsearch httpx wafw00f sqlmap nikto wpscan \
    hydra john hashcat medusa patator evil-winrm hash-identifier ophcrack crackmapexec \
    gdb radare2 binwalk ropgadget checksec strings objdump readelf xxd hexdump file \
    volatility3 foremost steghide exiftool photorec testdisk scalpel bulk-extractor sleuthkit \
    trivy kube-hunter kube-bench docker-bench-security prowler scout-suite checkov terrascan falco \
    awscli azure-cli google-cloud-cli kubectl helm \
    whois dnsutils host \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Go-based tools
# -----------------------------
RUN go install github.com/projectdiscovery/nuclei/v2/cmd/nuclei@latest \
 && go install github.com/projectdiscovery/httpx/cmd/httpx@latest \
 && go install github.com/projectdiscovery/katana/cmd/katana@latest \
 && go install github.com/hahwul/dalfox/v2@latest \
 && go install github.com/jaeles-project/jaeles/v2@latest

# -----------------------------
# Python virtual environment
# -----------------------------
RUN python3 -m venv /opt/hexstrike-venv

WORKDIR /app
COPY requirements.txt .

RUN /opt/hexstrike-venv/bin/pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy application code
# -----------------------------
COPY . .

# -----------------------------
# Chromium (stabilisé pour Docker)
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends chromium chromium-driver \
 && ln -s /usr/bin/chromium /usr/bin/google-chrome \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV CHROME_FLAGS="--no-sandbox --disable-dev-shm-usage --disable-gpu"

# -----------------------------
# Expose port
# -----------------------------
EXPOSE 8888

# -----------------------------
# Healthcheck
# -----------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8888/health || exit 1

# -----------------------------
# Start HexStrike AI
# -----------------------------
CMD ["/opt/hexstrike-venv/bin/python", "hexstrike_server.py", "--port", "8888"]
