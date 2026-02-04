# Docker Support for HexStrike-AI Community edition 🐳

---

##  Overview

This Dockerfile provides a **reliable container build** that:

- Uses **Kali Linux** as a base (credit to kali team),
- Installs CLI tools via `apt`,
- Installs Go-based tools via `go install`,
- Uses Python virtualenv for HexStrike dependencies,
- Includes headless browser support for browser agents,
- Includes a **health check** endpoint.

---

## 🧱  Installed Tools :

✔ **APT-installed tools**  
Core security and reconnaissance tools available in Kali repositories (e.g., `nmap`, `ffuf`, `wpscan`, `hydra`, `trivy`, `kube-hunter`, etc.). :contentReference[oaicite:3]{index=3}

✔ **Go-based tools**  
Nuclei, HTTPx, Katana, Dalfox, Jaeles — installed with `go install` for up-to-date versions.

✔ **Python dependencies**  
Installed from `requirements.txt` into a virtual environment for the HexStrike server and MCP client.

✔ **Headless browser (Chromium)**  
Supports headless browser automation for web agents.

---

## 🚫 Not Included :

- Burp Suite (full GUI)
- OWASP ZAP Proxy GUI
- Maltego

These tools require a desktop environment and cannot run in a standard headless Docker container. 
To use them, consider separate **VNC/GUI-enabled Docker images** or desktop installations.

---

## 📦 Build & Run Instructions

### 🔧 Build the Image :

From the root of the repository:

```bash
docker build -t hexstrike-ai-ce
```
This creates a container with all supported CLI tools and the HexStrike server.
