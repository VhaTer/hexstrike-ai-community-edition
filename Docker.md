# 📦 Docker Support for HexStrike-AI-CE

---

## 🚀 Overview

This Dockerfile aims to provide:
- a consistent, reproducible penetration-testing container,
- a headless environment suitable for MCP clients,
- easy integration with AI agents via MCP.

---

## 📌 What’s Included in the Docker Image

### 🧰 Installed Tools

The Docker image includes:

**1. Security tools installed via `apt` from Kali repositories**
- Network & Reconnaissance: `nmap`, `masscan`, `rustscan`, `amass`, etc.
- Web security: `gobuster`, `ffuf`, `dirsearch`, `sqlmap`, `wpscan`
- Auth & password tools: `hydra`, `john`, `hashcat`, `evil-winrm`, etc.
- Cloud & container tools: `trivy`, `kube-hunter`, `kube-bench`, `docker-bench-security`
- Forensics & CTF tools: `volatility3`, `foremost`, `photorec`, `sleuthkit`, etc.

**2. Go-based tools installed via `go install`**
- `nuclei`, `httpx`, `katana`, `dalfox`, `jaeles`

**3. Python dependencies via `pip`**
- HexStrike’s backend Python requirements inside a virtual environment.

**4. Headless browser support**
- Chrome/Chromium (for headless automation by agents).

---

The following tools are **not included** because they require a graphical user interface (GUI):
- **Burp Suite (full GUI)**
- **OWASP ZAP Proxy (desktop GUI)**
- **Maltego**

GUI tools are excluded because the Docker image is **headless**. If you need them, consider:
- a **VNC/GUI-enabled container**, or
- running them outside of Docker.

---

## 🛠 Build & Run Instructions

### 🔧 1. Build the Image

```sh
docker build -t hexstrike-ai-ce .
````

This will produce a Docker image with the HexStrike server and a large suite of security tools installed.

---

### ▶️ 2. Run the Container

```sh
docker run -it -p 8888:8888 hexstrike-ai-ce
```

This starts the HexStrike server on port **8888**, which can be accessed locally.

---

### 🔍 3. Check Health

Ensure the server is running:

```sh
curl http://localhost:8888/health
```

You should see a success response.

---

## 🧠 How HexStrike Works

HexStrike-AI uses the **MCP (Model Context Protocol)** to bridge AI agents and underlying security tools:

1. The HexStrike server listens for MCP connections.
2. AI clients execute MCP calls through a local MCP client script (`hexstrike_mcp.py`).
3. The server accepts tool requests and runs them, returning structured results. ([GitHub][1])

---

## 🤖 Example MCP Client Configurations

Below are configuration snippets for popular AI clients that support MCP.

---

### 🧠 Claude Desktop / Cursor

Edit `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "hexstrike-ai": {
      "command": "python3",
      "args": [
        "/path/to/hexstrike-ai/hexstrike_mcp.py",
        "--server",
        "http://localhost:8888"
      ],
      "description": "HexStrike AI MCP Server",
      "timeout": 300,
      "alwaysAllow": []
    }
  }
}
```

This tells Claude (or Cursor) how to start the MCP connection to your running Docker HexStrike server.

---

### 💻 VS Code Copilot

Add to your `.vscode/settings.json`:

```json
{
  "servers": {
    "hexstrike": {
      "type": "stdio",
      "command": "python3",
      "args": [
        "/path/to/hexstrike-ai/hexstrike_mcp.py",
        "--server",
        "http://localhost:8888"
      ]
    }
  },
  "inputs": []
}
```

This connects VS Code’s extension to the HexStrike server via MCP.

---

## ⚠️ Security & Best Practices

**Important:** The container gives AI clients access to low-level tool execution:

* Only run in isolated or controlled environments.
* Do *not* deploy this against systems you don’t own or are not authorized to test.

Always safeguard your local network and hosts when using powerful toolchains.

---

## 🧾 Troubleshooting

### ❌ Can’t reach server?

Check that Docker is running and the container logs:

```sh
docker logs <container-id>
```

### ❌ Tools not found inside container

Verify that the appropriate packages exist in the Kali repo or require manual installation via Go/pip.

---

## 📜 Summary

HexStrike-AI in Docker lets you automate security assessments with AI agents while keeping e
nvironments reproducible and consistent — ideal for development, testing, and controlled security research.

```
