Absolutely — here is your **detailed Docker Testing Roadmap** in **Markdown format**, formatted for easy inclusion in your docs (`DOCKER.md`, CONTRIBUTING.md, or a dedicated test plan). It walks through exactly how to verify that your Docker image works end‑to‑end, including Chromium/Chromium‑Driver behavior.

You can paste this whole block directly into a Markdown file.

---

````markdown
## 🧪 Docker Testing Roadmap

This section outlines a **structured plan** to validate your HexStrike‑AI Docker image. Follow each step carefully to ensure the build is reliable and that key components (including CLI tools and optional browser automation) work as expected.

---

### 🧱 1. Build the Docker Image

From the root of the project directory (where the `Dockerfile` lives):

```sh
docker build -t hexstrike-ai-custom .
````

* Make sure there are **no errors** during the build.
* If a tool fails to install, double‑check the package name in Kali (`apt search <tool>`).

---

### ▶️ 2. Run the Container

Start a container and expose the HexStrike server port:

```sh
docker run -it -p 8888:8888 hexstrike-ai-custom
```

* You should see the container start in interactive mode.
* The server will listen on **port 8888** inside the container.

---

### 🔍 3. Validate Server Health

From your host machine:

```sh
curl http://localhost:8888/health
```

* A successful response means the **healthcheck endpoint is working**.

---

### 📦 4. Verify Key CLI Tools Inside the Container

Open a shell in the running container:

```sh
docker exec -it <container_id> bash
```

Then test several installed tools:

```sh
nmap --version
ffuf -h
httpx -h
trivy --version
hydra -h
```

Each command should output version info or help text.
If any are missing or fail, the install step needs review.

---

### 🌐 5. Test Headless Chromium

Chromium was included so that any tool or script that **needs a headless browser** can run in Docker without a GUI.

Run Chromium in “headless” mode:

```sh
chromium --headless --no-sandbox --disable-gpu --remote-debugging-port=9222 \
  https://example.com
```

This should not crash. It confirms that **Chromium launches headlessly**, which is important for:

* browser‑based scraping
* automated testing
* Selenium/Puppeteer workflows

> **Note:** Chrome/Chromium must be run with flags like `--no-sandbox`, because Docker containers generally don’t support default sandboxing. ([Chainguard Containers][1])

---

### 🚗 6. Optional: Test Chromium‑Driver with Selenium

If you intend to use browser automation (e.g., Selenium):

1. Install Selenium inside the container:

```sh
pip install selenium
```

2. Create a test script (`test_selenium.py`):

```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=options)
driver.get("https://example.com")
print(driver.title)
driver.quit()
```

3. Run the script:

```sh
python test_selenium.py
```

Successful output (e.g., “Example Domain”) confirms that:

* Chromium launches head‑lessly
* ChromeDriver works with Selenium
* The browser automation stack is functional

> Running headless Chrome in a container sometimes requires additional flags like `--disable-dev‑shm‑usage` and shared memory adjustments. ([Chainguard Containers][1])

---

### 📘 7. Check Tool Execution via MCP

From inside the container or an external client, test a few MCP commands:

```sh
python3 hexstrike_mcp.py --server http://localhost:8888 --query "nmap_scan example.com"
```

Then verify that:

* The MCP client connects
* The server executes the tool
* Results are returned as structured JSON

If these succeed, the MCP pipeline (client → server → tool → output) is valid.

---

### 🧹 8. Verify Logging & Errors

Check container logs:

```sh
docker logs <container_id>
```

Make sure there are:

* no silent failures
* no repeated error stack traces
* clear logs for startup, healthchecks, and tool execution

---

### 🧠 9. Edge Cases & Negative Tests

Test error conditions:

* Use an invalid domain in a scanner (e.g., `nmap_scan invalid.domain`)
* Try a tool with missing parameters
* Check server response timeouts

This ensures graceful error handling.

---

### 📊 10. Optional CI Integration

Set up CI to automatically:

* build the Docker image
* run healthchecks
* execute a small MCP test
* report build/test status

This automates your testing checklist on *every commit*. Tools like GitHub Actions or GitLab CI can do this.

---

## 🧠 Summary

| Test Step         | Purpose                              |
| ----------------- | ------------------------------------ |
| Build Image       | Confirm all installs succeed         |
| Run Container     | Confirm container starts             |
| Healthcheck       | Confirm MCP server is alive          |
| CLI Tools         | Validate tool availability           |
| Headless Chromium | Confirm browser automation stack     |
| Selenium Test     | Validate browser automation workflow |
| MCP Commands      | Confirm agent interaction            |
| Logs              | Diagnose errors                      |
| Edge Cases        | Validate failure paths               |
| CI Integration    | Automate tests                       |

---

Following this roadmap will help you **ensure the Docker image reliably supports HexStrike‑AI** and the tools you intend to run inside it.

```

---

If you like, I can also generate a **GitHub Actions workflow file** that automatically executes many of these tests on every push or PR — useful for CI/CD verification.
::contentReference[oaicite:2]{index=2}
```

[1]: https://images.chainguard.dev/directory/image/selenium-standalone-chromium/overview?utm_source=chatgpt.com "selenium-standalone-chromium Secure-by-Default Container Image | Chainguard"
