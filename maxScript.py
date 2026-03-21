import re, glob, os

# Tools longs — phases adaptées par catégorie
CONFIGS = {
    "password_cracking/hashcat.py": {
        "emoji": "🔐", "phases": [
            (15, "📂 Loading hash file..."),
            (30, "🔑 Initializing hashcat engine..."),
            (55, "💥 Cracking in progress — GPU working..."),
            (80, "💥 Still cracking — this may take a while..."),
        ], "tick": 30,
        "success_hint": "💡 Use --show flag to display cracked passwords"
    },
    "password_cracking/john.py": {
        "emoji": "🔐", "phases": [
            (20, "📂 Loading hash file..."),
            (40, "🔍 Auto-detecting hash format..."),
            (65, "💥 Dictionary attack in progress..."),
            (85, "💥 Still cracking..."),
        ], "tick": 20,
        "success_hint": "💡 Run with --show to display cracked passwords"
    },
    "password_cracking/patator.py": {
        "emoji": "🔐", "phases": [
            (20, "🔌 Connecting to target service..."),
            (45, "💥 Brute-forcing credentials..."),
            (75, "💥 Attack in progress..."),
        ], "tick": 15,
        "success_hint": None
    },
    "password_cracking/hydra.py": {
        "emoji": "🔐", "phases": [
            (20, "🔌 Connecting to target service..."),
            (45, "💥 Brute-forcing credentials..."),
            (70, "💥 Attack in progress..."),
            (88, "📋 Finalizing results..."),
        ], "tick": 15,
        "success_hint": None
    },
    "password_cracking/medusa.py": {
        "emoji": "🔐", "phases": [
            (20, "🔌 Initializing parallel attack..."),
            (50, "💥 Brute-forcing in progress..."),
            (80, "💥 Still attacking..."),
        ], "tick": 15,
        "success_hint": None
    },
    "password_cracking/ophcrack.py": {
        "emoji": "🔐", "phases": [
            (20, "📂 Loading rainbow tables..."),
            (50, "💥 Cracking NTLM hashes..."),
            (80, "📋 Finalizing results..."),
        ], "tick": 15,
        "success_hint": None
    },
    "web_crawl/katana.py": {
        "emoji": "⚔️", "phases": [
            (20, "🌐 Fetching initial page..."),
            (40, "🔍 Crawling discovered links..."),
            (65, "⚙️ Rendering JavaScript endpoints..."),
            (85, "📋 Extracting forms and parameters..."),
        ], "tick": 10,
        "success_hint": None
    },
    "web_crawl/hakrawler.py": {
        "emoji": "🕷️", "phases": [
            (25, "🌐 Fetching pages..."),
            (55, "🔍 Extracting endpoints..."),
            (80, "📋 Processing results..."),
        ], "tick": 8,
        "success_hint": None
    },
    "url_recon/gau.py": {
        "emoji": "📡", "phases": [
            (20, "📚 Querying Wayback Machine..."),
            (45, "📚 Querying CommonCrawl..."),
            (70, "📚 Querying OTX and urlscan..."),
            (90, "📋 Deduplicating results..."),
        ], "tick": 10,
        "success_hint": "💡 Pipe results through uro to deduplicate"
    },
    "url_recon/waybackurls.py": {
        "emoji": "📡", "phases": [
            (30, "📚 Querying Wayback Machine..."),
            (65, "📋 Processing historical URLs..."),
            (90, "🔍 Filtering results..."),
        ], "tick": 10,
        "success_hint": None
    },
    "smb_enum/enum4linux.py": {
        "emoji": "🏴", "phases": [
            (20, "🔌 Connecting to SMB target..."),
            (45, "👥 Enumerating users and groups..."),
            (70, "📂 Enumerating shares..."),
            (88, "📋 Gathering policy info..."),
        ], "tick": 10,
        "success_hint": None
    },
    "smb_enum/netexec.py": {
        "emoji": "🏴", "phases": [
            (25, "🔌 Connecting to target..."),
            (55, "🔍 Running enumeration modules..."),
            (85, "📋 Processing results..."),
        ], "tick": 10,
        "success_hint": None
    },
    "smb_enum/smbmap.py": {
        "emoji": "🏴", "phases": [
            (25, "🔌 Connecting to SMB service..."),
            (55, "📂 Enumerating shares and permissions..."),
            (85, "📋 Processing results..."),
        ], "tick": 8,
        "success_hint": None
    },
    "cloud_audit/prowler.py": {
        "emoji": "☁️", "phases": [
            (15, "🔐 Authenticating to cloud provider..."),
            (35, "🔍 Running compliance checks..."),
            (60, "🛡️ Analyzing IAM policies..."),
            (80, "📋 Generating findings report..."),
        ], "tick": 30,
        "success_hint": "💡 Check prowler output for HIGH/CRITICAL findings"
    },
    "cloud_audit/scout_suite.py": {
        "emoji": "☁️", "phases": [
            (15, "🔐 Authenticating to cloud provider..."),
            (40, "🔍 Scanning cloud services..."),
            (70, "🛡️ Analyzing security posture..."),
            (88, "📋 Generating HTML report..."),
        ], "tick": 30,
        "success_hint": None
    },
    "web_fuzz/gobuster.py": {
        "emoji": "🔫", "phases": [
            (20, "🔍 Initializing wordlist..."),
            (45, "💥 Fuzzing directories and files..."),
            (70, "💥 Still fuzzing..."),
            (88, "📋 Processing results..."),
        ], "tick": 12,
        "success_hint": None
    },
    "web_fuzz/ffuf.py": {
        "emoji": "🔫", "phases": [
            (20, "🔍 Initializing wordlist..."),
            (45, "💥 Fast fuzzing in progress..."),
            (70, "💥 Still fuzzing..."),
            (88, "📋 Processing results..."),
        ], "tick": 10,
        "success_hint": None
    },
    "web_fuzz/feroxbuster.py": {
        "emoji": "🔫", "phases": [
            (20, "🔍 Starting recursive content discovery..."),
            (45, "💥 Fuzzing directories..."),
            (70, "💥 Recursing into discovered paths..."),
            (88, "📋 Filtering and processing..."),
        ], "tick": 12,
        "success_hint": None
    },
    "web_fuzz/dirsearch.py": {
        "emoji": "🔫", "phases": [
            (20, "🔍 Loading wordlist..."),
            (45, "💥 Scanning directories..."),
            (70, "💥 Still scanning..."),
            (88, "📋 Processing results..."),
        ], "tick": 10,
        "success_hint": None
    },
    "web_fuzz/dirb.py": {
        "emoji": "🔫", "phases": [
            (20, "🔍 Loading wordlist..."),
            (45, "💥 Directory brute-forcing..."),
            (70, "💥 Still scanning..."),
            (88, "📋 Processing results..."),
        ], "tick": 10,
        "success_hint": None
    },
    "web_fuzz/wfuzz.py": {
        "emoji": "🔫", "phases": [
            (20, "🔍 Preparing payload..."),
            (45, "💥 Fuzzing in progress..."),
            (70, "💥 Still fuzzing..."),
            (88, "📋 Filtering responses..."),
        ], "tick": 10,
        "success_hint": None
    },
    "web_fuzz/dotdotpwn.py": {
        "emoji": "🔫", "phases": [
            (25, "🔍 Preparing traversal payloads..."),
            (55, "💥 Testing path traversal..."),
            (85, "📋 Processing results..."),
        ], "tick": 10,
        "success_hint": None
    },
    "web_scan/nikto.py": {
        "emoji": "🌐", "phases": [
            (15, "🔌 Connecting to web server..."),
            (35, "🔍 Scanning for vulnerabilities..."),
            (60, "🔍 Running vulnerability checks..."),
            (85, "📋 Compiling findings..."),
        ], "tick": 15,
        "success_hint": None
    },
    "web_scan/sqlmap.py": {
        "emoji": "💉", "phases": [
            (15, "🔌 Testing connection..."),
            (35, "💉 Testing SQL injection vectors..."),
            (60, "💉 Exploiting injection point..."),
            (85, "📋 Extracting data..."),
        ], "tick": 15,
        "success_hint": "💡 Check sqlmap output for extracted data"
    },
    "web_scan/wpscan.py": {
        "emoji": "🌐", "phases": [
            (20, "🔌 Fingerprinting WordPress..."),
            (45, "🔍 Enumerating plugins and themes..."),
            (70, "🛡️ Checking for vulnerabilities..."),
            (88, "📋 Generating report..."),
        ], "tick": 12,
        "success_hint": None
    },
    "web_scan/dalfox.py": {
        "emoji": "🌐", "phases": [
            (20, "🔍 Analyzing parameters..."),
            (45, "💥 Testing XSS payloads..."),
            (70, "💥 Running DOM analysis..."),
            (88, "📋 Processing findings..."),
        ], "tick": 10,
        "success_hint": None
    },
    "web_scan/zap.py": {
        "emoji": "🌐", "phases": [
            (15, "🔌 Starting ZAP scanner..."),
            (35, "🕷️ Spider crawling target..."),
            (60, "🔍 Active scanning..."),
            (85, "📋 Generating report..."),
        ], "tick": 20,
        "success_hint": None
    },
    "web_scan/jaeles.py": {
        "emoji": "🌐", "phases": [
            (20, "📋 Loading signatures..."),
            (45, "🔍 Scanning with signatures..."),
            (75, "📋 Processing findings..."),
        ], "tick": 12,
        "success_hint": None
    },
    "web_scan/xsser.py": {
        "emoji": "🌐", "phases": [
            (20, "🔍 Analyzing parameters..."),
            (50, "💥 Testing XSS payloads..."),
            (82, "📋 Processing results..."),
        ], "tick": 10,
        "success_hint": None
    },
    "web_scan/burpsuite.py": {
        "emoji": "🌐", "phases": [
            (20, "🔌 Connecting to Burp Suite..."),
            (50, "🔍 Running active scan..."),
            (82, "📋 Processing findings..."),
        ], "tick": 20,
        "success_hint": None
    },
    "vuln_scan/nuclei.py": {
        "emoji": "⚡", "phases": [
            (15, "📋 Loading templates..."),
            (35, "⚡ Running nuclei templates..."),
            (60, "⚡ Scanning with all templates..."),
            (85, "📋 Processing findings..."),
        ], "tick": 15,
        "success_hint": "💡 Check severity: CRITICAL > HIGH > MEDIUM > LOW"
    },
    "recon/amass.py": {
        "emoji": "🔍", "phases": [
            (15, "🌐 Querying passive sources..."),
            (35, "🔍 DNS brute-force enumeration..."),
            (60, "🌐 Scraping additional sources..."),
            (85, "📋 Building subdomain graph..."),
        ], "tick": 20,
        "success_hint": None
    },
    "recon/subfinder.py": {
        "emoji": "🔍", "phases": [
            (25, "🌐 Querying passive sources..."),
            (55, "🔍 Aggregating results..."),
            (85, "📋 Deduplicating subdomains..."),
        ], "tick": 10,
        "success_hint": None
    },
    "recon/autorecon.py": {
        "emoji": "🔍", "phases": [
            (10, "🔌 Starting AutoRecon..."),
            (25, "🔍 Running initial scans..."),
            (45, "📡 Port scanning all ports..."),
            (65, "🔎 Running service enumeration..."),
            (85, "📋 Generating report structure..."),
        ], "tick": 30,
        "success_hint": "💡 Check results/ directory for detailed output"
    },
    "recon/theharvester.py": {
        "emoji": "🔍", "phases": [
            (20, "🌐 Querying search engines..."),
            (50, "📧 Harvesting emails and subdomains..."),
            (80, "📋 Processing results..."),
        ], "tick": 10,
        "success_hint": None
    },
    "exploit_framework/metasploit.py": {
        "emoji": "💣", "phases": [
            (15, "🚀 Loading Metasploit framework..."),
            (35, "🎯 Configuring exploit module..."),
            (60, "💣 Executing exploit..."),
            (85, "📋 Processing results..."),
        ], "tick": 20,
        "success_hint": "💡 Check for opened sessions with sessions -l"
    },
    "exploit_framework/msfvenom.py": {
        "emoji": "💣", "phases": [
            (25, "🔧 Configuring payload..."),
            (55, "💣 Generating payload..."),
            (85, "📋 Encoding and finalizing..."),
        ], "tick": 8,
        "success_hint": None
    },
    "exploit_framework/pwntools.py": {
        "emoji": "💣", "phases": [
            (25, "🔧 Setting up exploit environment..."),
            (55, "💣 Running exploit script..."),
            (85, "📋 Processing results..."),
        ], "tick": 10,
        "success_hint": None
    },
    "exploit_framework/pwninit.py": {
        "emoji": "💣", "phases": [
            (30, "📂 Patching binary..."),
            (65, "🔧 Setting up environment..."),
            (88, "📋 Finalizing..."),
        ], "tick": 8,
        "success_hint": None
    },
    "exploit_framework/exploit_db.py": {
        "emoji": "💣", "phases": [
            (30, "🔍 Searching ExploitDB..."),
            (65, "📋 Fetching exploit details..."),
            (88, "📋 Processing results..."),
        ], "tick": 8,
        "success_hint": None
    },
    "memory_forensics/volatility.py": {
        "emoji": "🧠", "phases": [
            (15, "📂 Loading memory image..."),
            (40, "🔍 Identifying profile..."),
            (65, "🧠 Running analysis plugins..."),
            (88, "📋 Processing artifacts..."),
        ], "tick": 20,
        "success_hint": None
    },
    "memory_forensics/volatility3.py": {
        "emoji": "🧠", "phases": [
            (15, "📂 Loading memory image..."),
            (40, "🔍 Auto-detecting OS profile..."),
            (65, "🧠 Running analysis plugins..."),
            (88, "📋 Processing artifacts..."),
        ], "tick": 20,
        "success_hint": None
    },
    "binary_analysis/angr.py": {
        "emoji": "🔬", "phases": [
            (15, "📂 Loading binary..."),
            (35, "🧠 Lifting to VEX IR..."),
            (60, "⚙️ Symbolic execution in progress..."),
            (85, "📋 Analyzing paths..."),
        ], "tick": 25,
        "success_hint": None
    },
    "binary_analysis/ghidra.py": {
        "emoji": "🔬", "phases": [
            (15, "📂 Loading binary into Ghidra..."),
            (40, "🔍 Auto-analyzing binary..."),
            (70, "🧠 Decompiling functions..."),
            (88, "📋 Exporting results..."),
        ], "tick": 20,
        "success_hint": None
    },
    "binary_analysis/binwalk.py": {
        "emoji": "🔬", "phases": [
            (25, "🔍 Scanning binary signatures..."),
            (55, "📦 Extracting embedded files..."),
            (85, "📋 Processing extracted content..."),
        ], "tick": 10,
        "success_hint": None
    },
    "binary_analysis/autopsy.py": {
        "emoji": "🔬", "phases": [
            (15, "📂 Loading disk image..."),
            (40, "🔍 Running ingest modules..."),
            (70, "🧠 Analyzing artifacts..."),
            (88, "📋 Generating report..."),
        ], "tick": 25,
        "success_hint": None
    },
    "binary_debug/gdb.py": {
        "emoji": "🔬", "phases": [
            (25, "📂 Loading binary in GDB..."),
            (55, "🔍 Running debug session..."),
            (85, "📋 Processing output..."),
        ], "tick": 10,
        "success_hint": None
    },
    "binary_debug/radare2.py": {
        "emoji": "🔬", "phases": [
            (20, "📂 Loading binary..."),
            (45, "🔍 Analyzing code..."),
            (75, "📋 Processing results..."),
        ], "tick": 12,
        "success_hint": None
    },
    "net_scan/masscan.py": {
        "emoji": "📡", "phases": [
            (20, "📡 Starting high-speed scan..."),
            (50, "📡 Scanning port ranges..."),
            (80, "📋 Processing discovered ports..."),
        ], "tick": 10,
        "success_hint": None
    },
    "net_scan/rustscan.py": {
        "emoji": "📡", "phases": [
            (25, "⚡ Fast port discovery..."),
            (60, "🔍 Identifying open ports..."),
            (88, "📋 Passing to nmap for service detection..."),
        ], "tick": 8,
        "success_hint": None
    },
    "k8s_scan/kube_hunter.py": {
        "emoji": "☸️", "phases": [
            (15, "🔌 Connecting to Kubernetes cluster..."),
            (40, "🔍 Hunting for vulnerabilities..."),
            (70, "🛡️ Testing attack vectors..."),
            (88, "📋 Generating findings..."),
        ], "tick": 20,
        "success_hint": None
    },
    "k8s_scan/kube_bench.py": {
        "emoji": "☸️", "phases": [
            (20, "🔌 Connecting to cluster..."),
            (50, "🔍 Running CIS benchmark checks..."),
            (80, "📋 Generating compliance report..."),
        ], "tick": 15,
        "success_hint": "💡 Check FAIL items for hardening recommendations"
    },
    "container_scan/trivy.py": {
        "emoji": "🐳", "phases": [
            (20, "📂 Loading image/filesystem..."),
            (45, "🔍 Scanning for CVEs..."),
            (70, "🛡️ Checking misconfigurations..."),
            (88, "📋 Generating report..."),
        ], "tick": 15,
        "success_hint": None
    },
    "container_scan/docker_bench.py": {
        "emoji": "🐳", "phases": [
            (25, "🔍 Running Docker CIS benchmarks..."),
            (60, "🛡️ Checking security configurations..."),
            (88, "📋 Generating report..."),
        ], "tick": 15,
        "success_hint": None
    },
    "container_scan/clair_vulnerability.py": {
        "emoji": "🐳", "phases": [
            (20, "🔌 Connecting to Clair..."),
            (50, "🔍 Scanning image layers..."),
            (82, "📋 Processing CVE findings..."),
        ], "tick": 15,
        "success_hint": None
    },
    "vuln_scan/nuclei.py": {
        "emoji": "⚡", "phases": [
            (15, "📋 Loading templates..."),
            (35, "⚡ Running nuclei templates..."),
            (60, "⚡ Scanning with all templates..."),
            (85, "📋 Processing findings..."),
        ], "tick": 15,
        "success_hint": "💡 Check severity: CRITICAL > HIGH > MEDIUM > LOW"
    },
    "param_discovery/arjun.py": {
        "emoji": "🔍", "phases": [
            (20, "🔍 Loading parameter wordlist..."),
            (50, "💥 Testing parameters..."),
            (80, "📋 Processing discovered parameters..."),
        ], "tick": 10,
        "success_hint": None
    },
    "param_discovery/paramspider.py": {
        "emoji": "🕷️", "phases": [
            (25, "🌐 Querying web archives..."),
            (60, "🔍 Extracting parameters..."),
            (88, "📋 Processing results..."),
        ], "tick": 10,
        "success_hint": None
    },
    "param_discovery/x8.py": {
        "emoji": "🔍", "phases": [
            (20, "🔍 Initializing parameter discovery..."),
            (50, "💥 Testing hidden parameters..."),
            (82, "📋 Processing results..."),
        ], "tick": 10,
        "success_hint": None
    },
    "iac_scan/checkov.py": {
        "emoji": "🛡️", "phases": [
            (25, "📂 Loading IaC files..."),
            (55, "🔍 Running security checks..."),
            (85, "📋 Generating findings report..."),
        ], "tick": 10,
        "success_hint": "💡 Check FAILED checks for security issues"
    },
    "iac_scan/terrascan.py": {
        "emoji": "🛡️", "phases": [
            (25, "📂 Scanning Terraform/K8s files..."),
            (55, "🔍 Running policy checks..."),
            (85, "📋 Generating report..."),
        ], "tick": 10,
        "success_hint": None
    },
    "cloud_visual/cloudmapper.py": {
        "emoji": "☁️", "phases": [
            (20, "🔐 Collecting AWS data..."),
            (50, "🗺️ Building network map..."),
            (82, "📋 Generating visualization..."),
        ], "tick": 15,
        "success_hint": None
    },
    "api_scan/api_schema_analyzer.py": {
        "emoji": "🌐", "phases": [
            (25, "📋 Fetching API schema..."),
            (55, "🔍 Analyzing endpoints..."),
            (85, "📋 Processing security findings..."),
        ], "tick": 10,
        "success_hint": None
    },
    "api_scan/graphql_scanner.py": {
        "emoji": "🌐", "phases": [
            (20, "🔍 Introspecting GraphQL schema..."),
            (50, "💥 Testing queries and mutations..."),
            (82, "📋 Processing findings..."),
        ], "tick": 10,
        "success_hint": None
    },
    "api_fuzz/api_fuzzer.py": {
        "emoji": "🔫", "phases": [
            (20, "📋 Loading API schema..."),
            (45, "💥 Fuzzing endpoints..."),
            (70, "💥 Still fuzzing..."),
            (88, "📋 Processing results..."),
        ], "tick": 12,
        "success_hint": None
    },
    "web_framework/browser_agent.py": {
        "emoji": "🌐", "phases": [
            (20, "🌐 Launching headless browser..."),
            (45, "🔍 Navigating and analyzing..."),
            (75, "📋 Processing DOM and network data..."),
        ], "tick": 10,
        "success_hint": None
    },
    "waf_detect/wafw00f.py": {
        "emoji": "🛡️", "phases": [
            (30, "🔍 Sending detection probes..."),
            (65, "🛡️ Analyzing WAF signatures..."),
            (88, "📋 Processing results..."),
        ], "tick": 8,
        "success_hint": None
    },
    "file_carving/foremost.py": {
        "emoji": "🔬", "phases": [
            (20, "📂 Loading disk image..."),
            (50, "🔍 Carving files by signature..."),
            (82, "📋 Recovering files..."),
        ], "tick": 15,
        "success_hint": None
    },
    "runtime_monitor/falco.py": {
        "emoji": "👁️", "phases": [
            (20, "🔌 Starting Falco monitor..."),
            (50, "👁️ Monitoring runtime events..."),
            (82, "📋 Processing alerts..."),
        ], "tick": 15,
        "success_hint": None
    },
    "stego_analysis/steghide.py": {
        "emoji": "🔍", "phases": [
            (30, "🔍 Analyzing steganographic data..."),
            (65, "📦 Extracting hidden content..."),
            (88, "📋 Processing results..."),
        ], "tick": 8,
        "success_hint": None
    },
    "credential_harvest/responder.py": {
        "emoji": "🎣", "phases": [
            (15, "🔌 Starting Responder listeners..."),
            (40, "🎣 Poisoning LLMNR/NBT-NS..."),
            (70, "🔑 Waiting for credential captures..."),
            (88, "📋 Processing captured hashes..."),
        ], "tick": 20,
        "success_hint": "💡 Check Responder logs for captured credentials"
    },
    "cloud_exploit/pacu.py": {
        "emoji": "☁️", "phases": [
            (15, "🔐 Authenticating to AWS..."),
            (40, "🔍 Running enumeration modules..."),
            (65, "💣 Executing attack modules..."),
            (88, "📋 Processing results..."),
        ], "tick": 20,
        "success_hint": None
    },
    "bugbounty_workflow/bug_bounty_recon.py": {
        "emoji": "🏆", "phases": [
            (15, "🔍 Starting recon workflow..."),
            (35, "📡 Subdomain enumeration..."),
            (55, "🌐 Web probing discovered assets..."),
            (75, "⚡ Vulnerability scanning..."),
            (90, "📋 Compiling findings..."),
        ], "tick": 20,
        "success_hint": "💡 Check output for potential bug bounty findings"
    },
    "recon_bot/bbot.py": {
        "emoji": "🤖", "phases": [
            (15, "🤖 Initializing BBOT scan..."),
            (35, "🌐 Running recon modules..."),
            (60, "🔍 Aggregating intelligence..."),
            (85, "📋 Building target graph..."),
        ], "tick": 20,
        "success_hint": None
    },
    # WiFi tools that still need progress
    "wifi_pentest/eaphammer.py": {
        "emoji": "🎭", "phases": [
            (20, "🔐 Setting up certificates..."),
            (45, "📡 Broadcasting rogue AP..."),
            (70, "🎣 Waiting for client connections..."),
            (88, "🔑 Capturing EAP credentials..."),
        ], "tick": 15,
        "success_hint": "💡 Check eaphammer.log for captured credentials"
    },
    "wifi_pentest/mdk4.py": {
        "emoji": "💥", "phases": [
            (25, "📡 Starting attack..."),
            (55, "💥 Attack in progress..."),
            (85, "📋 Processing results..."),
        ], "tick": 10,
        "success_hint": None
    },
    "wifi_pentest/airbase_ng.py": {
        "emoji": "🎭", "phases": [
            (25, "📡 Starting rogue AP..."),
            (55, "🎣 Waiting for clients..."),
            (85, "📋 Processing connections..."),
        ], "tick": 10,
        "success_hint": None
    },
    "wifi_pentest/bettercap_wifi.py": {
        "emoji": "📶", "phases": [
            (20, "🔌 Starting Bettercap..."),
            (50, "📡 Running WiFi modules..."),
            (82, "📋 Processing results..."),
        ], "tick": 10,
        "success_hint": None
    },
    "wifi_pentest/aireplay_ng.py": {
        "emoji": "💥", "phases": [
            (25, "📡 Preparing injection..."),
            (55, "💥 Injecting packets..."),
            (85, "📋 Finalizing attack..."),
        ], "tick": 8,
        "success_hint": None
    },
    "dns_enum/dnsenum.py": {
        "emoji": "🔍", "phases": [
            (25, "🌐 Querying DNS records..."),
            (55, "🔍 Brute-forcing subdomains..."),
            (85, "📋 Processing results..."),
        ], "tick": 10,
        "success_hint": None
    },
    "dns_enum/fierce.py": {
        "emoji": "🔍", "phases": [
            (25, "🌐 Probing DNS servers..."),
            (55, "🔍 Discovering subdomains..."),
            (85, "📋 Processing results..."),
        ], "tick": 8,
        "success_hint": None
    },
}


def add_progress(path, config):
    with open(path, 'r') as f:
        content = f.read()

    if "report_progress" in content:
        print(f"  ⏭️  {path} — already has report_progress")
        return

    if "run_in_executor" in content:
        print(f"  ⏭️  {path} — already has run_in_executor")
        return

    # Build phases string
    phases_str = "[\n"
    for prog, msg in config["phases"]:
        phases_str += f'            ({prog}, "{msg}"),\n'
    phases_str += "        ]"

    tick = config["tick"]
    hint = config.get("success_hint")
    hint_line = f'\n            await ctx.info("{hint}")' if hint else ""

    old_pattern = re.compile(
        r'(        await ctx\.info\([^)]+\)\n)'
        r'(        result = hexstrike_client\.safe_post\("([^"]+)", data\)\n)'
        r'(        if result\.get\("success"\):\n'
        r'            await ctx\.info\([^)]+\)\n'
        r'        else:\n'
        r'            await ctx\.error\([^)]+\)\n)'
        r'(        return result)',
        re.MULTILINE
    )

    def replace_block(m):
        endpoint = m.group(3)
        return (
            f'{m.group(1)}'
            f'        await ctx.report_progress(0, 100)\n\n'
            f'        loop = asyncio.get_running_loop()\n'
            f'        future = loop.run_in_executor(\n'
            f'            None, lambda: hexstrike_client.safe_post("{endpoint}", data)\n'
            f'        )\n\n'
            f'        phases = {phases_str}\n'
            f'        for progress, message in phases:\n'
            f'            done, _ = await asyncio.wait([future], timeout={tick})\n'
            f'            if done:\n'
            f'                break\n'
            f'            await ctx.report_progress(progress, 100)\n'
            f'            await ctx.info(message)\n\n'
            f'        result = await future\n'
            f'        await ctx.report_progress(100, 100)\n\n'
            f'        if result.get("success"):\n'
            f'            await ctx.info("✅ Completed successfully"){hint_line}\n'
            f'        else:\n'
            f'            await ctx.error(f"❌ Failed: {{result.get(\'error\', \'unknown\')}}")\n'
            f'        return result'
        )

    new_content = old_pattern.sub(replace_block, content)

    # Add asyncio import if missing
    if "import asyncio" not in new_content:
        new_content = new_content.replace(
            "from fastmcp import Context",
            "import asyncio\nfrom fastmcp import Context"
        )

    if new_content == content:
        print(f"  ⚠️  {path} — pattern not matched, skip")
        return

    with open(path, 'w') as f:
        f.write(new_content)
    print(f"  ✅ {path}")


base = "mcp_tools/"
for rel_path, config in CONFIGS.items():
    full_path = base + rel_path
    if os.path.exists(full_path):
        add_progress(full_path, config)
    else:
        print(f"  ❌ {full_path} not found")

print("\nDone.")
