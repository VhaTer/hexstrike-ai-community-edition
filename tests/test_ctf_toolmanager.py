"""Test suite for CTF Tool Manager"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from server_core.workflows.ctf.toolManager import CTFToolManager


class TestCTFToolManager:
    """Comprehensive tests for CTFToolManager"""

    @pytest.fixture
    def tool_manager(self):
        """Create CTFToolManager instance"""
        return CTFToolManager()

    def test_initialization(self, tool_manager):
        """Test CTFToolManager initialization"""
        assert tool_manager is not None
        assert hasattr(tool_manager, 'tool_commands')
        assert isinstance(tool_manager.tool_commands, dict)
        assert len(tool_manager.tool_commands) > 0

    def test_tool_commands_structure(self, tool_manager):
        """Test that tool_commands has expected entries"""
        expected_tools = ['httpx', 'katana', 'sqlmap', 'hashcat', 'john', 'checksec', 'pwntools']
        for tool in expected_tools:
            assert tool in tool_manager.tool_commands
            assert isinstance(tool_manager.tool_commands[tool], str)
            assert len(tool_manager.tool_commands[tool]) > 0

    def test_get_tool_command_simple(self, tool_manager):
        """Test getting simple tool command with target"""
        command = tool_manager.get_tool_command("hashcat", "target.txt")
        assert command is not None
        assert "hashcat" in command
        assert "target.txt" in command

    def test_get_tool_command_with_target(self, tool_manager):
        """Test getting tool command with target variable"""
        command = tool_manager.get_tool_command("nikto", "example.com")
        assert command is not None
        assert "nikto" in command

    def test_get_tool_command_with_additional_args(self, tool_manager):
        """Test getting tool command with additional arguments"""
        command = tool_manager.get_tool_command("sqlmap", "example.com", "--technique=B")
        assert command is not None
        assert "sqlmap" in command

    def test_get_tool_command_nonexistent_tool(self, tool_manager):
        """Test getting command for nonexistent tool"""
        command = tool_manager.get_tool_command("nonexistent_tool", "example.com")
        assert command is not None  # Should handle gracefully

    def test_get_category_tools_web(self, tool_manager):
        """Test getting web category tools"""
        tools = tool_manager.get_category_tools("web_recon")
        assert isinstance(tools, list)
        assert len(tools) > 0
        assert "httpx" in tools or any("web" in str(tool).lower() for tool in tools)

    def test_get_category_tools_crypto(self, tool_manager):
        """Test getting crypto category tools"""
        tools = tool_manager.get_category_tools("crypto_hash")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_category_tools_pwn(self, tool_manager):
        """Test getting pwn category tools"""
        tools = tool_manager.get_category_tools("pwn_analysis")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_category_tools_forensics(self, tool_manager):
        """Test getting forensics category tools"""
        tools = tool_manager.get_category_tools("forensics_file")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_category_tools_rev(self, tool_manager):
        """Test getting reverse engineering category tools"""
        tools = tool_manager.get_category_tools("rev_static")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_category_tools_misc(self, tool_manager):
        """Test getting misc category tools"""
        tools = tool_manager.get_category_tools("misc_encoding")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_category_tools_osint(self, tool_manager):
        """Test getting OSINT category tools"""
        tools = tool_manager.get_category_tools("osint_domain")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_category_tools_unknown(self, tool_manager):
        """Test getting tools for unknown category"""
        tools = tool_manager.get_category_tools("unknown_category")
        assert isinstance(tools, list)

    def test_suggest_tools_for_challenge_web(self, tool_manager):
        """Test suggesting tools for web challenge"""
        description = "SQL injection vulnerability in login form"
        tools = tool_manager.suggest_tools_for_challenge(description, "web")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_suggest_tools_for_challenge_crypto(self, tool_manager):
        """Test suggesting tools for crypto challenge"""
        description = "Break MD5 hash with dictionary attack"
        tools = tool_manager.suggest_tools_for_challenge(description, "crypto")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_suggest_tools_for_challenge_pwn(self, tool_manager):
        """Test suggesting tools for pwn challenge"""
        description = "Exploit buffer overflow in binary"
        tools = tool_manager.suggest_tools_for_challenge(description, "pwn")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_suggest_tools_for_challenge_forensics(self, tool_manager):
        """Test suggesting tools for forensics challenge"""
        description = "Extract hidden data from image using steganography"
        tools = tool_manager.suggest_tools_for_challenge(description, "forensics")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_suggest_tools_for_challenge_rev(self, tool_manager):
        """Test suggesting tools for reverse engineering challenge"""
        description = "Reverse engineer encrypted binary to find flag"
        tools = tool_manager.suggest_tools_for_challenge(description, "rev")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_suggest_tools_for_challenge_misc(self, tool_manager):
        """Test suggesting tools for misc challenge"""
        description = "Decode base64 encoded string"
        tools = tool_manager.suggest_tools_for_challenge(description, "misc")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_suggest_tools_for_challenge_osint(self, tool_manager):
        """Test suggesting tools for OSINT challenge"""
        description = "Find all subdomains for target domain"
        tools = tool_manager.suggest_tools_for_challenge(description, "osint")
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_suggest_tools_for_challenge_empty_description(self, tool_manager):
        """Test suggesting tools with empty description"""
        tools = tool_manager.suggest_tools_for_challenge("", "web")
        assert isinstance(tools, list)

    def test_suggest_tools_for_challenge_long_description(self, tool_manager):
        """Test suggesting tools with very long description"""
        description = "This is a very long description " * 50
        tools = tool_manager.suggest_tools_for_challenge(description, "web")
        assert isinstance(tools, list)

    def test_suggest_tools_returns_unique_tools(self, tool_manager):
        """Test that suggested tools are unique"""
        description = "Complex web challenge with multiple vectors"
        tools = tool_manager.suggest_tools_for_challenge(description, "web")
        assert len(tools) == len(set(tools))  # All tools should be unique

    def test_get_tool_command_with_empty_target(self, tool_manager):
        """Test get_tool_command with empty target"""
        command = tool_manager.get_tool_command("nikto", "")
        assert command is not None

    def test_get_tool_command_with_special_chars_target(self, tool_manager):
        """Test get_tool_command with special characters in target"""
        command = tool_manager.get_tool_command("sqlmap", "http://example.com:8080/test?id=1&name=value")
        assert command is not None

    def test_get_tool_command_preserves_base_command(self, tool_manager):
        """Test that base command is preserved"""
        tool = "sqlmap"
        original_command = tool_manager.tool_commands[tool]
        command = tool_manager.get_tool_command(tool, "example.com")
        assert tool in command or original_command in command

    def test_multiple_category_lookups(self, tool_manager):
        """Test multiple category lookups return consistent results"""
        categories = ["web_recon", "crypto_hash", "pwn_exploit", "forensics_image", "rev_static", "misc_encoding", "osint_domain"]
        results = {}
        for category in categories:
            tools = tool_manager.get_category_tools(category)
            results[category] = tools
            assert isinstance(tools, list)
            assert len(tools) > 0

        # Verify results are consistent
        for category in categories:
            tools1 = tool_manager.get_category_tools(category)
            tools2 = tool_manager.get_category_tools(category)
            assert tools1 == tools2

    def test_tool_commands_have_valid_strings(self, tool_manager):
        """Test all tool commands are valid strings"""
        for tool_name, command in tool_manager.tool_commands.items():
            assert isinstance(tool_name, str)
            assert isinstance(command, str)
            assert len(tool_name) > 0
            assert len(command) > 0
            assert not command.isspace()

    # --- get_tool_command branch coverage ---

    def test_get_tool_command_hashcat_adds_wordlist(self, tool_manager):
        cmd = tool_manager.get_tool_command("hashcat", "hash.txt")
        assert "wordlist" in cmd

    def test_get_tool_command_hashcat_adds_rules(self, tool_manager):
        cmd = tool_manager.get_tool_command("hashcat", "hash.txt")
        assert "rules-file" in cmd

    def test_get_tool_command_sqlmap_adds_tamper(self, tool_manager):
        cmd = tool_manager.get_tool_command("sqlmap", "http://target.com")
        assert "tamper" in cmd

    def test_get_tool_command_sqlmap_adds_threads(self, tool_manager):
        cmd = tool_manager.get_tool_command("sqlmap", "http://target.com")
        assert "--threads" in cmd

    def test_get_tool_command_gobuster_adds_threads(self, tool_manager):
        cmd = tool_manager.get_tool_command("gobuster", "http://target.com")
        assert "-t 50" in cmd

    def test_get_tool_command_dirsearch_adds_threads(self, tool_manager):
        cmd = tool_manager.get_tool_command("dirsearch", "http://target.com")
        assert "-t 50" in cmd

    def test_get_tool_command_feroxbuster_adds_threads(self, tool_manager):
        cmd = tool_manager.get_tool_command("feroxbuster", "http://target.com")
        assert "-t 50" in cmd

    def test_get_tool_command_no_additional_args(self, tool_manager):
        cmd = tool_manager.get_tool_command("nikto", "http://target.com")
        assert cmd.endswith("http://target.com")

    def test_get_tool_command_with_additional_args_branch(self, tool_manager):
        cmd = tool_manager.get_tool_command("nikto", "http://target.com", "-ssl")
        assert "-ssl" in cmd

    def test_get_tool_command_unknown_tool(self, tool_manager):
        cmd = tool_manager.get_tool_command("unknown_tool", "target")
        assert cmd == "unknown_tool target"

    def test_get_tool_command_sqlmap_no_threads_in_base(self):
        tm = CTFToolManager()
        tm.tool_commands["sqlmap"] = "sqlmap --batch --level 3 --risk 2"
        cmd = tm.get_tool_command("sqlmap", "http://target.com")
        assert "--threads" in cmd and "5" in cmd

    def test_get_tool_command_sqlmap_no_tamper_in_base(self):
        tm = CTFToolManager()
        tm.tool_commands["sqlmap"] = "sqlmap --batch --level 3 --risk 2"
        cmd = tm.get_tool_command("sqlmap", "http://target.com")
        assert "tamper" in cmd

    def test_get_tool_command_sqlmap_tamper_already_in_base(self):
        tm = CTFToolManager()
        tm.tool_commands["sqlmap"] = "sqlmap --batch --tamper=something"
        cmd = tm.get_tool_command("sqlmap", "http://target.com")
        assert "--tamper" in cmd

    def test_get_tool_command_dirsearch_no_t_flag(self):
        tm = CTFToolManager()
        tm.tool_commands["dirsearch"] = "dirsearch -u {} -e php"
        cmd = tm.get_tool_command("dirsearch", "http://target.com")
        assert "-t 50" in cmd

    # --- suggest_tools_for_challenge web keyword branches ---

    def test_suggest_web_sql(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("sql injection in database", "web")
        assert "sqlmap" in tools and "hashid" in tools

    def test_suggest_web_xss(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("xss via javascript dom", "web")
        assert "dalfox" in tools and "katana" in tools

    def test_suggest_web_wordpress(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("wordpress cms plugin", "web")
        assert "wpscan" in tools

    def test_suggest_web_directory(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("hidden admin files", "web")
        assert "gobuster" in tools and "dirsearch" in tools

    def test_suggest_web_parameter(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("parameter injection via get post", "web")
        assert "arjun" in tools and "paramspider" in tools

    def test_suggest_web_jwt(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("jwt token session", "web")
        assert "jwt-tool" in tools

    def test_suggest_web_graphql(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("graphql api endpoint", "web")
        assert "graphql-voyager" in tools

    # --- suggest_tools_for_challenge crypto keyword branches ---

    def test_suggest_crypto_hash(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("md5 sha password hash", "crypto")
        assert "hashcat" in tools and "hashid" in tools

    def test_suggest_crypto_rsa(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("rsa public key factorization", "crypto")
        assert "rsatool" in tools and "factordb" in tools and "yafu" in tools

    def test_suggest_crypto_cipher(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("cipher encryption decrypt substitution", "crypto")
        assert "cipher-identifier" in tools and "frequency-analysis" in tools

    def test_suggest_crypto_vigenere(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("vigenere polyalphabetic cipher", "crypto")
        assert "vigenere-solver" in tools

    def test_suggest_crypto_base64(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("base64 base32 encoding", "crypto")
        assert "base64" in tools and "base32" in tools

    def test_suggest_crypto_rot(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("rot caesar shift cipher", "crypto")
        assert "rot13" in tools

    def test_suggest_crypto_gpg(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("pgp gpg signature", "crypto")
        assert "gpg" in tools

    # --- suggest_tools_for_challenge pwn keyword branches ---

    def test_suggest_pwn_buffer(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("buffer overflow bof", "pwn")
        assert "pwntools" in tools and "ropper" in tools

    def test_suggest_pwn_format(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("printf format string", "pwn")
        assert "pwntools" in tools and "gdb-peda" in tools

    def test_suggest_pwn_heap(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("heap malloc free exploitation", "pwn")
        assert "pwntools" in tools and "gdb-gef" in tools

    def test_suggest_pwn_rop(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("rop gadget chain", "pwn")
        assert "ropper" in tools and "ropgadget" in tools

    def test_suggest_pwn_shellcode(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("shellcode exploit development", "pwn")
        assert "pwntools" in tools and "one-gadget" in tools

    def test_suggest_pwn_canary(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("canary stack protection bypass", "pwn")
        assert "checksec" in tools and "pwntools" in tools

    # --- suggest_tools_for_challenge forensics keyword branches ---

    def test_suggest_forensics_image(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("hidden data in png jpg gif steganography", "forensics")
        assert "steghide" in tools and "stegsolve" in tools and "zsteg" in tools

    def test_suggest_forensics_memory(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("memory dump ram analysis", "forensics")
        assert "volatility" in tools and "volatility3" in tools

    def test_suggest_forensics_network(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("network pcap wireshark traffic", "forensics")
        assert "wireshark" in tools and "tcpdump" in tools

    def test_suggest_forensics_file_recovery(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("deleted file recovery carving", "forensics")
        assert "binwalk" in tools and "foremost" in tools and "photorec" in tools

    def test_suggest_forensics_disk(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("disk filesystem partition", "forensics")
        assert "testdisk" in tools and "sleuthkit" in tools

    def test_suggest_forensics_audio(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("audio wav mp3 sound analysis", "forensics")
        assert "audacity" in tools and "sonic-visualizer" in tools

    # --- suggest_tools_for_challenge rev keyword branches ---

    def test_suggest_rev_packed(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("packed upx packer", "rev")
        assert "upx" in tools and "peid" in tools and "detect-it-easy" in tools

    def test_suggest_rev_android(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("android apk mobile", "rev")
        assert "apktool" in tools and "jadx" in tools and "dex2jar" in tools

    def test_suggest_rev_dotnet(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge(".net dotnet csharp", "rev")
        assert "dnspy" in tools and "ilspy" in tools

    def test_suggest_rev_java(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("java jar class decompile", "rev")
        assert "jd-gui" in tools and "jadx" in tools

    def test_suggest_rev_windows(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("windows exe dll binary", "rev")
        assert "ghidra" in tools and "x64dbg" in tools

    def test_suggest_rev_linux(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("linux elf binary analysis", "rev")
        assert "ghidra" in tools and "radare2" in tools and "gdb-peda" in tools

    # --- suggest_tools_for_challenge osint keyword branches ---

    def test_suggest_osint_username(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("username social media", "osint")
        assert "sherlock" in tools and "social-analyzer" in tools

    def test_suggest_osint_domain(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("domain subdomain dns", "osint")
        assert "sublist3r" in tools and "amass" in tools and "dig" in tools

    def test_suggest_osint_email(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("email harvest contact", "osint")
        assert "theHarvester" in tools

    def test_suggest_osint_ip(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("ip port service", "osint")
        assert "shodan" in tools and "censys" in tools

    def test_suggest_osint_whois(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("whois registration owner", "osint")
        assert "whois" in tools

    # --- suggest_tools_for_challenge misc keyword branches ---

    def test_suggest_misc_qr(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("qr barcode code", "misc")
        assert "qr-decoder" in tools

    def test_suggest_misc_zip(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("zip archive compressed", "misc")
        assert "zip" in tools and "7zip" in tools and "rar" in tools

    def test_suggest_misc_brainfuck(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("brainfuck bf esoteric", "misc")
        assert "brainfuck" in tools

    def test_suggest_misc_whitespace(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("whitespace ws", "misc")
        assert "whitespace" in tools

    def test_suggest_misc_piet(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("piet image program", "misc")
        assert "piet" in tools

    def test_suggest_misc_no_keywords(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("something completely unrelated", "misc")
        assert isinstance(tools, list)
        assert len(tools) == 0

    def test_suggest_unknown_category_reaches_misc_elif_false(self, tool_manager):
        tools = tool_manager.suggest_tools_for_challenge("anything", "nonexistent_category")
        assert isinstance(tools, list)
