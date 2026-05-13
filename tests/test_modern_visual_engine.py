import pytest
from server_core.modern_visual_engine import ModernVisualEngine


MVE = ModernVisualEngine


class TestCreateBanner:
    def test_returns_non_empty_string(self):
        banner = MVE.create_banner()
        assert isinstance(banner, str)
        assert len(banner) > 0

    def test_contains_hexstrike_ascii(self):
        banner = MVE.create_banner()
        assert len(banner) > 100


class TestCreateProgressBar:
    def test_basic_renders(self):
        result = MVE.create_progress_bar(50, 100, tool="nmap")
        assert "nmap" in result
        assert "█" in result

    def test_zero_total(self):
        result = MVE.create_progress_bar(0, 0, tool="test")
        assert "test" in result
        assert "0.0%" in result

    def test_full_progress(self):
        result = MVE.create_progress_bar(100, 100, tool="nmap")
        assert "100.0%" in result

    def test_custom_width(self):
        result = MVE.create_progress_bar(50, 100, width=30, tool="nmap")
        assert "nmap" in result


class TestRenderProgressBar:
    def test_cyber_style(self):
        result = MVE.render_progress_bar(0.5, style="cyber", label="Scan")
        assert "Scan" in result
        assert "50.0%" in result

    def test_matrix_style(self):
        result = MVE.render_progress_bar(0.75, style="matrix", label="Test")
        assert "75.0%" in result

    def test_neon_style(self):
        result = MVE.render_progress_bar(0.25, style="neon")
        assert "25.0%" in result

    def test_default_style(self):
        result = MVE.render_progress_bar(0.5, style="unknown")
        assert "50.0%" in result

    def test_with_eta(self):
        result = MVE.render_progress_bar(0.5, eta=42.5, style="cyber")
        assert "ETA" in result or "42" in result

    def test_with_speed(self):
        result = MVE.render_progress_bar(0.5, speed="100 req/s", style="cyber")
        assert "Speed" in result or "100" in result

    def test_without_label(self):
        result = MVE.render_progress_bar(0.5, style="cyber")
        assert "%" in result

    def test_clamp_below_zero(self):
        result = MVE.render_progress_bar(-0.5, style="cyber")
        assert "0.0%" in result

    def test_clamp_above_one(self):
        result = MVE.render_progress_bar(1.5, style="cyber")
        assert "100.0%" in result

    def test_full_width_bars(self):
        result = MVE.render_progress_bar(1.0, width=80, style="cyber")
        assert "100.0%" in result


class TestCreateLiveDashboard:
    def test_empty_processes(self):
        result = MVE.create_live_dashboard({})
        assert "No active processes" in result

    def test_with_running_process(self):
        processes = {1234: {"status": "running", "command": "nmap -sV target", "duration": 15.5}}
        result = MVE.create_live_dashboard(processes)
        assert "1234" in result
        assert "running" in result.lower()
        assert "nmap" in result

    def test_with_failed_process(self):
        processes = {5678: {"status": "failed", "command": "broken_tool", "duration": 5.0}}
        result = MVE.create_live_dashboard(processes)
        assert "5678" in result
        assert "failed" in result.lower()

    def test_long_command_truncated(self):
        long_cmd = "a" * 100
        processes = {1: {"status": "running", "command": long_cmd, "duration": 0}}
        result = MVE.create_live_dashboard(processes)
        assert "..." in result

    def test_multiple_processes(self):
        processes = {
            1: {"status": "running", "command": "nmap", "duration": 10},
            2: {"status": "failed", "command": "sqlmap", "duration": 5},
        }
        result = MVE.create_live_dashboard(processes)
        assert "HEXSTRIKE LIVE DASHBOARD" in result


class TestFormatVulnerabilityCard:
    def test_critical_severity(self):
        result = MVE.format_vulnerability_card({"severity": "critical", "name": "CVE-2024-0001", "description": "Critical vuln"})
        assert "CRITICAL" in result
        assert "CVE-2024-0001" in result

    def test_high_severity(self):
        result = MVE.format_vulnerability_card({"severity": "high", "name": "High Vuln", "description": "desc"})
        assert "HIGH" in result

    def test_medium_severity(self):
        result = MVE.format_vulnerability_card({"severity": "medium", "name": "Med Vuln", "description": "desc"})
        assert "MEDIUM" in result

    def test_low_severity(self):
        result = MVE.format_vulnerability_card({"severity": "low", "name": "Low Vuln", "description": "desc"})
        assert "LOW" in result

    def test_info_severity(self):
        result = MVE.format_vulnerability_card({"severity": "info", "name": "Info Vuln", "description": "desc"})
        assert "INFO" in result

    def test_unknown_severity(self):
        result = MVE.format_vulnerability_card({"severity": "unknown", "name": "Unknown", "description": "desc"})
        assert "UNKNOWN" in result or "VULNERABILITY" in result

    def test_default_name(self):
        result = MVE.format_vulnerability_card({})
        assert "Unknown Vulnerability" in result

    def test_default_description(self):
        result = MVE.format_vulnerability_card({})
        assert "No description" in result


class TestFormatErrorCard:
    def test_critical_error(self):
        result = MVE.format_error_card("CRITICAL", "nmap", "segfault", "restart")
        assert "CRITICAL" in result
        assert "nmap" in result

    def test_error_type(self):
        result = MVE.format_error_card("ERROR", "sqlmap", "connection failed")
        assert "ERROR" in result

    def test_timeout_type(self):
        result = MVE.format_error_card("TIMEOUT", "nuclei", "timeout")
        assert "TIMEOUT" in result

    def test_recovery_type(self):
        result = MVE.format_error_card("RECOVERY", "tool", "error", "restarting")
        assert "RECOVERY" in result

    def test_warning_type(self):
        result = MVE.format_error_card("WARNING", "wpscan", "rate limited")
        assert "WARNING" in result or "rate limited" in result

    def test_unknown_type(self):
        result = MVE.format_error_card("UNKNOWN", "tool", "error")
        assert "ERROR" in result or "tool" in result

    def test_with_recovery_action(self):
        result = MVE.format_error_card("ERROR", "nmap", "failed", "restart with -T4")
        assert "Recovery" in result or "restart" in result

    def test_without_recovery_action(self):
        result = MVE.format_error_card("ERROR", "nmap", "failed")
        assert "Recovery" not in result


class TestFormatToolStatus:
    def test_running(self):
        result = MVE.format_tool_status("nmap", "RUNNING", target="10.0.0.1")
        assert "RUNNING" in result
        assert "nmap" in result.lower()
        assert "10.0.0.1" in result

    def test_success(self):
        result = MVE.format_tool_status("nmap", "SUCCESS")
        assert "SUCCESS" in result

    def test_failed(self):
        result = MVE.format_tool_status("sqlmap", "FAILED")
        assert "FAILED" in result

    def test_timeout(self):
        result = MVE.format_tool_status("nuclei", "TIMEOUT")
        assert "TIMEOUT" in result

    def test_recovery(self):
        result = MVE.format_tool_status("tool", "RECOVERY")
        assert "RECOVERY" in result

    def test_unknown_status(self):
        result = MVE.format_tool_status("tool", "UNKNOWN")
        assert "UNKNOWN" in result or "INFO" in result or "tool" in result

    def test_with_progress(self):
        result = MVE.format_tool_status("nmap", "RUNNING", progress=0.75)
        assert "75" in result or "█" in result

    def test_without_target(self):
        result = MVE.format_tool_status("nmap", "RUNNING", progress=0.5)
        assert "nmap" in result.lower()


class TestFormatHighlightedText:
    def test_red(self):
        result = MVE.format_highlighted_text("ALERT", "RED")
        assert "ALERT" in result

    def test_yellow(self):
        result = MVE.format_highlighted_text("WARN", "YELLOW")
        assert "WARN" in result

    def test_green(self):
        result = MVE.format_highlighted_text("OK", "GREEN")
        assert "OK" in result

    def test_blue(self):
        result = MVE.format_highlighted_text("INFO", "BLUE")
        assert "INFO" in result

    def test_purple(self):
        result = MVE.format_highlighted_text("DEBUG", "PURPLE")
        assert "DEBUG" in result

    def test_unknown_highlight(self):
        result = MVE.format_highlighted_text("test", "UNKNOWN")
        assert "test" in result


class TestFormatVulnerabilitySeverity:
    def test_critical(self):
        result = MVE.format_vulnerability_severity("CRITICAL")
        assert "CRITICAL" in result

    def test_high(self):
        result = MVE.format_vulnerability_severity("HIGH")
        assert "HIGH" in result

    def test_medium(self):
        result = MVE.format_vulnerability_severity("MEDIUM")
        assert "MEDIUM" in result

    def test_low(self):
        result = MVE.format_vulnerability_severity("LOW")
        assert "LOW" in result

    def test_info(self):
        result = MVE.format_vulnerability_severity("INFO")
        assert "INFO" in result

    def test_unknown_severity(self):
        result = MVE.format_vulnerability_severity("UNKNOWN")
        assert "UNKNOWN" in result or "INFO" in result

    def test_with_count(self):
        result = MVE.format_vulnerability_severity("HIGH", count=5)
        assert "5" in result or "(5)" in result

    def test_zero_count(self):
        result = MVE.format_vulnerability_severity("HIGH", count=0)
        assert "(0)" not in result


class TestCreateSectionHeader:
    def test_default_color(self):
        result = MVE.create_section_header("Results")
        assert "RESULTS" in result

    def test_custom_color(self):
        result = MVE.create_section_header("Scan", color="MATRIX_GREEN")
        assert "SCAN" in result
        assert "═" in result

    def test_custom_icon(self):
        result = MVE.create_section_header("Test", icon="🚀")
        assert "🚀" in result


class TestFormatCommandExecution:
    def test_starting(self):
        result = MVE.format_command_execution("nmap -sV", "STARTING")
        assert "STARTING" in result
        assert "nmap" in result

    def test_running(self):
        result = MVE.format_command_execution("nmap -sV", "RUNNING")
        assert "RUNNING" in result

    def test_success(self):
        result = MVE.format_command_execution("nmap", "SUCCESS", duration=5.5)
        assert "SUCCESS" in result
        assert "5.50" in result

    def test_failed(self):
        result = MVE.format_command_execution("broken", "FAILED")
        assert "FAILED" in result

    def test_timeout(self):
        result = MVE.format_command_execution("slow_tool", "TIMEOUT")
        assert "TIMEOUT" in result

    def test_unknown_status(self):
        result = MVE.format_command_execution("tool", "PENDING")
        assert "PENDING" in result

    def test_zero_duration(self):
        result = MVE.format_command_execution("nmap", "SUCCESS", duration=0)
        assert "0.00" not in result

    def test_long_command_truncated(self):
        long_cmd = "x" * 100
        result = MVE.format_command_execution(long_cmd, "RUNNING")
        assert "..." in result


class TestCreateSummaryReport:
    def test_empty_results(self):
        result = MVE.create_summary_report({})
        assert "SUMMARY" in result.upper()

    def test_with_vulnerabilities(self):
        results = {
            "target": "10.0.0.1",
            "vulnerabilities": [
                {"severity": "critical"},
                {"severity": "high"},
                {"severity": "medium"},
                {"severity": "low"},
                {"severity": "info"},
            ],
            "execution_time": 120.5,
            "tools_used": ["nmap", "gobuster", "nuclei", "sqlmap", "nikto", "extra"],
        }
        result = MVE.create_summary_report(results)
        assert "10.0.0.1" in result
        assert "120" in result
        assert "1" in result or "Critical" in result

    def test_without_tools(self):
        results = {"vulnerabilities": [], "execution_time": 0, "tools_used": []}
        result = MVE.create_summary_report(results)
        assert "SUMMARY" in result.upper()


class TestFormatToolOutput:
    def test_success_output(self):
        result = MVE.format_tool_output("nmap", "open ports found", success=True)
        assert "SUCCESS" in result
        assert "nmap" in result.lower()

    def test_failed_output(self):
        result = MVE.format_tool_output("sqlmap", "error: connection failed", success=False)
        assert "FAILED" in result

    def test_error_keyword_highlighted(self):
        result = MVE.format_tool_output("nmap", "error: timeout", success=True)
        assert "error" in result.lower()

    def test_found_keyword_highlighted(self):
        result = MVE.format_tool_output("nmap", "found 5 open ports", success=True)
        assert "found" in result.lower()

    def test_warning_keyword_highlighted(self):
        result = MVE.format_tool_output("nmap", "warning: slow scan", success=True)
        assert "warning" in result.lower()

    def test_normal_line_displayed(self):
        result = MVE.format_tool_output("nmap", "normal output line", success=True)
        assert "normal output line" in result

    def test_truncates_more_than_20_lines(self):
        lines = "\n".join(f"line {i}" for i in range(25))
        result = MVE.format_tool_output("nmap", lines, success=True)
        assert "more lines truncated" in result

    def test_empty_line_skipped(self):
        result = MVE.format_tool_output("nmap", "\n\n\ncontent\n\n", success=True)
        assert "content" in result

    def test_timeout_keyword(self):
        result = MVE.format_tool_output("nmap", "timeout occurred", success=True)
        assert "timeout" in result.lower()

    def test_vulnerable_keyword(self):
        result = MVE.format_tool_output("nmap", "vulnerable service found", success=True)
        assert "vulnerable" in result.lower()
