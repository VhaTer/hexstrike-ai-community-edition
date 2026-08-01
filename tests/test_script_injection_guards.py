"""
tests/test_script_injection_guards.py

Script-injection guards for handlers that write user values into
command files (msfconsole .rc, pacu commands, bettercap caplet, gdb/r2
command files) before executing the tool.

Newline (\n) is the command separator in those files — a value with
\\n injects extra commands (e.g. "set RHOST x\nuse exploit/y\nrun").
Unlike shell injection (S86), this is not about shell metacharacters:
the file is interpreted by the tool itself.

Multi-line fields that are code by design (gdb/r2 commands, angr/pwntools
script_content) are intentionally NOT rejected — the caller already has
code execution by contract.
"""

from unittest.mock import mock_open, patch

import pytest

from pulse.tools.exploit_framework_direct import _metasploit, _pwntools
from pulse.tools.security_direct import _pacu
from pulse.tools.wifi_direct import _bettercap_wifi


# ---------------------------------------------------------------------------
# _metasploit — resource .rc file
# ---------------------------------------------------------------------------

class TestMetasploitResourceInjection:
    def test_module_with_newline_rejected(self):
        with patch("pulse.tools.exploit_framework_direct.execute_command") as m:
            result = _metasploit({"module": "exploit/a\nuse exploit/b\nrun"})
        assert result["success"] is False
        m.assert_not_called()

    def test_option_value_with_newline_rejected(self):
        with patch("pulse.tools.exploit_framework_direct.execute_command") as m:
            result = _metasploit({
                "module": "exploit/multi/handler",
                "options": {"PAYLOAD": "linux/x64/shell\nuse exploit/c\nrun"},
            })
        assert result["success"] is False
        m.assert_not_called()

    def test_option_key_with_newline_rejected(self):
        with patch("pulse.tools.exploit_framework_direct.execute_command") as m:
            result = _metasploit({
                "module": "exploit/multi/handler",
                "options": {"PAYLOAD\nset RHOST evil": "linux/x64"},
            })
        assert result["success"] is False
        m.assert_not_called()

    def test_option_value_with_shell_metachars_rejected(self):
        with patch("pulse.tools.exploit_framework_direct.execute_command") as m:
            result = _metasploit({
                "module": "exploit/multi/handler",
                "options": {"RHOSTS": "1.2.3.4;rm -rf /"},
            })
        assert result["success"] is False
        m.assert_not_called()

    def test_clean_options_written_to_resource_file(self):
        written = {}

        def fake_write(path, content):
            written["content"] = content

        with patch("pulse.tools.exploit_framework_direct.open", mock_open()) as mo:
            with patch("pulse.tools.exploit_framework_direct.execute_command",
                       return_value={"success": True}) as m:
                result = _metasploit({
                    "module": "exploit/multi/handler",
                    "options": {"PAYLOAD": "linux/x64/shell_reverse_tcp",
                                "LHOST": "10.0.0.1"},
                })
        assert result["success"] is True
        m.assert_called_once()
        handle = mo()
        assert "use exploit/multi/handler\n" in "".join(
            call.args[0] for call in handle.write.call_args_list
        )


# ---------------------------------------------------------------------------
# _pwntools — generated Python template (repr literals)
# ---------------------------------------------------------------------------

class TestPwntoolsTemplate:
    def test_quote_in_target_binary_escaped(self):
        payload = "/tmp/evil' + os.system('id') + '"
        with patch("builtins.open", mock_open()) as mo:
            with patch("pulse.tools.exploit_framework_direct.execute_command",
                       return_value={"success": True}) as m:
                result = _pwntools({"target_binary": payload})
        assert result["success"] is True
        m.assert_called_once()
        written = "".join(call.args[0] for call in mo().write.call_args_list)
        assert repr(payload) in written
        compile(written, "<template>", "exec")

    def test_clean_template_generated(self):
        with patch("builtins.open", mock_open()) as mo:
            with patch("pulse.tools.exploit_framework_direct.execute_command",
                       return_value={"success": True}) as m:
                result = _pwntools({"target_binary": "/bin/sh",
                                    "target_host": "10.0.0.1",
                                    "target_port": 4444})
        assert result["success"] is True
        written = "".join(call.args[0] for call in mo().write.call_args_list)
        assert "binary = '/bin/sh' if '/bin/sh' else None" in written
        assert "host = '10.0.0.1'" in written
        assert "port = 4444" in written


# ---------------------------------------------------------------------------
# Regression: already guarded by reject_shell_metachars (includes \n)
# ---------------------------------------------------------------------------

class TestAlreadyGuardedHandlers:
    def test_pacu_session_name_with_newline_rejected(self):
        with patch("pulse.tools.security_direct.execute_command") as m:
            result = _pacu({"session_name": "sess\nrun iam__privesc_scan"})
        assert result["success"] is False
        m.assert_not_called()

    def test_pacu_modules_with_newline_rejected(self):
        with patch("pulse.tools.security_direct.execute_command") as m:
            result = _pacu({"modules": "iam__privesc_scan\nrun ec2__enum"})
        assert result["success"] is False
        m.assert_not_called()

    def test_bettercap_bssid_with_newline_rejected(self):
        with patch("pulse.tools.wifi_direct.execute_command") as m:
            result = _bettercap_wifi({
                "interface": "wlan0",
                "mode": "deauth",
                "target_bssid": "aa:bb:cc\nnet.sniff on",
            })
        assert result["success"] is False
        m.assert_not_called()

    def test_bettercap_interface_with_newline_rejected(self):
        with patch("pulse.tools.wifi_direct.execute_command") as m:
            result = _bettercap_wifi({"interface": "wlan0;rm -rf /\n"})
        assert result["success"] is False
        m.assert_not_called()


# ---------------------------------------------------------------------------
# By design: multi-line code fields stay accepted (contract of the tool)
# ---------------------------------------------------------------------------

class TestMultiLineByDesign:
    def test_gdb_commands_multiline_accepted(self):
        from pulse.tools.misc_direct import _gdb
        with patch("builtins.open", mock_open()):
            with patch("pulse.tools.misc_direct.execute_command",
                       return_value={"success": True}) as m:
                result = _gdb({"binary": "/bin/ls", "commands": "break main\nrun\nbt"})
        assert result["success"] is True
        assert m.call_args[0][0].endswith("-batch")

    def test_radare2_commands_multiline_accepted(self):
        from pulse.tools.misc_direct import _radare2
        with patch("builtins.open", mock_open()):
            with patch("pulse.tools.misc_direct.execute_command",
                       return_value={"success": True}) as m:
                result = _radare2({"binary": "/bin/ls", "commands": "aaa\npd 10"})
        assert result["success"] is True

    def test_angr_script_content_accepted(self):
        from pulse.tools.misc_direct import _angr
        with patch("builtins.open", mock_open()):
            with patch("pulse.tools.misc_direct.execute_command",
                       return_value={"success": True}) as m:
                result = _angr({"binary": "/bin/ls", "script_content": "import angr\np = angr.Project('/bin/ls')\n"})
        assert result["success"] is True
        assert m.call_args[0][0].endswith("/tmp/angr_script.py")
