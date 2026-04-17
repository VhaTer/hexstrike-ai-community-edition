# Reference

## Example Calls

```python
checksec(file="/tmp/challenge")
strings(file="/tmp/challenge", additional_args="-n 8")
binwalk(file="/tmp/firmware.bin", additional_args="-e")
radare2(binary="/tmp/challenge", commands="aaa; afl; pdf @main")
ropgadget(binary="/tmp/challenge")
gdb(binary="/tmp/challenge", commands="run")
sqlite(db_path="/tmp/app.db", query="select * from users limit 5;")
volatility3(memory_file="/tmp/memdump.raw", plugin="windows.pslist.PsList")
```

## Tool Guide

| Tool | Common parameters | Use |
|---|---|---|
| `checksec` | `file` | mitigation triage |
| `strings` | `file`, `additional_args` | plaintext and indicator extraction |
| `binwalk` | `file`, `additional_args` | firmware and embedded data extraction |
| `radare2` | `file`, `commands` | static analysis |
| `ropgadget`, `ropper`, `one_gadget` | `file` | gadget hunting |
| `gdb` | `file`, `commands` | dynamic analysis and crash triage |
