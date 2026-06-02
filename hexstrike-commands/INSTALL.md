# Install Pulse Slash Commands

Copy the command files to your AI client's commands directory.

---

## opencode

```bash
# Project-level (recommended)
cp *.md .opencode/commands/

# Or global
cp *.md ~/.config/opencode/commands/
```

Restart opencode. Type `/` in the TUI to see available commands.

---

## Claude Code

```bash
mkdir -p ~/.claude/commands/
cp scan.md dashboard.md recon.md ~/.claude/commands/
```

Restart Claude Code. Type `/` to browse available commands.

---

## Cline

Cline does not support custom slash commands. Use Pulse MCP tools directly via `call_tool()`.

---

## Continue

Add to your `~/.continue/config.json`:

```json
{
  "experimental": {
    "slashCommands": [
      {
        "name": "scan",
        "description": "Full reconnaissance scan on a target via Pulse",
        "prompt": "Call scan() via Pulse MCP for target {{input}}"
      },
      {
        "name": "dashboard",
        "description": "Open the Pulse live dashboard",
        "prompt": "Call pulse_dashboard() via Pulse MCP"
      },
      {
        "name": "recon",
        "description": "Consolidated surface + findings + plan via Pulse",
        "prompt": "Call get_surface(), get_findings(), and get_plan() via Pulse MCP for target {{input}}"
      }
    ]
  }
}
```

Restart Continue.

---

## Creating Custom Commands

1. Copy `TEMPLATE.md` to `<name>.md`.
2. Edit the frontmatter `description`.
3. Replace the prompt body with your tool chain.
4. Copy to your client's commands directory.
5. Restart your client.

See `TEMPLATE.md` for the full structure.
