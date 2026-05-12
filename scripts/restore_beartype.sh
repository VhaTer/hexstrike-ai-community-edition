#!/bin/bash
# Restore original beartype from backup
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BAK_FILE="$SCRIPT_DIR/hexstrike-env/lib/python3.13/site-packages/beartype/claw/_importlib/_clawimpload.py.bak"
TARGET_FILE="$SCRIPT_DIR/hexstrike-env/lib/python3.13/site-packages/beartype/claw/_importlib/_clawimpload.py"

if [ ! -f "$BAK_FILE" ]; then
    echo "Error: backup not found at $BAK_FILE. Cannot restore." >&2
    echo "Reinstall with: pip install beartype==0.22.9" >&2
    exit 1
fi

cp "$BAK_FILE" "$TARGET_FILE"
echo "Restored original beartype from backup $BAK_FILE"
