#!/bin/bash
# Apply beartype 0.22.9 patch for coverage compatibility
# The circular import in _clawimpload.py::get_code() is triggered by
# coverage's import-tracking. This wraps the lazy imports in a try/except.
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BEARTYPE_FILE="$SCRIPT_DIR/hexstrike-env/lib/python3.13/site-packages/beartype/claw/_importlib/_clawimpload.py"

if [ ! -f "$BEARTYPE_FILE" ]; then
    echo "Error: $BEARTYPE_FILE not found." >&2
    echo "Run from project root with: bash scripts/patch_beartype_for_coverage.sh" >&2
    exit 1
fi

# Check if already patched
if grep -q "except ImportError" "$BEARTYPE_FILE"; then
    echo "Already patched."
    exit 0
fi

# Save backup
cp "$BEARTYPE_FILE" "${BEARTYPE_FILE}.bak"
echo "Backup saved to ${BEARTYPE_FILE}.bak"

# Apply patch
python3 -c "
content = open('$BEARTYPE_FILE').read()
old = '''        # Avoid circular import dependencies.
        from beartype.claw._clawstate import claw_state
        from beartype.claw._package.clawpkgtrie import get_package_conf_or_none'''

new = '''        # Avoid circular import dependencies.
        try:
            from beartype.claw._clawstate import claw_state
            from beartype.claw._package.clawpkgtrie import get_package_conf_or_none
        except ImportError:
            return super().get_code(fullname)'''

content = content.replace(old, new)
open('$BEARTYPE_FILE', 'w').write(content)
print('Patch applied.')
"
