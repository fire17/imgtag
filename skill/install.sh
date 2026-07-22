#!/usr/bin/env bash
# Install the imgtag skill globally (~/.claude/skills/imgtag) + its /imgtag-search alias.
# Idempotent: re-running converges to the same state. Default = symlink (live source of
# truth = this repo). --copy = detached copy, for machines without the repo.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SRC")"
SKILLS="${CLAUDE_SKILLS_DIR:-$HOME/.claude/skills}"
DEST="$SKILLS/imgtag"
ALIAS="$SKILLS/imgtag-search"
MODE="symlink"
[[ "${1:-}" == "--copy" ]] && MODE="copy"

[[ -f "$SRC/SKILL.md" ]] || { echo "install: no SKILL.md in $SRC" >&2; exit 1; }
mkdir -p "$SKILLS"

# --- primary skill dir ---------------------------------------------------------
if [[ "$MODE" == "symlink" ]]; then
  if [[ -e "$DEST" && ! -L "$DEST" ]]; then
    echo "install: $DEST exists and is a real directory — remove it or use --copy" >&2
    exit 1
  fi
  ln -sfn "$SRC" "$DEST"
else
  rm -rf "$DEST"
  mkdir -p "$DEST"
  cp "$SRC"/*.md "$DEST"/
fi

# --- alias twin: real dir + symlinked SKILL.md (house pattern; dir-symlinks de-dupe) ---
if [[ -e "$ALIAS" && ! -d "$ALIAS" ]]; then rm -f "$ALIAS"; fi
mkdir -p "$ALIAS"
ln -sfn "../imgtag/SKILL.md" "$ALIAS/SKILL.md"

# --- verify --------------------------------------------------------------------
echo "installed: $DEST ($MODE)"
ls -ld "$DEST" "$ALIAS" "$ALIAS/SKILL.md"

echo
if command -v imgtag >/dev/null 2>&1; then
  echo "cli: $(command -v imgtag)"
  CLI=(imgtag)
else
  echo "cli: not on PATH — skill documents the 'uv run --project $REPO imgtag' form"
  CLI=(uv run --project "$REPO" imgtag)
fi

if "${CLI[@]}" --help >/dev/null 2>&1; then
  echo "verify: '${CLI[*]} --help' OK"
else
  echo "verify: FAILED — '${CLI[*]} --help' does not run yet (engine incomplete?)" >&2
  exit 2
fi

if "${CLI[@]}" info --json >/tmp/imgtag-install-check.json 2>/dev/null \
   && python3 -c "import json,sys; json.load(open('/tmp/imgtag-install-check.json'))" 2>/dev/null; then
  echo "verify: 'info --json' returns valid JSON"
else
  echo "verify: 'info --json' not answering valid JSON yet (skill installed, engine pending)" >&2
fi
