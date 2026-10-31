#!/usr/bin/env bash
# Install Video Intelligence skill for Claude Code, Codex, or Grok Bot / custom environments.
set -euo pipefail

target="${1:-}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$target" in
  claude)
    destination="$HOME/.claude/skills/video-intelligence"
    ;;
  codex)
    destination="$HOME/.codex/skills/video-intelligence"
    ;;
  grok|grokbot|workspace)
    destination="/workspace/skills/video-intelligence"
    ;;
  *)
    echo "Usage: bash install_skill.sh [claude|codex|grok]"
    echo "Defaulting to local installation..."
    destination="$PWD"
    ;;
esac

if [ "$destination" != "$PWD" ]; then
  mkdir -p "$destination"
  rsync -a --delete \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude 'output' \
    --exclude 'tmp' \
    --exclude 'media' \
    "$root/" "$destination/"
  echo "✓ Installed Video Intelligence to $destination"
fi

echo "Installing Python dependencies..."
python3 -m pip install -r "$root/requirements.txt" || python -m pip install -r "$root/requirements.txt"

echo "✓ Video Intelligence ready."
echo "Ensure GEMINI_API_KEY is exported before running."
