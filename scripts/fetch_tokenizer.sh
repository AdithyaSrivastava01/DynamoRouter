#!/usr/bin/env bash
set -euo pipefail
DEST="${1:-data}"
mkdir -p "$DEST"
if [ ! -f "$DEST/tokenizer.json" ]; then
  curl -fL -o "$DEST/tokenizer.json" \
    "https://huggingface.co/gpt2/resolve/main/tokenizer.json"
fi
echo "tokenizer at $DEST/tokenizer.json"
