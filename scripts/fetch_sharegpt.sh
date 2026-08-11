#!/usr/bin/env bash
set -euo pipefail
DEST="${1:-data}"
mkdir -p "$DEST"
if [ ! -f "$DEST/sharegpt.json" ]; then
  echo "downloading ShareGPT (~640MB, one-time)..."
  curl -fL -o "$DEST/sharegpt.json" \
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
fi
echo "trace at $DEST/sharegpt.json"
