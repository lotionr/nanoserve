#!/usr/bin/env bash
# Downloads Qwen2.5-0.5B-Instruct (Apache-2.0, ungated) from Hugging Face into models/.
# No account or token required. ~1 GB total.
set -euo pipefail

REPO="Qwen/Qwen2.5-0.5B-Instruct"
DEST="$(cd "$(dirname "$0")/.." && pwd)/models/qwen2.5-0.5b-instruct"
BASE="https://huggingface.co/${REPO}/resolve/main"

FILES=(
    config.json
    generation_config.json
    tokenizer_config.json
    tokenizer.json
    vocab.json
    merges.txt
    model.safetensors
)

mkdir -p "$DEST"
echo "Downloading ${REPO} -> ${DEST}"

for f in "${FILES[@]}"; do
    out="${DEST}/${f}"
    if [ -s "$out" ]; then
        echo "  [skip] ${f} (already present)"
        continue
    fi
    echo "  [get ] ${f}"
    curl -L --fail --retry 3 --progress-bar -o "${out}.part" "${BASE}/${f}"
    mv "${out}.part" "$out"
done

echo "Done. Files:"
ls -lh "$DEST"
