#!/usr/bin/env bash
# NeRF-synthetic "ficus" (a potted plant, Blender render, known poses, RGBA on transparent bg).
# Mirror: nerfbaselines on HuggingFace. Output: data/nerf_synthetic/ficus/{train,val,test,transforms_*.json}
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="$ROOT/data/nerf_synthetic"
URL="https://huggingface.co/datasets/nerfbaselines/nerfbaselines-data/resolve/main/blender/ficus.zip"
mkdir -p "$DST/_dl"
[ -f "$DST/_dl/ficus.zip" ] || curl -L --fail -o "$DST/_dl/ficus.zip" "$URL"
# Extract into an empty staging dir first (untrusted archive), then move the expected folder.
STAGE="$(mktemp -d "$DST/_dl/stage.XXXX")"
python3 -I -c 'import sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); bad=[n for n in z.namelist() if n.startswith("/") or ".." in n.split("/")]; assert not bad, bad; z.extractall(sys.argv[2])' "$DST/_dl/ficus.zip" "$STAGE"
SRC="$(dirname "$(find "$STAGE" -name transforms_train.json | head -1)")"
rm -rf "$DST/ficus" && mv "$SRC" "$DST/ficus" && rm -rf "$STAGE"
ls "$DST/ficus"
