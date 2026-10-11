#!/usr/bin/env bash
# Multispectral Greenhouse dataset (LGDWT-GS, Texas A&M, CC-BY-4.0): https://doi.org/10.5281/zenodo.18838172
# One plant per scene on a turntable, black background, MSIS-AGRI-1-A snapshot camera:
#   580 nm green, 660 nm red, 735 nm red-edge, 820 nm NIR. COLMAP poses included.
# Scenes: cotton grape houseplant sorghum tomato. Only the requested scene is pulled (HTTP range requests).
# usage: scripts/download_greenhouse.sh [scene]   (default: houseplant)
set -euo pipefail
SCENE="${1:-houseplant}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
URL="https://zenodo.org/api/records/18838173/files/Multispectral_Greenhouse.zip/content"
DST="$ROOT/data/greenhouse"
mkdir -p "$DST"
python3 -I "$ROOT/scripts/remote_zip.py" get "$URL" "^Multispectral_Greenhouse/${SCENE}/" "$DST/_stage"
rm -rf "$DST/$SCENE" && mv "$DST/_stage/Multispectral_Greenhouse/$SCENE" "$DST/$SCENE" && rm -rf "$DST/_stage"
find "$DST/$SCENE" -maxdepth 2 | sort | head -40
