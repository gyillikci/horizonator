#!/usr/bin/env bash
# Reproduce the skyline-matching environment on a normal machine.
#
#   ./setup_local.sh [--dems] [--assets] [--all]
#
# Nothing here is destructive: every step is skipped when its output is
# already present. Total footprint with --all is roughly 4 GB, almost
# all of it DEM tiles and pretrained weights that do not belong in git.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS="${HORIZONATOR_ASSETS:-$HOME/.horizonator/assets}"
DEMS="$HOME/.horizonator"
DO_ASSETS=0; DO_DEMS=0
for a in "$@"; do
  case "$a" in
    --assets) DO_ASSETS=1;;
    --dems)   DO_DEMS=1;;
    --all)    DO_ASSETS=1; DO_DEMS=1;;
    *) echo "unknown flag: $a" >&2; exit 2;;
  esac
done
[ $# -eq 0 ] && { DO_ASSETS=1; DO_DEMS=1; }

say(){ printf '\n=== %s\n' "$*"; }

say "python packages"
# Versions this campaign actually ran on. numpy 2.x and OpenCV 5 are both
# recent enough that older pins fail in ways that are not obvious.
python3 -m pip install --upgrade \
  'numpy>=2.2' 'scipy>=1.15' 'pillow>=11' 'matplotlib>=3.10' \
  'opencv-python>=4.11' 'rasterio>=1.4' \
  'torch>=2.6' 'torchvision>=0.21' 'timm>=1.0.15' 'pytorch_lightning>=2.4'

say "native ray-marcher"
# fastmarch.so is rebuilt automatically by skyline.CMarcher whenever the
# source is newer, so this is only a check that a compiler exists.
command -v cc >/dev/null || { echo "no C compiler; apt install build-essential" >&2; exit 1; }
cc -O3 -march=native -fopenmp -shared -fPIC "$HERE/fastmarch.c" -o "$HERE/fastmarch.so" 2>/dev/null \
  || cc -O3 -fopenmp -shared -fPIC "$HERE/fastmarch.c" -o "$HERE/fastmarch.so"
echo "built $HERE/fastmarch.so"

if [ "$DO_ASSETS" = 1 ]; then
  mkdir -p "$ASSETS"

  say "eWaSR (Apache 2.0, MaSTr1325 weights) -> $ASSETS"
  if [ ! -d "$ASSETS/eWaSR" ]; then
    git clone --depth 1 https://github.com/tersekmatija/eWaSR "$ASSETS/eWaSR"
    # two patches the clone needs against current dependencies, both
    # documented at the top of ewasr_bridge.py
    sed -i 's/from timm\.models\.layers import to_2tuple/from timm.layers import to_2tuple/' \
      "$ASSETS/eWaSR/wasr/metaformer.py" || true
    sed -i 's/pretrained=True/pretrained=False/g' "$ASSETS/eWaSR/wasr/models.py" || true
  fi
  [ -f "$ASSETS/ewasr_resnet18.pth" ] || curl -L -o "$ASSETS/ewasr_resnet18.pth" \
    https://github.com/tersekmatija/eWaSR/releases/download/0.1.0/ewasr_resnet18.pth

  say "MobileSAM (Apache 2.0) -> $ASSETS"
  [ -d "$ASSETS/MobileSAM" ] || \
    git clone --depth 1 https://github.com/ChaoningZhang/MobileSAM "$ASSETS/MobileSAM"
  # the checkpoint ships inside that repo at weights/mobile_sam.pt

  echo
  echo "export HORIZONATOR_ASSETS=$ASSETS   # add to your shell profile"
fi

if [ "$DO_DEMS" = 1 ]; then
  say "SRTM1 tiles (AWS elevation-tiles-prod mirror) -> $DEMS"
  python3 "$HERE/fetch_dems.py" "$DEMS"

  say "Copernicus GLO-30 DSM -> $DEMS/DEMs_COP30"
  # the Aegean + Marmara boxes this campaign used; widen as needed
  python3 "$HERE/copernicus_to_hgt.py" --bbox 36 27 38 29
  python3 "$HERE/copernicus_to_hgt.py" --bbox 40 28 42 30
fi

say "smoke test"
cd "$HERE"
python3 - <<'PY'
import os, numpy as np, skyline as S
d = os.path.expanduser('~/.horizonator/DEMs_SRTM1_WM')
if not os.path.isdir(d):
    raise SystemExit('no DEM store at %s — run with --dems' % d)
cm = S.CMarcher(d, (36.4, 37.6), (27.0, 28.6), d_min=300.)
el, r = cm.skyline(37.01992, 27.44426, 9.0, np.arange(160., 200., 1.0))
print('ray-marcher ok: %d azimuths, elevation %.2f..%.2f deg, range p50 %.1f km'
      % (el.size, np.degrees(el).min(), np.degrees(el).max(),
         np.nanpercentile(r, 50) / 1e3))
PY
echo
echo "done. Next: python3 skyfix.py --help"
