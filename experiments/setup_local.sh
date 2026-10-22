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
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
  || { echo "need Python >= 3.10 (the campaign ran on 3.11)" >&2; exit 1; }
# Versions this campaign actually ran on. numpy 2.x and OpenCV 5 are both
# recent enough that older pins fail in ways that are not obvious.
python3 -m pip install --upgrade \
  'numpy>=2.2' 'scipy>=1.15' 'pillow>=11' 'matplotlib>=3.10' \
  'opencv-python>=4.11' 'rasterio>=1.4'
# torch is only for the segmenters. On Linux without an NVIDIA GPU the
# CPU wheel avoids ~2 GB of CUDA libraries; everything in the study ran
# on CPU. Set TORCH_INDEX to override (a CUDA index URL), or to the
# empty string to use plain PyPI.
if [ "$(uname -s)" = Linux ] && ! command -v nvidia-smi >/dev/null; then
  TORCH_INDEX="${TORCH_INDEX-https://download.pytorch.org/whl/cpu}"
fi
python3 -m pip install --upgrade ${TORCH_INDEX:+--index-url "$TORCH_INDEX"} \
  'torch>=2.6' 'torchvision>=0.21'
python3 -m pip install --upgrade 'timm>=1.0.15' 'pytorch_lightning>=2.4' \
  'ultralytics>=8.3'
# gtsam is only for the fusion experiments (e5_fusion.py, skynav.py);
# its wheels lag new Python releases, so a failure here is not fatal
python3 -m pip install 'gtsam>=4.2' \
  || echo "WARNING: gtsam did not install; only e5_fusion/skynav need it" >&2

say "native ray-marcher"
# fastmarch.so is rebuilt automatically by skyline.CMarcher whenever the
# source is newer, so this is only a check that a compiler exists.
command -v cc >/dev/null || { echo "no C compiler; apt install build-essential" >&2; exit 1; }
# Apple clang has no -fopenmp: the last fallback builds single-threaded
# (identical results, slower). brew install libomp + gcc restores it.
cc -O3 -march=native -fopenmp -shared -fPIC "$HERE/fastmarch.c" -o "$HERE/fastmarch.so" 2>/dev/null \
  || cc -O3 -mcpu=native -fopenmp -shared -fPIC "$HERE/fastmarch.c" -o "$HERE/fastmarch.so" 2>/dev/null \
  || cc -O3 -fopenmp -shared -fPIC "$HERE/fastmarch.c" -o "$HERE/fastmarch.so" 2>/dev/null \
  || { echo "no OpenMP: building the marcher single-threaded" >&2;
       cc -O3 -shared -fPIC "$HERE/fastmarch.c" -o "$HERE/fastmarch.so"; }
echo "built $HERE/fastmarch.so"

if [ "$DO_ASSETS" = 1 ]; then
  mkdir -p "$ASSETS"

  say "eWaSR (Apache 2.0, MaSTr1325 weights) -> $ASSETS"
  if [ ! -d "$ASSETS/eWaSR" ]; then
    git clone --depth 1 https://github.com/tersekmatija/eWaSR "$ASSETS/eWaSR"
    # two patches the clone needs against current dependencies, both
    # documented at the top of ewasr_bridge.py
    # the clone imports from timm.models.layers.helpers, which timm 1.x
    # removed; match any submodule so this survives both spellings
    sed -i -E 's/from timm\.models\.layers(\.[a-z_]+)? import to_2tuple/from timm.layers import to_2tuple/' \
      "$ASSETS/eWaSR/wasr/metaformer.py"
    grep -q "from timm.layers import to_2tuple" "$ASSETS/eWaSR/wasr/metaformer.py" \
      || { echo "eWaSR timm patch did not apply; inspect wasr/metaformer.py" >&2; exit 1; }
    sed -i 's/pretrained=True/pretrained=False/g' "$ASSETS/eWaSR/wasr/models.py" || true
  fi
  [ -f "$ASSETS/ewasr_resnet18.pth" ] || curl -L -o "$ASSETS/ewasr_resnet18.pth" \
    https://github.com/tersekmatija/eWaSR/releases/download/0.1.0/ewasr_resnet18.pth

  say "MobileSAM (Apache 2.0) -> $ASSETS"
  [ -d "$ASSETS/MobileSAM" ] || \
    git clone --depth 1 https://github.com/ChaoningZhang/MobileSAM "$ASSETS/MobileSAM"
  # the checkpoint ships inside that repo at weights/mobile_sam.pt

  say "SAM 2.1 tiny + base (Apache 2.0, Ultralytics release) -> $ASSETS/sam2"
  mkdir -p "$ASSETS/sam2"
  for w in sam2.1_t.pt sam2.1_b.pt; do
    [ -f "$ASSETS/sam2/$w" ] || curl -fL -o "$ASSETS/sam2/$w" \
      "https://github.com/ultralytics/assets/releases/download/v8.3.0/$w"
  done

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
# any store will do; COP30 is what --dems builds, SRTM1_WM is the
# campaign's historical store and is NOT produced by this script
cands = [os.path.expanduser('~/.horizonator/' + n)
         for n in ('DEMs_COP30', 'DEMs_SRTM1_WM', 'DEMs_SRTM1')]
d = next((c for c in cands if os.path.isdir(c) and os.listdir(c)), None)
if d is None:
    raise SystemExit('no DEM store under ~/.horizonator — run with --dems')
print('DEM store:', d)
cm = S.CMarcher(d, (36.4, 37.6), (27.0, 28.6), d_min=300.)
el, r = cm.skyline(37.01992, 27.44426, 9.0, np.arange(160., 200., 1.0))
print('ray-marcher ok: %d azimuths, elevation %.2f..%.2f deg, range p50 %.1f km'
      % (el.size, np.degrees(el).min(), np.degrees(el).max(),
         np.nanpercentile(r, 50) / 1e3))
PY
echo
echo "done. Next: python3 skyfix.py --help"
