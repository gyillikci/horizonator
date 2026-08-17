# User-side downloads

Everything the study still wants that the development container cannot
reach. The container's egress allows GitHub (repo, raw, releases,
codeload), `s3.amazonaws.com`, `storage.googleapis.com` and PyPI, and
blocks academic hosts, Google Drive, Kaggle, HuggingFace, Zenodo and
Overpass — so the items below have to be fetched from a normal
connection and committed (or dropped into `~/.horizonator/`).

Ordered by what each one unlocks, not by size.

## 1. OSM landmark extracts — no download, one command

Highest value, zero license friction (ODbL, attribution only), and it
is the only item the repo can consume immediately. Run on any machine
with network access:

    cd experiments
    python3 fetch_osm_landmarks.py --region bodrum        # or --bbox S W N E
    git add data/*.json && git commit -m "OSM landmarks: Bodrum"

Fetches all four charted classes in one pass — navigation lights
(night channel, E5f/E5g), wind turbines (E5h), transmission pylons and
communication masts (E5i) — into `experiments/data/*.json`. Presets:
`bodrum`, `gokova`, `bafa`, `izmir`, `canakkale`, `istanbul-bosphorus`.
`--dry-run` prints the plan without touching the network.

Unlocks: the landmark web on real coastline instead of synthetic demo
farms — i.e. the first honest measurement of how many charted points
are actually visible from a boat in the operating area.

## 2. MaSTr1325 — real sea imagery with IMU horizons

- Landing page: <https://www.vicos.si/resources/mastr1325/>
- 1325 annotated maritime images (512×384) + per-image IMU data
- Companion evaluation set MODS/MODD2: <https://github.com/bborja/modd>
- License: research use, see the page (cite Bovcon et al. 2019)

Unlocks **E4q**, the outstanding validation: auto-levelling (E4g) has
only ever been checked against its own geometry, never against
independently measured horizons. MaSTr1325's IMU gives exactly that
ground truth, so E4q measures the auto-leveller's real error instead of
assuming the ±2 mrad window. The eWaSR weights this study already
pulled (Apache-2.0, GitHub releases) were trained on this dataset.

## 3. GeoPose3K — photos with known camera poses

- Landing page: <https://cphoto.fit.vutbr.cz/geoPose3K/>
- ~3000 mountain photos with full camera pose (position + orientation)
- Only photos and poses are needed; the rendered depth/normal maps are
  large and irrelevant here

Unlocks a CH1-style audit where the attitude prior is *measured*
rather than simulated. E4f/E4u corrupt a solved reference attitude with
synthetic noise; GeoPose3K supplies real poses, which turns the
instrumented-regime numbers from "plausible" into "validated".

## 4. CrossLocate — cross-modal retrieval weights

- Project site: <http://cphoto.fit.vutbr.cz/crosslocate/>
- Code: <https://github.com/JanTomesek/CrossLocate> (reachable; the
  weights and datasets live on the project site, via its download
  scripts)
- Paper: WACV 2022, Tomešek & Čadík

Unlocks a retrieval baseline to compare against E4t's brute-force
whole-coast search (which already fixes to 10 m over a 222 km coast in
~200 s without any prior). Nice-to-have, not load-bearing: E4t's result
suggests retrieval is unnecessary in the coastal regime.

## 5. Pylon / power-line fine-tune material

Only needed if E5j's conclusion is revisited — that single-frame
open-vocabulary detection (YOLOE, already working here) is a near-field
proposal generator and distant landmarks are identified temporally and
geometrically instead.

- **TTPLA** (aerial towers + lines, 1100 images, YOLACT weights in six
  configs): <https://github.com/R3ab/ttpla_dataset> — data and weights
  on Google Drive; dataset zip id `1Yz59yXCiPKS0_X4K3x9mW22NLnxjvrr0`;
  license unspecified, so treat as research-only until confirmed
- **PLD-UAV** (ground/UAV wire segmentation):
  <https://github.com/SnorkerHeng/PLD-UAV> — PLDU
  <https://drive.google.com/open?id=1XjoWvHm2I8Y4RV_i9gEd93ZP-KryjJlm>,
  PLDM
  <https://drive.google.com/open?id=1bKFEuXKHRsy0tnOnoEVW6oRi7hS5oekr>;
  no explicit license

## 6. OpenCellID — coverage cross-check only

- <https://www.opencellid.org/downloads.php> (free API token, CC BY-SA)

**Not** a source of bearing landmarks: positions are crowd-triangulated
and hundreds of meters off. Useful only to check whether OSM's
`man_made=mast` coverage has gaps in a region.

## Reachable from the container — no action needed

Recorded so they are not re-hunted: eWaSR weights (Apache-2.0, GitHub
releases, already pulled and assayed in E4r), YOLOE + MobileCLIP
TorchScript (GitHub release assets, pulled and assayed in E5j),
GroundingDINO `swint_ogc.pth` (694 MB, GitHub releases — reachable if
a heavier open-vocabulary detector is ever wanted), SRTM1/SRTM3 skadi
tiles (`s3.amazonaws.com`, fetched automatically by the pipeline).

## Licenses to keep flagged

- **FABDEM** (canopy-free DEM): CC BY-NC-SA — non-commercial only. The
  DIY substitute is ETH canopy height minus GLO-30.
- **Ahmad skyline_detection** (CH1's origin): non-commercial license —
  the dataset is used here for evaluation only; none of its code is
  vendored, and none should be.
- **YOLOE / Ultralytics**: AGPL-3.0 — fine for research, needs a
  commercial license if the instrument ships with it.
