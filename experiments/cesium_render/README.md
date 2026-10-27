# Synthetic frames from Google Photorealistic 3D Tiles

Renders camera poses over Google Photorealistic 3D Tiles (Cesium ion asset
2275207, the same tiles NavionSIM's Unreal scenes stream) with CesiumJS in
headless Chrome, without Unreal or AirSim. Every pose gives two images:

| file | content |
|---|---|
| `<tag>_photo.png` | the render, with sky atmosphere; no fog (CesiumJS coarsens and drops distant tiles with it) |
| `<tag>_seg.png` | sky magenta, sea blue, land green: the exact skyline and sea line |
| `<tag>.json` | the pose, whether every tile loaded, timing |

Conventions the page handles:

- **Heights.** The tiles are WGS84-ellipsoidal. Poses carry `h_ellipsoid` =
  height above mean sea level + the EGM96 geoid separation `geoid_n` (30-40 m
  over the Aegean). The open sea has no tile geometry, so it is a globe on an
  ellipsoid raised by `geoid_n`: the render's sea sits at mean sea level
  (the tiles' own coastal water samples at -0.1 m).
- **Attitude.** Heading, pitch and roll in skyfix's convention; the page
  flips roll for Cesium. Field of view is horizontal.
- **No refraction.** CesiumJS draws straight rays over a curved Earth, so a
  model compared with a render takes `k = 0`.
- **The tiles are a surface model.** Trees and buildings are in them: a camera
  2-4 m above a wooded or built-up shore is inside them. Captures are kept on
  the water, 2-3 m above sea level (`sea_poses.py`).
- **Camera collision is off.** With it on, CesiumJS lifted a camera 2.5 m above
  the sea to 185 m (above a coarse tile's surface), which lowered every near
  ridge in the render while leaving far ones nearly in place. Each `<tag>.json`
  records `camera_used` as Cesium read it back; `compare.py` skips a frame
  whose camera moved by more than 0.5 m.
- **Drawing buffer.** Short of GPU memory, WebGL silently allocates a
  smaller drawing buffer than the canvas asks for: four workers gave five
  1600x3479 frames 800x1739 buffers, and so half the resolution and
  coarser tiles. The driver checks the buffer before and after each pose
  and retries a pose that got a smaller one; frames over 4 Mpx render at
  most two at a time. `<tag>.json` records `buffer`, and `compare.py` skips a frame
  whose images are not the pose's size.
- **Level of detail.** Dynamic and foveated screen-space error are off, so
  distant ridges are not coarsened, and `maximumScreenSpaceError` is 16
  (CesiumJS's default). Against SSE 4 on five sea-level frames the exact
  skyline moved 0.3 px on average (p95 1 px, max 3 px) and SegFormer's
  skyline on the photo 0.4-0.6 px, for a tenth of the tiles: SSE 4 took
  2-2.5 GB of tiles and 150-250 s per view, and the heaviest views
  (KWHC9160, PF_20260824_122139) never settled in the tile cache. Each
  `<tag>.json` records `lod`: the error asked for, the error used (higher
  when a view needs more than the 1.25 GB cache), tiles and tile memory.

## Running

```bash
export CESIUM_ION_TOKEN=...            # never written to a file by these scripts
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json   # NVIDIA, not the iGPU
python3 make_poses.py out/poses_truth44.json            # the 44 campaign frames at their truth
node driver.mjs out/poses_truth44.json out/w --gl egl --water 300   # nearest water per pose
python3 sea_poses.py out/poses_truth44.json out/w out/poses_sea.json --z 2.5
node driver.mjs out/poses_sea.json out/r --gl egl                    # render, 3 workers (resumes)
python3 compare.py out/poses_sea.json out/r --sheets                 # render vs photo vs COP30
```

`--workers N` (default 3) renders N poses at once, each in its own browser.
The time goes to the page's JavaScript, which traverses and draws the
tileset every frame (downloads ran at 2-4 MB/s), so workers scale with
cores until the GPU's memory runs out: a heavy view holds 1.1 GB of tiles
(the cache keeps 0.75 GB between poses), and a 1600x3479 frame's buffers
come on top. On the 8 GB RTX PRO 2000 the 29 sea-level poses took 283 s
with three workers (3-76 s each); with four, 251 s once and 485 s the next
time, when four portrait frames met at 7.7 GB of GPU memory and took
2-4 min each. A pose alone takes 10-15 s. Browser profiles are temporary
and removed at exit, so every run downloads its tiles again. `--gl
swiftshader` renders on the CPU (minutes per frame at 1600 px).

Google's terms require attribution (recorded in every `<tag>.json`) and limit
caching and derived datasets: check them before generating data in bulk.
