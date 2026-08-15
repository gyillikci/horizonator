#!/bin/sh
# Turnkey benchmark for the native skyline solver on target hardware
# (built for the Raspberry Pi CM5, runs anywhere with cc + python3-numpy).
#
#   sh bench_cm5.sh
#
# Fetches the Bodrum/Kos DEM tiles if missing, compiles fastmarch.c, then
# times: single skyline synthesis (1 and all threads), a full 1 km-box fix
# (106 evaluations), and the E3 100 km-box coarse stage (2336 evaluations).
# Compare against the study-doc reference numbers (4-core x86 Xeon 2.8GHz:
# 13.8 / 4.4 ms per skyline, ~0.5 s and ~10.6 s per fix).
set -e
cd "$(dirname "$0")"

python3 -c "import numpy" 2>/dev/null || { echo "need python3-numpy"; exit 1; }
command -v cc >/dev/null || { echo "need a C compiler (apt install gcc)"; exit 1; }

python3 fetch_dems.py
# on ARM, -march=native is rejected by some gcc versions; fastmarch falls
# back gracefully if skyline.py's compile line needs -mcpu=native instead
python3 - <<'EOF'
import time, os, json, subprocess, numpy as np
import skyline as S

az = np.arange(-180.0, 180.0, 0.1) + 0.05
res = {'cpu': open('/proc/cpuinfo').read().count('processor'),
       'platform': os.uname().machine}

cm = S.CMarcher(os.path.expanduser('~/.horizonator/DEMs_SRTM3'),
                (36.4, 37.4), (26.6, 27.9))
cm.skyline(36.95, 27.25, 5.0, az)  # warm-up + compile check

def bench(n=50):
    t0 = time.time()
    for i in range(n):
        cm.skyline(36.95 + i * 1e-5, 27.25, 5.0, az)
    return (time.time() - t0) / n * 1e3

res['ms_per_skyline_allthreads'] = round(bench(), 2)
env = dict(os.environ, OMP_NUM_THREADS='1')
out = subprocess.run(['python3', '-c', '''
import time, numpy as np, sys; sys.path.insert(0, ".")
import skyline as S
import os
az = np.arange(-180.,180.,0.1)+0.05
cm = S.CMarcher(os.path.expanduser("~/.horizonator/DEMs_SRTM3"),(36.4,37.4),(26.6,27.9))
cm.skyline(36.95,27.25,5.,az)
t0=time.time()
for i in range(30): cm.skyline(36.95+i*1e-5,27.25,5.,az)
print((time.time()-t0)/30*1e3)'''], env=env, capture_output=True, text=True)
res['ms_per_skyline_1thread'] = round(float(out.stdout.strip().split()[-1]), 2)

# 1 km-box fix: 106 evaluations (coarse 9x9 + fine 5x5) + cost overhead
el_obs, _ = cm.skyline(36.9512, 27.2489, 5.0, az)
t0 = time.time()
n_eval = 0
mlat, mlon = S.meters_per_degree(36.95)
def sky(dn, de):
    global n_eval; n_eval += 1
    return cm.skyline(36.95 + dn / mlat, 27.25 + de / mlon, 5.0, az)[0]
S.solve_position(sky, el_obs, 5.0, box_m=1000.0, coarse_n=9, fine_step_m=25.0)
res['s_per_1km_fix'] = round(time.time() - t0, 2)
res['evals_1km'] = n_eval

# 100 km-box coarse stage: 2336 sea candidates at 2 km pitch (E3 workload)
t0 = time.time()
for i in range(200):   # 200-candidate sample, extrapolated
    cm.skyline(36.95 + (i % 20) * 0.018, 27.25 + (i // 20) * 0.02, 5.0, az)
res['s_per_100km_L0_extrap'] = round((time.time() - t0) / 200 * 2336, 1)

print(json.dumps(res, indent=1))
json.dump(res, open('out/bench_results.json', 'w'), indent=1)
EOF
echo "done -> out/bench_results.json"
