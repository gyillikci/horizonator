#!/usr/bin/env bash
# Sail every planned route (routes/*.json) headless, recording a pose-tagged frame every
# EVERY seconds into OUT (out/aegean), three routes at a time; LOOK=1 turns the camera as
# look_planner.py planned (into out/aegean_look by default).
cd "$(dirname "$0")"
EVERY=${EVERY:-15}; SPEED=${SPEED:-12}; LOOK=${LOOK:+--look}
OUT=${OUT:-out/aegean${LOOK:+_look}}
export CESIUM_ION_TOKEN=$(grep '^CESIUM_ION_TOKEN=' /home/user/workspace/NavionSIM/sim/env.local | cut -d= -f2-)
mkdir -p $OUT
ls ${ROUTES:-routes/*.json} | xargs -P 3 -I{} sh -c 'n=$(basename {} .json); node cruise.mjs --route {} --speed '$SPEED' --wobble 1 '$LOOK' --record '$OUT' --every '$EVERY' > '$OUT'/$n.log 2>&1; tail -n 1 '$OUT'/$n.log'
