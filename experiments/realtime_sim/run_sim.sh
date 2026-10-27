#!/usr/bin/env bash
# Start the Blocks/Cesium level in AirSim's ComputerVision mode with this folder's settings.
# Usage: run_sim.sh [seconds]   (background it; kill with pkill -f UnrealEditor)
cd "$(dirname "$0")"
P=/home/user/workspace/NavionSIM/sim/workspace/Cosys-AirSim/Unreal/Environments/Blocks/Blocks.uproject
exec timeout "${1:-3000}" /home/user/UnrealEngine/launch-editor.sh "$P" -game -RenderOffScreen -nosound -nosplash \
  -ResX=1280 -ResY=720 -settings="$PWD/settings_boat.json" -log -abslog="$PWD/out/sim.log"
