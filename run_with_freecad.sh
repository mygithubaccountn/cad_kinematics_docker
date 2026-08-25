#!/usr/bin/env bash
# Run pipeline.py with a Python that can `import FreeCAD` (required for real
# STEP). Does NOT inherit a polluted shell PYTHONPATH (that breaks system
# Python later). Tries, in order: macOS FreeCAD.app (unchanged default
# behavior), a python3 that already has FreeCAD importable (Docker image /
# conda env already activated), conda-forge env "fc" (see Dockerfile), then
# the Linux apt package location.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECT_PYTHONPATH="$ROOT/src:$ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: ./run_with_freecad.sh run <STEP> --out out/step" >&2
  echo "Example: ./run_with_freecad.sh run \"$HOME/Desktop/robot_assembly.stp\" --out out/step" >&2
  echo "Iterate:  ./run_with_freecad.sh run \"$HOME/Desktop/robot_assembly.stp\" --out out/step --from-stage joints" >&2
  echo "Meshes:   cached by topology (joint edits reuse GLB); --remesh | --final-meshes | --no-meshes" >&2
  exit 1
fi

# Reject obvious placeholder paths
for arg in "$@"; do
  if [[ "$arg" == *"/path/to/"* ]]; then
    echo "Error: replace /path/to/... with a real STEP file path on your machine." >&2
    echo "Example: $HOME/Desktop/robot_assembly.stp" >&2
    exit 1
  fi
done

cd "$ROOT"

# 1) macOS FreeCAD.app (default/original behavior, unchanged).
FC_APP="${FREECAD_APP:-/Applications/FreeCAD.app}"
FC_RES="$FC_APP/Contents/Resources"
if [[ -x "$FC_RES/bin/python" ]]; then
  export PYTHONPATH="$PROJECT_PYTHONPATH:$FC_RES/lib:$FC_RES/lib/python3.11/site-packages"
  exec "$FC_RES/bin/python" pipeline.py "$@"
fi

# 2) Already usable as-is (Docker image, conda env active, Linux system
#    install with FreeCAD already on PYTHONPATH)?
if python3 -c "import FreeCAD" >/dev/null 2>&1; then
  export PYTHONPATH="$PROJECT_PYTHONPATH:${PYTHONPATH:-}"
  exec python3 pipeline.py "$@"
fi

# 3) conda-forge FreeCAD env named "fc" (see Dockerfile), not yet activated.
if [[ -x "/opt/conda/envs/fc/bin/python" ]]; then
  export PYTHONPATH="$PROJECT_PYTHONPATH:/opt/conda/envs/fc/lib"
  exec /opt/conda/envs/fc/bin/python pipeline.py "$@"
fi

# 4) Ubuntu apt package location (older FreeCAD, best-effort).
if [[ -d "/usr/lib/freecad-python3/lib" ]]; then
  export PYTHONPATH="$PROJECT_PYTHONPATH:/usr/lib/freecad-python3/lib"
  exec python3 pipeline.py "$@"
fi

echo "FreeCAD Python module not found." >&2
echo "macOS: set FREECAD_APP to your FreeCAD.app path." >&2
echo "Linux: install FreeCAD via conda-forge: conda create -n fc -c conda-forge freecad" >&2
exit 1
