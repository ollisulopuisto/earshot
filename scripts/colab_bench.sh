#!/bin/bash
# The bench on a Colab GPU, one engine per process so the card holds one
# model at a time (a T4 ran out of memory holding two, 2026-10-03).
# Run by scripts/colab.py on the VM, from the repository root:
#   uv run python scripts/colab.py --extras universr,unipase,novasr,lavasr,speaker,perceptual \
#       --run "bash scripts/colab_bench.sh" --result out/bench
set -u
DAMAGE="--damage clean --damage room --damage wideband-voip --damage platform-upload \
--damage narrowband-voip --damage voip-call --damage landline --damage overload --damage overload-call"
# Overridable, so the same script runs on a CPU without UniverSR (about 12
# minutes per call on an M1 Max) and on another set of voices.
INPUT=${INPUT:-material/local/ears}
OUT=${OUT:-out/bench}
ENGINES=${ENGINES:-"passthrough unipase router:unipase keepzero:unipase lavasr router:lavasr universr novasr chain:declip+unipase"}
for engine in $ENGINES; do
  name=$(echo "$engine" | tr ':+' '__')
  echo "== $engine"
  uv run earshot bench --engine "$engine" --input "$INPUT" --excerpts 6 --seconds 8 \
    $DAMAGE --out "$OUT/$name" || echo "!! $engine failed"
done
