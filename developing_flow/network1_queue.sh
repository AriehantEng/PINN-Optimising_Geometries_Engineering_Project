#!/usr/bin/env bash
# Retrain Network 1 (P7: G5, then three 16,384-point polishes of it) to measure
# its outlet profile against the Poiseuille parabola; no trained Network 1 was
# kept. One job at a time (4 cores). Safe to re-run after a container restart:
# finished steps are skipped and an interrupted run resumes from its checkpoint.
#
#     nohup bash developing_flow/network1_queue.sh > docs/re_study/logs/network1.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
LOGS=docs/re_study/logs
RUNS=developing_flow/runs
REF=$RUNS/reference_re10_m8
mkdir -p "$LOGS"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

if [ ! -f "$REF/reference.npz" ]; then
    echo "$(stamp) start reference"
    python -m developing_flow.reference --config developing_flow/re10.yaml --nx 50 --ny 12 \
        --x-grading 5.0 --levels 4 --grid 2001 1001 --out "$REF" > "$LOGS/network1_reference.log" 2>&1 \
        && echo "$(stamp) done reference" || { echo "$(stamp) FAILED reference"; exit 1; }
fi

for name in re10_lm_fresh_inputmap re10_polish_g5_points re10_polish_g5_points_seed2 \
            re10_polish_g5_points_seed3; do
    run=$RUNS/developing_$name
    if [ -f "$run/params.npz" ]; then echo "$(stamp) skip $name (finished)"; continue; fi
    resume=""
    [ -f "$run/checkpoint.npz" ] && resume="--resume"
    echo "$(stamp) start $name $resume"
    python -m developing_flow.pinn --config developing_flow/$name.yaml $resume \
        >> "$LOGS/network1_$name.log" 2>&1 \
        && echo "$(stamp) done $name" || { echo "$(stamp) FAILED $name"; exit 1; }
done

P7="$RUNS/developing_re10_polish_g5_points $RUNS/developing_re10_polish_g5_points_seed2 $RUNS/developing_re10_polish_g5_points_seed3"
labels=""
for r in $P7; do labels="$labels P7=$r"; done
python -m developing_flow.summary_reference --out docs/re_study/network1_retrain --solutions "$REF" \
    --reference-setup G5 --title "Network 1, retrained" "G5=$RUNS/developing_re10_lm_fresh_inputmap" \
    $labels > "$LOGS/network1_summary.log" 2>&1 && echo "$(stamp) summary written" \
    || echo "$(stamp) summary FAILED"
python -m developing_flow.outlet_error --solution "$REF" --out docs/developing_re10-outlet $P7 \
    > "$LOGS/network1_outlet.log" 2>&1 && echo "$(stamp) outlet error written" \
    || echo "$(stamp) outlet error FAILED"
echo "$(stamp) queue finished"
