#!/usr/bin/env bash
# Tier A amendment 3 (docs/re_study/tierA/preregistration.md): one job at a
# time (a run peaks near 9.8 GB of 15); resumable like bifurcation/tier_a_queue.sh.
#
#     nohup bash bifurcation/tier_a_amend3_queue.sh > docs/re_study/logs/tierA_amend3.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
OUT=docs/re_study/tierA_amend3
LOGS=$OUT/logs
mkdir -p "$LOGS"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

fe_step() {        # name, output stem, command...
    local name=$1 stem=$2; shift 2
    if [ -f "$stem.json" ]; then echo "$(stamp) skip $name (done)"; return; fi
    if [ -f "$stem.FAILED" ]; then echo "$(stamp) skip $name (failed before)"; return; fi
    echo "$(stamp) start $name"
    if "$@" > "$LOGS/$name.log" 2>&1; then echo "$(stamp) done $name"
    else tail -40 "$LOGS/$name.log" > "$stem.FAILED"; echo "$(stamp) FAILED $name"; fi
}

pinn_step() {      # config tag
    local tag=$1 config=bifurcation/p_$1.yaml run=bifurcation/runs/parametric_$1
    if [ -f "$run/metrics.json" ] || [ -f "$run/FAILED" ] || [ -f "$run/SKIPPED" ]; then
        echo "$(stamp) skip $tag (finished, failed or skipped before)"
    else
        local resume=""
        [ -f "$run/checkpoint.npz" ] && resume="--resume"
        echo "$(stamp) start $tag $resume"
        if python -m bifurcation.parametric --config "$config" $resume >> "$LOGS/$tag.log" 2>&1; then
            echo "$(stamp) done $tag"
        else
            mkdir -p "$run"
            { echo "exit $? at $(stamp)"; tail -40 "$LOGS/$tag.log"; } > "$run/FAILED"
            echo "$(stamp) FAILED $tag"
        fi
    fi
    if [ -d "$run" ] && ! grep -qs "\"run\": \"$run\"" $OUT/pinn_*_summary.json; then  # grep -s: no summary yet means not exported
        python -m bifurcation.re_export --run "$run" --out $OUT --index $OUT >> "$LOGS/$tag.export.log" 2>&1 \
            && echo "$(stamp) exported $tag" || echo "$(stamp) export FAILED $tag"
    fi
}

for tag in re10_3090_q0.35_width_corners_s21 re10_3090_q0.2_width_corners_s21 \
           re0.1_3090_q0.5_cut0.15_lm400_corners_s22; do
    pinn_step $tag
done
echo "$(stamp) queue finished"
