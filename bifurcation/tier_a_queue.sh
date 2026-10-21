#!/usr/bin/env bash
# Tier A of the Reynolds-number study (docs/re_study/tierA/preregistration.md):
# answer keys for the cases that passed screening, then the PINN runs, one job
# at a time (4 cores). Safe to re-run after a container restart: finished steps
# are skipped and an interrupted PINN run resumes from its checkpoint. A run that
# failed screening carries SKIPPED (bifurcation/tier_a.py configs) and is not
# trained; a step that fails is recorded beside its run and not rerun.
#
#     nohup bash bifurcation/tier_a_queue.sh > docs/re_study/logs/tierA_queue.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
OUT=docs/re_study/tierA
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

for key in 30-90_q0.2 30-90_q0.8 15-15_q0.2 30-90_q0.35 30-90_q0.65 0-120_q0.5; do
    if [ -f bifurcation/answer_keys_tierA_$key.yaml ]; then
        fe_step answer_keys_$key $OUT/answer_keys_$key \
            python -m bifurcation.answer_key --config bifurcation/answer_keys_tierA_$key.yaml \
                --out $OUT/answer_keys_$key
    fi
done
# the main gap first (q = 0.2 and 0.8), then the null test, then the rest; Re_h 10 before 0.1
for case in 3090_q0.2 3090_q0.8 1515_q0.2 3090_q0.35 3090_q0.65 0120_q0.5; do
    pinn_step re10_${case}_corners_s21
    pinn_step re0.1_${case}_corners_s21
done
echo "$(stamp) queue finished"
