#!/usr/bin/env bash
# The Reynolds-number study, one job at a time (4 cores), in priority order.
# Safe to re-run after a container restart: finished steps are skipped and an
# interrupted PINN run resumes from its checkpoint. A step that fails is
# recorded (FAILED beside the run, its log in docs/re_study/logs) and not rerun.
#
#     nohup bash bifurcation/re_study_queue.sh > docs/re_study/logs/queue.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
OUT=docs/re_study
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
    if [ -d "$run" ] && ! ls $OUT/pinn_*"_summary.json" 2>/dev/null | xargs -r grep -l "\"run\": \"$run\"" >/dev/null; then
        python -m bifurcation.re_export --run "$run" --out $OUT --index $OUT >> "$LOGS/$tag.export.log" 2>&1 \
            && echo "$(stamp) exported $tag" || echo "$(stamp) export FAILED $tag"
    fi
}

junction() {
    python -m bifurcation.junction_re --config bifurcation/junction_re_all.yaml \
        --out $OUT/closed-junction-all > "$LOGS/junction.log" 2>&1 \
        && echo "$(stamp) junction report written" || echo "$(stamp) junction report FAILED"
}

fe_step answer_keys $OUT/answer_keys \
    python -m bifurcation.answer_key --config bifurcation/answer_keys_re.yaml --out $OUT/answer_keys
fe_step closed_re0.1 docs/bifurcation_closed-re0.1 \
    python -m bifurcation.design_study --config bifurcation/study_closed_re0.1.yaml --out docs/bifurcation_closed-re0.1
pinn_step re0.1_3090_corners_s21        # run 1
pinn_step re100_3090_corners_s21        # run 4
fe_step closed_re1 docs/bifurcation_closed-re1 \
    python -m bifurcation.design_study --config bifurcation/study_closed_re1.yaml --out docs/bifurcation_closed-re1
junction
pinn_step re10_30120_corners_s21        # run 7
pinn_step re0.1_30120_corners_s21       # run 5: SKIPPED (its in-range optimum is the range's bound)
pinn_step re100_30120_corners_s21       # run 8
pinn_step re1_3090_corners_s21          # run 2
for seed in 22 23; do
    pinn_step re0.1_3090_corners_s$seed
    pinn_step re100_3090_corners_s$seed
    pinn_step re10_3090_corners_s$seed
done
pinn_step re1_30120_corners_s21         # run 6, last: it has run 5's problem; skip it if asked
echo "$(stamp) queue finished"
