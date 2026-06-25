#!/usr/bin/env bash
# @file run_seed_leg.sh
# @brief Drive one complete seed's experiment leg: train every ablation arm
#        through the full curriculum (resume-chained), then evaluate the FINAL
#        stage only and generate the evaluation-suite tables.
#
# The seed is read once from agent_config.yaml (the single source of truth); set
# it there before running. Every run leaf is <stage>_<seed>_<timestamp>, so the
# per-seed output trees (seed_<N>/) are routed automatically.
#
# Why a script and not one make line: each curriculum stage resumes from the
# previous stage's checkpoint, whose leaf carries a runtime timestamp. The leaf
# cannot be known ahead of time, so the chain is discovered on disk after each
# stage completes.
#
# DRY_RUN=1 prints every command without executing - use it to preview the plan.
#
# @warning Long-running (multi-day). Run in a detachable session (tmux/screen).
#          Stops at the first failing stage so a broken chain never silently
#          trains the wrong weights.

set -euo pipefail

# DRY_RUN=1 -> echo commands instead of running them.
DRY_RUN="${DRY_RUN:-0}"

# @brief Run a command, or just print it when DRY_RUN=1.
run() {
    if [ "${DRY_RUN}" = "1" ]; then
        echo "    [dry-run] $*"
    else
        "$@"
    fi
}

# ---------------------------------------------------------------------------
# Configuration - mirror the seed-42 leg so seeds are comparable.
# ---------------------------------------------------------------------------
# All four ablation arms, in dependency-free order.
ARMS=(vanilla_ppo input_uncertainty output_uncertainty full_method)
# Evidential arms get BOTH SafetyWrapper variants at eval (the with/without A/B);
# standard arms have no uncertainty head, so only the free-running variant.
EDL_ARMS=(output_uncertainty full_method)
# The full curriculum. Stage 1 trains from scratch; 2..6 resume the prior stage.
STAGES=(1 2 3 4 5 6)
# Only the FINAL stage is evaluated - it is the deployed policy and the headline
# deliverable. Mid-stage evals add nothing to a per-seed endpoint comparison.
EVAL_STAGE=6
# Per-step uncertainty trace cap (no-op on standard heads; full trace on EDL).
PER_STEP_CAP=440

LAYOUT=rectangle

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# @brief Read the master seed from agent_config.yaml (the single source).
read_seed() {
    grep -E '^seed:' configs/deployment/agent_config.yaml | head -1 \
        | sed -E 's/^seed:[[:space:]]*([0-9]+).*/\1/'
}

# @brief Newest checkpoint leaf for an arm at a given stage (the run just made).
# @param $1 arm, $2 stage, $3 seed -> echoes the leaf name (<stage>_<seed>_<ts>).
# The glob pins BOTH stage and seed, so a coexisting prior seed's leaves (same
# directory) and other stages are excluded; -t + head pick the youngest, i.e.
# the one just trained. Under DRY_RUN a placeholder is synthesised when nothing
# is on disk yet, so the preview still shows the resume-chain structure.
latest_leaf() {
    local arm="$1" stage="$2" seed="$3" found
    found="$(ls -1dt "checkpoints/${arm}/${stage}_${seed}_"* 2>/dev/null \
        | head -1 | xargs -r basename)"
    if [ -z "${found}" ] && [ "${DRY_RUN}" = "1" ]; then
        found="${stage}_${seed}_DDMMYYYY-HHMM"  # placeholder for the preview
    fi
    echo "${found}"
}

SEED="$(read_seed)"
if [ -z "${SEED}" ]; then
    echo "ERROR: could not read seed from configs/deployment/agent_config.yaml" >&2
    exit 1
fi

echo "=================================================================="
echo " Seed leg: SEED=${SEED}  arms=${ARMS[*]}  stages=${STAGES[*]}"
echo " Eval: stage ${EVAL_STAGE} only, PER_STEP_CAP=${PER_STEP_CAP}"
[ "${DRY_RUN}" = "1" ] && echo " DRY_RUN=1: printing commands only, nothing executes."
echo "=================================================================="

# ---------------------------------------------------------------------------
# 1. Train every arm through the full curriculum, resume-chained.
# ---------------------------------------------------------------------------
for arm in "${ARMS[@]}"; do
    prev_leaf=""
    for stage in "${STAGES[@]}"; do
        echo ">>> [1/2] TRAIN arm=${arm} stage=${stage} seed=${SEED}"
        if [ "${stage}" = "1" ]; then
            # Stage 1 trains from scratch (no resume).
            run make docker-train LAYOUT="${LAYOUT}" STAGE="${stage}" BASELINE="${arm}"
        else
            # Resume from the leaf the previous stage wrote.
            if [ -z "${prev_leaf}" ]; then
                echo "ERROR: no previous-stage leaf for ${arm} stage ${stage}" >&2
                exit 1
            fi
            run make docker-train LAYOUT="${LAYOUT}" STAGE="${stage}" \
                BASELINE="${arm}" CHECKPOINT="${prev_leaf}"
        fi
        # Discover the leaf this stage just produced for the next resume.
        prev_leaf="$(latest_leaf "${arm}" "${stage}" "${SEED}")"
        if [ -z "${prev_leaf}" ]; then
            echo "ERROR: ${arm} stage ${stage} produced no checkpoint leaf" >&2
            exit 1
        fi
        echo "    -> resume leaf: ${prev_leaf}"
    done
done

# ---------------------------------------------------------------------------
# 2. Evaluate the FINAL stage of every arm (200 episodes/condition from
#    eval_config). Evidential arms run twice: without_wrapper (free-running, the
#    headline) and with_wrapper (the SafetyWrapper A/B). Standard arms run once.
# ---------------------------------------------------------------------------
declare -A FINAL_LEAF
for arm in "${ARMS[@]}"; do
    leaf="$(latest_leaf "${arm}" "${EVAL_STAGE}" "${SEED}")"
    FINAL_LEAF["${arm}"]="${leaf}"
    is_edl=0
    for e in "${EDL_ARMS[@]}"; do [ "${e}" = "${arm}" ] && is_edl=1; done

    echo ">>> [2/2] EVAL arm=${arm} leaf=${leaf} (without_wrapper)"
    run make docker-eval LAYOUT="${LAYOUT}" BASELINE="${arm}" CHECKPOINT="${leaf}" \
        PER_STEP_CAP="${PER_STEP_CAP}" NO_SAFETY=1

    if [ "${is_edl}" = "1" ]; then
        echo ">>> [2/2] EVAL arm=${arm} leaf=${leaf} (with_wrapper)"
        run make docker-eval LAYOUT="${LAYOUT}" BASELINE="${arm}" CHECKPOINT="${leaf}" \
            PER_STEP_CAP="${PER_STEP_CAP}"
    fi
done

# ---------------------------------------------------------------------------
# 3. Evaluation-suite tables (host-side, read the eval CSVs just written).
#    Cross-arm: ablation + gate at the eval stage. Per-EDL-arm: calibration +
#    handover. SEED routes the cross-arm targets to this seed's tree.
# ---------------------------------------------------------------------------
echo ">>> [suite] Cross-arm: ablation + gate (stage ${EVAL_STAGE})"
run make analyse-ablation STAGE="${EVAL_STAGE}" SEED="${SEED}" \
    SLOPE_CLEAN=gnss_fixed SLOPE_DEGRADED=gnss_degraded
run make analyse-gate STAGE="${EVAL_STAGE}" SEED="${SEED}"

for arm in "${EDL_ARMS[@]}"; do
    leaf="${FINAL_LEAF[${arm}]}"
    echo ">>> [suite] EDL: calibration + handover arm=${arm} leaf=${leaf}"
    run make analyse-calibration ARM="${arm}" CHECKPOINT="${leaf}"
    run make handover-timing ARM="${arm}" CHECKPOINT="${leaf}"
done

echo "=================================================================="
echo " Seed leg COMPLETE for SEED=${SEED}."
echo " Results under outputs/<tree>/seed_${SEED}/"
echo "=================================================================="
