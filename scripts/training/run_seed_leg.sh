#!/usr/bin/env bash
# @file run_seed_leg.sh
# @brief Drive the multi-seed experiment: for each seed, train every ablation arm
#        through the full curriculum (resume-chained), then evaluate the FINAL
#        stage and generate the evaluation-suite tables.
#
# IDEMPOTENT: every step checks for its own completion marker on disk and skips
# work already done, so a crashed run is resumed simply by re-running this script
# - it picks up at the first unfinished stage/eval. A training stage is "done"
# only if its checkpoint leaf has a final_model.zip; a partial leaf (e.g. from a
# CARLA timeout) lacks one and is retrained. An eval is "done" if its
# evaluation_results.csv exists.
#
# The seed is the single source of truth in agent_config.yaml. This script
# OVERWRITES that one line per seed (and leaves it on the last seed of the list),
# so the in-container training/noise processes pick up the active seed. Run leaves
# are <stage>_<seed>_<timestamp>, routing the per-seed output trees (seed_<N>/)
# automatically.
#
# Does NOT build images - run `make docker-build-no-cache` first.
#
# DRY_RUN=1 prints every command (and skip decisions) without executing.
#
# @warning Long-running (days). Run in a detachable session (tmux/screen).

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Seeds to run, in order. Done work is skipped, so listing an already-finished
# seed is a cheap no-op. The file is left on the LAST seed when the leg ends.
SEEDS=(42 123 7)
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
AGENT_CONFIG=configs/deployment/agent_config.yaml
EVAL_ROOT=outputs/evaluation_results

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# @brief Run a command, or just print it when DRY_RUN=1.
run() {
    if [ "${DRY_RUN}" = "1" ]; then
        echo "    [dry-run] $*"
    else
        "$@"
    fi
}

# @brief Write the active seed into agent_config.yaml (the single source).
# @param $1 seed. Rewrites the single `seed:` line, preserving its comment.
set_seed() {
    local seed="$1"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "    [dry-run] set agent_config seed -> ${seed}"
        return
    fi
    sed -i -E "s/^seed:[[:space:]]*[0-9]+(.*)$/seed: ${seed}\1/" "${AGENT_CONFIG}"
}

# @brief Newest COMPLETE checkpoint leaf for an arm/stage/seed, else empty.
# @param $1 arm, $2 stage, $3 seed. "Complete" = the leaf holds final_model.zip,
#        so a partial crash leaf (no final_model) is ignored - it will be
#        retrained and a fresh leaf written. -t picks the youngest match.
complete_leaf() {
    local arm="$1" stage="$2" seed="$3" d
    for d in $(ls -1dt "checkpoints/${arm}/${stage}_${seed}_"* 2>/dev/null); do
        if [ -f "${d}/final_model.zip" ]; then
            basename "${d}"
            return 0
        fi
    done
    return 0  # nothing complete -> empty stdout
}

# @brief The leaf the chain should use after a stage: the real complete leaf if
#        one exists on disk, else a placeholder (only meaningful under DRY_RUN,
#        where a stage that "would run" has not actually written a leaf). Using
#        the real leaf when present keeps DRY_RUN skip decisions identical to a
#        real run - the placeholder appears only for stages a real run would
#        genuinely train next.
# @param $1 arm, $2 stage, $3 seed.
resume_leaf() {
    local arm="$1" stage="$2" seed="$3" real
    real="$(complete_leaf "${arm}" "${stage}" "${seed}")"
    if [ -n "${real}" ]; then
        echo "${real}"
    else
        echo "${stage}_${seed}_DDMMYYYY-HHMM"
    fi
}

# @brief True if an eval variant has already been written for a leaf.
# @param $1 arm, $2 leaf, $3 variant (with_wrapper|without_wrapper).
eval_done() {
    local arm="$1" leaf="$2" variant="$3" seed
    seed="$(printf '%s' "${leaf}" | cut -d_ -f2)"
    [ -f "${EVAL_ROOT}/seed_${seed}/${arm}/${leaf}/${variant}/evaluation_results.csv" ]
}

is_edl() {
    local arm="$1" e
    for e in "${EDL_ARMS[@]}"; do [ "${e}" = "${arm}" ] && return 0; done
    return 1
}

# ---------------------------------------------------------------------------
# Main: per seed -> train all arms/stages -> eval final stage -> suite tables.
# ---------------------------------------------------------------------------
echo "=================================================================="
echo " Multi-seed leg: SEEDS=${SEEDS[*]}  arms=${ARMS[*]}  stages=${STAGES[*]}"
echo " Eval: stage ${EVAL_STAGE} only, PER_STEP_CAP=${PER_STEP_CAP}"
echo " Idempotent: completed stages/evals are skipped."
[ "${DRY_RUN}" = "1" ] && echo " DRY_RUN=1: printing commands + skip decisions only."
echo "=================================================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "################## SEED ${SEED} ##################"
    set_seed "${SEED}"

    # --- Train every arm through the curriculum, resume-chained, skip-if-done.
    for arm in "${ARMS[@]}"; do
        prev_leaf=""
        for stage in "${STAGES[@]}"; do
            done_leaf="$(complete_leaf "${arm}" "${stage}" "${SEED}")"
            if [ -n "${done_leaf}" ]; then
                echo ">>> SKIP train ${arm} stage ${stage} seed ${SEED} (done: ${done_leaf})"
                prev_leaf="${done_leaf}"
                continue
            fi

            echo ">>> TRAIN ${arm} stage ${stage} seed ${SEED}"
            if [ "${stage}" = "1" ]; then
                run make docker-train LAYOUT="${LAYOUT}" STAGE="${stage}" BASELINE="${arm}"
            else
                if [ -z "${prev_leaf}" ]; then
                    echo "ERROR: no complete previous-stage leaf for ${arm} stage ${stage}" >&2
                    exit 1
                fi
                run make docker-train LAYOUT="${LAYOUT}" STAGE="${stage}" \
                    BASELINE="${arm}" CHECKPOINT="${prev_leaf}"
            fi

            # Leaf the next stage resumes from: the complete leaf just written
            # (real run), or a placeholder under DRY_RUN where nothing was made.
            prev_leaf="$(resume_leaf "${arm}" "${stage}" "${SEED}")"
            if [ "${DRY_RUN}" != "1" ] && [ -z "$(complete_leaf "${arm}" "${stage}" "${SEED}")" ]; then
                echo "ERROR: ${arm} stage ${stage} produced no complete checkpoint" >&2
                exit 1
            fi
            echo "    -> leaf: ${prev_leaf}"
        done
    done

    # --- Evaluate the FINAL stage of every arm, skip-if-done.
    declare -A FINAL_LEAF=()
    for arm in "${ARMS[@]}"; do
        # Real final-stage leaf when present (so eval-done skips match reality),
        # placeholder only under DRY_RUN when the arm was not trained on disk.
        leaf="$(resume_leaf "${arm}" "${EVAL_STAGE}" "${SEED}")"
        FINAL_LEAF["${arm}"]="${leaf}"

        if eval_done "${arm}" "${leaf}" without_wrapper; then
            echo ">>> SKIP eval ${arm} ${leaf} (without_wrapper done)"
        else
            echo ">>> EVAL ${arm} ${leaf} (without_wrapper)"
            run make docker-eval LAYOUT="${LAYOUT}" BASELINE="${arm}" \
                CHECKPOINT="${leaf}" PER_STEP_CAP="${PER_STEP_CAP}" NO_SAFETY=1
        fi

        if is_edl "${arm}"; then
            if eval_done "${arm}" "${leaf}" with_wrapper; then
                echo ">>> SKIP eval ${arm} ${leaf} (with_wrapper done)"
            else
                echo ">>> EVAL ${arm} ${leaf} (with_wrapper)"
                run make docker-eval LAYOUT="${LAYOUT}" BASELINE="${arm}" \
                    CHECKPOINT="${leaf}" PER_STEP_CAP="${PER_STEP_CAP}"
            fi
        fi
    done

    # --- Evaluation-suite tables (cheap host-side; always re-run, they overwrite
    #     in place and must reflect the full arm set once all evals exist).
    echo ">>> SUITE cross-arm: ablation + gate (stage ${EVAL_STAGE}, seed ${SEED})"
    run make analyse-ablation STAGE="${EVAL_STAGE}" SEED="${SEED}" \
        SLOPE_CLEAN=gnss_fixed SLOPE_DEGRADED=gnss_degraded
    run make analyse-gate STAGE="${EVAL_STAGE}" SEED="${SEED}"

    for arm in "${EDL_ARMS[@]}"; do
        leaf="${FINAL_LEAF[${arm}]}"
        echo ">>> SUITE EDL: calibration + handover ${arm} ${leaf}"
        run make analyse-calibration ARM="${arm}" CHECKPOINT="${leaf}"
        run make handover-timing ARM="${arm}" CHECKPOINT="${leaf}"
    done

    echo "################## SEED ${SEED} COMPLETE ##################"
done

# ---------------------------------------------------------------------------
# Cross-seed suite: pool EVERY seed's stage-EVAL_STAGE eval into the seed-robust
# headline tables (pooled bootstrap contrast + per-seed mean/range). Gated on the
# full SEEDS x ARMS matrix being evaluated so a partial leg never aggregates half
# the data. Host-side and cheap, so it always re-runs and overwrites in place once
# complete (like the per-seed suite tables). FINAL_LEAF holds only the LAST seed's
# leaves (it is declared inside the per-seed loop), so completeness is recomputed
# here in a fresh double loop.
# ---------------------------------------------------------------------------
missing=()
for SEED in "${SEEDS[@]}"; do
    for arm in "${ARMS[@]}"; do
        leaf="$(complete_leaf "${arm}" "${EVAL_STAGE}" "${SEED}")"
        if [ -z "${leaf}" ]; then
            missing+=("${SEED}/${arm} (no stage-${EVAL_STAGE} checkpoint)")
        elif ! eval_done "${arm}" "${leaf}" without_wrapper; then
            missing+=("${SEED}/${arm} (eval missing)")
        fi
    done
done

if [ "${DRY_RUN}" != "1" ] && [ "${#missing[@]}" -ne 0 ]; then
    echo ">>> SKIP cross-seed (waiting on: ${missing[*]})"
else
    echo ">>> SUITE cross-seed: pooled headline + robustness (stage ${EVAL_STAGE}, seeds ${SEEDS[*]})"
    run make analyse-cross-seed STAGE="${EVAL_STAGE}" \
        SLOPE_CLEAN=gnss_fixed SLOPE_DEGRADED=gnss_degraded
fi

echo "=================================================================="
echo " Multi-seed leg COMPLETE for seeds: ${SEEDS[*]}"
echo " agent_config.yaml left at seed: $(grep -E '^seed:' ${AGENT_CONFIG} | sed -E 's/seed:[[:space:]]*([0-9]+).*/\1/')"
echo "=================================================================="
