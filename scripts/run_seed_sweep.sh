#!/bin/bash
# Sequential multi-seed sweep for the noise floor (HANDOFF.md §6 / docs/LESION_TOKENS.md §8).
#
# Two architecturally identical runs of this config scored micro-F1 0.235 and
# 0.439. Until the run-to-run spread is quantified, no B0-vs-B2 comparison can be
# interpreted, because the spread is wider than the effect being claimed.
#
# Runs strictly one at a time: a single run already takes ~62 GB of the 80 GB A100,
# so two concurrently would OOM and take the whole sweep down with them.
#
# Usage:  bash run_seed_sweep.sh ncku_a100 [seed ...]
set -u

config_file="${1:-ncku_a100}"
shift || true
seeds=("$@")
if [ ${#seeds[@]} -eq 0 ]; then
    seeds=(1 2 3)
fi

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
log_dir="../outputs/seed_sweep_logs"
mkdir -p "$log_dir"

# train_radgenome.sh invokes a bare `python`, so the venv has to be active in this
# process for the child to inherit it. Activating here rather than relying on the
# caller's shell means an unattended launch (nohup, cron, another session) behaves
# the same as an interactive one -- otherwise it fails several seconds in with
# ModuleNotFoundError: peft, which looks like a code bug rather than a missing env.
VENV="${VENV:-/root/notebooks/automl/env}"
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -f "$VENV/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV/bin/activate"
        echo "[sweep] activated venv: $VENV"
    else
        echo "[sweep] !! no venv at $VENV and none active; set VENV=/path/to/env" >&2
        exit 1
    fi
fi
python -c 'import peft, transformers, torch' 2>/dev/null || {
    echo "[sweep] !! python at $(command -v python) cannot import peft/transformers/torch" >&2
    exit 1
}

# Offline W&B, per HANDOFF.md §10: metrics stay on this machine, which is what we
# want for clinical data, and the curves are still there for ablation comparisons.
# It is also required for an unattended run -- online mode tries to prompt for an
# API key and aborts with "api_key not configured (no-tty)" when there is no TTY.
export WANDB_MODE="${WANDB_MODE:-offline}"
echo "[sweep] WANDB_MODE=$WANDB_MODE"

echo "[sweep] config=$config_file seeds=${seeds[*]}"
echo "[sweep] started $(date -Is)"

for s in "${seeds[@]}"; do
    log="$log_dir/train_seed${s}.log"
    echo "[sweep] === seed $s -> $log ($(date -Is)) ==="

    seed=$s bash train_radgenome.sh "$config_file" > "$log" 2>&1
    status=$?

    if [ $status -ne 0 ]; then
        # Keep going: one seed dying (OOM, a transient CUDA fault -- see HANDOFF.md
        # §9) should not cost the seeds that would have run after it. The summary
        # below shows which ones actually produced checkpoints.
        echo "[sweep] !! seed $s exited $status, see $log"
    else
        echo "[sweep] seed $s done ($(date -Is))"
    fi
done

echo "[sweep] finished $(date -Is)"
echo "[sweep] checkpoints produced:"
for s in "${seeds[@]}"; do
    d="../outputs/Reg2RG_ncku_lobe5_seed${s}"
    n=$(ls -d "$d"/checkpoint-* 2>/dev/null | wc -l)
    echo "  seed $s: $n checkpoint(s) in $d"
done
