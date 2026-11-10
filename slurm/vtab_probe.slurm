#!/bin/bash -l
#SBATCH --job-name=vtab-probe
#SBATCH --account=jiminxiao
#SBATCH --partition=gpu4090,gpu40901t --qos=4gpus --gres=gpu:1
#SBATCH --cpus-per-gpu=4 --mem=64G --time=6:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
#
# VTAB linear probes (paper App. C.1) for every model of $MODELSET, sharded over the array.
# Needs ~/datasets/vtab-224 from task_adaptation/scripts/export_vtab_arrays.slurm.
# Resumes per (model, task): rerun the same command after a timeout.
#   sbatch slurm/vtab_probe.slurm                      # vv_ext, 4 shards
#   MODELSET=vv_smoke sbatch --array=0 slurm/vtab_probe.slurm
# Then: uv run python analyze_vv_vtab.py --modelset vv_ext
set -euo pipefail
cd ~/repos/plato/platonic-rep

MODELSET=${MODELSET:-vv_ext}
export PLATO_CKPT_ROOT=~/checkpoints PLATO_DATA_ROOT=~/datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
echo "host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-} modelset=$MODELSET shard=$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT"
~/.local/bin/uv run python vtab_probe.py --modelset $MODELSET --shard "$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT" ${TASKS:+--tasks $TASKS}
