#!/bin/bash -l
#SBATCH --job-name=prh-vv
#SBATCH --account=jiminxiao
#SBATCH --partition=gpu4090,gpu40901t --qos=4gpus --gres=gpu:1
#SBATCH --cpus-per-gpu=4 --mem=48G --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#
# Vision-vision alignment on 1000 Places365 val images (paper Fig. 2 / App. C.1).
# Usage (from the repo root on hpc-2; pick --time / partition with hpcmon fit first):
#   sbatch slurm/prh_vv.slurm                                   # vv modelset: extract + align
#   MODELSET=vv_smoke sbatch --time=0:30:00 slurm/prh_vv.slurm
#   STAGES=align METRICS="mutual_knn cka cknna" sbatch slurm/prh_vv.slurm
# Extraction resumes per model, so short --time + resubmit is safe.
set -euo pipefail
cd ~/repos/plato/platonic-rep

MODELSET=${MODELSET:-vv}
STAGES=${STAGES:-"extract align"}
METRICS=${METRICS:-mutual_knn}
LAYERS=${LAYERS:-"final all"}
DATA="--dataset places365 --subset val_1000 --modelset $MODELSET"

# everything is pre-fetched: timm/ + emb/img/ in ~/checkpoints, ~/datasets/places365/val_256
export PLATO_CKPT_ROOT=~/checkpoints PLATO_DATA_ROOT=~/datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
UV=~/.local/bin/uv

echo "host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-} modelset=$MODELSET stages=$STAGES metrics=$METRICS layers=$LAYERS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for stage in $STAGES; do
  case $stage in
    extract) $UV run python extract_features.py $DATA --modality vision --pool cls --batch_size 50 ;;
    align)
      for m in $METRICS; do for l in $LAYERS; do
        $UV run python measure_vv_alignment.py $DATA --metric "$m" --layers "$l"
      done; done ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
