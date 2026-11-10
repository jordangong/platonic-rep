#!/bin/bash -l
#SBATCH --job-name=prh-vl
#SBATCH --account=jiminxiao
#SBATCH --partition=gpua800,gpua8001t --qos=4gpus --gres=gpu:4
#SBATCH --cpus-per-gpu=8 --mem=200G --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#
# Vision-language alignment on minhuh/prh wit_1024 (paper Fig. 2 setup).
# Usage (from the repo root on hpc-2):
#   sbatch slurm/prh_vl.slurm                                  # full val set, mutual_knn
#   MODELSET=smoke sbatch --gres=gpu:1 --mem=32G --time=1:00:00 slurm/prh_vl.slurm
#   STAGES="align" METRICS="mutual_knn cka cknna" sbatch --gres=gpu:1 --mem=64G slurm/prh_vl.slurm
set -euo pipefail
cd ~/repos/plato/platonic-rep

MODELSET=${MODELSET:-val}
STAGES=${STAGES:-"vision language align"}
METRICS=${METRICS:-mutual_knn}
DATA="--dataset minhuh/prh --subset wit_1024 --modelset $MODELSET"

# everything is pre-fetched: checkpoints in ~/checkpoints, parquet in ~/datasets/minhuh--prh
export PLATO_CKPT_ROOT=~/checkpoints PLATO_DATA_ROOT=~/datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
UV=~/.local/bin/uv

echo "host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-} modelset=$MODELSET stages=$STAGES metrics=$METRICS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for stage in $STAGES; do
  case $stage in
    vision)   $UV run python extract_features.py $DATA --modality vision   --pool cls ;;
    language) $UV run python extract_features.py $DATA --modality language --pool avg ;;
    align)
      for m in $METRICS; do
        $UV run python measure_alignment.py $DATA --metric "$m" \
          --modality_x language --pool_x avg --modality_y vision --pool_y cls
      done ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
