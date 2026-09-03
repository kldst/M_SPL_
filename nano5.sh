#!/bin/bash

#SBATCH --job-name=vggt
#SBATCH --partition=8gpus
#SBATCH --time=2-00:00:00
#SBATCH --account=MST114564
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1

# Email 通知
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=crayon715@gmail.com

ml load miniconda3/24.11.1
CONDA_DEFAULT_ENV="vggt"

cd /work/crayon715/vggt/training

conda run --no-capture-output \
    -n "$CONDA_DEFAULT_ENV" \
    python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 \
    launch.py \
    --config mamma_harmony4d_mask_dpt