#!/bin/bash

#SBATCH --job-name=vggt		# 工作名稱
#SBATCH --partition=dev				# 使用的 partition (請根據你的系統修改)
#SBATCH --time=02:00:00				# 執行時間上限 (小時:分鐘:秒)
#SBATCH --account=MST114564	####### 請記得換成您的計畫代碼 #######
#SBATCH --nodes=1				# (-N) Maximum number of nodes to be allocated
#SBATCH --gpus-per-node=4			# Gpus per node
#SBATCH --cpus-per-task=12			# (-c) Number of cores per MPI task
#SBATCH --ntasks-per-node=1			# Maximum number of tasks on each nodes

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