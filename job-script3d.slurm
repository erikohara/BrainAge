#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --time=11:00:00
#SBATCH --mem=64GB
#SBATCH --partition=gpu-a100 --gres=gpu:2

####### Set environment variables ###############
module load python/anaconda3-2018.12
#export MASTER_PORT=12340
echo "world size="${WORLD_SIZE}

####### Run your script #########################
source ~/software/init-conda
conda activate /home/finn.vamosi/software/miniconda3/envs/DiT
torchrun /home/finn.vamosi/3Brain/train.py --nnodes=2 --nproc_per_node=2 $1
#python -m torch.distributed.launch --use-env /home/finn.vamosi/3Brain/train.py
